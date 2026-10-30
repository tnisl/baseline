"""CPU integration: scripted node outputs, real environment, no GPU weights."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_pipeline import Agent, ContractError, select_tools, check_call, parse_object
from common import load_catalog, import_toolenv, evaluate_agent_case
from inference_errors import InferenceBackendError, ContextLimitError

ROOT = Path(__file__).resolve().parent.parent / 'train+public_test'


def sample(sample_id='train_0236'):
    with (ROOT / 'train/train.jsonl').open() as handle:
        return next(row for line in handle if (row := json.loads(line))['id'] == sample_id)


def pool(*tools):
    return {'tools': list(tools)}


def call(name, **arguments):
    return {'type': 'call', 'name': name, 'arguments': arguments}


class ScriptedLLM:
    model_id, backend, dtype, temperature, seed = 'scripted', 'test', 'none', 0, 42
    structured_outputs = True

    def __init__(self, outputs):
        self.outputs, self.inputs = iter(outputs), []

    def count(self, messages):
        return 100

    def complete(self, messages, max_new_tokens=512, output_schema=None):
        self.inputs.append((deepcopy(messages), deepcopy(output_schema)))
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return {'text': output if isinstance(output, str) else json.dumps(output, ensure_ascii=False),
                'prompt_tokens': self.count(messages), 'generation_tokens': 5, 'latency_s': .01}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)
        cls.ToolEnv = staticmethod(import_toolenv(ROOT))

    def run_script(self, outputs, **options):
        llm, row = ScriptedLLM(outputs), sample()
        outcome = Agent(llm, self.catalog, self.ToolEnv, ROOT, **options).run(row, row['audio_text'])
        return outcome, llm, row

    def test_one_planner_and_complete_caller_actions_match_real_gold(self):
        tools = ['sms.send', 'contacts.search', 'bill_payment.pay', 'ewallet.get_balance', 'bill_payment.list_bills']
        result, llm, row = self.run_script([
            pool(*tools), call('bill_payment.list_bills'), call('ewallet.get_balance'),
            call('bill_payment.pay', bill_id='bill_01', source='ewallet'),
            call('contacts.search', query='Sếp Dũng'),
            call('sms.send', contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'), {'type': 'done'},
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['selected_tools'], sorted(tools))
        self.assertEqual(result['planner_calls'], 1)
        self.assertEqual(result['caller_calls'], 6)
        self.assertEqual(result['llm_calls'], 7)
        self.assertEqual(result['decisions'], 7)
        self.assertEqual([c['name'] for c in result['tool_calls']], [
            'bill_payment.list_bills', 'ewallet.get_balance', 'bill_payment.pay', 'contacts.search', 'sms.send'])
        case = evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)
        self.assertEqual(case['score_proxy'], 1)
        self.assertEqual(case['tool_set_recall'], 1)
        for event in result['trace']:
            self.assertEqual(len(event['node_outputs']), 1)
            self.assertIn(event['event_type'], ('plan', 'call'))
        for messages, _ in llm.inputs[1:]:
            payload = json.loads(messages[1]['content'])
            self.assertEqual({s['name'] for s in payload['available_tools']}, set(tools))
            self.assertEqual(payload['selected_tools'], sorted(tools))
            for key in ('label', 'task_plan', 'reason', 'issue', 'selected_arguments', 'value_reasons'):
                self.assertNotIn(key, payload)
            self.assertTrue(all(set(f) == {'stage', 'tool', 'arguments', 'response'} for f in payload['env_feedback']))

    def test_caller_can_use_subset_and_choose_order_independently(self):
        result, _, _ = self.run_script([
            pool('contacts.search', 'ewallet.get_balance'), call('ewallet.get_balance'), {'type': 'done'},
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual([x['name'] for x in result['tool_calls']], ['ewallet.get_balance'])
        self.assertEqual(result['planner_calls'], 1)

    def test_lookup_observation_is_exact_and_only_available_after_execution(self):
        result, llm, _ = self.run_script([
            pool('contacts.search', 'sms.send'), call('contacts.search', query='Sếp Dũng'),
            call('sms.send', contact_id='c_04', content='text'), {'type': 'done'},
        ], save_node_inputs=True)
        before = json.loads(llm.inputs[1][0][1]['content'])
        after = json.loads(llm.inputs[2][0][1]['content'])
        self.assertEqual(before['env_feedback'], [])
        self.assertEqual(after['env_feedback'][-1]['response'], result['trace'][1]['tool_result'])
        self.assertEqual(after['env_feedback'][-1]['stage'], 'call')
        for event, (messages, _) in zip(result['trace'], llm.inputs):
            node = event['event_type']
            self.assertEqual(event['node_outputs'][node]['input_messages'], messages)
        self.assertNotIn('env_feedback', json.loads(llm.inputs[0][0][1]['content']))

    def test_planner_empty_set_finishes_without_caller(self):
        result, llm, _ = self.run_script([pool()])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['caller_calls'], 0)
        self.assertEqual(result['tool_calls'], [])
        self.assertEqual(len(llm.inputs), 1)

    def test_tool_outside_pool_never_validated_or_executed(self):
        result, llm, _ = self.run_script([pool('contacts.search'), call('sms.send', phone_number='0900000000', content='text')])
        self.assertEqual(result['stop_reason'], 'invalid_model_output')
        self.assertEqual(result['tool_calls'], [])
        self.assertEqual(len(llm.inputs), 2)
        self.assertNotIn('sms.send', json.dumps(llm.inputs[-1][1]))

    def test_schema_failure_stops_without_retry_or_replan(self):
        result, llm, _ = self.run_script([pool('sms.send'), call('sms.send', content='missing recipient')])
        self.assertEqual(result['stop_reason'], 'validation_failed')
        self.assertFalse(result['trace'][-1]['validation']['ok'])
        self.assertEqual(result['tool_calls'], [])
        self.assertEqual(len(llm.inputs), 2)

    def test_business_failure_stops_even_when_ok_true(self):
        result, llm, _ = self.run_script([
            pool('bill_payment.pay', 'contacts.search'), call('bill_payment.pay', bill_id='unknown', source='ewallet'),
        ])
        self.assertEqual(result['stop_reason'], 'tool_failed')
        self.assertTrue(result['trace'][-1]['tool_result']['ok'])
        self.assertFalse(result['trace'][-1]['tool_result']['done'])
        self.assertEqual(result['tool_failures'], 1)
        self.assertEqual(len(llm.inputs), 2)

    def test_ask_uses_real_fields_in_selected_target_tool(self):
        result, _, _ = self.run_script([pool('sms.send'), {'type': 'ask', 'tool_name': 'sms.send', 'params': ['contact_id']}])
        self.assertEqual(result['stop_reason'], 'ask_missing_info')
        self.assertEqual(result['tool_calls'], [{'name': 'sys.ask_missing_info',
             'arguments': {'tool_name': 'sms.send', 'params': ['contact_id']}}])
        for action in ({'type': 'ask', 'tool_name': 'sms.send', 'params': ['unknown']},
                       {'type': 'ask', 'tool_name': 'sms.send', 'params': ['content', 'content']},
                       {'type': 'call', 'name': 'sys.ask_missing_info', 'arguments': {}}):
            with self.assertRaises(ContractError):
                check_call(action, {'sms.send': self.catalog['sms.send']})

    def test_planner_rejects_duplicate_unknown_tools_and_control_text(self):
        for output in (pool('contacts.search', 'contacts.search'), pool('unknown'),
                       dict(pool('contacts.search'), reason='explanation'),
                       {'type': 'done'}, {'tools': 'contacts.search'}, pool('sys.ask_missing_info')):
            with self.subTest(output=output):
                result, _, _ = self.run_script([output])
                self.assertEqual(result['stop_reason'], 'invalid_model_output')
                self.assertEqual(result['caller_calls'], 0)
        self.assertEqual(select_tools(pool('sms.send', 'contacts.search'), self.catalog), ['contacts.search', 'sms.send'])

    def test_caller_rejects_extra_control_fields_and_unknown_arguments(self):
        for output in (dict(call('contacts.search'), tool_reason='explanation'),
                       {'issue': {'type': 'missing_data'}},
                       call('contacts.search', nonexistent='value'),
                       {'type': 'call', 'name': 'contacts.search'},
                       {'type': 'done', 'note': 'finished'}):
            with self.subTest(output=output):
                result, _, _ = self.run_script([pool('contacts.search'), output])
                self.assertEqual(result['stop_reason'], 'invalid_model_output')
                self.assertEqual(result['tool_calls'], [])

    def test_same_tool_can_run_for_multiple_objects_but_duplicate_write_stops(self):
        result, _, _ = self.run_script([
            pool('sms.send'), call('sms.send', content='first', phone_number='0900000000'),
            call('sms.send', content='second', phone_number='0900000000'), {'type': 'done'},
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(len(result['tool_calls']), 2)
        result, _, _ = self.run_script([
            pool('sms.send'), call('sms.send', content='same', phone_number='0900000000'),
            call('sms.send', phone_number='0900000000', content='same'),
        ])
        self.assertEqual(result['stop_reason'], 'duplicate_write')
        self.assertEqual(len(result['tool_calls']), 1)

    def test_tool_budget_permits_done_but_blocks_further_execution(self):
        for ending, expected in (({'type': 'done'}, 'done'), (call('ewallet.get_balance'), 'tool_limit')):
            result, _, _ = self.run_script([
                pool('ewallet.get_balance'), call('ewallet.get_balance'), ending,
            ], max_tool_calls=1)
            self.assertEqual(result['stop_reason'], expected)
            self.assertEqual(len(result['tool_calls']), 1)

    def test_repeated_reads_and_total_requests_are_bounded(self):
        result, _, _ = self.run_script([pool('ewallet.get_balance')] + [call('ewallet.get_balance')] * 3)
        self.assertEqual(result['stop_reason'], 'environment_no_progress')
        result, llm, _ = self.run_script([pool('ewallet.get_balance')], max_decisions=1)
        self.assertEqual(result['stop_reason'], 'decision_limit')
        self.assertEqual(len(llm.inputs), 1)

    def test_hardware_errors_propagate_and_json_is_strict(self):
        for error in (InferenceBackendError('failed'), ContextLimitError('too long')):
            with self.assertRaises(type(error)):
                self.run_script([error])
        for text in ('text {}', '{} trailing', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]'):
            with self.assertRaises((ValueError, ContractError)):
                parse_object(text)

    def test_prompt_trimming_keeps_tool_set_latest_response_and_hides_state(self):
        class LongLLM(ScriptedLLM):
            def count(self, messages):
                p = json.loads(messages[1]['content'])
                return 100 + 10 * len(p['history']) + 10 * len(p.get('env_feedback', []))
        row = sample()
        row['history'] = [{'role': 'user', 'audio_text': 'older'}] * 3
        row['context']['installed_apps']['hidden'] = {'tools': [], 'state': {'secret': 999}}
        agent = Agent(LongLLM([]), self.catalog, self.ToolEnv, ROOT, max_prompt_tokens=110)
        specs = {'contacts.search': self.catalog['contacts.search']}
        msgs, cut = agent.messages(row, 'current', specs,
            [{'stage': 'call', 'response': 1}, {'stage': 'call', 'response': 2}],
            10, 'call', selected_tools=['contacts.search'])
        payload = json.loads(msgs[1]['content'])
        self.assertTrue(cut)
        self.assertEqual(payload['env_feedback'], [{'stage': 'call', 'response': 2}])
        self.assertEqual(payload['selected_tools'], ['contacts.search'])
        self.assertNotIn('secret', msgs[1]['content'])
        self.assertNotIn('label', payload)
        agent.max_prompt_tokens = 1
        with self.assertRaises(ContextLimitError):
            agent.messages(row, 'current', specs, [], 10, 'plan')

    def test_runner_reports_two_roles_and_pool_coverage(self):
        from run import agent_experiment
        row, llm = sample(), ScriptedLLM([pool('ewallet.get_balance'), call('ewallet.get_balance'), {'type': 'done'}])
        agent = Agent(llm, self.catalog, self.ToolEnv, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            agent_experiment([row], {row['id']: row['audio_text']}, ROOT, directory,
                             'workflow_gold', agent, self.catalog, self.ToolEnv)
            dest = Path(directory) / 'workflow_gold'
            trace = json.loads((dest / 'traces.jsonl').read_text())
            summary = json.loads((dest / 'summary.json').read_text())
            self.assertEqual(trace['selected_tools'], ['ewallet.get_balance'])
            self.assertEqual(summary['architecture'], 'planner_tool_set_caller')
            self.assertEqual(summary['planner_calls_per_sample']['mean'], 1)
            self.assertEqual(summary['caller_calls_per_sample']['mean'], 2)
            self.assertEqual(json.loads((dest / 'run_status.json').read_text())['status'], 'complete')
            self.assertNotIn('replans_per_sample', summary)
        self.assertNotIn('input_messages', llm.inputs[0][0][0])


if __name__ == '__main__':
    unittest.main()
