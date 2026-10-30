"""CPU integration: ordered plan execution against the real ToolEnv."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_pipeline import Agent, ContractError, select_plan, check_call, parse_object
from common import load_catalog, import_toolenv, evaluate_agent_case, ordered_matches
from inference_errors import InferenceBackendError, ContextLimitError

ROOT = Path(__file__).resolve().parent.parent / 'train+public_test'


def sample(sample_id='train_0236'):
    with (ROOT / 'train/train.jsonl').open() as handle:
        return next(row for line in handle if (row := json.loads(line))['id'] == sample_id)


def plan(*tools):
    return {'tools': list(tools)}


def call(**arguments):
    return {'type': 'call', 'arguments': arguments}


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

    def test_one_planner_then_fixed_calls_match_real_gold_without_done_request(self):
        tools = ['bill_payment.list_bills', 'ewallet.get_balance', 'bill_payment.pay',
                 'contacts.search', 'sms.send']
        result, llm, row = self.run_script([
            plan(*tools), call(), call(), call(bill_id='bill_01', source='ewallet'),
            call(query='Sếp Dũng'), call(contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'),
        ], save_node_inputs=True)
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['tool_plan'], tools)
        self.assertEqual((result['planner_calls'], result['caller_calls'], result['llm_calls']), (1, 5, 6))
        self.assertEqual((result['processed_steps'], result['skipped_steps']), (5, 0))
        self.assertEqual([c['name'] for c in result['tool_calls']], tools)
        case = evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)
        self.assertEqual(case['score_proxy'], 1)
        self.assertEqual(case['tool_order_recall'], 1)
        self.assertTrue(case['tool_plan_exact_match'])
        for index, (messages, schema) in enumerate(llm.inputs[1:]):
            payload = json.loads(messages[1]['content'])
            self.assertEqual(payload['selected_tool']['name'], tools[index])
            self.assertEqual(payload['step_index'], index)
            self.assertEqual(payload['tool_plan'], tools)
            self.assertEqual(len(payload['step_results']), index)
            for key in ('available_tools', 'label', 'final_state', 'reason', 'issue', 'planner_instruction'):
                self.assertNotIn(key, payload)
            self.assertNotIn('"name"', json.dumps(schema))
            self.assertTrue(all(set(f) == {'stage', 'tool', 'arguments', 'response'} for f in payload['env_feedback']))
        self.assertEqual(result['trace'][2]['node_outputs']['call']['input_messages'], llm.inputs[2][0])

    def test_plan_order_is_preserved_instead_of_sorted(self):
        tools = ['ewallet.get_balance', 'bill_payment.list_bills']
        result, _, _ = self.run_script([plan(*tools), call(), call()])
        self.assertEqual(result['tool_plan'], tools)
        self.assertEqual([c['name'] for c in result['tool_calls']], tools)
        self.assertEqual(result['stop_reason'], 'done')

    def test_conditional_slots_can_be_skipped_while_executed_branch_matches_gold(self):
        tools = ['bill_payment.list_bills', 'ewallet.get_balance', 'bill_payment.pay',
                 'contacts.search', 'sms.send', 'reminders.create']
        result, llm, row = self.run_script([
            plan(*tools), call(), call(), call(bill_id='bill_01', source='ewallet'),
            call(query='Sếp Dũng'), call(contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'),
            {'type': 'skip'},
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['processed_steps'], 6)
        self.assertEqual(result['skipped_steps'], 1)
        self.assertEqual(len(result['tool_calls']), 5)
        self.assertEqual(result['trace'][2]['tool_result']['balance'], 210009)
        case = evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)
        self.assertEqual(case['score_proxy'], 1)
        self.assertEqual(case['tool_order_recall'], 1)
        self.assertFalse(case['tool_plan_exact_match'])

    def test_lookup_observation_only_available_after_execution(self):
        result, llm, _ = self.run_script([
            plan('contacts.search', 'sms.send'), call(query='Sếp Dũng'), call(contact_id='c_04', content='text'),
        ], save_node_inputs=True)
        before = json.loads(llm.inputs[1][0][1]['content'])
        after = json.loads(llm.inputs[2][0][1]['content'])
        self.assertEqual(before['env_feedback'], [])
        self.assertEqual(after['env_feedback'][-1]['response'], result['trace'][1]['tool_result'])
        self.assertEqual(after['env_feedback'][-1]['tool'], 'contacts.search')
        self.assertNotIn('env_feedback', json.loads(llm.inputs[0][0][1]['content']))
        self.assertNotIn('step_results', json.loads(llm.inputs[0][0][1]['content']))

    def test_empty_plan_finishes_without_caller(self):
        result, llm, _ = self.run_script([plan()])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual((result['caller_calls'], result['tool_calls']), (0, []))
        self.assertEqual(len(llm.inputs), 1)

    def test_caller_cannot_choose_tool_reorder_ask_target_or_done_early(self):
        for output in ({'type': 'call', 'name': 'sms.send', 'arguments': {}},
                       {'type': 'call', 'name': 'contacts.search', 'arguments': {}},
                       {'type': 'done'}, {'tools': ['sms.send']},
                       {'type': 'ask', 'tool_name': 'sms.send', 'params': ['query']}):
            with self.subTest(output=output):
                result, llm, _ = self.run_script([plan('contacts.search', 'sms.send'), output])
                self.assertEqual(result['stop_reason'], 'invalid_model_output')
                self.assertEqual(result['tool_calls'], [])
                self.assertNotIn('sms.send', json.dumps(llm.inputs[-1][1]))

    def test_skip_advances_one_slot_without_env_call_or_fake_observation(self):
        result, llm, _ = self.run_script([
            plan('ewallet.get_balance', 'contacts.search', 'sms.send'), call(),
            {'type': 'skip'}, call(phone_number='0900000000', content='text'),
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual([c['name'] for c in result['tool_calls']], ['ewallet.get_balance', 'sms.send'])
        self.assertEqual((result['processed_steps'], result['skipped_steps']), (3, 1))
        last = json.loads(llm.inputs[-1][0][1]['content'])
        self.assertEqual(last['step_results'][-1], {'step_index': 1, 'tool': 'contacts.search', 'status': 'skipped'})
        self.assertTrue(all(f['tool'] == 'ewallet.get_balance' for f in last['env_feedback']))
        self.assertNotIn('action', result['trace'][2])

    def test_blocked_does_not_execute_skip_or_replan(self):
        result, llm, _ = self.run_script([plan('sms.send', 'contacts.search'), {'type': 'blocked'}])
        self.assertEqual(result['stop_reason'], 'caller_blocked')
        self.assertEqual(result['processed_steps'], 0)
        self.assertEqual(result['tool_calls'], [])
        self.assertEqual(len(llm.inputs), 2)

    def test_schema_and_business_failure_stop_without_retry(self):
        result, llm, _ = self.run_script([plan('sms.send'), call(content='missing recipient')])
        self.assertEqual(result['stop_reason'], 'validation_failed')
        self.assertFalse(result['trace'][-1]['validation']['ok'])
        self.assertEqual(result['tool_calls'], [])
        result, llm, _ = self.run_script([
            plan('bill_payment.pay', 'contacts.search'), call(bill_id='unknown', source='ewallet'),
        ])
        self.assertEqual(result['stop_reason'], 'tool_failed')
        self.assertTrue(result['trace'][-1]['tool_result']['ok'])
        self.assertFalse(result['trace'][-1]['tool_result']['done'])
        self.assertEqual((result['tool_failures'], len(llm.inputs)), (1, 2))

    def test_ask_target_is_current_tool_and_fields_are_real_unique_nonempty(self):
        result, _, _ = self.run_script([plan('sms.send'), {'type': 'ask', 'params': ['contact_id']}])
        self.assertEqual(result['stop_reason'], 'ask_missing_info')
        self.assertEqual(result['tool_calls'], [{'name': 'sys.ask_missing_info',
             'arguments': {'tool_name': 'sms.send', 'params': ['contact_id']}}])
        for fields in (['unknown'], ['content', 'content'], []):
            with self.assertRaises(ContractError):
                check_call({'type': 'ask', 'params': fields}, self.catalog, 'sms.send')

    def test_repeated_tool_slots_are_distinct_but_duplicate_write_is_blocked(self):
        result, _, _ = self.run_script([
            plan('sms.send', 'sms.send'), call(content='first', phone_number='0900000000'),
            call(content='second', phone_number='0900000000'),
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['tool_plan'], ['sms.send', 'sms.send'])
        self.assertEqual(result['selected_tools'], ['sms.send'])
        self.assertEqual(len(result['tool_calls']), 2)
        result, _, _ = self.run_script([
            plan('sms.send', 'sms.send'), call(content='same', phone_number='0900000000'),
            call(phone_number='0900000000', content='same'),
        ])
        self.assertEqual(result['stop_reason'], 'duplicate_write')
        self.assertEqual(len(result['tool_calls']), 1)

    def test_planner_rejects_unknown_names_prose_and_excessive_length(self):
        for output in (plan('unknown'), dict(plan('contacts.search'), reason='explanation'),
                       {'type': 'done'}, {'tools': 'contacts.search'}, plan('sys.ask_missing_info'),
                       plan(*(['contacts.search'] * 25))):
            with self.subTest(output=output):
                result, _, _ = self.run_script([output])
                self.assertEqual(result['stop_reason'], 'invalid_model_output')
                self.assertEqual(result['caller_calls'], 0)
        self.assertEqual(select_plan(plan('sms.send', 'contacts.search', 'sms.send'), self.catalog, 24),
                         ['sms.send', 'contacts.search', 'sms.send'])

    def test_extra_control_fields_and_unknown_arguments_rejected(self):
        for output in (dict(call(), reason='explanation'), {'issue': {'type': 'missing_data'}},
                       call(nonexistent='value'), {'type': 'call'}, {'type': 'skip', 'note': 'why'}):
            result, _, _ = self.run_script([plan('contacts.search'), output])
            self.assertEqual(result['stop_reason'], 'invalid_model_output')
            self.assertEqual(result['tool_calls'], [])

    def test_tool_budget_can_finish_or_skip_but_cannot_execute_extra_call(self):
        for ending, expected in (({'type': 'skip'}, 'done'), (call(), 'tool_limit')):
            result, _, _ = self.run_script([plan('ewallet.get_balance', 'ewallet.get_balance'), call(), ending],
                                           max_tool_calls=1)
            self.assertEqual(result['stop_reason'], expected)
            self.assertEqual(len(result['tool_calls']), 1)
        result, _, _ = self.run_script([plan('ewallet.get_balance'), call()], max_tool_calls=1)
        self.assertEqual(result['stop_reason'], 'done')

    def test_repeated_reads_and_total_requests_are_bounded_without_forcing_empty_plan(self):
        result, _, _ = self.run_script([plan(*(['ewallet.get_balance'] * 3))] + [call()] * 3)
        self.assertEqual(result['stop_reason'], 'environment_no_progress')
        result, llm, _ = self.run_script([plan('ewallet.get_balance')], max_decisions=1)
        self.assertEqual(result['stop_reason'], 'decision_limit')
        self.assertEqual(len(llm.inputs), 1)

    def test_hardware_errors_propagate_and_json_is_strict(self):
        for error in (InferenceBackendError('failed'), ContextLimitError('too long')):
            for outputs in ([error], [plan('ewallet.get_balance'), error]):
                with self.assertRaises(type(error)):
                    self.run_script(outputs)
        for text in ('text {}', '{} trailing', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]'):
            with self.assertRaises((ValueError, ContractError)):
                parse_object(text)

    def test_prompt_trimming_keeps_plan_current_schema_and_latest_response(self):
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
            10, 'call', tool_plan=['contacts.search'])
        payload = json.loads(msgs[1]['content'])
        self.assertTrue(cut)
        self.assertEqual(payload['env_feedback'], [{'stage': 'call', 'response': 2}])
        self.assertEqual(payload['selected_tool']['name'], 'contacts.search')
        self.assertEqual(payload['tool_plan'], ['contacts.search'])
        self.assertNotIn('secret', msgs[1]['content'])
        self.assertNotIn('label', payload)
        agent.max_prompt_tokens = 1
        with self.assertRaises(ContextLimitError):
            agent.messages(row, 'current', specs, [], 10, 'plan')

    def test_offline_order_metric_counts_repetitions_and_order(self):
        self.assertEqual(ordered_matches(['a', 'b', 'a'], ['a', 'a']), 2)
        self.assertEqual(ordered_matches(['b', 'a'], ['a', 'b']), 1)
        self.assertEqual(ordered_matches([], []), 0)

    def test_runner_reports_order_steps_and_two_roles(self):
        from run import agent_experiment
        row, llm = sample(), ScriptedLLM([plan('ewallet.get_balance'), call()])
        agent = Agent(llm, self.catalog, self.ToolEnv, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            agent_experiment([row], {row['id']: row['audio_text']}, ROOT, directory,
                             'workflow_gold', agent, self.catalog, self.ToolEnv)
            dest = Path(directory) / 'workflow_gold'
            trace = json.loads((dest / 'traces.jsonl').read_text())
            summary = json.loads((dest / 'summary.json').read_text())
            self.assertEqual(trace['tool_plan'], ['ewallet.get_balance'])
            self.assertEqual(trace['processed_steps'], 1)
            self.assertEqual(summary['architecture'], 'ordered_tool_plan_caller')
            self.assertFalse(summary['caller_tool_choice'])
            self.assertEqual(summary['planner_calls_per_sample']['mean'], 1)
            self.assertEqual(summary['caller_calls_per_sample']['mean'], 1)
            self.assertIn('mean_tool_order_recall', summary)
            self.assertEqual(json.loads((dest / 'run_status.json').read_text())['status'], 'complete')
            self.assertNotIn('replans_per_sample', summary)
        self.assertNotIn('input_messages', trace['trace'][0]['node_outputs']['plan'])


if __name__ == '__main__':
    unittest.main()
