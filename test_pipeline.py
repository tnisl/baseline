"""Ordered execution and replanning integration against the real ToolEnv."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_pipeline import Agent, ContractError, check_plan, select_fields, parse_object
from common import load_catalog, import_toolenv, evaluate_agent_case
from inference_errors import InferenceBackendError, ContextLimitError

ROOT = Path(__file__).resolve().parent.parent / 'train+public_test'


def sample(sample_id='train_0236'):
    with (ROOT / 'train/train.jsonl').open() as handle:
        return next(row for line in handle if (row := json.loads(line))['id'] == sample_id)


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


def plan(*tools, after='done'):
    return {'type': 'plan', 'tools': list(tools), 'after': after}


def args(*fields):
    return {'selected_arguments': list(fields)}


def vals(**fields):
    return {'arguments': fields}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)
        cls.ToolEnv = staticmethod(import_toolenv(ROOT))

    def run_script(self, outputs, **limits):
        llm, row = ScriptedLLM(outputs), sample()
        result = Agent(llm, self.catalog, self.ToolEnv, ROOT, **limits).run(row, row['audio_text'])
        return result, llm, row

    def test_single_plan_executes_full_order_with_no_per_tool_planner(self):
        tools = ['bill_payment.list_bills', 'ewallet.get_balance', 'bill_payment.pay', 'contacts.search', 'sms.send']
        result, llm, row = self.run_script([
            plan(*tools), args(), vals(bill_id='bill_01', source='ewallet'),
            args('query'), vals(query='Sếp Dũng'), args('contact_id'),
            vals(contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'),
        ], plan_mode='fixed')
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['planner_calls'], 1)
        self.assertEqual(result['replans'], 0)
        self.assertEqual([c['name'] for c in result['tool_calls']], tools)
        self.assertEqual(len(llm.inputs), 7)
        self.assertEqual(result['decisions'], 6)
        self.assertEqual(evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)['score_proxy'], 1)
        self.assertTrue(result['trace'][2]['argument_selector_skipped'])
        self.assertTrue(result['trace'][2]['value_selector_skipped'])
        for messages, _ in llm.inputs[1:]:
            payload = json.loads(messages[1]['content'])
            self.assertEqual(payload['execution_plan']['tools'], tools)
            self.assertEqual(payload['selected_tool']['name'], tools[payload['step_index']])
            self.assertNotIn('available_tools', payload)
            self.assertNotIn('label', payload)

    def test_declared_barrier_replans_after_reads_and_before_branch_write(self):
        result, llm, row = self.run_script([
            plan('bill_payment.list_bills', 'ewallet.get_balance', after='replan'), args(),
            plan('bill_payment.pay', 'contacts.search', 'sms.send'),
            vals(bill_id='bill_01', source='ewallet'),
            args('query'), vals(query='Sếp Dũng'), args('contact_id'),
            vals(contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'),
        ])
        self.assertEqual(result['planner_calls'], 2)
        self.assertEqual(result['replans'], 1)
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)['score_proxy'], 1)
        payload = json.loads(llm.inputs[2][0][1]['content'])
        self.assertEqual(payload['previous_plan']['next_step'], 2)
        self.assertEqual(payload['env_feedback'][-1]['tool'], 'ewallet.get_balance')
        self.assertEqual(payload['env_feedback'][-1]['response']['balance'], 210009)

    def test_missing_value_discards_unexecuted_tail_then_replans_from_real_validation(self):
        result, llm, _ = self.run_script([
            plan('sms.send', 'bill_payment.pay'), args('contact_id'), vals(content='Đã đóng'),
            plan('contacts.search'), args('query'), vals(query='Sếp Dũng'),
        ])
        self.assertEqual(result['repair_attempts'], 1)
        self.assertEqual([c['name'] for c in result['tool_calls']], ['contacts.search'])
        payload = json.loads(llm.inputs[3][0][1]['content'])
        expected = self.ToolEnv('train', sample()['id'], root=ROOT).validate('sms.send', {'content': 'Đã đóng'})
        self.assertEqual(payload['env_feedback'], [{'stage': 'validate', 'tool': 'sms.send',
            'arguments': {'content': 'Đã đóng'}, 'response': expected}])
        self.assertEqual(payload['previous_plan']['next_step'], 0)

    def test_business_failure_is_not_advanced_to_next_planned_tool(self):
        result, llm, _ = self.run_script([
            plan('bill_payment.pay', 'contacts.search'), vals(bill_id='unknown', source='ewallet'),
            {'type': 'ask', 'tool_name': 'bill_payment.pay', 'params': ['bill_id']},
        ])
        self.assertEqual(result['stop_reason'], 'ask_missing_info')
        self.assertEqual([c['name'] for c in result['tool_calls']], ['bill_payment.pay', 'sys.ask_missing_info'])
        response = result['trace'][1]['tool_result']
        self.assertTrue(response['ok'])
        self.assertFalse(response['done'])
        self.assertEqual(json.loads(llm.inputs[-1][0][1]['content'])['env_feedback'][-1]['response'], response)

    def test_fixed_mode_stops_on_env_validation_or_business_error(self):
        for script, reason in [
            ([plan('sms.send', 'contacts.search'), args('contact_id'), vals(content='text')], 'validation_failed'),
            ([plan('bill_payment.pay', 'contacts.search'), vals(bill_id='unknown', source='ewallet')], 'tool_failed')]:
            with self.subTest(reason=reason):
                result, _, _ = self.run_script(script, plan_mode='fixed')
                self.assertEqual(result['stop_reason'], reason)
                self.assertEqual(result['planner_calls'], 1)
                self.assertNotIn('contacts.search', [c['name'] for c in result['tool_calls']])

    def test_repeated_tool_names_are_allowed_and_step_index_distinguishes_occurrences(self):
        result, llm, _ = self.run_script([
            plan('sms.send', 'sms.send'), args('phone_number'), vals(content='first', phone_number='0900000000'),
            args('phone_number'), vals(content='second', phone_number='0900000000'),
        ], plan_mode='fixed')
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(len(result['tool_calls']), 2)
        self.assertEqual([json.loads(llm.inputs[i][0][1]['content'])['step_index'] for i in (1, 3)], [0, 1])

    def test_duplicate_write_is_local_stop_without_model_feedback(self):
        result, _, _ = self.run_script([
            plan('sms.send', 'sms.send'), args('phone_number'), vals(content='same', phone_number='0900000000'),
            args('phone_number'), vals(phone_number='0900000000', content='same'),
        ])
        self.assertEqual(result['stop_reason'], 'duplicate_write')
        self.assertEqual(len(result['tool_calls']), 1)

    def test_ask_and_direct_done_need_only_one_planner_call(self):
        result, _, _ = self.run_script([{'type': 'ask', 'tool_name': 'sms.send', 'params': ['contact_id']}])
        self.assertEqual(result['stop_reason'], 'ask_missing_info')
        result, _, _ = self.run_script([{'type': 'done'}])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['plans'], [])

    def test_contract_rejects_reason_unknown_tools_empty_plan_and_fixed_barrier(self):
        for output in (dict(plan('contacts.search'), reason='text'), plan('unknown'), plan(),
                       {'type': 'call', 'name': 'contacts.search'}, plan('contacts.search', after='replan')):
            with self.subTest(output=output):
                result, _, _ = self.run_script([output], plan_mode='fixed')
                self.assertEqual(result['stop_reason'], 'invalid_model_output')
                self.assertEqual(result['tool_calls'], [])
        for script in ([plan('sms.send'), {'issue': {'reason': 'text'}}],
                       [plan('bill_payment.pay'), {'arguments': {}, 'value_reasons': {}}]):
            result, _, _ = self.run_script(script)
            self.assertEqual(result['stop_reason'], 'invalid_model_output')

    def test_repeated_reads_and_validation_are_bounded_across_replans(self):
        result, _, _ = self.run_script([plan('ewallet.get_balance', 'ewallet.get_balance', 'ewallet.get_balance')])
        self.assertEqual(result['stop_reason'], 'environment_no_progress')
        result, _, _ = self.run_script([x for _ in range(3) for x in [plan('sms.send'), args('contact_id'), vals()]], max_repairs=3)
        self.assertEqual(result['stop_reason'], 'validation_no_progress')
        self.assertEqual(result['tool_calls'], [])

    def test_exact_budget_and_decision_limit_complete_without_extra_done_call(self):
        result, llm, _ = self.run_script([plan('ewallet.get_balance')], max_tool_calls=1, max_decisions=2)
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(len(llm.inputs), 1)
        result, _, _ = self.run_script([plan('ewallet.get_balance', after='replan')], max_tool_calls=1)
        self.assertEqual(result['stop_reason'], 'tool_limit')
        result, _, _ = self.run_script([plan('ewallet.get_balance', 'ewallet.get_balance')], max_tool_calls=1)
        self.assertEqual(result['stop_reason'], 'invalid_model_output')

    def test_hardware_errors_propagate_and_json_is_strict(self):
        for error in (InferenceBackendError('failed'), ContextLimitError('too long')):
            with self.assertRaises(type(error)):
                self.run_script([error])
        for text in ('text {}', '{} trailing', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]'):
            with self.assertRaises((ValueError, ContractError)):
                parse_object(text)

    def test_field_rules_remain_required_plus_extras(self):
        self.assertEqual(select_fields(args('contact_id'), self.catalog['sms.send']), ['content', 'contact_id'])
        with self.assertRaises(ContractError):
            select_fields(args('content'), self.catalog['sms.send'])
        with self.assertRaises(ContractError):
            select_fields(args('phone_number', 'phone_number'), self.catalog['sms.send'])

    def test_prompt_trimming_keeps_plan_latest_feedback_and_hides_state(self):
        class LongLLM(ScriptedLLM):
            def count(self, messages):
                p = json.loads(messages[1]['content'])
                return 100 + 10 * len(p['history']) + 10 * len(p['env_feedback'])
        row = sample()
        row['history'] = [{'role': 'user', 'audio_text': 'older'}] * 3
        row['context']['installed_apps']['hidden'] = {'tools': [], 'state': {'secret': 999}}
        agent = Agent(LongLLM([]), self.catalog, self.ToolEnv, ROOT, max_prompt_tokens=110)
        msgs, cut = agent.messages(row, 'current', {}, [{'stage': 'call', 'response': 1}, {'stage': 'call', 'response': 2}],
                                   10, 'plan', execution_plan={'tools': ['contacts.search'], 'after': 'done'})
        payload = json.loads(msgs[1]['content'])
        self.assertTrue(cut)
        self.assertEqual(len(payload['env_feedback']), 1)
        self.assertEqual(payload['previous_plan']['tools'], ['contacts.search'])
        self.assertNotIn('secret', msgs[1]['content'])
        self.assertNotIn('label', payload)

    def test_runner_reports_plan_and_planner_cost(self):
        from run import agent_experiment
        row, llm = sample(), ScriptedLLM([plan('ewallet.get_balance')])
        agent = Agent(llm, self.catalog, self.ToolEnv, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            agent_experiment([row], {row['id']: row['audio_text']}, ROOT, directory,
                             'workflow_gold', agent, self.catalog, self.ToolEnv)
            dest = Path(directory) / 'workflow_gold'
            trace = json.loads((dest / 'traces.jsonl').read_text())
            summary = json.loads((dest / 'summary.json').read_text())
            self.assertEqual(len(trace['plans']), 1)
            self.assertEqual(summary['plan_mode'], 'adaptive')
            self.assertEqual(summary['planner_calls_per_sample']['mean'], 1)
            self.assertEqual(json.loads((dest / 'run_status.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
