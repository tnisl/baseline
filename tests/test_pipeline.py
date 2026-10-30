"""CPU integration against real ToolEnv; the LLM is scripted, not fine-tuned."""
import json
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, ScriptedLLM, sample, plan, call
from lifemate_v6.pipeline import Agent, ContractError, check_plan, check_call, parse_object
from lifemate_v6.common import load_catalog, import_toolenv, evaluate_agent_case
from lifemate_v6.errors import InferenceBackendError, ContextLimitError


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)
        cls.ToolEnv = staticmethod(import_toolenv(ROOT))

    def run_script(self, outputs, **options):
        llm, row = ScriptedLLM(outputs), sample()
        outcome = Agent(llm, self.catalog, self.ToolEnv, ROOT, **options).run(row, row['audio_text'])
        return outcome, llm, row

    def test_full_workflow_alternates_planner_and_caller_and_matches_gold(self):
        result, llm, row = self.run_script([
            plan('Tra hóa đơn truyền hình.'), call('bill_payment.list_bills'),
            plan('Tra số dư ví.'), call('ewallet.get_balance'),
            plan('Thanh toán hóa đơn truyền hình theo nhánh đã thỏa mãn.'),
            call('bill_payment.pay', bill_id='bill_01', source='ewallet'),
            plan('Tìm liên hệ người nhận SMS.'), call('contacts.search', query='Sếp Dũng'),
            plan('Gửi SMS thông báo đã đóng tiền.'),
            call('sms.send', contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'), {'next': 'done'},
        ], save_node_inputs=True)
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['planner_calls'], 6)
        self.assertEqual(result['caller_calls'], 5)
        self.assertEqual(result['llm_calls'], 11)
        self.assertEqual(result['decisions'], 6)
        self.assertEqual(evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)['score_proxy'], 1)
        for event in result['trace'][:-1]:
            self.assertEqual(set(event['node_outputs']), {'plan', 'call'})
            caller_input = json.loads(event['node_outputs']['call']['input_messages'][1]['content'])
            self.assertEqual(caller_input['planner_instruction'], event['planner_output']['instruction'])
        self.assertEqual(set(result['trace'][-1]['node_outputs']), {'plan'})

    def test_next_planner_gets_exact_observations_not_previous_instruction_as_feedback(self):
        instruction = 'Chỉ dẫn tạm thời không phải observation.'
        result, llm, _ = self.run_script([plan(instruction), call('ewallet.get_balance'), {'next': 'done'}])
        p0, c0, p1 = [json.loads(messages[1]['content']) for messages, _ in llm.inputs]
        self.assertEqual(p0['env_feedback'], [])
        self.assertEqual(c0['planner_instruction'], instruction)
        expected = result['trace'][0]['tool_result']
        self.assertEqual(p1['env_feedback'][-1], {'stage': 'call', 'tool': 'ewallet.get_balance',
                                                'arguments': {}, 'response': expected})
        self.assertNotIn(instruction, json.dumps(p1, ensure_ascii=False))
        for key in ('selected_tools', 'label', 'final_state', 'previous_plan', 'task_plan'):
            self.assertNotIn(key, p1)

    def test_caller_chooses_tool_and_arguments_without_planner_whitelist(self):
        result, llm, _ = self.run_script([
            plan('Tra dữ kiện phù hợp để thực hiện yêu cầu.'), call('contacts.search', query='Dũng'), {'next': 'done'},
        ])
        p, c = [json.loads(messages[1]['content']) for messages, _ in llm.inputs[:2]]
        self.assertEqual({s['name'] for s in p['available_tools']}, {s['name'] for s in c['available_tools']})
        self.assertIn('parameters', next(s for s in c['available_tools'] if s['name'] == 'contacts.search'))
        self.assertEqual(result['tool_calls'][0]['arguments'], {'query': 'Dũng'})

    def test_only_planner_can_done_or_give_up(self):
        for next_role in ('done', 'give_up'):
            result, llm, _ = self.run_script([{'next': next_role}])
            self.assertEqual(result['stop_reason'], next_role)
            self.assertEqual(result['caller_calls'], 0)
            self.assertEqual(len(llm.inputs), 1)
        result, _, _ = self.run_script([plan(), {'type': 'done'}])
        self.assertEqual(result['stop_reason'], 'invalid_model_output')
        self.assertEqual(result['tool_calls'], [])

    def test_ask_is_a_real_system_tool_call_with_valid_target_fields(self):
        result, _, _ = self.run_script([plan('Yêu cầu thông tin người nhận SMS.'),
            call('sys.ask_missing_info', tool_name='sms.send', params=['contact_id'])])
        self.assertEqual(result['stop_reason'], 'ask_missing_info')
        self.assertEqual(result['tool_calls'], [call('sys.ask_missing_info', tool_name='sms.send', params=['contact_id'])])
        for fields in (['unknown'], ['content', 'content'], []):
            with self.assertRaises(ContractError):
                check_call(call('sys.ask_missing_info', tool_name='sms.send', params=fields), self.catalog)

    def test_invalid_guidance_and_old_contracts_stop_without_execution(self):
        for output in (plan(''), plan(' '), plan('x' * 241), plan('two\nlines'),
                       dict(plan(), reason='why'), {'tools': ['contacts.search']},
                       {'next': 'done', 'instruction': 'extra'}):
            with self.subTest(output=output):
                result, _, _ = self.run_script([output])
                self.assertEqual(result['stop_reason'], 'invalid_model_output')
                self.assertEqual(result['caller_calls'], 0)
        for output in (dict(call('contacts.search'), reason='why'), call('unknown'),
                       call('contacts.search', contact_id='wrong field'),
                       {'type': 'call', **call('contacts.search')}):
            result, _, _ = self.run_script([plan(), output])
            self.assertEqual(result['stop_reason'], 'invalid_model_output')
            self.assertEqual(result['tool_calls'], [])

    def test_validation_error_stops_without_automatic_repair(self):
        result, llm, _ = self.run_script([plan(), call('sms.send', content='missing target')])
        self.assertEqual(result['stop_reason'], 'validation_failed')
        self.assertFalse(result['trace'][-1]['validation']['ok'])
        self.assertEqual(result['tool_calls'], [])
        self.assertEqual(len(llm.inputs), 2)

    def test_business_failure_stops_instead_of_being_rewritten_as_success(self):
        result, llm, _ = self.run_script([plan(), call('bill_payment.pay', bill_id='unknown', source='ewallet')])
        self.assertEqual(result['stop_reason'], 'tool_failed')
        self.assertTrue(result['trace'][-1]['tool_result']['ok'])
        self.assertFalse(result['trace'][-1]['tool_result']['done'])
        self.assertEqual(len(llm.inputs), 2)

    def test_duplicate_successful_write_and_repeated_reads_are_bounded(self):
        result, _, _ = self.run_script([
            plan(), call('sms.send', content='same', phone_number='0900000000'),
            plan(), call('sms.send', phone_number='0900000000', content='same')])
        self.assertEqual(result['stop_reason'], 'duplicate_write')
        self.assertEqual(len(result['tool_calls']), 1)
        result, _, _ = self.run_script([item for _ in range(3) for item in [plan(), call('ewallet.get_balance')]])
        self.assertEqual(result['stop_reason'], 'environment_no_progress')

    def test_tool_budget_allows_planner_done_and_blocks_next_caller(self):
        for ending, stop in (({'next': 'done'}, 'done'), (plan(), 'tool_limit')):
            result, llm, _ = self.run_script([plan(), call('ewallet.get_balance'), ending], max_tool_calls=1)
            self.assertEqual(result['stop_reason'], stop)
            self.assertEqual(result['caller_calls'], 1)
            self.assertEqual(len(llm.inputs), 3)
        result, _, _ = self.run_script([plan(), call('ewallet.get_balance')], max_decisions=1)
        self.assertEqual(result['stop_reason'], 'decision_limit')

    def test_infrastructure_errors_propagate_and_parser_rejects_non_json(self):
        for error in (InferenceBackendError('failed'), ContextLimitError('too long')):
            for script in ([error], [plan(), error]):
                with self.assertRaises(type(error)):
                    self.run_script(script)
        for text in ('text {}', '{} trailing', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]'):
            with self.assertRaises((ValueError, ContractError)):
                parse_object(text)

    def test_prompt_trimming_keeps_instruction_schema_and_latest_observation(self):
        class LongLLM(ScriptedLLM):
            def count(self, messages):
                p = json.loads(messages[1]['content'])
                return 100 + 10 * len(p['history']) + 10 * len(p['env_feedback'])
        row = sample()
        row['history'] = [{'role': 'user', 'audio_text': 'older'}] * 3
        row['context']['installed_apps']['hidden'] = {'tools': [], 'state': {'secret': 999}}
        agent = Agent(LongLLM([]), self.catalog, self.ToolEnv, ROOT, max_prompt_tokens=110)
        specs = {'contacts.search': self.catalog['contacts.search']}
        msgs, cut = agent.messages(row, 'current', specs,
            [{'stage': 'call', 'response': 1}, {'stage': 'call', 'response': 2}], 10, 'call', instruction='Current step')
        payload = json.loads(msgs[1]['content'])
        self.assertTrue(cut)
        self.assertEqual(payload['env_feedback'], [{'stage': 'call', 'response': 2}])
        self.assertEqual(payload['planner_instruction'], 'Current step')
        self.assertNotIn('secret', msgs[1]['content'])
        agent.max_prompt_tokens = 1
        with self.assertRaises(ContextLimitError):
            agent.messages(row, 'current', specs, [], 10, 'plan')

    def test_trace_input_capture_is_opt_in(self):
        result, _, _ = self.run_script([{'next': 'done'}])
        self.assertNotIn('input_messages', result['trace'][0]['node_outputs']['plan'])
        result, llm, _ = self.run_script([plan(), call('ewallet.get_balance'), {'next': 'done'}], save_node_inputs=True)
        self.assertEqual(result['trace'][0]['node_outputs']['call']['input_messages'], llm.inputs[1][0])

    def test_runner_report_has_round_and_node_costs_without_pool_metrics(self):
        from lifemate_v6.runner import agent_experiment, workflow_sha256
        row = sample()
        agent = Agent(ScriptedLLM([plan(), call('ewallet.get_balance'), {'next': 'done'}]),
                      self.catalog, self.ToolEnv, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            agent_experiment([row], {row['id']: row['audio_text']}, ROOT, directory,
                             'workflow_gold', agent, self.catalog, self.ToolEnv)
            dest = Path(directory) / 'workflow_gold'
            data = json.loads((dest / 'summary.json').read_text())
            self.assertEqual(data['architecture'], 'step_planner_caller')
            self.assertEqual(data['planner_calls_per_sample']['mean'], 2)
            self.assertEqual(data['caller_calls_per_sample']['mean'], 1)
            self.assertEqual(data['llm_calls_per_sample']['mean'], 3)
            self.assertEqual(data['workflow_sha256'], workflow_sha256())
            self.assertNotIn('mean_tool_set_recall', data)


if __name__ == '__main__':
    unittest.main()
