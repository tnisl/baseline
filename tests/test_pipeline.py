"""CPU integration with scripted roles and real ToolEnv responses."""
import json
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, ScriptedLLM, sample, plan, call, review, handoff
from lifemate_v7.pipeline import Agent, ContractError, check_plan, check_call, check_summary, parse_object
from lifemate_v7.common import load_catalog, import_toolenv, evaluate_agent_case
from lifemate_v7.errors import InferenceBackendError, ContextLimitError


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)
        cls.ToolEnv = staticmethod(import_toolenv(ROOT))

    def run_script(self, outputs, **options):
        llm, row = ScriptedLLM(outputs), sample()
        result = Agent(llm, self.catalog, self.ToolEnv, ROOT, **options).run(row, row['audio_text'])
        return result, llm, row

    def test_full_three_role_workflow_matches_real_gold(self):
        result, llm, row = self.run_script([
            plan('Tra hóa đơn truyền hình.'), call('bill_payment.list_bills'), review(),
            plan('Tra số dư ví.'), call('ewallet.get_balance'), review(),
            plan('Thanh toán hóa đơn truyền hình bằng ví.'),
            call('bill_payment.pay', bill_id='bill_01', source='ewallet'), review(),
            plan('Tìm liên hệ người nhận SMS.'), call('contacts.search', query='Sếp Dũng'), review(),
            plan('Gửi SMS báo đã thanh toán.'),
            call('sms.send', contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'),
            review('done', 'Đã thanh toán truyền hình bằng ví và gửi SMS cho sếp Dũng.'),
        ], save_node_inputs=True)
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual((result['planner_calls'], result['caller_calls'], result['summarizer_calls']), (5, 5, 5))
        self.assertEqual((result['llm_calls'], result['decisions']), (15, 5))
        self.assertEqual(evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)['score_proxy'], 1)
        for event in result['trace']:
            self.assertEqual(set(event['node_outputs']), {'plan', 'call', 'summarize'})
            caller_input = json.loads(event['node_outputs']['call']['input_messages'][1]['content'])
            self.assertEqual(caller_input['planner_instruction'], event['planner_output']['instruction'])
        self.assertIn('sếp Dũng', result['final_answer'])

    def test_summarizer_can_reject_planner_handoff_and_return_to_planner(self):
        result, llm, _ = self.run_script([
            handoff(), review(), plan('Tra số dư ví.'), call('ewallet.get_balance'), review('done'),
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual((result['planner_calls'], result['caller_calls'], result['summarizer_calls']), (2, 1, 2))
        self.assertEqual(set(result['trace'][0]['node_outputs']), {'plan', 'summarize'})
        self.assertEqual(result['trace'][0]['summarizer_output'], {'next': 'planner'})
        second = json.loads(llm.inputs[2][0][1]['content'])
        self.assertEqual(second['previous_review'], {'next': 'planner'})

    def test_terminal_decisions_belong_only_to_summarizer(self):
        for role in ('done', 'give_up'):
            result, _, _ = self.run_script([handoff(), review(role)])
            self.assertEqual(result['stop_reason'], role)
            self.assertEqual(result['caller_calls'], 0)
            result, _, _ = self.run_script([{'next': role}])
            self.assertEqual(result['stop_reason'], 'invalid_model_output')
            self.assertEqual(result['summarizer_calls'], 0)
        result, _, _ = self.run_script([plan(), {'next': 'done'}])
        self.assertEqual(result['stop_reason'], 'invalid_model_output')

    def test_feedback_is_exact_env_response_not_model_summary(self):
        instruction = 'Chỉ dẫn tạm thời không phải observation.'
        result, llm, _ = self.run_script([
            plan(instruction), call('ewallet.get_balance'), review(), handoff(), review('done', 'Đã đọc số dư ví.'),
        ], save_node_inputs=True)
        p0, c0, s0, p1, s1 = [json.loads(messages[1]['content']) for messages, _ in llm.inputs]
        self.assertEqual(p0['env_feedback'], [])
        self.assertEqual(c0['env_feedback'], [])
        self.assertEqual(c0['planner_instruction'], instruction)
        fact = {'stage': 'call', 'tool': 'ewallet.get_balance', 'arguments': {},
                'response': result['trace'][0]['tool_result']}
        self.assertEqual(s0['env_feedback'][-1], fact)
        self.assertEqual(p1['env_feedback'][-1], fact)
        self.assertEqual(s0['current_step']['action'], call('ewallet.get_balance'))
        self.assertNotIn(instruction, json.dumps(p1, ensure_ascii=False))
        for payload in (p0, c0, s0, p1, s1):
            self.assertNotIn('label', payload)
            self.assertNotIn('final_state', payload)
            self.assertNotIn('final_answer', payload)
        self.assertEqual({x['name'] for x in s0['available_tools']}, {x['name'] for x in p0['available_tools']})
        self.assertTrue(all(set(x) == {'name', 'kind', 'description'} for x in s0['available_tools']))

    def test_caller_retains_entire_catalog_not_a_planner_whitelist(self):
        result, llm, _ = self.run_script([plan(), call('contacts.search', query='Dũng'), review('done')])
        p, c = [json.loads(messages[1]['content']) for messages, _ in llm.inputs[:2]]
        self.assertEqual({s['name'] for s in p['available_tools']}, {s['name'] for s in c['available_tools']})
        self.assertIn('parameters', next(s for s in c['available_tools'] if s['name'] == 'contacts.search'))

    def test_real_ask_forces_wait_review_and_no_more_tools(self):
        result, llm, _ = self.run_script([
            plan('Hỏi người nhận SMS.'), call('sys.ask_missing_info', tool_name='sms.send', params=['contact_id']),
            review('wait', 'Bạn muốn gửi SMS cho liên hệ nào?'),
        ])
        self.assertEqual(result['stop_reason'], 'ask_missing_info')
        self.assertEqual(len(result['tool_calls']), 1)
        self.assertEqual(result['summarizer_calls'], 1)
        summary_input = json.loads(llm.inputs[-1][0][1]['content'])
        self.assertTrue(summary_input['waiting_for_user'])
        self.assertNotIn('planner', json.dumps(llm.inputs[-1][1]))
        for wrong in (review(), review('done'), review('give_up')):
            with self.assertRaises(ContractError):
                check_summary(wrong, waiting_for_user=True)
        for fields in (['unknown'], ['content', 'content'], []):
            with self.assertRaises(ContractError):
                check_call(call('sys.ask_missing_info', tool_name='sms.send', params=fields), self.catalog)

    def test_validation_error_is_visible_to_summarizer_and_can_be_corrected(self):
        result, llm, _ = self.run_script([
            plan(), call('sms.send', content='text'), review(),
            plan('Hỏi thông tin người nhận SMS.'),
            call('sys.ask_missing_info', tool_name='sms.send', params=['contact_id']), review('wait'),
        ])
        self.assertEqual(result['stop_reason'], 'ask_missing_info')
        self.assertEqual(result['invalid_schema_proposals'], 1)
        self.assertEqual(len(result['tool_calls']), 1)
        s0 = json.loads(llm.inputs[2][0][1]['content'])
        p1 = json.loads(llm.inputs[3][0][1]['content'])
        self.assertFalse(s0['env_feedback'][-1]['response']['ok'])
        self.assertEqual(s0['env_feedback'], p1['env_feedback'])
        self.assertNotIn('tool_result', result['trace'][0])

    def test_business_failure_can_return_to_planner_without_faking_success(self):
        result, llm, _ = self.run_script([
            plan(), call('bill_payment.pay', bill_id='unknown', source='ewallet'), review(),
            plan('Tra hóa đơn cần trả.'), call('bill_payment.list_bills'), review('give_up'),
        ])
        self.assertEqual(result['stop_reason'], 'give_up')
        self.assertEqual(result['tool_failures'], 1)
        self.assertTrue(result['trace'][0]['tool_result']['ok'])
        self.assertFalse(result['trace'][0]['tool_result']['done'])
        self.assertFalse(json.loads(llm.inputs[2][0][1]['content'])['env_feedback'][-1]['response']['done'])
        self.assertEqual(len(result['tool_calls']), 2)

    def test_duplicate_writes_repeated_reads_and_repeated_errors_are_bounded(self):
        result, _, _ = self.run_script([
            plan(), call('sms.send', content='same', phone_number='0900000000'), review(),
            plan(), call('sms.send', phone_number='0900000000', content='same'),
        ])
        self.assertEqual(result['stop_reason'], 'duplicate_write')
        self.assertEqual(len(result['tool_calls']), 1)
        self.assertEqual(result['summarizer_calls'], 1)
        for name, args in (('ewallet.get_balance', {}), ('sms.send', {'content': 'missing recipient'}),
                           ('bill_payment.pay', {'bill_id': 'unknown', 'source': 'ewallet'})):
            outputs = [plan(), call(name, **args), review()] * 2 + [plan(), call(name, **args)]
            result, _, _ = self.run_script(outputs)
            self.assertEqual(result['stop_reason'], 'environment_no_progress')
            self.assertEqual(result['summarizer_calls'], 2)
        result, _, _ = self.run_script([handoff(), review()] * 2 + [handoff()])
        self.assertEqual(result['stop_reason'], 'control_no_progress')

    def test_failed_write_can_be_followed_by_lookup_and_successful_corrected_write(self):
        result, _, _ = self.run_script([
            plan(), call('bill_payment.pay', bill_id='unknown', source='ewallet'), review(),
            plan('Tra hóa đơn đúng.'), call('bill_payment.list_bills'), review(),
            plan('Kiểm tra số dư ví.'), call('ewallet.get_balance'), review(),
            plan('Thanh toán hóa đơn đã xác định bằng ví.'),
            call('bill_payment.pay', bill_id='bill_01', source='ewallet'), review('done'),
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['tool_failures'], 1)
        self.assertEqual(result['summarizer_calls'], 4)
        self.assertEqual(len(result['tool_calls']), 4)
        self.assertFalse(result['trace'][0]['tool_result']['done'])
        self.assertTrue(result['trace'][-1]['tool_succeeded'])
        self.assertTrue(result['trace'][-1]['tool_result']['bill']['paid'])
        self.assertEqual(result['trace'][-1]['tool_result']['bill']['paid_with'], 'ewallet')

    def test_tool_budget_still_allows_summarizer_to_conclude_after_last_call(self):
        result, _, _ = self.run_script([plan(), call('ewallet.get_balance'), review('done')], max_tool_calls=1)
        self.assertEqual(result['stop_reason'], 'done')
        result, _, _ = self.run_script([plan(), call('ewallet.get_balance'), review(), plan()], max_tool_calls=1)
        self.assertEqual(result['stop_reason'], 'tool_limit')
        self.assertEqual(result['caller_calls'], 1)
        result, _, _ = self.run_script([plan(), call('ewallet.get_balance'), review()], max_decisions=1)
        self.assertEqual(result['stop_reason'], 'decision_limit')
        self.assertIsNone(result['final_answer'])

    def test_contracts_reject_old_outputs_control_prose_and_malformed_terminal_answer(self):
        for output in (plan(''), plan(' '), plan('x' * 241), plan('a\nb'),
                       dict(plan(), reason='why'), {'next': 'done'}, {'tools': ['contacts.search']},
                       {'next': 'summarizer', 'instruction': 'extra'}):
            with self.assertRaises(ContractError):
                check_plan(output, self.catalog)
        for output in (dict(call('contacts.search'), reason='why'), call('unknown'),
                       call('contacts.search', wrong='field'), {'type': 'call', **call('contacts.search')}):
            with self.assertRaises(ContractError):
                check_call(output, self.catalog)
        for output in ({'next': 'done'}, review('done', ' '), review('done', 'x' * 481),
                       review('done', 'a\nb'), {'next': 'planner', 'answer': 'extra'}, review('wait')):
            with self.assertRaises(ContractError):
                check_summary(output)
        result, _, _ = self.run_script([handoff(), {'next': 'done'}])
        self.assertEqual(result['stop_reason'], 'invalid_model_output')
        self.assertEqual(result['trace'][0]['local_diagnostic']['stage'], 'summarize')

    def test_infrastructure_errors_remain_experiment_errors_in_all_three_nodes(self):
        for error in (InferenceBackendError('failed'), ContextLimitError('too long')):
            for outputs in ([error], [plan(), error], [plan(), call('ewallet.get_balance'), error]):
                with self.assertRaises(type(error)):
                    self.run_script(outputs)
        for text in ('text {}', '{} trailing', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]'):
            with self.assertRaises((ValueError, ContractError)):
                parse_object(text)

    def test_trimming_protects_current_step_latest_observation_and_hides_state(self):
        class LongLLM(ScriptedLLM):
            def count(self, messages):
                p = json.loads(messages[1]['content'])
                return 100 + 10 * len(p['history']) + 10 * len(p['env_feedback'])
        row = sample()
        row['history'] = [{'role': 'user', 'audio_text': 'older'}] * 3
        row['context']['installed_apps']['hidden'] = {'tools': [], 'state': {'secret': 999}}
        agent = Agent(LongLLM([]), self.catalog, self.ToolEnv, ROOT, max_prompt_tokens=110)
        for phase in ('call', 'summarize'):
            msgs, cut = agent.messages(row, 'current', self.catalog,
                [{'stage': 'call', 'response': 1}, {'stage': 'call', 'response': 2}], 10, phase,
                instruction='Current step', current_step={'action': call('ewallet.get_balance')})
            p = json.loads(msgs[1]['content'])
            self.assertTrue(cut)
            self.assertEqual(p['env_feedback'], [{'stage': 'call', 'response': 2}])
            self.assertNotIn('secret', msgs[1]['content'])
            if phase == 'summarize':
                self.assertEqual(p['current_step']['action'], call('ewallet.get_balance'))
            else:
                self.assertEqual(p['planner_instruction'], 'Current step')
        agent.max_prompt_tokens = 1
        with self.assertRaises(ContextLimitError):
            agent.messages(row, 'current', self.catalog, [], 10, 'summarize')

    def test_runner_logs_three_role_costs_and_final_answer(self):
        from lifemate_v7.runner import agent_experiment, workflow_sha256
        result, _, row = self.run_script([handoff(), review('done')])
        agent = Agent(ScriptedLLM([handoff(), review('done', 'Không có tác vụ cần thực hiện.')]),
                      self.catalog, self.ToolEnv, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            agent_experiment([row], {row['id']: row['audio_text']}, ROOT, directory,
                             'workflow_gold', agent, self.catalog, self.ToolEnv)
            dest = Path(directory) / 'workflow_gold'
            data = json.loads((dest / 'summary.json').read_text())
            trace = json.loads((dest / 'traces.jsonl').read_text())
            self.assertEqual(data['architecture'], 'planner_caller_summarizer')
            self.assertEqual(data['termination_owner'], 'summarizer')
            self.assertEqual(data['summarizer_calls_per_sample']['mean'], 1)
            self.assertEqual(data['llm_calls_per_sample']['mean'], 2)
            self.assertEqual(data['workflow_sha256'], workflow_sha256())
            self.assertEqual(trace['final_answer'], 'Không có tác vụ cần thực hiện.')
            self.assertEqual(trace['summarizer_calls'], 1)
            self.assertNotIn('input_messages', trace['trace'][0]['node_outputs']['summarize'])


if __name__ == '__main__':
    unittest.main()
