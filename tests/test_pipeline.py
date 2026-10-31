"""Behavioral integration for fixed order, scoped values and summarizer routing."""
import json
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, ScriptedLLM, sample, plan, call, review
from lifemate_v8_finetune.pipeline import Agent, ContractError, select_plan, check_call, check_summary, parse_object
from lifemate_v8_finetune.common import load_catalog, import_toolenv, evaluate_agent_case, ordered_matches
from lifemate_v8_finetune.errors import InferenceBackendError, ContextLimitError


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)
        cls.ToolEnv = staticmethod(import_toolenv(ROOT))

    def run_script(self, outputs, **options):
        llm, row = ScriptedLLM(outputs), sample()
        result = Agent(llm, self.catalog, self.ToolEnv, ROOT, **options).run(row, row['audio_text'])
        return result, llm, row

    def test_full_fixed_plan_with_three_roles_matches_real_gold(self):
        tools = ['bill_payment.list_bills', 'ewallet.get_balance', 'bill_payment.pay', 'contacts.search', 'sms.send']
        result, llm, row = self.run_script([
            plan(*tools), call(), review(), call(), review(),
            call(bill_id='bill_01', source='ewallet'), review(), call(query='Sếp Dũng'), review(),
            call(contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'), review('done'),
        ], save_node_inputs=True)
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['tool_plan'], tools)
        self.assertEqual([c['name'] for c in result['tool_calls']], tools)
        self.assertEqual((result['planner_calls'], result['caller_calls'], result['summarizer_calls']), (1, 5, 5))
        self.assertEqual((result['llm_calls'], result['decisions']), (11, 6))
        self.assertEqual(result['processed_steps'], 5)
        case = evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)
        self.assertEqual(case['score_proxy'], 1)
        self.assertEqual(case['tool_order_recall'], 1)
        self.assertTrue(case['tool_plan_exact_match'])
        self.assertEqual(set(result['trace'][0]['node_outputs']), {'plan'})
        for index, event in enumerate(result['trace'][1:]):
            self.assertEqual(set(event['node_outputs']), {'call', 'summarize'})
            p = json.loads(event['node_outputs']['call']['input_messages'][1]['content'])
            s = json.loads(event['node_outputs']['summarize']['input_messages'][1]['content'])
            self.assertEqual(p['selected_tool']['name'], tools[index])
            self.assertEqual(p['step_index'], index)
            self.assertEqual(s['step_index'], index + 1)
            self.assertEqual(s['current_step']['step_index'], index)
            self.assertEqual(s['env_feedback'][-1]['response'], event['tool_result'])
            self.assertNotIn('available_tools', p)
            self.assertNotIn('planner_instruction', p)
            self.assertNotIn('label', p)

    def test_plan_order_repetition_and_scope_are_preserved(self):
        tools = ['sms.send', 'ewallet.get_balance', 'sms.send']
        result, llm, _ = self.run_script([
            plan(*tools), call(content='first', phone_number='0900000000'), review(), call(), review(),
            call(content='second', phone_number='0900000000'), review('done'),
        ])
        self.assertEqual([c['name'] for c in result['tool_calls']], tools)
        self.assertEqual(result['selected_tools'], ['sms.send', 'ewallet.get_balance'])
        self.assertEqual(result['planner_calls'], 1)
        self.assertNotIn('ewallet.get_balance', json.dumps(llm.inputs[1][1]))
        for wrong in ({'name': 'ewallet.get_balance', 'arguments': {}}, {'type': 'done'},
                      {'type': 'call', 'name': 'sms.send', 'arguments': {}}, {'tools': []}):
            result, _, _ = self.run_script([plan('sms.send'), wrong])
            self.assertEqual(result['stop_reason'], 'invalid_model_output')
            self.assertEqual(result['tool_calls'], [])

    def test_summarizer_cannot_done_before_plan_end_or_advance_after_failure(self):
        result, _, _ = self.run_script([plan('ewallet.get_balance', 'sms.send'), call(), review('done')])
        self.assertEqual(result['stop_reason'], 'invalid_model_output')
        self.assertEqual(result['trace'][-1]['local_diagnostic']['stage'], 'summarize')
        for output in (review(), review('done'), {'next': 'planner'}):
            with self.assertRaises(ContractError):
                check_summary(output, 'tool_failed', 1)
        with self.assertRaises(ContractError):
            check_summary(review(), 'executed', 0)
        with self.assertRaises(ContractError):
            check_summary(review('retry'), 'executed', 1)

    def test_empty_plan_still_requires_summarizer_and_cannot_create_tool(self):
        for route in ('done', 'give_up'):
            result, llm, _ = self.run_script([plan(), review(route)])
            self.assertEqual(result['stop_reason'], route)
            self.assertEqual((result['planner_calls'], result['caller_calls'], result['summarizer_calls']), (1, 0, 1))
            self.assertEqual(result['llm_calls'], 2)
            self.assertEqual(result['tool_calls'], [])
            self.assertEqual(set(result['trace'][0]['node_outputs']), {'plan', 'summarize'})
        result, _, _ = self.run_script([plan(), review()])
        self.assertEqual(result['stop_reason'], 'invalid_model_output')

    def test_skip_has_no_env_observation_and_still_passes_through_summarizer(self):
        result, llm, _ = self.run_script([
            plan('ewallet.get_balance', 'contacts.search', 'sms.send'), call(), review(),
            {'type': 'skip'}, review(), call(content='text', phone_number='0900000000'), review('done'),
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual((result['processed_steps'], result['skipped_steps']), (3, 1))
        self.assertEqual([c['name'] for c in result['tool_calls']], ['ewallet.get_balance', 'sms.send'])
        skipped_summary = json.loads(llm.inputs[4][0][1]['content'])
        self.assertEqual(skipped_summary['step_status'], 'skipped')
        self.assertEqual(skipped_summary['step_results'][-1]['status'], 'skipped')
        self.assertTrue(all(f['tool'] == 'ewallet.get_balance' for f in skipped_summary['env_feedback']))
        self.assertNotIn('action', result['trace'][2])

    def test_conditional_unused_slot_can_be_skipped_without_losing_gold_score(self):
        tools = ['bill_payment.list_bills', 'ewallet.get_balance', 'bill_payment.pay',
                 'contacts.search', 'sms.send', 'reminders.create']
        result, _, row = self.run_script([
            plan(*tools), call(), review(), call(), review(), call(bill_id='bill_01', source='ewallet'), review(),
            call(query='Sếp Dũng'), review(), call(contact_id='c_04', content='đã đóng tiền truyền hình rồi nhé'),
            review(), {'type': 'skip'}, review('done'),
        ])
        self.assertEqual(result['processed_steps'], 6)
        self.assertEqual(result['skipped_steps'], 1)
        case = evaluate_agent_case(row, result, ROOT, self.catalog, self.ToolEnv)
        self.assertEqual(case['score_proxy'], 1)
        self.assertFalse(case['tool_plan_exact_match'])

    def test_ask_is_for_current_tool_and_forces_terminal_wait(self):
        result, llm, _ = self.run_script([
            plan('sms.send', 'contacts.search'), {'type': 'ask', 'params': ['contact_id']}, review('wait'),
        ])
        self.assertEqual(result['stop_reason'], 'ask_missing_info')
        self.assertEqual(result['tool_calls'], [{'name': 'sys.ask_missing_info',
             'arguments': {'tool_name': 'sms.send', 'params': ['contact_id']}}])
        self.assertEqual(result['processed_steps'], 0)
        self.assertNotIn('caller', json.dumps(llm.inputs[-1][1]))
        for fields in (['unknown'], ['content', 'content'], []):
            with self.assertRaises(ContractError):
                check_call({'type': 'ask', 'params': fields}, self.catalog, 'sms.send')
        for output in (review(), review('done'), review('give_up'), review('retry')):
            with self.assertRaises(ContractError):
                check_summary(output, 'waiting_for_user', 2)

    def test_validation_retry_keeps_tool_slot_and_exact_error_without_replanning(self):
        result, llm, _ = self.run_script([
            plan('sms.send'), call(content='text'), review('retry'),
            call(content='text', phone_number='0900000000'), review('done'),
        ], save_node_inputs=True)
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual((result['planner_calls'], result['retry_requests'], result['invalid_schema_proposals']), (1, 1, 1))
        self.assertEqual(len(result['tool_calls']), 1)
        self.assertEqual(result['processed_steps'], 1)
        before = json.loads(llm.inputs[1][0][1]['content'])
        after = json.loads(llm.inputs[3][0][1]['content'])
        summary = json.loads(llm.inputs[2][0][1]['content'])
        self.assertEqual(before['selected_tool'], after['selected_tool'])
        self.assertEqual(after['step_index'], 0)
        self.assertEqual(after['previous_review'], {'next': 'retry'})
        self.assertEqual(after['env_feedback'], summary['env_feedback'])
        self.assertFalse(after['env_feedback'][-1]['response']['ok'])
        self.assertNotIn('answer', after)
        self.assertNotIn('final_answer', after)

    def test_business_failure_retry_can_correct_value_in_same_tool(self):
        result, _, _ = self.run_script([
            plan('bill_payment.list_bills', 'ewallet.get_balance', 'bill_payment.pay'), call(), review(), call(), review(),
            call(bill_id='unknown', source='ewallet'), review('retry'),
            call(bill_id='bill_01', source='ewallet'), review('done'),
        ])
        self.assertEqual(result['stop_reason'], 'done')
        self.assertEqual(result['tool_failures'], 1)
        self.assertEqual(result['retry_requests'], 1)
        self.assertEqual(result['processed_steps'], 3)
        self.assertFalse(result['trace'][3]['tool_result']['done'])
        self.assertTrue(result['trace'][4]['tool_result']['bill']['paid'])
        self.assertEqual(result['trace'][3]['step_index'], result['trace'][4]['step_index'])

    def test_blocked_or_incomplete_plan_can_only_conclude_give_up(self):
        result, _, _ = self.run_script([plan('sms.send'), {'type': 'blocked'}, review('give_up')])
        self.assertEqual(result['stop_reason'], 'give_up')
        self.assertEqual(result['tool_calls'], [])
        result, _, _ = self.run_script([plan('ewallet.get_balance'), call(), review('give_up', 'Kế hoạch còn thiếu bước cần thiết.')])
        self.assertEqual(result['stop_reason'], 'give_up')
        self.assertEqual(result['processed_steps'], 1)
        self.assertIn('thiếu bước', result['final_answer'])
        with self.assertRaises(ContractError):
            check_summary(review('done'), 'blocked', 1)

    def test_duplicates_repeated_reads_and_same_error_retry_are_bounded(self):
        result, _, _ = self.run_script([
            plan('sms.send', 'sms.send'), call(content='same', phone_number='0900000000'), review(),
            call(phone_number='0900000000', content='same'),
        ])
        self.assertEqual(result['stop_reason'], 'duplicate_write')
        self.assertEqual(len(result['tool_calls']), 1)
        result, _, _ = self.run_script([
            plan(*(['ewallet.get_balance'] * 3)), call(), review(), call(), review(), call(),
        ])
        self.assertEqual(result['stop_reason'], 'environment_no_progress')
        for tool, args in [('sms.send', {'content': 'text'}),
                           ('bill_payment.pay', {'bill_id': 'unknown', 'source': 'ewallet'})]:
            result, _, _ = self.run_script([
                plan(tool), call(**args), review('retry'), call(**args), review('retry'), call(**args),
            ])
            self.assertEqual(result['stop_reason'], 'environment_no_progress')
            self.assertEqual(result['processed_steps'], 0)
            self.assertEqual(result['summarizer_calls'], 2)

    def test_tool_and_event_budgets_do_not_force_false_done(self):
        result, _, _ = self.run_script([plan('ewallet.get_balance'), call(), review('done')], max_tool_calls=1)
        self.assertEqual(result['stop_reason'], 'done')
        result, _, _ = self.run_script([
            plan('ewallet.get_balance', 'sms.send'), call(), review(), {'type': 'skip'}, review('done'),
        ], max_tool_calls=1)
        self.assertEqual(result['stop_reason'], 'done')
        result, _, _ = self.run_script([
            plan('ewallet.get_balance', 'ewallet.get_balance'), call(), review(), call(),
        ], max_tool_calls=1)
        self.assertEqual(result['stop_reason'], 'tool_limit')
        result, _, _ = self.run_script([plan('ewallet.get_balance')], max_decisions=1)
        self.assertEqual(result['stop_reason'], 'decision_limit')
        self.assertIsNone(result['final_answer'])

    def test_json_control_fields_plan_limit_and_terminal_answer_contracts(self):
        for output in (plan('unknown'), dict(plan(), reason='why'), {'tools': 'sms.send'},
                       plan('sys.ask_missing_info'), plan(*(['sms.send'] * 25))):
            with self.assertRaises(ContractError):
                select_plan(output, self.catalog, 24)
        for output in (dict(call(), reason='why'), call(balance=100), {'type': 'call'},
                       {'type': 'ask', 'tool_name': 'sms.send', 'params': ['content']}):
            with self.assertRaises(ContractError):
                check_call(output, self.catalog, 'sms.send')
        for output in ({'next': 'done'}, review('done', ' '), review('done', 'x' * 481),
                       review('done', 'a\nb'), {'next': 'caller', 'answer': 'extra'}):
            with self.assertRaises(ContractError):
                check_summary(output, 'executed', 0)
        self.assertEqual(ordered_matches(['a', 'b', 'a'], ['a', 'a']), 2)
        for text in ('text {}', '{} trailing', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]'):
            with self.assertRaises((ValueError, ContractError)):
                parse_object(text)

    def test_infrastructure_errors_abort_in_each_of_three_roles(self):
        for error in (InferenceBackendError('failed'), ContextLimitError('too long')):
            for outputs in ([error], [plan('ewallet.get_balance'), error],
                            [plan('ewallet.get_balance'), call(), error]):
                with self.assertRaises(type(error)):
                    self.run_script(outputs)

    def test_trimming_preserves_plan_current_step_latest_fact_and_hides_state(self):
        class LongLLM(ScriptedLLM):
            def count(self, messages):
                p = json.loads(messages[1]['content'])
                return 100 + 10 * len(p['history']) + 10 * len(p.get('env_feedback', []))
        row = sample(); row['history'] = [{'role': 'user', 'audio_text': 'older'}] * 3
        row['context']['installed_apps']['hidden'] = {'tools': [], 'state': {'secret': 999}}
        agent = Agent(LongLLM([]), self.catalog, self.ToolEnv, ROOT, max_prompt_tokens=110)
        for phase in ('call', 'summarize'):
            msgs, cut = agent.messages(row, 'current', self.catalog,
                [{'stage': 'call', 'response': 1}, {'stage': 'call', 'response': 2}], 10, phase,
                tool_plan=['contacts.search'], step_status='tool_failed', current_step={'step_index': 0})
            p = json.loads(msgs[1]['content'])
            self.assertTrue(cut)
            self.assertEqual(p['env_feedback'], [{'stage': 'call', 'response': 2}])
            self.assertEqual(p['tool_plan'], ['contacts.search'])
            self.assertNotIn('secret', msgs[1]['content'])
            self.assertNotIn('label', p)
        agent.max_prompt_tokens = 1
        with self.assertRaises(ContextLimitError):
            agent.messages(row, 'current', self.catalog, [], 10, 'plan')

    def test_runner_reports_plan_summarizer_retry_and_final_answer(self):
        from lifemate_v8_finetune.runner import agent_experiment, workflow_sha256
        row = sample()
        agent = Agent(ScriptedLLM([plan('ewallet.get_balance'), call(), review('done', 'Đã đọc số dư ví.')]),
                      self.catalog, self.ToolEnv, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            agent_experiment([row], {row['id']: row['audio_text']}, ROOT, directory,
                             'workflow_gold', agent, self.catalog, self.ToolEnv)
            dest = Path(directory) / 'workflow_gold'
            data = json.loads((dest / 'summary.json').read_text())
            trace = json.loads((dest / 'traces.jsonl').read_text())
            self.assertEqual(data['architecture'], 'ordered_plan_caller_summarizer')
            self.assertEqual(data['termination_owner'], 'summarizer')
            self.assertFalse(data['caller_tool_choice'])
            self.assertEqual(data['summarizer_calls_per_sample']['mean'], 1)
            self.assertEqual(data['llm_calls_per_sample']['mean'], 3)
            self.assertEqual(data['workflow_sha256'], workflow_sha256())
            self.assertEqual(trace['tool_plan'], ['ewallet.get_balance'])
            self.assertEqual(trace['final_answer'], 'Đã đọc số dư ví.')
            self.assertEqual(trace['retry_requests'], 0)
            self.assertNotIn('input_messages', trace['trace'][1]['node_outputs']['summarize'])


if __name__ == '__main__':
    unittest.main()
