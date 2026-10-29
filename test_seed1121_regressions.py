"""Regressions for plan repair, hint certainty and conditional writes."""
import json
import unittest
import test_pipeline as support
from agent_pipeline import Agent, added_goals_error
from pipeline_rules import value_hints, trusted_value_hints, semantic_value_error
from evidence_rules import write_evidence_issue

class SeedRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = support.PipelineTests.catalog if hasattr(support.PipelineTests, 'catalog') else support.load_catalog(support.ROOT)

    def test_inline_rewrite_preserves_goal_and_adds_new_one(self):
        old = [{'id': 'g1', 'task': 'Thanh toán', 'when': 'always', 'kind': 'write'}]
        proposal = {'goals': [{'id': 'g1', 'task': 'Tra ID', 'when': 'always', 'kind': 'read'},
                             {'id': 'g2', 'task': 'Nhắn tin', 'when': 'always', 'kind': 'write'}]}
        merged, error = added_goals_error(proposal, old)
        self.assertIsNone(error)
        self.assertEqual(merged[0], old[0])
        self.assertEqual(len(merged), 2)

    def test_four_inline_additions_do_not_hit_revision_cap(self):
        row = support.sample('train_0236')
        answers = []
        for i, tool in enumerate(['ewallet.get_balance', 'banking.get_balance', 'banking.list_beneficiaries', 'bill_payment.list_bills']):
            answers.append({'type': 'call', 'name': tool, 'tool_reason': 'Đọc số dư',
                            'goals': [{'id': f'g{i}', 'task': 'Đọc số dư', 'when': 'always', 'kind': 'read'}]})
            if tool == 'bill_payment.list_bills':
                answers.append({'selected_arguments': []})
        llm = support.ScriptedLLM(answers)
        out = Agent(llm, self.catalog, support.ToolEnv, support.ROOT,
                    max_decisions=4, max_repairs=5).run(row, row['audio_text'])
        self.assertEqual(len(out['task_plan']), 4)
        self.assertFalse(any(t.get('error', {}).get('error') == 'invalid_plan_revision' for t in out['trace']))
        self.assertEqual(out['plan_revisions'], 0)

    def test_person_name_cannot_be_a_province_hint(self):
        row = support.sample('train_1537')
        hints = value_hints('lottery.check_ticket', row['audio_text'], row, [], self.catalog['lottery.check_ticket'])
        self.assertNotIn('province', hints)
        self.assertNotIn('date', hints)
        from evidence_rules import lottery_reference_issue
        self.assertEqual(lottery_reference_issue('lottery.check_ticket', row['audio_text'], row, [])['fields'], ['date'])

    def test_advisory_query_does_not_constrain_value(self):
        trusted = trusted_value_hints('podcast_radio.search', {'query': 'một suy đoán'})
        self.assertEqual(trusted, {})
        self.assertEqual(semantic_value_error('podcast_radio.search', {'query': 'chủ đề khác'}, '', trusted), (None, None))

    def test_ambiguous_contact_is_blocked_and_qualified_contact_allowed(self):
        spec = self.catalog['phone.call']
        obs = [{'name': 'contacts.search', 'result': {'ok': True, 'contacts': [
            {'id': 'a', 'name': 'Linh', 'relation': 'bạn'},
            {'id': 'b', 'name': 'Linh', 'relation': 'đồng nghiệp'}]}}]
        issue, _ = write_evidence_issue('phone.call', {'contact_id': 'a'}, spec, 'Gọi Linh', {}, obs)
        self.assertEqual(issue['type'], 'ambiguous_target')
        self.assertEqual(write_evidence_issue('phone.call', {'contact_id': 'b'}, spec,
                         'Gọi Linh đồng nghiệp', {}, obs), (None, None))

    def test_single_recipient_cannot_expand_to_both_addresses(self):
        issue, _ = write_evidence_issue('email.create_draft', {'to': ['a@test', 'b@test']},
                     self.catalog['email.create_draft'], 'Chỉ bán một mối, soạn thư cho người ta', {}, [])
        self.assertEqual(issue['fields'], ['to'])

    def test_previous_successful_payment_is_not_repeated(self):
        row = support.sample('train_1358')
        args = {'from_account_id': 'acc_8319', 'beneficiary_id': 'ben_2468', 'amount': 180000}
        issue, prior = write_evidence_issue('banking.transfer', args, self.catalog['banking.transfer'],
                                           row['audio_text'], row, [])
        self.assertEqual(issue['type'], 'already_completed')
        self.assertEqual(prior['result']['transaction']['id'], 'tx_01')
        # Explicit new transfer instructions must still be executable.
        self.assertEqual(write_evidence_issue('banking.transfer', args, self.catalog['banking.transfer'],
                         'Chuyển thêm 180 nghìn cho Trang', row, []), (None, None))

    def test_history_proof_can_finish_without_env_transfer(self):
        row = support.sample('train_1358')
        first = {'type': 'call', 'name': 'banking.transfer', 'tool_reason': 'Chuyển nếu chưa chuyển',
                 'goals': [{'id': 'g1', 'task': 'Chuyển nếu chưa chuyển', 'when': 'always', 'kind': 'write'}]}
        values = {'arguments': {'from_account_id': 'acc_8319', 'beneficiary_id': 'ben_2468', 'amount': 180000},
                  'value_reasons': {'from_account_id': 'Tài khoản đã thấy', 'beneficiary_id': 'Đúng Trang', 'amount': 'Khoản đã nói'}}
        llm = support.ScriptedLLM([first, {'selected_arguments': []}, values,
                                   {'type': 'done', 'completed_goals': {'g1': 0}}])
        out = Agent(llm, self.catalog, support.ToolEnv, support.ROOT).run(row, row['audio_text'])
        self.assertEqual(out['tool_calls'], [])
        self.assertEqual(out['stop_reason'], 'done')
        payload = json.loads(llm.prompts[-1][1]['content'])
        self.assertEqual(payload['executed_calls'], [])
        self.assertEqual(payload['tool_observations'][0]['origin'], 'history')

    def test_created_ride_in_history_is_valid_cancellation_evidence(self):
        from evidence_rules import reference_error
        row = {'history': [{'tool_calls': [{'name': 'ride_hailing.book', 'arguments': {}}],
                            'tool_results': [{'ok': True, 'id': 'rA', 'item': {'id': 'rA', 'status': 'booked'}}]}]}
        self.assertEqual(reference_error('ride_hailing.cancel', {'ride_id': 'rA'},
                         self.catalog['ride_hailing.cancel'], 'Hủy xe đó', row, []), (None, None))

if __name__ == '__main__':
    unittest.main()
