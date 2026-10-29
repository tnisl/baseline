"""Behavioral checks for request coverage, write meaning and local field repair."""
import unittest
from intent_contract import (intent_fields, request_done_error, intent_value_error,
                             branch_write_error, effect_error, request_requirements)
from agent_pipeline import Agent
import test_pipeline as support

class IntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = support.load_catalog(support.ROOT)

    def wallet_observations(self):
        return [
            {'name': 'bill_payment.list_bills', 'result': {'ok': True, 'bills': [
                {'id': 'billA', 'type': 'tv', 'amount': 100, 'paid': False}]}},
            {'name': 'ewallet.get_balance', 'result': {'ok': True, 'balance': 200}},
            {'name': 'bill_payment.pay', 'arguments': {'bill_id': 'billA'},
             'result': {'ok': True, 'bill': {'id': 'billA', 'paid': True}}},
        ]

    def test_paid_bill_does_not_finish_missing_sms(self):
        text = 'Nếu ví đủ thì trả tiền truyền hình bằng ví và nhắn SMS Linh đã đóng, còn thiếu thì trả qua ngân hàng và nhắc tôi nạp ví.'
        obs = self.wallet_observations()
        self.assertIn('sms.send', request_done_error(text, self.catalog, obs))
        self.assertIsNotNone(branch_write_error('reminders.create', text, obs))
        self.assertIsNone(branch_write_error('sms.send', text, obs))

    def test_message_about_users_payment_does_not_require_agent_to_pay(self):
        text = 'Nhắn SMS Linh là tôi đã đóng tiền mạng rồi'
        self.assertIsNone(branch_write_error('sms.send', text, []))

    def test_inactive_gold_branch_is_blocked_before_write(self):
        text = 'Vàng SJC bán ra trên 100 triệu thì nhắc tôi bán, chưa tới thì chuyển 2 triệu từ tài khoản.'
        obs = [{'name': 'gold_price.get_price', 'result': {'ok': True,
                'prices': [{'type': 'sjc', 'sell': 90000000}]}}]
        self.assertIsNotNone(branch_write_error('reminders.create', text, obs))
        self.assertIsNone(branch_write_error('banking.transfer', text, obs))

    def test_schema_optional_can_be_intent_required(self):
        fields = intent_fields('smart_home.light_set', 'Bật giúp em cái đèn ở 40 phần trăm', self.catalog['smart_home.light_set'])
        self.assertEqual(fields, ['brightness', 'power'])
        self.assertIsNotNone(intent_value_error('smart_home.light_set', {'brightness': 40}, 'Bật đèn ở 40 phần trăm')[0])
        fields = intent_fields('medication.log_taken', 'Ghi đã uống hồi bảy giờ', self.catalog['medication.log_taken'])
        self.assertIn('time', fields)

    def test_scheduled_power_not_applied_to_immediate_brightness(self):
        text = 'Cài tự bật đèn lúc 19h hằng ngày, còn giờ chỉnh đèn xuống 40 phần trăm'
        self.assertEqual(intent_fields('smart_home.light_set', text, self.catalog['smart_home.light_set']), ['brightness'])

    def test_tool_ok_does_not_override_mismatching_result(self):
        error = effect_error('smart_home.light_set', {'power': 'on'},
                             {'ok': True, 'item': {'state': {'power': 'off'}}})
        self.assertIsNotNone(error)

    def test_all_eligible_bills_need_separate_evidence(self):
        text = 'Trả tất cả hóa đơn trong năm ngày tới, hơn 500 nghìn thì chưa trả'
        row = {'context': {'current_time': '2026-01-01T10:00:00+07:00'}}
        obs = [{'name': 'bill_payment.list_bills', 'result': {'ok': True, 'bills': [
            {'id': 'a', 'amount': 100000, 'due_date': '2026-01-02', 'paid': False},
            {'id': 'b', 'amount': 200000, 'due_date': '2026-01-03', 'paid': False}]}},
            {'name': 'bill_payment.pay', 'arguments': {'bill_id': 'a'}, 'result': {'ok': True}}]
        self.assertIn('bill_payment.pay:b', request_done_error(text, self.catalog, obs, row))
        obs.append({'name': 'bill_payment.pay', 'arguments': {'bill_id': 'b'}, 'result': {'ok': True}})
        self.assertIsNone(request_done_error(text, self.catalog, obs, row))

    def test_reading_notes_does_not_require_note_creation(self):
        requirements = request_requirements('Đọc ghi chú mới nhất cho tôi', self.catalog, [])
        self.assertNotIn('notes.create', [item['tool'] for item in requirements])

    def test_value_extra_field_repairs_argument_node_once(self):
        row = {**support.sample('train_0075'), 'audio_text': 'Bật đèn phòng khách màu đỏ'}
        goal = [{'id': 'g1', 'task': 'Bật đèn đỏ', 'when': 'always', 'kind': 'write'}]
        values = {'arguments': {'device_id': 'light_living', 'power': 'on', 'color': 'red'},
                  'value_reasons': {'device_id': 'Đèn đã thấy', 'power': 'User yêu cầu bật', 'color': 'User yêu cầu đỏ'}}
        llm = support.ScriptedLLM([
            {'type': 'call', 'name': 'smart_home.get_status', 'tool_reason': 'Tra đèn', 'goals': goal},
            {'selected_arguments': []},
            {'type': 'call', 'name': 'smart_home.light_set', 'tool_reason': 'Bật đèn đỏ'},
            {'selected_arguments': ['power']}, values,
            {'selected_arguments': ['power', 'color']}, values,
            {'type': 'done', 'completed_goals': {'g1': 1}},
        ])
        out = Agent(llm, self.catalog, support.ToolEnv, support.ROOT).run(row, row['audio_text'])
        self.assertEqual(out['stop_reason'], 'done')
        self.assertIn('selector_repair', out['trace'][1])
        self.assertEqual(out['tool_calls'][-1]['arguments']['color'], 'red')
        self.assertEqual(len(out['tool_calls']), 2)

    def test_rain_branch_sms_is_not_marked_unconditional(self):
        text = 'Mai mưa thì hủy họp rồi nhắn SMS Linh, còn không mưa thì đặt xe'
        requirement = next(item for item in request_requirements(text, self.catalog, [])
                           if item['tool'] == 'sms.send')
        self.assertEqual(requirement['status'], 'unknown')

    def test_known_mismatching_effect_cannot_complete_goal(self):
        from agent_pipeline import done_evidence_error
        goal = [{'id': 'g1', 'task': 'Bật đèn', 'when': 'always', 'kind': 'write'}]
        obs = [{'name': 'smart_home.light_set', 'kind': 'write', 'result': {'ok': True},
                'effect_verified': False}]
        self.assertIsNotNone(done_evidence_error({'type': 'done', 'completed_goals': {'g1': 0}}, goal, obs))

if __name__ == '__main__':
    unittest.main()
