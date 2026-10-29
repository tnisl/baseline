"""Regression checks for namespace, lookup routing and cache dependencies."""
import unittest
from pathlib import Path
from common import load_catalog
from agent_pipeline import available_lookup_for_ask
from evidence_rules import reference_error, read_invalidated
from pipeline_rules import tool_action_error, value_hints

ROOT = Path(__file__).resolve().parent.parent / 'train+public_test'

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT)

    def test_lookup_matches_account_collection_not_transactions(self):
        self.assertEqual(available_lookup_for_ask('banking.transfer', ['from_account_id'],
                         self.catalog, []), 'banking.get_balance')
        self.assertEqual(available_lookup_for_ask('home_automation.create_rule', ['device_id'],
                         self.catalog, []), 'smart_home.get_status')
        self.assertEqual(available_lookup_for_ask('food_delivery.place_order', ['items'],
                         self.catalog, []), 'food_delivery.get_menu')

    def test_contact_id_cannot_be_a_chat_id(self):
        row = {'history': [{'tool_calls': [{'name': 'contacts.search'}],
                            'tool_results': [{'ok': True, 'contacts': [{'id': 'c_A'}]}]}]}
        field, _ = reference_error('chat_ott.send_message', {'chat_id': 'c_A'},
                     self.catalog['chat_ott.send_message'], 'Nhắn Zalo cho chị ấy', row, [])
        self.assertEqual(field, 'chat_id')

    def test_device_type_is_checked_after_id_exists(self):
        obs = [{'name': 'smart_home.get_status', 'result': {'ok': True,
                'devices': [{'id': 'lamp_A', 'type': 'light'}]}}]
        field, _ = reference_error('smart_home.climate_set', {'device_id': 'lamp_A'},
                     self.catalog['smart_home.climate_set'], 'Bật điều hòa', {}, obs)
        self.assertEqual(field, 'device_id')

    def test_nested_menu_id_is_scoped_to_restaurant(self):
        spec = self.catalog['food_delivery.place_order']
        obs = [{'name': 'food_delivery.list_restaurants', 'result': {'ok': True,
                'restaurants': [{'id': 'r_A'}, {'id': 'r_B'}]}},
               {'name': 'food_delivery.get_menu', 'arguments': {'restaurant_id': 'r_A'},
                'result': {'ok': True, 'menu': [{'id': 'dish_A'}]}}]
        args = {'restaurant_id': 'r_B', 'items': [{'item_id': 'dish_A', 'quantity': 1}]}
        self.assertEqual(reference_error('food_delivery.place_order', args, spec, '', {}, obs)[0], 'items')
        args['restaurant_id'] = 'r_A'
        self.assertEqual(reference_error('food_delivery.place_order', args, spec, '', {}, obs), (None, None))

    def test_unrelated_write_does_not_invalidate_read(self):
        write = {'name': 'notes.create', 'kind': 'write', 'result': {'ok': True}}
        self.assertFalse(read_invalidated('home_security.list_alerts', write))
        self.assertTrue(read_invalidated('notes.list', write))

    def test_tomorrow_and_explicit_channel(self):
        hints = value_hints('weather.get_forecast', 'Mai mưa không?',
                            {'context': {'current_time': '2026-02-28T23:00:00+07:00'}}, [])
        self.assertEqual(hints['date'], '2026-03-01')
        self.assertIsNotNone(tool_action_error('chat_ott.send_message', 'Rep mail cho chị ấy'))
        self.assertIsNone(tool_action_error('email.reply', 'Rep mail cho chị ấy'))

    def test_exact_alert_copy_hint_selects_location_before_newest(self):
        obs = [{'name': 'home_security.list_alerts', 'result': {'ok': True, 'alerts': [
            {'location': 'phòng làm việc', 'time': '2026-01-01T08:00:00+07:00', 'message': 'Đúng bản ghi'},
            {'location': 'sân sau', 'time': '2026-01-01T09:00:00+07:00', 'message': 'Sai bản ghi'}]}}]
        hints = value_hints('notes.create', 'Chép nguyên cảnh báo mới nhất phòng làm việc', {}, obs)
        self.assertEqual(hints['content'], 'Đúng bản ghi')

    def test_ambiguous_storage_cannot_be_cancelled(self):
        obs = [{'name': 'subscriptions.list', 'result': {'ok': True, 'subscriptions': [
            {'id': 's_A', 'name': 'Cloud A 10GB', 'cycle': 'monthly', 'status': 'active'},
            {'id': 's_B', 'name': 'Cloud B 20GB', 'cycle': 'monthly', 'status': 'active'}]}}]
        field, reason = reference_error('subscriptions.cancel', {'subscription_id': 's_A'},
                         self.catalog['subscriptions.cancel'], 'Hủy gói lưu trữ đám mây', {}, obs)
        self.assertEqual(field, 'subscription_id')
        self.assertIn('ambiguous_target', reason)

    def test_repeated_blocked_volume_call_becomes_valid_ask(self):
        import test_pipeline as support
        row = support.sample('train_1715')
        first = {'type': 'call', 'name': 'tv.control', 'tool_reason': 'Giảm âm lượng',
                 'goals': [{'id': 'g1', 'task': 'Giảm âm lượng', 'kind': 'write', 'when': 'always'}]}
        llm = support.ScriptedLLM([first, {'selected_arguments': ['volume']},
               {'arguments': {'action': 'volume', 'volume': 1},
                'value_reasons': {'action': 'Chỉnh âm lượng', 'volume': 'Mức nhỏ'}},
               {'type': 'call', 'name': 'tv.control', 'tool_reason': 'Thử lại'}])
        out = support.Agent(llm, self.catalog, support.ToolEnv, ROOT).run(row, row['audio_text'])
        self.assertEqual(out['stop_reason'], 'ask_missing_info')
        self.assertEqual(out['tool_calls'][0]['name'], 'sys.ask_missing_info')

if __name__ == '__main__':
    unittest.main()
