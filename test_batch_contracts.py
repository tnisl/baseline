"""Regression tests for cross-sample scheduling and bounded semantic guards."""
import unittest
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_inference import BatchedLLM
from output_contracts import node_schema
from semantic_checks import semantic_write_error, temporal_values, goal_result_error
from pipeline_rules import repeated_write
from intent_contract import request_done_error


class FakeModel:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail
    def complete_batch(self, requests):
        self.calls.append(requests)
        if self.fail:
            raise ValueError('batch failed')
        return [{'text': r[0], 'prompt_tokens': 1, 'generation_tokens': 1} for r in requests]


class BatchContractTests(unittest.TestCase):
    def test_batches_distinct_requests_and_preserves_ownership(self):
        model = FakeModel()
        scheduler = BatchedLLM(model, batch_size=4, wait_ms=100)
        barrier = Barrier(4)
        def work(i):
            barrier.wait()
            return scheduler.complete(str(i), output_schema={'const': str(i)})['text']
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(work, i) for i in range(4)]
                self.assertEqual([f.result(timeout=3) for f in futures], ['0', '1', '2', '3'])
            self.assertEqual(len(model.calls), 1)
            for messages, _, schema in model.calls[0]:
                self.assertEqual(schema['const'], messages)
        finally:
            scheduler.close()

    def test_partial_batch_and_exception_release_waiters(self):
        for fail in (False, True):
            scheduler = BatchedLLM(FakeModel(fail), 4, 1)
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(scheduler.complete, 'single')
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'batch failed'):
                            future.result(timeout=3)
                    else:
                        self.assertEqual(future.result(timeout=3)['text'], 'single')
            finally:
                scheduler.close()

    def test_done_schema_indices_are_integer(self):
        schema = node_schema('select', {'tool': {}}, goals=[])
        done = schema['anyOf'][2]
        self.assertEqual(done['properties']['completed_goals']['additionalProperties']['type'], 'integer')

    def test_duplicate_email_punctuation(self):
        prior = {'name': 'email.reply', 'arguments': {'email_id': 'x', 'body': 'Mai hai giờ.'}, 'result': {'ok': True}}
        self.assertTrue(repeated_write(prior, 'email.reply', {'email_id': 'x', 'body': 'Mai hai giờ'}))
        self.assertFalse(repeated_write(prior, 'email.reply', {'email_id': 'y', 'body': 'Mai hai giờ'}))
        self.assertFalse(repeated_write(prior, 'email.reply', {'email_id': 'x', 'body': 'Mai ba giờ'}))

    def test_corrected_study_duration(self):
        request = 'Học từ chín giờ, à nhầm, chín rưỡi đến mười giờ hai mươi, cộng vào nhé.'
        self.assertEqual(semantic_write_error('learning.log_progress', {'minutes': 130}, request, {}, [])[0], 'minutes')
        self.assertIsNone(semantic_write_error('learning.log_progress', {'minutes': 50}, request, {}, [])[0])

    def test_invalid_calendar_interval(self):
        obs = [{'name': 'calendar.list_events', 'result': {'ok': True, 'events': [
            {'id': 'e', 'start': '2026-10-01T09:00:00+07:00', 'end': '2026-10-01T11:00:00+07:00'}]}}]
        self.assertEqual(semantic_write_error('calendar.update_event', {'event_id': 'e', 'start': '2026-10-01T14:00:00+07:00'}, '', {}, obs)[0], 'end')

    def test_missing_climate_and_notification_block_done(self):
        self.assertIn('climate_set', request_done_error('Bật đèn, quạt thì đừng bật, với bật điều hoà 26 độ.', {'smart_home.climate_set': {}}, []))
        self.assertIn('send', request_done_error('Sửa lịch, xong nhắn chị điều phối.', {'chat_ott.send_message': {}}, []))

    def test_relative_time_keeps_timezone(self):
        row = {'context': {'current_time': '2026-10-05T09:00:00+07:00'},
               'history': [{'role': 'user', 'audio_text': 'Thứ tư họp lúc chín giờ sáng.'}]}
        self.assertEqual(temporal_values('video_meeting.create', 'Vẫn đúng giờ đó.', row, [])['start'],
                         '2026-10-07T09:00:00+07:00')
        row['history'].append({'role': 'user', 'audio_text': 'Thứ năm họp lúc mười giờ.'})
        self.assertEqual(temporal_values('video_meeting.create', 'Vẫn đúng giờ đó.', row, []), {})

    def test_goal_cannot_use_other_bill_type(self):
        observation = {'name': 'bill_payment.pay', 'kind': 'write',
                       'result': {'ok': True, 'bill': {'type': 'internet'}}}
        self.assertIsNotNone(goal_result_error('Trả tiền nước', observation))
        self.assertIsNone(goal_result_error('Trả tiền mạng', observation))


if __name__ == '__main__':
    unittest.main()
