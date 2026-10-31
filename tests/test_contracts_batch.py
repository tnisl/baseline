"""Fixed caller schemas, state-dependent summary routes and concurrent isolation."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from threading import Barrier
from types import SimpleNamespace
import unittest

from helpers import ROOT, plan, call, review
from lifemate_v8_finetune.batching import BatchedLLM, iter_outcomes
from lifemate_v8_finetune.common import load_catalog, load_train, import_toolenv
from lifemate_v8_finetune.pipeline import Agent
from lifemate_v8_finetune.contracts import node_schema, preflight_schemas, validate_decoder_schema
from lifemate_v8_finetune.errors import StructuredOutputError, InferenceBackendError, ContextLimitError
from lifemate_v8_finetune.models import QwenVLLM


class BatchModel:
    structured_outputs = True

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def complete_batch(self, requests):
        self.calls.append(requests)
        if self.fail:
            raise InferenceBackendError('batch failed')
        return [{'text': r[0], 'prompt_tokens': 1, 'generation_tokens': 1} for r in requests]


class ContractBatchTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('llguidance'), 'llguidance not installed')
    def test_full_catalog_and_every_fixed_caller_schema_compile(self):
        catalog = load_catalog(ROOT)
        self.assertEqual(preflight_schemas(catalog), 12 + len(set(catalog) - {'sys.ask_missing_info'}))

    @unittest.skipUnless(importlib.util.find_spec('llguidance'), 'llguidance not installed')
    def test_unsupported_keyword_is_not_silently_stripped(self):
        with self.assertRaisesRegex(StructuredOutputError, 'uniqueItems'):
            validate_decoder_schema({'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True})

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_current_tool_arguments_are_enforced_without_name_or_done(self):
        from jsonschema import validate, ValidationError
        cat = load_catalog(ROOT)
        schema = node_schema('call', cat, selected_tool='sms.send')
        for output in (call(content='text', contact_id='id'), call(content='text', phone_number='0900000000'),
                       {'type': 'skip'}, {'type': 'blocked'}, {'type': 'ask', 'params': ['contact_id']}):
            validate(output, schema)
        for output in (call(content='text'), call(contact_id='id'), call(query='wrong tool'),
                       call(), dict(call(content='text', contact_id='id'), name='sms.send'),
                       {'type': 'done'}, dict(call(content='text', contact_id='id'), reason='text'),
                       {'type': 'ask', 'params': ['unknown']}, {'type': 'skip', 'note': 'text'}):
            with self.subTest(output=output), self.assertRaises(ValidationError):
                validate(output, schema)
        validate(call(), node_schema('call', cat, selected_tool='ewallet.get_balance'))
        with self.assertRaises(ValidationError):
            validate(call(balance=1), node_schema('call', cat, selected_tool='ewallet.get_balance'))
        for selected in (None, 'unknown', 'sys.ask_missing_info'):
            with self.assertRaises(ValueError):
                node_schema('call', cat, selected_tool=selected)

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_planner_allows_repeated_ordered_slots_and_closes_control_fields(self):
        from jsonschema import validate, ValidationError
        cat = load_catalog(ROOT)
        schema = node_schema('plan', cat, max_plan_steps=3)
        validate(plan(), schema)
        validate(plan('sms.send', 'contacts.search', 'sms.send'), schema)
        for output in (plan('unknown'), dict(plan(), reason='text'), {'type': 'done'},
                       plan('sys.ask_missing_info'), plan(*(['sms.send'] * 4))):
            with self.assertRaises(ValidationError):
                validate(output, schema)
        validate(plan(), node_schema('plan', {}))
        with self.assertRaises(ValidationError):
            validate(plan('sms.send'), node_schema('plan', {}))

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_x_one_of_group_means_all_members_in_one_group(self):
        from jsonschema import validate, ValidationError
        spec = {'parameters': {'properties': {'id': {'type': 'string'}, 'a': {'type': 'number'},
                                             'b': {'type': 'number'}}, 'x-one-of': [['id'], ['a', 'b']]}}
        schema = node_schema('call', {'test.lookup': spec}, selected_tool='test.lookup')
        validate(call(id='x'), schema)
        validate(call(a=1, b=2), schema)
        with self.assertRaises(ValidationError):
            validate(call(a=1), schema)
        with self.assertRaises(ValidationError):
            validate({'type': 'ask', 'params': ['id']}, schema)  # sys.ask_missing_info unavailable

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_field_type_enum_and_time_are_constrained(self):
        from jsonschema import validate, ValidationError
        spec = {'parameters': {'properties': {'at': {'type': 'string', 'format': 'time'},
                  'kind': {'type': 'string', 'enum': ['a', 'b']}, 'count': {'type': 'integer', 'minimum': 1}},
                  'required': ['at', 'kind', 'count']}}
        schema = node_schema('call', {'test.create': spec}, selected_tool='test.create')
        validate(call(at='20:00', kind='a', count=1), schema)
        for values in ({'at': '25:90', 'kind': 'a', 'count': 1},
                       {'at': '20:00', 'kind': 'other', 'count': 1},
                       {'at': '20:00', 'kind': 'a', 'count': '1'}):
            with self.assertRaises(ValidationError):
                validate(call(**values), schema)

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_summary_schema_binds_routes_to_actual_step_status_and_remaining_plan(self):
        from jsonschema import validate, ValidationError
        catalog = load_catalog(ROOT)
        scenarios = [
            ('executed', 1, [review(), review('give_up')], [review('done'), review('retry'), review('wait')]),
            ('skipped', 0, [review('done'), review('give_up')], [review(), review('retry')]),
            ('tool_failed', 1, [review('retry'), review('give_up')], [review(), review('done')]),
            ('validation_failed', 1, [review('retry')], [review(), review('done')]),
            ('blocked', 1, [review('give_up')], [review(), review('retry'), review('done')]),
            ('waiting_for_user', 1, [review('wait')], [review(), review('retry'), review('done'), review('give_up')]),
            ('empty_plan', 0, [review('done'), review('give_up')], [review(), review('retry')]),
        ]
        for status, remaining, valid, invalid in scenarios:
            schema = node_schema('summarize', catalog, step_status=status, remaining_plan_steps=remaining)
            for output in valid:
                validate(output, schema)
            for output in invalid:
                with self.subTest(status=status, output=output), self.assertRaises(ValidationError):
                    validate(output, schema)
        with self.assertRaises(ValueError):
            node_schema('summarize', catalog)

    def test_mixed_role_batch_preserves_prompt_schema_ownership(self):
        model = BatchModel()
        scheduler = BatchedLLM(model, 4, 100)
        barrier = Barrier(4)
        def work(i):
            barrier.wait()
            return scheduler.complete(str(i), output_schema={'const': str(i)})['text']
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(work, i) for i in range(4)]
                self.assertEqual([f.result(timeout=3) for f in futures], ['0', '1', '2', '3'])
            self.assertEqual(len(model.calls), 1)
            for messages, _, schema in model.calls[0]:
                self.assertEqual(messages, schema['const'])
        finally:
            scheduler.close()

    def test_partial_batch_and_failed_batch_release_waiters(self):
        for fail in (False, True):
            scheduler = BatchedLLM(BatchModel(fail), 4, 1)
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(scheduler.complete, 'single')
                    if fail:
                        with self.assertRaises(InferenceBackendError):
                            future.result(timeout=3)
                    else:
                        self.assertEqual(future.result(timeout=3)['text'], 'single')
            finally:
                scheduler.close()

    def test_shared_agent_keeps_plan_cursor_caller_schema_and_summary_route_per_sample(self):
        class AutoModel(BatchModel):
            def count(self, messages):
                return 100

            def complete_batch(self, requests):
                outputs = []
                for messages, _, schema in requests:
                    p = json.loads(messages[1]['content'])
                    self.calls.append(p)
                    if 'tool_plan' not in p:
                        tools = (['sms.send', 'ewallet.get_balance'] if p['request'].startswith('sms:')
                                 else ['bill_payment.list_bills', 'ewallet.get_balance', 'sms.send'])
                        output = plan(*tools)
                    elif 'step_status' in p:
                        for f in p['env_feedback']:
                            if f['tool'] == 'sms.send':
                                assert f['arguments']['content'] == p['request']
                        assert p['step_status'] == 'executed'
                        assert len(p['step_results']) == p['step_index']
                        output = review('done') if p['remaining_plan_steps'] == 0 else review()
                    else:
                        assert p['tool_plan'][p['step_index']] == p['selected_tool']['name']
                        if p['selected_tool']['name'] == 'sms.send':
                            output = call(content=p['request'], phone_number='0900000000')
                        else:
                            output = call()
                    outputs.append({'text': json.dumps(output), 'prompt_tokens': 100,
                                    'generation_tokens': 5, 'latency_s': .01})
                return outputs

        wanted = {'sms.send', 'ewallet.get_balance', 'bill_payment.list_bills'}
        rows = [r for r in load_train(ROOT, 0)
                if wanted <= {n for a in r['context']['installed_apps'].values() for n in a['tools']}][:2]
        self.assertEqual(len(rows), 2)
        transcripts = {rows[0]['id']: 'sms:' + rows[0]['id'], rows[1]['id']: 'balance:' + rows[1]['id']}
        scheduler = BatchedLLM(AutoModel(), 2, 10)
        try:
            agent = Agent(scheduler, load_catalog(ROOT), import_toolenv(ROOT), ROOT)
            outcomes = list(iter_outcomes(agent, rows, transcripts, workers=2))
            self.assertEqual([o['stop_reason'] for o in outcomes], ['done', 'done'])
            self.assertEqual([o['tool_plan'] for o in outcomes],
                             [['sms.send', 'ewallet.get_balance'],
                              ['bill_payment.list_bills', 'ewallet.get_balance', 'sms.send']])
            self.assertEqual([o['planner_calls'] for o in outcomes], [1, 1])
            self.assertEqual([o['processed_steps'] for o in outcomes], [2, 3])
            self.assertEqual([o['summarizer_calls'] for o in outcomes], [2, 3])
            self.assertEqual([o['llm_calls'] for o in outcomes], [5, 7])
            self.assertEqual([len(o['tool_calls']) for o in outcomes], [2, 3])
        finally:
            scheduler.close()

    def test_backend_error_is_distinct_from_context_failure(self):
        adapter = QwenVLLM.__new__(QwenVLLM)
        adapter.temperature, adapter.seed, adapter.max_model_len = 0, 42, 10
        adapter.encode = lambda messages: [1, 2]
        adapter.SamplingParams = lambda **kwargs: kwargs
        def fail(*args, **kwargs):
            raise ValueError('engine failed')
        adapter.engine = SimpleNamespace(generate=fail)
        with self.assertRaisesRegex(InferenceBackendError, 'engine failed'):
            adapter.complete([], max_new_tokens=2)
        with self.assertRaises(ContextLimitError):
            adapter.complete([], max_new_tokens=20)


if __name__ == '__main__':
    unittest.main()
