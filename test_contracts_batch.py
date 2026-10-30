"""Actual decoder compatibility and two-role batch isolation, without GPU weights."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_inference import BatchedLLM, iter_outcomes
from common import load_catalog, load_train, import_toolenv
from agent_pipeline import Agent
from output_contracts import node_schema, preflight_schemas, validate_decoder_schema
from inference_errors import StructuredOutputError, InferenceBackendError, ContextLimitError
from models import QwenVLLM
from test_pipeline import ROOT, pool, call


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
    def test_full_catalog_and_each_caller_schema_compile(self):
        catalog = load_catalog(ROOT)
        self.assertEqual(preflight_schemas(catalog), 3 + len(set(catalog) - {'sys.ask_missing_info'}))

    @unittest.skipUnless(importlib.util.find_spec('llguidance'), 'llguidance not installed')
    def test_unsupported_keyword_is_not_silently_stripped(self):
        with self.assertRaisesRegex(StructuredOutputError, 'uniqueItems'):
            validate_decoder_schema({'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True})

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_call_schema_binds_tool_name_to_its_own_arguments(self):
        from jsonschema import validate, ValidationError
        cat = load_catalog(ROOT)
        schema = node_schema('call', {n: cat[n] for n in ['sms.send', 'contacts.search', 'ewallet.get_balance']})
        for output in (call('contacts.search'), call('contacts.search', query='a'),
                       call('sms.send', content='text', contact_id='id'),
                       call('sms.send', content='text', phone_number='0900000000'),
                       call('ewallet.get_balance'), {'type': 'done'},
                       {'type': 'ask', 'tool_name': 'sms.send', 'params': ['contact_id']}):
            validate(output, schema)
        for output in (call('sms.send', content='text'),  # missing x-one-of
                       call('sms.send', contact_id='id'),  # missing required
                       call('contacts.search', contact_id='id'),  # wrong tool's field
                       call('bill_payment.list_bills'),  # outside selected set
                       call('ewallet.get_balance', balance=1),  # result is not input
                       dict(call('contacts.search'), reason='text'),
                       {'type': 'ask', 'tool_name': 'sms.send', 'params': ['unknown']},
                       {'type': 'done', 'note': 'text'}):
            with self.subTest(output=output), self.assertRaises(ValidationError):
                validate(output, schema)

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_planner_can_select_empty_set_and_cannot_emit_control_prose(self):
        from jsonschema import validate, ValidationError
        cat = load_catalog(ROOT)
        schema = node_schema('plan', cat)
        validate(pool(), schema)
        validate(pool('sms.send', 'contacts.search'), schema)
        for output in (pool('unknown'), dict(pool(), reason='text'), {'type': 'done'}):
            with self.assertRaises(ValidationError):
                validate(output, schema)
        validate({'type': 'done'}, node_schema('call', {}))
        with self.assertRaises(ValidationError):
            validate(call('sms.send', content='text', contact_id='id'), node_schema('call', {}))

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_x_one_of_group_means_all_members_in_one_group(self):
        from jsonschema import validate, ValidationError
        spec = {'parameters': {'properties': {'id': {'type': 'string'}, 'a': {'type': 'number'},
                                             'b': {'type': 'number'}}, 'x-one-of': [['id'], ['a', 'b']]}}
        schema = node_schema('call', {'test.lookup': spec})
        validate(call('test.lookup', id='x'), schema)
        validate(call('test.lookup', a=1, b=2), schema)
        with self.assertRaises(ValidationError):
            validate(call('test.lookup', a=1), schema)

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_field_type_enum_and_time_are_constrained(self):
        from jsonschema import validate, ValidationError
        spec = {'parameters': {'properties': {'at': {'type': 'string', 'format': 'time'},
                  'kind': {'type': 'string', 'enum': ['a', 'b']}, 'count': {'type': 'integer', 'minimum': 1}},
                  'required': ['at', 'kind', 'count']}}
        schema = node_schema('call', {'test.create': spec})
        validate(call('test.create', at='20:00', kind='a', count=1), schema)
        for values in ({'at': '25:90', 'kind': 'a', 'count': 1},
                       {'at': '20:00', 'kind': 'other', 'count': 1},
                       {'at': '20:00', 'kind': 'a', 'count': '1'}):
            with self.assertRaises(ValidationError):
                validate(call('test.create', **values), schema)

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

    def test_shared_agent_isolates_each_tool_set_and_environment(self):
        class AutoModel(BatchModel):
            def count(self, messages):
                return 100

            def complete_batch(self, requests):
                outputs = []
                for messages, _, schema in requests:
                    p = json.loads(messages[1]['content'])
                    self.calls.append(p)
                    if 'selected_tools' not in p:
                        # Different sample gets a different tool set, not just different text.
                        target = 'sms.send' if p['request'].startswith('sms:') else 'ewallet.get_balance'
                        output = pool(target)
                    elif p['env_feedback']:
                        target = p['selected_tools'][0]
                        assert all(f['tool'] == target for f in p['env_feedback'])
                        if target == 'sms.send':
                            assert p['env_feedback'][-1]['arguments']['content'] == p['request']
                        output = {'type': 'done'}
                    elif p['selected_tools'] == ['sms.send']:
                        output = call('sms.send', content=p['request'], phone_number='0900000000')
                    else:
                        output = call('ewallet.get_balance')
                    outputs.append({'text': json.dumps(output), 'prompt_tokens': 100,
                                    'generation_tokens': 5, 'latency_s': .01})
                return outputs

        rows = [r for r in load_train(ROOT, 0) if any('sms.send' in a['tools'] for a in r['context']['installed_apps'].values())
                and any('ewallet.get_balance' in a['tools'] for a in r['context']['installed_apps'].values())][:2]
        self.assertEqual(len(rows), 2)
        transcripts = {rows[0]['id']: 'sms:' + rows[0]['id'], rows[1]['id']: 'balance:' + rows[1]['id']}
        model, ToolEnv = AutoModel(), import_toolenv(ROOT)
        scheduler = BatchedLLM(model, 2, 10)
        try:
            agent = Agent(scheduler, load_catalog(ROOT), ToolEnv, ROOT)
            outcomes = list(iter_outcomes(agent, rows, transcripts, workers=2))
            self.assertEqual([o['stop_reason'] for o in outcomes], ['done', 'done'])
            self.assertEqual([o['selected_tools'] for o in outcomes], [['sms.send'], ['ewallet.get_balance']])
            self.assertEqual([o['planner_calls'] for o in outcomes], [1, 1])
            self.assertEqual([len(o['tool_calls']) for o in outcomes], [1, 1])
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
