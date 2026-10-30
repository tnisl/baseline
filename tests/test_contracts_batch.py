"""Constrained decoding and multi-sample scheduler contracts, no GPU required."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from threading import Barrier
from types import SimpleNamespace
import unittest

from helpers import ROOT, plan, call
from lifemate_v6.batching import BatchedLLM, iter_outcomes
from lifemate_v6.common import load_catalog, load_train, import_toolenv
from lifemate_v6.pipeline import Agent
from lifemate_v6.contracts import node_schema, preflight_schemas, validate_decoder_schema
from lifemate_v6.errors import StructuredOutputError, InferenceBackendError, ContextLimitError
from lifemate_v6.models import QwenVLLM


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
    def test_actual_guidance_compiles_catalog_and_instruction_schema(self):
        catalog = load_catalog(ROOT)
        self.assertEqual(preflight_schemas(catalog), 2 + len(set(catalog) - {'sys.ask_missing_info'}))

    @unittest.skipUnless(importlib.util.find_spec('llguidance'), 'llguidance not installed')
    def test_unsupported_decoder_keywords_fail_explicitly(self):
        with self.assertRaisesRegex(StructuredOutputError, 'uniqueItems'):
            validate_decoder_schema({'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True})

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_planner_schema_closes_fields_and_caps_instruction(self):
        from jsonschema import validate, ValidationError
        schema = node_schema('plan', load_catalog(ROOT))
        for output in (plan(), {'next': 'done'}, {'next': 'give_up'}):
            validate(output, schema)
        for output in (plan(''), plan('x' * 241), plan('a\nb'), dict(plan(), reason='why'),
                       {'tools': ['contacts.search']}, {'next': 'caller'}, {'next': 'done', 'instruction': 'extra'}):
            with self.subTest(output=output), self.assertRaises(ValidationError):
                validate(output, schema)
        with self.assertRaises(ValidationError):
            validate(plan(), node_schema('plan', {}))

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_caller_schema_requires_tool_specific_fields_and_does_not_allow_done(self):
        from jsonschema import validate, ValidationError
        catalog = load_catalog(ROOT)
        specs = {n: catalog[n] for n in ['sms.send', 'contacts.search', 'ewallet.get_balance', 'sys.ask_missing_info']}
        schema = node_schema('call', specs)
        for output in (call('contacts.search'), call('sms.send', content='text', contact_id='id'),
                       call('sms.send', content='text', phone_number='0900000000'), call('ewallet.get_balance'),
                       call('sys.ask_missing_info', tool_name='sms.send', params=['contact_id'])):
            validate(output, schema)
        for output in (call('sms.send', content='text'), call('sms.send', contact_id='id'),
                       call('contacts.search', contact_id='wrong field'), call('unknown'),
                       {'next': 'done'}, dict(call('contacts.search'), reason='why'),
                       {'type': 'call', **call('contacts.search')},
                       call('sys.ask_missing_info', tool_name='sms.send', params=['unknown'])):
            with self.subTest(output=output), self.assertRaises(ValidationError):
                validate(output, schema)

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_group_members_type_enum_and_hhmm(self):
        from jsonschema import validate, ValidationError
        spec = {'parameters': {'properties': {'id': {'type': 'string'}, 'a': {'type': 'number'},
            'b': {'type': 'number'}, 'at': {'type': 'string', 'format': 'time'},
            'kind': {'type': 'string', 'enum': ['a', 'b']}}, 'x-one-of': [['id'], ['a', 'b']]}}
        schema = node_schema('call', {'test.lookup': spec})
        validate(call('test.lookup', id='x', at='20:00', kind='a'), schema)
        validate(call('test.lookup', a=1, b=2), schema)
        for output in (call('test.lookup', a=1), call('test.lookup', a='1', b=2),
                       call('test.lookup', id='x', at='25:90'), call('test.lookup', id='x', kind='unknown')):
            with self.assertRaises(ValidationError):
                validate(output, schema)

    def test_batch_owns_each_prompt_and_schema(self):
        model, barrier = BatchModel(), Barrier(4)
        scheduler = BatchedLLM(model, 4, 100)
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

    def test_partial_batches_and_failures_release_waiters(self):
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

    def test_shared_agent_keeps_current_guidance_and_env_per_sample(self):
        class AutoModel(BatchModel):
            def count(self, messages):
                return 100

            def complete_batch(self, requests):
                outputs = []
                for messages, _, schema in requests:
                    p = json.loads(messages[1]['content'])
                    self.calls.append(p)
                    if 'planner_instruction' in p:
                        assert p['request'] in p['planner_instruction']
                        if p['request'].startswith('sms:'):
                            output = call('sms.send', content=p['request'], phone_number='0900000000')
                        else:
                            output = call('ewallet.get_balance')
                    elif p['env_feedback']:
                        target = 'sms.send' if p['request'].startswith('sms:') else 'ewallet.get_balance'
                        assert all(f['tool'] == target for f in p['env_feedback'])
                        if target == 'sms.send':
                            assert p['env_feedback'][-1]['arguments']['content'] == p['request']
                        output = {'next': 'done'}
                    else:
                        output = plan('Thực hiện ' + p['request'])
                    outputs.append({'text': json.dumps(output), 'prompt_tokens': 100,
                                    'generation_tokens': 5, 'latency_s': .01})
                return outputs

        rows = [r for r in load_train(ROOT, 0) if any('sms.send' in a['tools'] for a in r['context']['installed_apps'].values())
                and any('ewallet.get_balance' in a['tools'] for a in r['context']['installed_apps'].values())][:2]
        self.assertEqual(len(rows), 2)
        transcripts = {rows[0]['id']: 'sms:' + rows[0]['id'], rows[1]['id']: 'balance:' + rows[1]['id']}
        scheduler = BatchedLLM(AutoModel(), 2, 10)
        try:
            agent = Agent(scheduler, load_catalog(ROOT), import_toolenv(ROOT), ROOT)
            outcomes = list(iter_outcomes(agent, rows, transcripts, workers=2))
            self.assertEqual([o['stop_reason'] for o in outcomes], ['done', 'done'])
            self.assertEqual([o['planner_calls'] for o in outcomes], [2, 2])
            self.assertEqual([o['caller_calls'] for o in outcomes], [1, 1])
            self.assertEqual([len(o['tool_calls']) for o in outcomes], [1, 1])
        finally:
            scheduler.close()

    def test_backend_error_and_context_error_remain_distinct(self):
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
