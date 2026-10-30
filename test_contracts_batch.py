"""Real grammar compilation and batch isolation, without loading model weights."""
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
from test_pipeline import ROOT, sample, ScriptedLLM, call, args, vals


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
    def test_full_catalog_compiles_on_actual_decoder(self):
        catalog = load_catalog(ROOT)
        self.assertEqual(preflight_schemas(catalog), 1 + 4 * len(catalog))

    @unittest.skipUnless(importlib.util.find_spec('llguidance'), 'llguidance not installed')
    def test_real_decoder_unsupported_keyword_is_not_silently_stripped(self):
        with self.assertRaisesRegex(StructuredOutputError, 'uniqueItems'):
            validate_decoder_schema({'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True})

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_schemas_accept_only_control_keys_and_allow_missing_values(self):
        from jsonschema import validate, ValidationError
        catalog = load_catalog(ROOT)
        schema = node_schema('select', catalog)
        for action in (call('contacts.search'), {'type': 'done'},
                       {'type': 'ask', 'tool_name': 'sms.send', 'params': ['contact_id']}):
            validate(action, schema)
        for action in (dict(call('contacts.search'), reason='text'), {'type': 'revise'},
                       {'type': 'ask', 'tool_name': 'sms.send', 'params': ['unknown']}):
            with self.assertRaises(ValidationError):
                validate(action, schema)
        schema = node_schema('fill', catalog, catalog['sms.send'], ['content', 'contact_id'])
        validate({'arguments': {}}, schema)
        validate({'arguments': {'content': 'task text'}}, schema)
        for output in ({'arguments': {}, 'value_reasons': {}}, {'issue': {}}, {'arguments': {'phone_number': '1'}}):
            with self.assertRaises(ValidationError):
                validate(output, schema)

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_hhmm_compatible_with_env_format_instead_of_rfc_time(self):
        from jsonschema import validate, ValidationError
        spec = {'parameters': {'properties': {'at': {'type': 'string', 'format': 'time'}}}}
        schema = node_schema('fill', {}, spec, ['at'])
        validate({'arguments': {'at': '20:00'}}, schema)
        with self.assertRaises(ValidationError):
            validate({'arguments': {'at': '25:90'}}, schema)

    def test_mixed_node_batch_preserves_request_and_schema_ownership(self):
        model = BatchModel()
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
                self.assertEqual(messages, schema['const'])
        finally:
            scheduler.close()

    def test_partial_batch_and_failed_batch_release_waiters(self):
        for fail in (False, True):
            scheduler = BatchedLLM(BatchModel(fail), 4, 1)
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(scheduler.complete, 'only one')
                    if fail:
                        with self.assertRaises(InferenceBackendError):
                            future.result(timeout=3)
                    else:
                        self.assertEqual(future.result(timeout=3)['text'], 'only one')
            finally:
                scheduler.close()

    def test_shared_agent_keeps_environment_feedback_isolated_between_samples(self):
        class AutoModel(BatchModel):
            def count(self, messages):
                return 100

            def complete_batch(self, requests):
                outputs = []
                for messages, _, schema in requests:
                    payload = json.loads(messages[1]['content'])
                    request = payload['request']
                    self.calls.append(payload)
                    if 'selected_arguments' in payload:
                        output = vals(content=request)
                    elif 'selected_tool' in payload:
                        output = args()
                    elif payload['env_feedback']:
                        # Any cross-sample feedback would fail this assertion.
                        self_arg = payload['env_feedback'][-1]['arguments']
                        assert self_arg['content'] == request
                        output = {'type': 'done'}
                    else:
                        output = call('sms.send')
                    # For SMS use phone_number from the request, with a custom branch.
                    if 'selected_tool' in payload and 'selected_arguments' not in payload:
                        output = args('phone_number')
                    if 'selected_arguments' in payload:
                        output = vals(content=request, phone_number='0900000000')
                    outputs.append({'text': json.dumps(output), 'prompt_tokens': 100,
                                    'generation_tokens': 5, 'latency_s': .01})
                return outputs

        model = AutoModel()
        scheduler = BatchedLLM(model, 2, 10)
        rows = [r for r in load_train(ROOT, 0) if any('sms.send' in app['tools']
                for app in r['context']['installed_apps'].values())][:2]
        self.assertEqual(len(rows), 2)
        transcripts = {row['id']: 'independent task ' + row['id'] for row in rows}
        try:
            agent = Agent(scheduler, load_catalog(ROOT), import_toolenv(ROOT), ROOT)
            outcomes = list(iter_outcomes(agent, rows, transcripts, workers=2))
            self.assertEqual([o['stop_reason'] for o in outcomes], ['done', 'done'])
            self.assertEqual([len(o['tool_calls']) for o in outcomes], [1, 1])
            self.assertEqual([len(o['trace']) for o in outcomes], [2, 2])
            first_payloads = [p for p in model.calls if 'selected_tool' not in p and not p['env_feedback']]
            self.assertEqual(len(first_payloads), 2)
        finally:
            scheduler.close()

    def test_vllm_backend_errors_are_distinct_from_context_failures(self):
        adapter = QwenVLLM.__new__(QwenVLLM)
        adapter.temperature, adapter.seed, adapter.max_model_len = 0, 42, 10
        adapter.encode = lambda messages: [1, 2]
        adapter.SamplingParams = lambda **kw: kw
        def fail(*a, **kw):
            raise ValueError('engine failed')
        adapter.engine = SimpleNamespace(generate=fail)
        with self.assertRaisesRegex(InferenceBackendError, 'engine failed'):
            adapter.complete([], max_new_tokens=2)
        with self.assertRaises(ContextLimitError):
            adapter.complete([], max_new_tokens=20)


if __name__ == '__main__':
    unittest.main()
