"""Compile with the actual grammar dependency, without loading a GPU model."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_contracts import preflight_schemas, validate_decoder_schema, node_schema
from inference_errors import ContextLimitError, StructuredOutputError, InferenceBackendError
from models import QwenVLLM
from common import load_catalog
from agent import valid_ask
from agent_pipeline import Agent, select_argument_fields
from test_pipeline import ScriptedLLM, sample, ROOT, CALL, VALUES


class DecoderIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('llguidance'), 'Install llguidance>=1.3,<1.4 to compile grammars')
    def test_catalog_compiles_on_actual_guidance_backend(self):
        self.assertGreater(preflight_schemas(load_catalog(ROOT)), 500)

    @unittest.skipUnless(importlib.util.find_spec('llguidance'), 'llguidance not installed')
    def test_original_failure_is_detected_before_inference(self):
        with self.assertRaisesRegex(StructuredOutputError, 'uniqueItems'):
            validate_decoder_schema({'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True})

    def test_duplicate_validation_remains_in_python(self):
        spec = {'parameters': {'properties': {'query': {'type': 'string'}}}}
        self.assertIsNotNone(select_argument_fields({'selected_arguments': ['query', 'query']}, spec)[1])
        self.assertFalse(valid_ask('search', ['query', 'query'], {'search': spec}))

    def test_context_and_backend_errors_are_distinct(self):
        adapter = QwenVLLM.__new__(QwenVLLM)
        adapter.temperature, adapter.seed, adapter.max_model_len = 0, 42, 10
        adapter.encode = lambda messages: [1, 2]
        adapter.SamplingParams = lambda **kw: kw
        def fail(*args, **kw):
            raise ValueError('Grammar error: test')
        adapter.engine = SimpleNamespace(generate=fail)
        with self.assertRaisesRegex(InferenceBackendError, 'Grammar error'):
            adapter.complete([], max_new_tokens=2)
        with self.assertRaises(ContextLimitError):
            adapter.complete([], max_new_tokens=20)

    def test_backend_error_at_any_node_aborts_instead_of_scoring_case(self):
        from zaic_tools import ToolEnv
        row = sample('train_0486')
        catalog = load_catalog(ROOT)
        for fail_at in range(1, 4):
            class FailureLLM(ScriptedLLM):
                def __init__(self):
                    super().__init__([CALL, {'selected_arguments': []}, VALUES])
                    self.calls = 0
                def complete(self, *args, **kwargs):
                    self.calls += 1
                    if self.calls == fail_at:
                        raise InferenceBackendError('Grammar error: failed compilation')
                    return super().complete(*args, **kwargs)
            with self.subTest(fail_at=fail_at), self.assertRaises(InferenceBackendError):
                Agent(FailureLLM(), catalog, ToolEnv, ROOT).run(row, row['audio_text'])

    def test_true_context_overflow_keeps_per_sample_stop_reason(self):
        from zaic_tools import ToolEnv
        row = sample('train_0486')
        class TooLong(ScriptedLLM):
            def count(self, messages):
                return 100000
        result = Agent(TooLong([]), load_catalog(ROOT), ToolEnv, ROOT).run(row, row['audio_text'])
        self.assertEqual(result['stop_reason'], 'context_overflow')
        self.assertEqual(result['tool_calls'], [])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'report dependencies not installed')
    def test_aborted_run_removes_stale_summary(self):
        import tempfile
        from unittest.mock import patch
        import run
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'workflow_gold'
            output.mkdir()
            (output / 'summary.json').write_text('{"old":true}')
            with patch.object(run, '_agent_experiment', side_effect=InferenceBackendError('grammar failed')):
                with self.assertRaises(InferenceBackendError):
                    run.agent_experiment([], {}, None, directory, 'workflow_gold', None, {}, None)
            self.assertFalse((output / 'summary.json').exists())
            status = json.loads((output / 'run_status.json').read_text())
            self.assertEqual(status['status'], 'aborted')
            self.assertEqual(status['error_type'], 'InferenceBackendError')


if __name__ == '__main__':
    unittest.main()
