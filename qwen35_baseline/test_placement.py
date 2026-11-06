"""Verify vLLM scheduling, unrestricted decoding, and per-row errors without GPUs."""
import contextlib
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from run import VLLMGenerator


def completion(text="Done", finish="stop"):
    return SimpleNamespace(outputs=[SimpleNamespace(
        text=text, finish_reason=finish, token_ids=[11, 99])])


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = {"input_ids": [7, 8]}
        self.tokenizer.convert_tokens_to_ids.return_value = 99
        self.model = MagicMock()
        self.model.get_tokenizer.return_value = self.tokenizer
        self.model.llm_engine.model_config.hf_config._commit_hash = "resolved"
        self.model.generate.return_value = [completion()]
        self.factory = MagicMock(return_value=self.model)
        self.sampling = MagicMock()
        self.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                    mem_get_info=lambda i: (2**30, 4*2**30))
        self.modules = {"torch": SimpleNamespace(cuda=self.cuda), "vllm": SimpleNamespace(
            LLM=self.factory, SamplingParams=self.sampling)}

    def load(self, count=2):
        with patch.dict(sys.modules, self.modules), contextlib.redirect_stdout(io.StringIO()):
            return VLLMGenerator("model", "revision", 16384, 768, count)

    def test_tensor_parallel_memory_settings_and_no_format_enforcer(self):
        generator = self.load()
        kwargs = self.factory.call_args.kwargs
        for key, value in {"tensor_parallel_size": 2, "dtype": "float16",
                           "max_model_len": 17152, "max_num_seqs": 2,
                           "max_num_batched_tokens": 1024, "gpu_memory_utilization": 0.80,
                           "enable_chunked_prefill": True, "enforce_eager": True,
                           "language_model_only": True, "enable_prefix_caching": False,
                           "disable_custom_all_reduce": True}.items():
            self.assertEqual(kwargs[key], value)
        self.assertEqual(self.sampling.call_args.kwargs, {
            "temperature": 0, "max_tokens": 768,
            "structured_outputs": None, "stop_token_ids": [99]})
        self.assertEqual(generator([], []), "Done")
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [], tools=[], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(generator.resolved_revision, "resolved")

    def test_four_requests_have_independent_unpadded_prompts_and_results(self):
        generator = self.load()
        encoded = [{"input_ids": [1]}, {"input_ids": [2, 3]},
                   {"input_ids": [4, 5, 6]}, {"input_ids": [7]}]
        self.tokenizer.side_effect = encoded
        self.model.generate.return_value = [completion("A"), completion("B"),
                                            completion("truncated", "length"), completion("D")]
        results = generator.generate_batch([([], [])] * 4)
        self.assertEqual(results[:2], ["A", "B"])
        self.assertIsInstance(results[2], ValueError)
        self.assertEqual(results[3], "D")
        self.model.generate.assert_called_once_with(
            [{"prompt_token_ids": row["input_ids"]} for row in encoded],
            sampling_params=generator.sampling, use_tqdm=False)
        self.tokenizer.pad.assert_not_called()
        self.assertEqual(generator.stats[0]["input_tokens"], [1, 2, 3, 1])

    def test_overlong_prompt_fails_only_its_own_row(self):
        generator = self.load()
        self.tokenizer.side_effect = [{"input_ids": [1]}, {"input_ids": [1] * 16385}]
        results = generator.generate_batch([([], [])] * 2)
        self.assertEqual(results[0], "Done")
        self.assertIsInstance(results[1], ValueError)
        self.assertEqual(len(self.model.generate.call_args.args[0]), 1)

    def test_requires_two_visible_gpus_before_loading(self):
        self.cuda.device_count = lambda: 1
        with self.assertRaisesRegex(RuntimeError, "T4 x2"):
            self.load()
        self.factory.assert_not_called()

    def test_shared_failure_is_reported_without_retry(self):
        generator = self.load()
        self.model.generate.side_effect = RuntimeError("engine failure")
        results = generator.generate_batch([([], [])] * 4)
        self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
        self.model.generate.assert_called_once()

    def test_driver_memory_samples_include_worker_usage(self):
        generator = self.load()
        self.assertEqual(generator.observed_gpu_used_gib_per_gpu, {"0": 3, "1": 3})
        self.cuda.mem_get_info = lambda i: (2**30, 6*2**30)
        generator.sample_gpu_memory()
        self.assertEqual(generator.observed_gpu_used_gib_per_gpu, {"0": 5, "1": 5})


if __name__ == "__main__":
    unittest.main()
