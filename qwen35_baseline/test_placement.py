"""Test dispatch and input routing without allocating a real model or GPU."""
import contextlib
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from run import HFGenerator


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.model.eval.return_value = self.model
        self.model.hf_device_map = {"layers.0": 0, "layers.16": 1}
        self.model.get_input_embeddings.return_value.weight.device = "cuda:1"
        self.model.generation_config.eos_token_id = 99
        self.model.generate.return_value.__getitem__.return_value.tolist.return_value = [[99]]
        self.tokens = MagicMock()
        self.tokens.shape = (1, 2)
        self.tokenizer = MagicMock()
        self.mask = MagicMock()
        self.tokenizer.return_value = {"input_ids": [7, 8], "attention_mask": [1, 1]}
        self.tokenizer.pad.return_value = {"input_ids": self.tokens, "attention_mask": self.mask}
        self.tokenizer.pad_token_id = 0
        self.tokenizer.decode.return_value = "Done"
        self.factory = MagicMock()
        self.factory.from_pretrained.return_value = self.model
        tokenizer_factory = MagicMock()
        tokenizer_factory.from_pretrained.return_value = self.tokenizer
        self.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
        self.torch = SimpleNamespace(cuda=self.cuda, float16="float16",
                                     inference_mode=contextlib.nullcontext)
        self.modules = {"torch": self.torch, "transformers": SimpleNamespace(
            AutoModelForImageTextToText=self.factory, AutoTokenizer=tokenizer_factory)}

    def load(self, count=2):
        with patch.dict(sys.modules, self.modules), contextlib.redirect_stdout(io.StringIO()):
            return HFGenerator("model", "revision", 8192, 768, count)

    def test_two_gpu_dispatch_and_embedding_input_device(self):
        generator = self.load()
        kwargs = self.factory.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], "balanced")
        self.assertEqual(kwargs["max_memory"], {0: "10GiB", 1: "10GiB"})
        self.assertEqual(kwargs["dtype"], "float16")
        self.assertEqual(generator([], []), "Done")
        self.tokens.to.assert_called_once_with("cuda:1")
        self.mask.to.assert_called_once_with("cuda:1")
        self.assertEqual(self.tokenizer.padding_side, "left")
        generator.tokenizer.apply_chat_template.assert_called_once_with(
            [], tools=[], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def test_batch_generates_once_and_trims_each_row_at_first_eos(self):
        generator = self.load()
        rows = [[10, 99, 0], [11, 12, 99], [13, 14, 15], [99, 0, 0]]
        self.model.generate.return_value.__getitem__.return_value.tolist.return_value = rows
        self.tokenizer.decode.side_effect = lambda tokens, **kwargs: str(tokens)
        results = generator.generate_batch([([], [])] * 4)
        self.model.generate.assert_called_once()
        self.assertEqual(results[:2], ["[10, 99]", "[11, 12, 99]"])
        self.assertIsInstance(results[2], ValueError)
        self.assertEqual(results[3], "[99]")
        self.assertEqual(generator.stats[0]["batch_size"], 4)
        self.assertEqual(generator.stats[0]["output_tokens"], [2, 3, 3, 1])
        self.tokenizer.pad.assert_called_once_with(
            [self.tokenizer.return_value] * 4, padding=True, return_tensors="pt")
        self.model.generate.return_value.__getitem__.assert_called_once_with(
            (slice(None), slice(2, None)))

    def test_overlong_prompt_fails_only_its_own_row(self):
        generator = self.load()
        short = {"input_ids": [7, 8], "attention_mask": [1, 1]}
        self.tokenizer.side_effect = [short, {"input_ids": [1] * 8193}, short]
        self.model.generate.return_value.__getitem__.return_value.tolist.return_value = [[99], [99]]
        results = generator.generate_batch([([], [])] * 3)
        self.assertEqual(results[0], "Done")
        self.assertIsInstance(results[1], ValueError)
        self.assertIn("8193", str(results[1]))
        self.assertEqual(results[2], "Done")
        self.tokenizer.pad.assert_called_once_with([short, short], padding=True, return_tensors="pt")

    def test_shared_cuda_failure_is_reported_for_every_affected_row(self):
        generator = self.load()
        self.model.generate.side_effect = RuntimeError("CUDA out of memory")
        results = generator.generate_batch([([], [])] * 4)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
        self.model.generate.assert_called_once()

    def test_requires_two_visible_gpus_before_loading(self):
        self.cuda.device_count = lambda: 1
        with self.assertRaisesRegex(RuntimeError, "T4 x2"):
            self.load()
        self.factory.from_pretrained.assert_not_called()

    def test_rejects_single_gpu_map_and_cpu_offload(self):
        for mapping in ({"": 0}, {"a": 0, "b": 1, "c": "cpu"}):
            self.model.hf_device_map = mapping
            with self.subTest(mapping=mapping), self.assertRaisesRegex(RuntimeError, "must span"):
                self.load()

    def test_explicit_one_gpu_mode(self):
        self.model.hf_device_map = {"": 0}
        generator = self.load(1)
        self.assertEqual(generator.num_gpus, 1)
        self.assertEqual(self.factory.from_pretrained.call_args.kwargs["device_map"], {"": 0})


if __name__ == "__main__":
    unittest.main()
