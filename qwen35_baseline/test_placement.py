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
        self.model.generate.return_value.__getitem__.return_value.tolist.return_value = [99]
        self.tokens = MagicMock()
        self.tokens.shape = (1, 2)
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = {"input_ids": self.tokens}
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
        generator.tokenizer.apply_chat_template.assert_called_once_with(
            [], tools=[], tokenize=False, add_generation_prompt=True, enable_thinking=False)

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
