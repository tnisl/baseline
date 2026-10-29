"""Check sampling arguments without loading model weights or requiring a GPU."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import Qwen, QwenVLLM  # noqa: E402


class SamplingTests(unittest.TestCase):
    def test_vllm_uses_selected_temperature(self):
        adapter = QwenVLLM.__new__(QwenVLLM)
        adapter.temperature = 0.7
        adapter.seed = 42
        adapter.max_model_len = 100
        adapter.tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: [1, 2],
        )
        adapter.SamplingParams = lambda **kwargs: kwargs
        captured = {}

        def generate(prompt, params, use_tqdm):
            captured.update(params)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="{}", token_ids=[3])])]

        adapter.engine = SimpleNamespace(generate=generate)
        adapter.complete([{"role": "user", "content": "test"}], max_new_tokens=8)
        self.assertEqual(captured["temperature"], 0.7)
        self.assertEqual(captured["top_p"], 0.8)
        self.assertEqual(captured["top_k"], 20)
        self.assertEqual(captured["seed"], 42)

    def test_transformers_samples_at_point_seven_and_can_use_greedy(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Sequence:
            def __getitem__(self, key):
                return SimpleNamespace(shape=(1,))

        for temperature in (0.7, 0):
            with self.subTest(temperature=temperature):
                adapter = Qwen.__new__(Qwen)
                adapter.temperature = temperature
                adapter.torch = SimpleNamespace(inference_mode=nullcontext)
                adapter.tokenizer = SimpleNamespace(
                    eos_token_id=0, decode=lambda *args, **kwargs: "{}",
                )
                captured = {}

                def generate(**kwargs):
                    captured.update(kwargs)
                    return [Sequence()]

                adapter.model = SimpleNamespace(device="cpu", generate=generate)
                adapter.encode = lambda messages: Inputs(
                    input_ids=SimpleNamespace(shape=(1, 2)),
                )
                adapter.complete([{"role": "user", "content": "test"}],
                                 max_new_tokens=8)
                self.assertEqual(captured["do_sample"], temperature > 0)
                if temperature > 0:
                    self.assertEqual(captured["temperature"], temperature)
                    self.assertEqual(captured["top_p"], 0.8)
                    self.assertEqual(captured["top_k"], 20)
                else:
                    self.assertNotIn("temperature", captured)


if __name__ == "__main__":
    unittest.main()
