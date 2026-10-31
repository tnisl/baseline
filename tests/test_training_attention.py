"""Backend selection and exact output/gradient equivalence of KV repetition."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from helpers import BASE
from lifemate_v8_finetune.training.attention import configure_training_attention

AVAILABLE = all(importlib.util.find_spec(x) for x in ("torch", "transformers"))


@unittest.skipUnless(AVAILABLE, "Install torch/transformers for attention tests")
class TrainingAttentionTests(unittest.TestCase):
    def test_t4_uses_explicit_kv_and_fused_kernel_without_changing_ampere_policy(self):
        from transformers.integrations import sdpa_attention
        backend = SimpleNamespace(**{name: Mock() for name in (
            "enable_mem_efficient_sdp", "enable_flash_sdp", "enable_cudnn_sdp", "enable_math_sdp")})
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
            get_device_capability=lambda: (7, 5)), backends=SimpleNamespace(cuda=backend))
        with patch.object(sdpa_attention, "use_gqa_in_sdpa", return_value=True):
            record = configure_training_attention(fake)
            self.assertFalse(sdpa_attention.use_gqa_in_sdpa(None, None))
            self.assertEqual(record["policy"], "explicit_kv_fused_efficient_sdpa")
        backend.enable_mem_efficient_sdp.assert_called_once_with(True)
        backend.enable_math_sdp.assert_called_once_with(False)
        for call in vars(backend).values():
            call.reset_mock()
        fake.cuda.get_device_capability = lambda: (8, 0)
        with patch.object(sdpa_attention, "use_gqa_in_sdpa", return_value=True):
            record = configure_training_attention(fake)
            self.assertTrue(sdpa_attention.use_gqa_in_sdpa(None, None))
            self.assertEqual(record["policy"], "standard_sdpa")
        self.assertFalse(any(call.called for call in vars(backend).values()))

    def test_explicit_kv_preserves_attention_outputs_and_parameter_gradients(self):
        import torch
        from transformers.integrations import sdpa_attention
        torch.manual_seed(42)
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        originals = [torch.randn(1, h, 7, 8, dtype=torch.float64) for h in (4, 2, 2)]
        results = []
        for native_gqa in (True, False):
            tensors = [x.clone().requires_grad_() for x in originals]
            with patch.object(sdpa_attention, "use_gqa_in_sdpa", return_value=native_gqa):
                output, _ = sdpa_attention.sdpa_attention_forward(module, *tensors, None)
            output.square().sum().backward()
            results.append((output.detach(), [x.grad for x in tensors]))
        torch.testing.assert_close(results[0][0], results[1][0], atol=1e-10, rtol=1e-10)
        for a, b in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
