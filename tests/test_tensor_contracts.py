"""Small CPU checks for masking, real Qwen3.5 PEFT export/reload and generation."""
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import transformers
    import peft
    TENSORS = True
except ImportError:
    TENSORS = False

from twonode.tokenization import CompletionCollator, encode_example


class TinyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["return_dict"] is False
        return [3, 4, 5]

    def encode(self, text, **kwargs):
        return [6, 7]


@unittest.skipUnless(TENSORS, "Requires Torch, Transformers and PEFT")
class TensorContracts(unittest.TestCase):
    def test_completion_mask_and_padding(self):
        row = {"key": "e:0", "messages": [], "target": {"stop": True}}
        encoded = encode_example(TinyTokenizer(), row, 32, 4)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 6, 7, 2])
        batch = CompletionCollator(0)([encoded, {"input_ids": [3, 6, 2], "attention_mask": [1]*3, "labels": [-100, 6, 2]}])
        self.assertEqual(batch["input_ids"].shape, (2, 8))
        self.assertEqual(batch["labels"][1].tolist(), [-100, 6, 2, -100, -100, -100, -100, -100])
        with self.assertRaises(ValueError):
            encode_example(TinyTokenizer(), row, 5, 4)

    def test_qwen35_lora_updates_reload_with_plain_transformers(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, AutoModelForImageTextToText
        from peft import LoraConfig, PeftModel, get_peft_model
        torch.manual_seed(42)
        torch.set_num_threads(1)
        config = Qwen3_5Config(
            text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                         "head_dim": 8, "layer_types": ["linear_attention", "full_attention"], "full_attention_interval": 2,
                         "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                         "linear_key_head_dim": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4,
                         "max_position_embeddings": 64, "eos_token_id": 2, "pad_token_id": 0,
                         "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                                             "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                           "out_hidden_size": 32, "patch_size": 2, "spatial_merge_size": 1,
                           "num_position_embeddings": 16},
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        base = Qwen3_5ForConditionalGeneration(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base.save_pretrained(root / "base")
            model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
            ids = torch.tensor([[3, 4, 5, 6, 2]])
            labels = torch.tensor([[-100, -100, -100, 6, 2]])
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            model.train()
            loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad and p.grad is not None))
            optimizer.step()
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before))
            model.eval()
            with torch.no_grad():
                expected = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            model.save_pretrained(root / "adapter")
            plain = AutoModelForImageTextToText.from_pretrained(root / "base")
            restored = PeftModel.from_pretrained(plain, root / "adapter").eval()
            with torch.no_grad():
                actual = restored(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
                output = restored.generate(ids[:, :3], attention_mask=torch.ones_like(ids[:, :3]),
                                           do_sample=False, max_new_tokens=2, pad_token_id=0, eos_token_id=2)
            torch.testing.assert_close(expected, actual)
            self.assertGreater(output.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
