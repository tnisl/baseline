"""CPU-sized Qwen3 test for the actual completion loss and PEFT global-to-local copy."""
import importlib.util
import tempfile
from pathlib import Path
import unittest

HAS_STACK=all(importlib.util.find_spec(x) is not None for x in ("torch","transformers","peft"))

@unittest.skipUnless(HAS_STACK,"Install torch/transformers/peft to verify actual tensor updates")
class TorchIntegration(unittest.TestCase):
    def test_local_continuations_start_at_global_and_update_independently(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from peft import LoraConfig,get_peft_model,PeftModel
        from umilife.train import completion_loss,adapter_state
        torch.manual_seed(42)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            max_position_embeddings=128,attention_dropout=0.)
        original=Qwen3ForCausalLM(cfg);base_state={k:v.detach().clone() for k,v in original.state_dict().items()}
        model=get_peft_model(original,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
            target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
        batch={"input_ids":torch.tensor([[1,2,3,4,5,6]]),"attention_mask":torch.ones(1,6,dtype=torch.long),
               "labels":torch.tensor([[-100,-100,-100,4,5,6]])}
        before=adapter_state(model)
        optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        for _ in range(2):
            optim.zero_grad();loss=completion_loss(model,batch);self.assertTrue(torch.isfinite(loss));loss.backward();optim.step()
        global_state=adapter_state(model)
        self.assertNotEqual(before["sha256"],global_state["sha256"])
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            finals=[]
            for _ in range(3):
                backbone=Qwen3ForCausalLM(cfg);backbone.load_state_dict(base_state)
                local=PeftModel.from_pretrained(backbone,tmp,is_trainable=True)
                self.assertEqual(adapter_state(local)["sha256"],global_state["sha256"])
                optimizer=torch.optim.AdamW([p for p in local.parameters() if p.requires_grad],lr=.005)
                optimizer.zero_grad();completion_loss(local,batch).backward();optimizer.step()
                result=adapter_state(local);self.assertTrue(result["finite"])
                self.assertNotEqual(result["sha256"],global_state["sha256"]);finals.append(result)
            self.assertEqual(adapter_state(model)["sha256"],global_state["sha256"])
if __name__=="__main__":unittest.main()
