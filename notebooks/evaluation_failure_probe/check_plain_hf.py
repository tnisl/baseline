"""Offline CPU integration with the pinned real Qwen3/PEFT generation APIs."""
import tempfile,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'runtime'))
import umilife
umilife.__path__.append(str(H.parents[1]/'umilife'))
from umilife.eval_probe import cast_lora_dtype,verify_lora_dtype,restore_source_lora
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import LoraConfig,get_peft_model,PeftModel,get_peft_model_state_dict
config=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=48,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=64,pad_token_id=0,eos_token_id=2,attention_dropout=0.)
torch.manual_seed(42);base=Qwen3ForCausalLM(config)
model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
with tempfile.TemporaryDirectory() as temp:
 model.save_pretrained(temp)
 source_weights={k:v.detach().clone() for k,v in get_peft_model_state_dict(model).items()}
 loaded=PeftModel.from_pretrained(Qwen3ForCausalLM(config),temp,is_trainable=False,autocast_adapter_dtype=False).eval()
 base_before={n:p.detach().clone() for n,p in loaded.named_parameters() if 'lora_' not in n}
 for dtype in (torch.float16,torch.float32):
  cast_lora_dtype(loaded,dtype)
  restore_source_lora(loaded,source_weights,dtype)
  try:verify_lora_dtype(loaded,torch.float16 if dtype==torch.float32 else torch.float32);raise RuntimeError('Incorrect dtype accepted')
  except AssertionError:pass
  for ids,mask in [(torch.tensor([[3,4,5]]),torch.ones(1,3,dtype=torch.long)),(torch.tensor([[0,3,4],[3,4,5]]),torch.tensor([[0,1,1],[1,1,1]]))]:
   with torch.inference_mode():out=loaded.generate(input_ids=ids,attention_mask=mask,max_new_tokens=4,do_sample=False,num_beams=1,use_cache=True,logits_to_keep=1,pad_token_id=0,eos_token_id=[2],synced_gpus=False,return_dict_in_generate=False)
   assert out.shape[0]==len(ids) and 3<out.shape[1]<=7
   verify_lora_dtype(loaded,dtype)
  for n,p in loaded.named_parameters():
   if n in base_before:assert p.dtype==base_before[n].dtype and torch.equal(p,base_before[n])
assert not any(n=='unsloth' or n.startswith(('unsloth.','unsloth_zoo')) for n in sys.modules)
print('PASS: real pinned Qwen3+PEFT adapter load, explicit FP32/FP16 LoRA casts preserve base weights, left-padded batched greedy generate, logits_to_keep=1, no Unsloth import (CPU only).')
