from __future__ import annotations
import json
import math
import os
from pathlib import Path
import time

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def seed_all(seed=42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def load_base(model_id=MODEL_ID, four_bit=True, training=False, revision=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from .attention import configure_compatible_attention,native_bf16_supported
    if not torch.cuda.is_available(): raise RuntimeError("Qwen3-4B requires an enabled CUDA GPU")
    attention_policy = configure_compatible_attention(torch)
    print(json.dumps({"attention_policy": attention_policy}), flush=True)
    dtype = torch.bfloat16 if native_bf16_supported(torch) else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa")
    if revision: kwargs["revision"] = revision
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = not training
    model.attention_policy_report = attention_policy
    return tokenizer, model

class Backend:
    def __init__(self, model_id=MODEL_ID, adapter_dir=None, roles=(), four_bit=True,
                 max_context=8192, max_new_tokens=512, revision=None):
        seed_all()
        self.tokenizer, self.model = load_base(model_id, four_bit, revision=revision)
        self.base_parameters = self.model.num_parameters()
        self.roles = set(roles)
        if adapter_dir:
            from peft import PeftModel
            for i, role in enumerate(roles):
                path = Path(adapter_dir)/role
                meta = json.loads((path/"lineage.json").read_text())
                if meta["stage"] != role: raise ValueError("Adapter role/lineage mismatch")
                if i == 0:
                    self.model = PeftModel.from_pretrained(self.model, path, adapter_name=role)
                else:
                    self.model.load_adapter(path, adapter_name=role)
        self.model.eval()
        self.grammar_tokenizer=None
        if self.roles:
            from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
            self.grammar_tokenizer=build_token_enforcer_tokenizer_data(self.tokenizer)
        self.max_context, self.max_new_tokens = max_context, max_new_tokens
        if not 1 < max_context <= 16384: raise ValueError("Competition context limit is 16384")
        adapter_parameters = sum(p.numel() for n,p in self.model.named_parameters() if "lora_" in n)
        self.parameter_report = {"base": self.base_parameters, "adapters": adapter_parameters,
                                 "total_llm": self.base_parameters + adapter_parameters}
        if self.parameter_report["total_llm"] >= 5_000_000_000:
            raise ValueError("LLM parameter budget exceeded")

    def generate(self, role, messages, remaining_s=30, max_new_tokens=None):
        return self.generate_batch(role, [messages], remaining_s, max_new_tokens)[0]

    def generate_batch(self, role, conversations, remaining_s=600, max_new_tokens=None):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        if self.roles:
            if role not in self.roles: raise ValueError("Missing adapter: " + role)
            self.model.set_adapter(role)
        self.tokenizer.padding_side = "left"
        texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in conversations]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        new_tokens = max_new_tokens or (192 if role=="planner" else self.max_new_tokens)
        if inputs["input_ids"].shape[1] + new_tokens > self.max_context:
            raise ValueError("Prompt exceeds context budget; refusing evidence truncation")
        inputs = {k:v.to(self.model.device) for k,v in inputs.items()}
        deadline = time.monotonic() + remaining_s
        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline
        kwargs={}
        if self.grammar_tokenizer is not None:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            from .grammar import role_schema
            kwargs["prefix_allowed_tokens_fn"]=build_transformers_prefix_allowed_tokens_fn(
                self.grammar_tokenizer,JsonSchemaParser(role_schema(role)))
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False,
                use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([Deadline()]),**kwargs)
        if time.monotonic() >= deadline: raise TimeoutError("Generation exceeded sample deadline")
        return self.tokenizer.batch_decode(output[:,inputs["input_ids"].shape[1]:], skip_special_tokens=True)

class ASR:
    def __init__(self, model_path="Systran/faster-whisper-small", device="cuda"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_path, device=device,
                                  compute_type="int8_float16" if device=="cuda" else "int8")
        self.parameter_count = 244_000_000

    def transcribe(self, path):
        segments, _ = self.model.transcribe(str(path), language="vi", beam_size=1,
            best_of=1, temperature=0, condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()
