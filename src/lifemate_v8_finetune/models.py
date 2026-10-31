"""Lazy-loaded PhoWhisper and Qwen inference adapters."""
from __future__ import annotations

import time
from threading import RLock
from .errors import ContextLimitError, InferenceBackendError
from .adapters import adapter_rank, role_for_messages
import wave

from .common import audio_path


class PhoWhisper:
    def __init__(self, model_id="vinai/PhoWhisper-medium"):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.torch = torch
        self.model_id = model_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=self.dtype).to(self.device)
        self.model.eval()

    @staticmethod
    def read_wav(path):
        import numpy as np
        with wave.open(str(path), "rb") as handle:
            fmt = (handle.getframerate(), handle.getnchannels(),
                   handle.getsampwidth(), handle.getcomptype())
            if fmt != (16000, 1, 2, "NONE"):
                raise ValueError(f"Expected 16 kHz mono PCM16 WAV, got {fmt}: {path}")
            duration = handle.getnframes() / 16000
            raw = handle.readframes(handle.getnframes())
        array = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return array, duration

    def transcribe(self, rows, root, batch_size=4):
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            audio = [self.read_wav(audio_path(root, row)) for row in batch]
            start_time = time.perf_counter()
            features = self.processor(
                [array for array, _ in audio], sampling_rate=16000,
                return_tensors="pt", truncation=False, padding="longest",
                return_attention_mask=True,
            )
            model_inputs = {
                key: value.to(self.device, dtype=self.dtype) if value.is_floating_point()
                else value.to(self.device)
                for key, value in features.items()
            }
            with self.torch.inference_mode():
                tokens = self.model.generate(
                    **model_inputs, language="vi", task="transcribe",
                    do_sample=False)
            texts = self.processor.batch_decode(tokens, skip_special_tokens=True)
            elapsed = time.perf_counter() - start_time
            if len(texts) != len(batch):
                raise RuntimeError(f"ASR returned {len(texts)} outputs for {len(batch)} inputs")
            for row, transcript, (_, duration) in zip(batch, texts, audio):
                yield {"id": row["id"], "text": transcript.strip(),
                       "duration_s": duration, "latency_s": elapsed,
                       "batch_size": len(batch), "requested_batch_size": batch_size,
                       "latency_basis": "full_batch_wall_time",
                       "model_id": self.model_id}


class Qwen:
    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507", *,
                 temperature=0.7, seed=42, role_adapters=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.temperature = temperature
        self.seed = seed
        self.role_adapters = dict(role_adapters or {})
        self._adapter_lock = RLock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32)
        self.dtype = str(dtype).replace("torch.", "")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        if self.role_adapters:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.role_adapters['planner'], adapter_name='planner')
            for role in ('caller', 'summarizer'):
                self.model.load_adapter(self.role_adapters[role], adapter_name=role)
        self.model.eval()
        torch.manual_seed(seed)

    def encode(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )

    def count(self, messages):
        return int(self.encode(messages)["input_ids"].shape[-1])

    def complete(self, messages, max_new_tokens=384):
        with self._adapter_lock:
            if self.role_adapters:
                self.model.set_adapter(role_for_messages(messages))
            return self._complete(messages, max_new_tokens)

    def _complete(self, messages, max_new_tokens):
        inputs = self.encode(messages)
        input_tokens = int(inputs["input_ids"].shape[-1])
        inputs = inputs.to(self.model.device)
        start = time.perf_counter()
        sampling = {"do_sample": self.temperature > 0}
        if self.temperature > 0:
            sampling.update(temperature=self.temperature, top_p=0.8, top_k=20)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **sampling,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start
        new_ids = output[0][input_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return {"text": text, "prompt_tokens": input_tokens,
                "generation_tokens": int(new_ids.shape[-1]), "latency_s": elapsed}


class QwenVLLM:
    """Offline vLLM adapter with the same count/complete interface as Qwen."""

    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507", *,
                 max_model_len=12288, gpu_memory_utilization=0.82,
                 tensor_parallel_size=1, enforce_eager=True,
                 attention_backend="TRITON_ATTN", dtype="float16", seed=42,
                 temperature=0.7, max_num_seqs=1, structured_outputs=True, role_adapters=None):
        from vllm import LLM, SamplingParams
        self.model_id = model_id
        self.backend = "vllm"
        self.max_model_len = max_model_len
        self.SamplingParams = SamplingParams
        self.structured_outputs = structured_outputs
        self.dtype = dtype
        self.temperature = temperature
        self.seed = seed
        self.role_adapters = dict(role_adapters or {})
        self.lora_requests = {}
        lora_options = {}
        if self.role_adapters:
            from vllm.lora.request import LoRARequest
            self.lora_requests = {role: LoRARequest(role, i, self.role_adapters[role])
                                  for i, role in enumerate(('planner', 'caller', 'summarizer'), 1)}
            lora_options = {'enable_lora': True, 'max_loras': 3,
                            'max_lora_rank': max(8, adapter_rank(self.role_adapters))}
        print(f"Loading vLLM: dtype={dtype}, attention_backend={attention_backend}", flush=True)
        self.engine = LLM(
            model=model_id,
            dtype=self.dtype,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=4096,
            enable_chunked_prefill=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            enable_prefix_caching=True,
            attention_backend=attention_backend,
            seed=seed,
            structured_outputs_config={"backend": "guidance"},
            **lora_options,
        )
        self.tokenizer = self.engine.get_tokenizer()

    def encode(self, messages):
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return ids

    def count(self, messages):
        return len(self.encode(messages))

    def complete(self, messages, max_new_tokens=256, output_schema=None):
        return self.complete_batch([(messages, max_new_tokens, output_schema)])[0]

    def complete_batch(self, requests):
        prompts, sampling, counts = [], [], []
        for messages, limit, schema in requests:
            ids = self.encode(messages)
            if len(ids) + limit > self.max_model_len:
                raise ContextLimitError(f"Prompt ({len(ids)}) + generation ({limit}) exceeds context")
            options = dict(temperature=self.temperature, top_p=0.8, top_k=20,
                           seed=self.seed, max_tokens=limit)
            if schema is not None and self.structured_outputs:
                from .contracts import validate_decoder_schema
                validate_decoder_schema(schema)
                from vllm.sampling_params import StructuredOutputsParams
                options["structured_outputs"] = StructuredOutputsParams(json=schema)
            prompts.append({"prompt_token_ids": ids})
            sampling.append(self.SamplingParams(**options))
            counts.append(len(ids))
        start = time.perf_counter()
        try:
            extra = {}
            if getattr(self, 'lora_requests', None):
                extra['lora_request'] = [self.lora_requests[role_for_messages(r[0])] for r in requests]
            responses = self.engine.generate(prompts, sampling[0] if len(sampling) == 1 else sampling,
                                             use_tqdm=False, **extra)
        except Exception as exc:
            raise InferenceBackendError("vLLM generation failed: " + str(exc)) from exc
        elapsed = time.perf_counter() - start
        return [{"text": response.outputs[0].text.strip(), "prompt_tokens": count,
                 "generation_tokens": len(response.outputs[0].token_ids),
                 "latency_s": elapsed} for response, count in zip(responses, counts)]
