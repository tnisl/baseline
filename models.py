"""Lazy-loaded PhoWhisper and Qwen inference adapters."""
from __future__ import annotations

import time
import wave

from common import audio_path


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
                 temperature=0.7, seed=42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.temperature = temperature
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32)
        self.dtype = str(dtype).replace("torch.", "")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
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
                 temperature=0.7):
        from vllm import LLM, SamplingParams
        self.model_id = model_id
        self.backend = "vllm"
        self.max_model_len = max_model_len
        self.SamplingParams = SamplingParams
        self.dtype = dtype
        self.temperature = temperature
        self.seed = seed
        print(f"Loading vLLM: dtype={dtype}, attention_backend={attention_backend}", flush=True)
        self.engine = LLM(
            model=model_id,
            dtype=self.dtype,
            max_model_len=max_model_len,
            max_num_seqs=1,  # Tool-dependent steps are sequential.
            max_num_batched_tokens=4096,
            enable_chunked_prefill=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            enable_prefix_caching=True,
            attention_backend=attention_backend,
            seed=seed,
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

    def complete(self, messages, max_new_tokens=256):
        ids = self.encode(messages)
        if len(ids) + max_new_tokens > self.max_model_len:
            raise ValueError(
                f"Prompt ({len(ids)}) + generation ({max_new_tokens}) exceeds "
                f"vLLM max_model_len ({self.max_model_len})")
        params = self.SamplingParams(
            temperature=self.temperature, top_p=0.8, top_k=20,
            seed=self.seed, max_tokens=max_new_tokens,
        )
        start = time.perf_counter()
        response = self.engine.generate(
            {"prompt_token_ids": ids}, params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        completion = response.outputs[0]
        return {"text": completion.text.strip(), "prompt_tokens": len(ids),
                "generation_tokens": len(completion.token_ids), "latency_s": elapsed}
