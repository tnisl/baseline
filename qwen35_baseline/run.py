"""CLI: python qwen35_baseline/run.py --help."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from agent import messages_for, run_batch
from scoring import score

class VLLMGenerator:
    def __init__(self, model_id, revision, max_input, max_new_tokens, num_gpus=2,
                 gpu_memory_utilization=0.80, max_num_seqs=2, max_num_batched_tokens=1024):
        import os
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        import torch
        from vllm import LLM, SamplingParams
        if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
            raise RuntimeError(f"Requested {num_gpus} GPUs; select GPU T4 x2 in Kaggle settings")
        self.torch, self.num_gpus = torch, num_gpus
        self.max_input, self.max_new_tokens = max_input, max_new_tokens
        self.engine_settings = dict(
            tensor_parallel_size=num_gpus, distributed_executor_backend="mp",
            dtype="float16", max_model_len=max_input + max_new_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=True, enforce_eager=True,
            language_model_only=True, enable_prefix_caching=False,
            disable_custom_all_reduce=True, generation_config="vllm", seed=42)
        self.model = LLM(model=model_id, revision=revision, **self.engine_settings)
        self.tokenizer = self.model.get_tokenizer()
        # Free text generation: no JSON schema, grammar, guided decoder, or tool-choice enforcer.
        # End at the actual chat-turn boundary, rather than generating a simulated conversation.
        stop_ids = [self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
        self.sampling = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                      structured_outputs=None, stop_token_ids=stop_ids)
        self.resolved_revision = getattr(self.model.llm_engine.model_config.hf_config, "_commit_hash", None)
        self.stats = []
        self.observed_gpu_used_gib_per_gpu = {}
        self.sample_gpu_memory()
        print(json.dumps({"stage": "model_loaded", "backend": "vllm",
                          "engine_settings": self.engine_settings,
                          "structured_outputs": None}), flush=True)

    def sample_gpu_memory(self):
        # Driver-level usage includes vLLM worker processes. Samples are not transient peaks.
        for i in range(self.num_gpus):
            free, total = self.torch.cuda.mem_get_info(i)
            used = (total - free)/2**30
            self.observed_gpu_used_gib_per_gpu[str(i)] = max(
                used, self.observed_gpu_used_gib_per_gpu.get(str(i), 0))

    def __call__(self, messages, tools):
        result = self.generate_batch([(messages, tools)])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def generate_batch(self, requests):
        results = [None] * len(requests)
        encoded, indices = [], []
        for i, (messages, tools) in enumerate(requests):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                tokens = self.tokenizer(prompt, add_special_tokens=False)
                length = len(tokens["input_ids"])
                if length > self.max_input:
                    raise ValueError(f"Input {length} tokens exceeds {self.max_input}; no truncation")
                encoded.append(tokens)
                indices.append(i)
            except Exception as exc:
                results[i] = exc
        if not encoded:
            return results
        try:
            decoded = self._generate_encoded(encoded)
            for i, result in zip(indices, decoded, strict=True):
                results[i] = result
        except Exception as exc:
            for i in indices:
                results[i] = exc.with_traceback(None)
        return results

    def _generate_encoded(self, encoded):
        started = time.monotonic()
        outputs = self.model.generate(
            [{"prompt_token_ids": row["input_ids"]} for row in encoded],
            sampling_params=self.sampling, use_tqdm=False)
        results, lengths = [], []
        for request in outputs:
            output = request.outputs[0]
            lengths.append(len(output.token_ids))
            if output.finish_reason != "stop":
                results.append(ValueError("Generation did not reach a stop; no truncated action executed"))
            else:
                results.append(output.text)
        self.sample_gpu_memory()
        self.stats.append({"batch_size": len(encoded),
                           "input_tokens": [len(row["input_ids"]) for row in encoded],
                           "output_tokens": lengths, "seconds": time.monotonic() - started})
        return results

def read_rows(root, split, ids=None, limit=None):
    rows = [json.loads(line) for line in (root/split/f"{split}.jsonl").read_text().splitlines()
            if line.strip()]
    if ids:
        selected = set(Path(ids).read_text().split())
        available = {r["id"] for r in rows}
        if selected - available:
            raise ValueError(f"Unknown selected IDs: {sorted(selected - available)}")
        rows = [r for r in rows if r["id"] in selected]
    return rows[:limit] if limit else rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", choices=["train", "public_test"], default="train")
    p.add_argument("--ids", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--mode", choices=["oracle", "asr"], default="oracle")
    p.add_argument("--transcripts", type=Path)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--revision", default="main")
    p.add_argument("--max-input", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--num-gpus", type=int, choices=[1, 2], default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-num-seqs", type=int, default=2)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.split != "train" and a.mode == "oracle":
        p.error("Oracle transcripts exist only in train; public_test requires --mode asr")
    if a.mode == "asr" and not a.transcripts:
        p.error("--mode asr requires --transcripts")
    if min(a.max_input, a.max_new_tokens, a.max_steps, a.batch_size) < 1 or (a.limit is not None and a.limit < 1):
        p.error("Token, step, batch and sample budgets must be positive")
    if not 0 < a.gpu_memory_utilization < 1 or min(a.max_num_seqs, a.max_num_batched_tokens) < 1:
        p.error("Memory utilization must be between 0 and 1; scheduler budgets must be positive")
    root = a.data_root.resolve()
    rows = read_rows(root, a.split, a.ids, a.limit)
    if not rows:
        p.error("No selected samples")
    cache = json.loads(a.transcripts.read_text()) if a.transcripts else None
    # Check every transcript before allocating the model or opening environments.
    for row in rows:
        messages_for(row, a.mode, cache)
    catalog_file = root/a.split/f"{a.split}_tools_catalog.json"
    catalog = json.loads(catalog_file.read_text())
    scoring_catalog = {t["name"]: t for app in catalog["apps"] for t in app["tools"]}
    # Import exactly the BTC package for this split in this fresh CLI process.
    sys.path.insert(0, str(root/a.split))
    from zaic_tools import ToolEnv, state_hash
    if a.output.exists():
        p.error("Output already exists; choose a fresh directory (no implicit reset/resume)")
    a.output.mkdir(parents=True)
    generator = VLLMGenerator(a.model, a.revision, a.max_input, a.max_new_tokens, a.num_gpus,
                              a.gpu_memory_utilization, a.max_num_seqs, a.max_num_batched_tokens)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(resolved_revision=generator.resolved_revision,
                  catalog_sha256=hashlib.sha256(catalog_file.read_bytes()).hexdigest(),
                  selected_ids=[r["id"] for r in rows],
                  packages={n: importlib.metadata.version(n) for n in
                            ("torch", "transformers", "vllm")},
                  gpus=[generator.torch.cuda.get_device_name(i) for i in range(a.num_gpus)],
                  backend="vllm", structured_outputs=None,
                  engine_settings=generator.engine_settings)
    (a.output/"config.json").write_text(json.dumps(config, indent=2))
    summaries = []
    started = time.monotonic()
    with (a.output/"submission.jsonl").open("w") as submissions, \
         (a.output/"traces.jsonl").open("w") as traces:
        for offset in range(0, len(rows), a.batch_size):
            batch = rows[offset:offset + a.batch_size]
            envs = [ToolEnv(a.split, row["id"], root=root) for row in batch]
            results = run_batch(batch, catalog, envs, generator.generate_batch,
                                a.mode, cache, a.max_steps)
            for i, (row, (prediction, detail)) in enumerate(zip(batch, results), offset + 1):
                # Labels are inspected only AFTER inference. Never supplied to the agent.
                if a.split == "train":
                    initial = json.loads((root/a.split/f"{a.split}_environment"/row["id"]/"state.json").read_text())
                    initial.update(row["label"]["final_state"])
                    reference = {"id": row["id"], "tool_calls": row["label"]["tool_calls"],
                                 "state_hash": state_hash(initial)}
                    metrics = score(prediction, reference, scoring_catalog)
                    detail.update({key: metrics[key] for key in
                                   ("score", "hash_match", "tool_f1", "matched", "pred_calls", "gold_calls")})
                    detail["canonical_state_match"] = metrics["hash_match"]
                    detail["exact_call_sequence_match"] = prediction["tool_calls"] == row["label"]["tool_calls"]
                submissions.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                traces.write(json.dumps(detail, ensure_ascii=False) + "\n")
                submissions.flush()
                traces.flush()
                summaries.append({k: v for k, v in detail.items() if k != "trace"})
                print(json.dumps({"completed": i, "total": len(rows), **summaries[-1]}), flush=True)
    from collections import Counter
    report = {"samples": len(rows), "mode": a.mode, "batch_size": a.batch_size,
              "seconds": time.monotonic()-started,
              "statuses": dict(Counter(r["status"] for r in summaries)),
              "note": "Local approximation of published hash-gated F1; fuzzy tokenization may differ from BTC. No latency gate; model load and ASR excluded.",
              "generation": generator.stats,
              "observed_gpu_used_gib_per_gpu": generator.observed_gpu_used_gib_per_gpu,
              "gpu_memory_note": "Driver usage sampled after engine load and completed generate batches, includes vLLM workers; not transient peak allocation."}
    if a.split == "train":
        for metric in ("canonical_state_match", "exact_call_sequence_match"):
            report[metric] = sum(r[metric] for r in summaries)/len(rows)
        report["score_100"] = 100 * sum(r["score"] for r in summaries)/len(rows)
        report["hash_accuracy"] = sum(r["hash_match"] for r in summaries)/len(rows)
        report["hash_accuracy_percent"] = 100 * report["hash_accuracy"]
        report["tool_call_f1"] = sum(r["tool_f1"] for r in summaries)/len(rows)
    (a.output/"report.json").write_text(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
