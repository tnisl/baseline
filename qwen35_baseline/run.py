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

class HFGenerator:
    def __init__(self, model_id, revision, max_input, max_new_tokens, num_gpus=2):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; run the Kaggle notebook")
        if torch.cuda.device_count() < num_gpus:
            raise RuntimeError(f"Requested {num_gpus} GPUs; select GPU T4 x2 in Kaggle settings")
        self.torch = torch
        self.num_gpus = num_gpus
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        placement = {"device_map": {"": 0}} if num_gpus == 1 else {
            "device_map": "balanced", "max_memory": {0: "10GiB", 1: "10GiB"}}
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=torch.float16,
            attn_implementation="sdpa", **placement).eval()
        self.device_map = {key: str(value) for key, value in self.model.hf_device_map.items()}
        used = {str(i) for i in range(num_gpus)}
        actual = {value.removeprefix("cuda:") for value in self.device_map.values()}
        if actual != used:
            raise RuntimeError(f"Model must span GPUs {used}, without CPU/disk offload: {self.device_map}")
        self.input_device = self.model.get_input_embeddings().weight.device
        print(json.dumps({"stage": "model_loaded", "num_gpus": num_gpus,
                          "device_map": self.device_map,
                          "input_device": str(self.input_device)}), flush=True)
        self.max_input, self.max_new_tokens = max_input, max_new_tokens
        self.stats = []
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

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
        inputs = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        length = inputs["input_ids"].shape[-1]
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id, logits_to_keep=1)
        rows = output[:, length:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        results, output_lengths = [], []
        for tokens in rows:
            stop = next((i + 1 for i, token in enumerate(tokens) if token in eos), None)
            output_lengths.append(stop or len(tokens))
            if stop is None:
                results.append(ValueError("Generation did not reach EOS; no truncated action executed"))
            else:
                results.append(self.tokenizer.decode(tokens[:stop], skip_special_tokens=True))
        self.stats.append({"batch_size": len(encoded),
                           "input_tokens": [len(row["input_ids"]) for row in encoded],
                           "padded_input_tokens": length, "output_tokens": output_lengths,
                           "seconds": time.monotonic() - started})
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
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.split != "train" and a.mode == "oracle":
        p.error("Oracle transcripts exist only in train; public_test requires --mode asr")
    if a.mode == "asr" and not a.transcripts:
        p.error("--mode asr requires --transcripts")
    if min(a.max_input, a.max_new_tokens, a.max_steps, a.batch_size) < 1 or (a.limit is not None and a.limit < 1):
        p.error("Token, step, batch and sample budgets must be positive")
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
    generator = HFGenerator(a.model, a.revision, a.max_input, a.max_new_tokens, a.num_gpus)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(resolved_revision=generator.resolved_revision,
                  catalog_sha256=hashlib.sha256(catalog_file.read_bytes()).hexdigest(),
                  selected_ids=[r["id"] for r in rows],
                  packages={n: importlib.metadata.version(n) for n in
                            ("torch", "transformers", "accelerate")},
                  gpus=[generator.torch.cuda.get_device_name(i) for i in range(a.num_gpus)],
                  device_map=generator.device_map,
                  input_device=str(generator.input_device))
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
              "peak_cuda_gib_per_gpu": {
                  str(i): generator.torch.cuda.max_memory_allocated(i)/2**30
                  for i in range(a.num_gpus)}}
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
