"""Ordinary Transformers generation in independent GPU workers, after training exits."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import time

from .config import RunConfig
from .diagnostics import disable_progress_bars
from .io import file_hash, read_jsonl, write_json, write_jsonl
from .metrics import grade, summarize
from .prompts import PROMPT_HASH
from .tokenization import prompt_ids


def evaluate(config, worker_rank):
    # Importing this module never imports Unsloth. The subprocess has a fresh runtime.
    import sys
    if any(name == "unsloth" or name.startswith("unsloth.") for name in sys.modules):
        raise RuntimeError("Validation must run in a fresh process without Unsloth patches")
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from peft import PeftModel

    disable_progress_bars()
    set_seed(config.seed)
    lineage = json.loads((config.path / "adapter/lineage.json").read_text())
    manifest = json.loads((config.prepared / "manifest.json").read_text())
    if (lineage["model"], lineage["revision"], lineage["prompt_sha256"], lineage["thinking"]) != (
            config.model, config.revision, PROMPT_HASH, False):
        raise ValueError("Adapter model, revision, prompt or thinking mode mismatch")
    if lineage["data_contract_sha256"] != manifest["contract_sha256"]:
        raise ValueError("Adapter/dataset contract mismatch")
    validation_path = config.prepared / "validation.jsonl"
    if file_hash(validation_path) != manifest["files_sha256"]["validation"]:
        raise ValueError("Validation examples changed after preparation")
    rows = read_jsonl(validation_path)[worker_rank::config.world_size]
    tokenizer = AutoTokenizer.from_pretrained(config.path / "adapter")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForImageTextToText.from_pretrained(
        config.model, revision=config.revision, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, config.path / "adapter", is_trainable=False)
    model.eval()
    model.config.use_cache = True
    modes = ("base", "adapter") if config.evaluate_base else ("adapter",)
    directory = config.path / "validation"
    for mode in modes:
        predictions = []
        context = model.disable_adapter() if mode == "base" else nullcontext()
        with context, torch.inference_mode():
            for start in range(0, len(rows), config.eval_batch):
                group = rows[start:start + config.eval_batch]
                inputs = [prompt_ids(tokenizer, row["messages"]) for row in group]
                if any(len(ids) + config.max_new_tokens > config.max_length for ids in inputs):
                    raise ValueError("Overlength validation prompt; refusing truncation")
                batch = tokenizer.pad({"input_ids": inputs}, padding=True, return_tensors="pt").to("cuda:0")
                torch.cuda.synchronize()
                started = time.monotonic()
                output = model.generate(
                    **batch, do_sample=False, num_beams=1, max_new_tokens=config.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                torch.cuda.synchronize()
                duration = time.monotonic() - started
                for row, prefix, sequence in zip(group, inputs, output, strict=True):
                    generated = sequence[batch["input_ids"].shape[1]:].tolist()
                    if tokenizer.eos_token_id in generated:
                        generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
                    # Preserve literal generated control/thinking tokens so the parser
                    # cannot silently repair a non-JSON answer by stripping them.
                    text_tokens = generated[:-1] if generated and generated[-1] == tokenizer.eos_token_id else generated
                    raw = tokenizer.decode(text_tokens, skip_special_tokens=False)
                    predictions.append(grade(row, raw, prompt_tokens=len(prefix),
                        generated_tokens=len(generated), prompt_sha256=hashlib.sha256(
                            json.dumps(prefix).encode()).hexdigest(),
                        latency_s=duration/len(group), batch_seconds=duration,
                        length_limited=len(generated) >= config.max_new_tokens and
                                       tokenizer.eos_token_id not in generated))
                print(json.dumps({"worker": worker_rank, "mode": mode,
                                  "completed": len(predictions), "total": len(rows)}), flush=True)
        write_jsonl(directory / f"{mode}.rank{worker_rank}.jsonl", predictions)
        write_json(directory / f"{mode}.rank{worker_rank}.metrics.json", summarize(predictions))
    write_json(directory / f"worker{worker_rank}.json", {
        "backend": "transformers+peft", "unsloth_imported": False,
        "gpu": torch.cuda.get_device_name(0), "worker_rank": worker_rank,
        "thinking": False, "generation": {"do_sample": False, "num_beams": 1,
        "max_new_tokens": config.max_new_tokens, "repair": False, "guided_decoding": False},
    })


def combine(config):
    source = read_jsonl(config.prepared / "validation.jsonl")
    expected = {row["key"] for row in source}
    reports = {}
    for mode in (("base", "adapter") if config.evaluate_base else ("adapter",)):
        rows = [row for rank in range(config.world_size)
                for row in read_jsonl(config.path / "validation" / f"{mode}.rank{rank}.jsonl")]
        if len(rows) != len(expected) or {row["key"] for row in rows} != expected:
            raise ValueError("Missing or duplicate validation predictions")
        lookup = {row["key"]: row for row in rows}
        ordered = [lookup[row["key"]] for row in source]
        write_jsonl(config.path / "validation" / f"{mode}.predictions.jsonl", ordered)
        reports[mode] = summarize(ordered)
        reports[mode]["profile"] = config.profile
        reports[mode]["quality_benchmark"] = config.profile == "full"
        write_json(config.path / "validation" / f"{mode}.metrics.json", reports[mode])
    if config.evaluate_base:
        base = read_jsonl(config.path / "validation/base.predictions.jsonl")
        adapter = read_jsonl(config.path / "validation/adapter.predictions.jsonl")
        if any(a["prompt_sha256"] != b["prompt_sha256"] for a, b in zip(base, adapter, strict=True)):
            raise ValueError("Base/adapter validation prompt IDs differ")
        comparison = {"samples": len(base), "both_correct": 0, "base_only_correct": 0,
                      "adapter_only_correct": 0, "both_wrong": 0}
        for a, b in zip(base, adapter, strict=True):
            key = "both_correct" if a["correct"] and b["correct"] else "base_only_correct" if a["correct"] else "adapter_only_correct" if b["correct"] else "both_wrong"
            comparison[key] += 1
        comparison["accuracy_delta"] = reports["adapter"]["accuracy"] - reports["base"]["accuracy"]
        write_json(config.path / "validation/comparison.json", comparison)
    write_json(config.path / "validation/report.json", reports)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    evaluate(RunConfig.load(args.config), args.worker_rank)
