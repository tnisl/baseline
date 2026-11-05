"""Completion-only encoding with the exact non-thinking inference prefix."""
import argparse
from collections import Counter
import json

from .config import RunConfig
from .io import canonical, file_hash, read_jsonl, write_json, write_jsonl


def prompt_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False, return_dict=False)


def encode_example(tokenizer, row, max_length, max_new_tokens):
    prefix = list(prompt_ids(tokenizer, row["messages"]))
    completion = tokenizer.encode(canonical(row["target"]), add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must have an EOS token")
    completion.append(eos)
    total = len(prefix) + len(completion)
    if max(total, len(prefix) + max_new_tokens) > max_length:
        raise ValueError(f"Overlength {row['key']}: prompt={len(prefix)}, target={len(completion)}, "
                         f"output_budget={max_new_tokens}, context={max_length}; no evidence was truncated")
    return {"input_ids": prefix + completion, "attention_mask": [1] * total,
            "labels": [-100] * len(prefix) + completion}


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        length = ((max(len(row["input_ids"]) for row in rows) + 7) // 8) * 8
        result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in rows:
            padding = length - len(row["input_ids"])
            for key, value in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                result[key].append(row[key] + [value] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def tokenize(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(config.path / "tokenizer")
    report = {"thinking": False, "truncation": False, "partitions": {}}
    for partition in ("train", "validation"):
        source = config.prepared / f"{partition}.jsonl"
        rows = read_jsonl(source)
        encoded = [encode_example(tokenizer, row, config.max_length, config.max_new_tokens) for row in rows]
        write_jsonl(config.prepared / f"{partition}.tokens.jsonl", encoded)
        lengths = sorted(len(row["input_ids"]) for row in encoded)
        report["partitions"][partition] = {
            "examples": len(rows), "source_sha256": file_hash(source),
            "max_tokens": max(lengths), "p95_tokens": lengths[min(len(lengths)-1, int(.95*len(lengths)))],
            "assistant_tokens": sum(sum(x != -100 for x in row["labels"]) for row in encoded),
            "length_buckets": dict(Counter(str((length // 1024) * 1024) for length in lengths)),
        }
    write_json(config.path / "tokenization_report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    tokenize(RunConfig.load(parser.parse_args().config))
