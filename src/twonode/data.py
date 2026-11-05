"""Derive supervised decisions from episodes without accessing environment files."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import unicodedata

from .io import canonical, file_hash, fingerprint, read_jsonl, write_json, write_jsonl
from .prompts import PROMPT_HASH, permitted_tools, planner_messages


def episode_examples(row, catalog):
    specs = permitted_tools(row, catalog)
    calls, results = row["label"]["tool_calls"], row["label"]["tool_results"]
    if len(calls) != len(results):
        raise ValueError(f"Call/result length mismatch: {row['id']}")
    trajectory, examples = [], []
    for step, (action, observation) in enumerate(zip(calls, results, strict=True)):
        meta = action["name"] == "sys.ask_missing_info"
        tool = action["arguments"]["tool_name"] if meta else action["name"]
        if tool not in specs:
            raise ValueError(f"Unavailable target {tool}: {row['id']}")
        if meta:
            fields = action["arguments"]["params"]
            if step != len(calls) - 1 or not fields or len(fields) != len(set(fields)):
                raise ValueError(f"Invalid clarification: {row['id']}")
            if not set(fields) <= specs[tool]["parameters"].get("properties", {}).keys():
                raise ValueError(f"Unknown clarification fields: {row['id']}")
        examples.append(make_example(row, specs, trajectory, {"tool": tool}, clarification=meta))
        trajectory.append({"action": deepcopy(action), "observation": deepcopy(observation)})
    if not calls or calls[-1]["name"] != "sys.ask_missing_info":
        examples.append(make_example(row, specs, trajectory, {"stop": True}))
    return examples


def make_example(row, specs, trajectory, target, clarification=False):
    step = len(trajectory)
    previous = trajectory[-1]["action"]["name"] if trajectory else None
    kind = specs[previous].get("kind", "unknown") if previous in specs else None
    return {
        "key": f"{row['id']}:{step}", "episode_id": row["id"], "step": step,
        "episode_calls": len(row["label"]["tool_calls"]),
        "position": "terminal" if "stop" in target else "initial" if step == 0 else "intermediate",
        "previous_kind": kind, "clarification_target": clarification,
        "allowed_tools": list(specs), "messages": planner_messages(row, specs, trajectory),
        "target": target,
    }


def normalize_request(text):
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def prepare(root, split_file, output, smoke=False):
    root, output = Path(root), Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Prepared data already exists: {output}; use a new run directory")
    rows = read_jsonl(root / "train/train.jsonl")
    ids = [row["id"] for row in rows]
    valid_ids = [line.strip() for line in Path(split_file).read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)) or len(valid_ids) != 200 or len(set(valid_ids)) != 200:
        raise ValueError("Expected unique episode IDs and exactly 200 unique validation IDs")
    valid = set(valid_ids)
    if not valid <= set(ids):
        raise ValueError("Validation IDs are missing from training_data")
    raw_catalog = json.loads((root / "train/train_tools_catalog.json").read_text())
    catalog = {spec["name"]: spec for app in raw_catalog["apps"] for spec in app["tools"]}
    partitions = {"train": [], "validation": []}
    quarantine, groups = [], defaultdict(list)
    for row in rows:
        partition = "validation" if row["id"] in valid else "train"
        groups[normalize_request(row["audio_text"])].append(row["id"])
        # Official gold may faithfully replay a failed operation. Such episodes cannot
        # supervise our no-recovery controller as successful normal trajectories.
        failed = [i for i, result in enumerate(row["label"]["tool_results"])
                  if result.get("ok") is False or result.get("error") or result.get("done") is False]
        if failed:
            quarantine.append({"id": row["id"], "partition": partition,
                               "failed_steps": failed, "excluded_from_train": partition == "train"})
            if partition == "train":
                continue
        partitions[partition].append(row)
    raw_counts = {"train": len(rows) - len(valid), "validation": len(valid)}
    if smoke:
        # Episode-level selection includes long inputs and the largest trajectory.
        # It changes neither the fixed split nor the batch/accumulation settings.
        for partition, limit in (("train", 16), ("validation", 8)):
            candidates = sorted(partitions[partition], key=lambda r: (-len(canonical(r)), r["id"]))
            special = []
            for predicate in (lambda r: not r["label"]["tool_calls"],
                              lambda r: any(a["name"] == "sys.ask_missing_info" for a in r["label"]["tool_calls"]),
                              lambda r: len(r["label"]["tool_calls"]) >= 5):
                match = next((r for r in candidates if predicate(r) and r not in special), None)
                if match:
                    special.append(match)
            selected = special + [r for r in candidates if r not in special]
            partitions[partition] = selected[:limit]
    counts, fingerprints = {}, {}
    for name, episodes in partitions.items():
        examples = [e for row in episodes for e in episode_examples(row, catalog)]
        write_jsonl(output / f"{name}.jsonl", examples)
        counts[name] = {"episodes": len(episodes), "examples": len(examples),
                        "targets": dict(Counter("stop" if "stop" in e["target"] else "tool" for e in examples)),
                        "clarification_targets": sum(e["clarification_target"] for e in examples),
                        "episode_ids": sorted(r["id"] for r in episodes)}
        fingerprints[name] = file_hash(output / f"{name}.jsonl")
    duplicates = [members for members in groups.values()
                  if any(x in valid for x in members) and any(x not in valid for x in members)]
    manifest = {"raw_episode_counts": raw_counts, "counts": counts, "smoke": smoke,
                "input_mode": "provided_clean_train_transcripts_and_gold_prefixes",
                "thinking": False, "prompt_sha256": PROMPT_HASH,
                "train_sha256": file_hash(root / "train/train.jsonl"),
                "catalog_sha256": file_hash(root / "train/train_tools_catalog.json"),
                "split_sha256": file_hash(split_file), "files_sha256": fingerprints,
                "quarantine": quarantine, "exact_request_groups_crossing_split": duplicates,
                "note": "The supplied split is fixed; duplicate requests are reported, not moved."}
    manifest["contract_sha256"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    (output / "split.txt").write_bytes(Path(split_file).read_bytes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.split_file, args.output, args.smoke), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
