"""Shared LifeMate dataset, evaluation, and reporting helpers."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import statistics
import sys
import unicodedata
import wave


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def leaves(data, path=""):
    """Yield scalar values with their JSON path and immediate record."""
    if isinstance(data, dict):
        for key, value in data.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(value, (dict, list)):
                yield from leaves(value, child)
            else:
                yield child, value, data
    elif isinstance(data, list):
        for index, value in enumerate(data):
            child = f"{path}[{index}]"
            if isinstance(value, (dict, list)):
                yield from leaves(value, child)
            else:
                yield child, value, data


def result_succeeded(result):
    return isinstance(result, dict) and result.get("ok") is not False and result.get("done") is not False


def result_facts(result):
    if not isinstance(result, (dict, list)):
        return set()
    return {(path, str(value)) for path, value, _ in leaves(result)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def find_root(given=None):
    if given:
        root = Path(given).expanduser().resolve()
        if (root / "train" / "train.jsonl").is_file():
            return root
        raise FileNotFoundError(f"Expected train/train.jsonl under {root}")
    candidates = [Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent.parent]
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        candidates.extend(p.parent.parent for p in kaggle.rglob("train.jsonl") if p.parent.name == "train")
    for root in candidates:
        if (root / "train" / "train.jsonl").is_file():
            return root.resolve()
        if (root / "train+public_test" / "train" / "train.jsonl").is_file():
            return (root / "train+public_test").resolve()
    raise FileNotFoundError("Cannot find extracted LifeMate data. Pass --data-root PATH to the folder containing train/.")


def load_train(root, limit=50, seed=42):
    path = Path(root) / "train" / "train.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return sorted(rows, key=lambda row: row["id"])


def load_catalog(root):
    path = Path(root) / "train" / "train_tools_catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {tool["name"]: tool for app in data["apps"] for tool in app["tools"]}


def import_toolenv(root):
    package_parent = str(Path(root) / "train")
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    from zaic_tools import ToolEnv
    return ToolEnv


def audio_path(root, row):
    path = Path(root) / "train" / row["audio"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def wav_duration(path):
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def normalize_vi(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


def edit_counts(reference, hypothesis):
    """Levenshtein alignment: return substitutions, deletions, insertions."""
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], op[i][0] = i, "D"
    for j in range(1, m + 1):
        dp[0][j], op[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (dp[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]), "H" if reference[i - 1] == hypothesis[j - 1] else "S"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]
            dp[i][j], op[i][j] = min(choices, key=lambda pair: pair[0])
    counts = Counter()
    i, j = n, m
    while i or j:
        action = op[i][j]
        counts[action] += 1
        if action in ("H", "S"):
            i -= 1
            j -= 1
        elif action == "D":
            i -= 1
        else:
            j -= 1
    return {key: counts[key] for key in ("S", "D", "I")}


def asr_case(row, prediction):
    reference = row["audio_text"]
    ref = normalize_vi(reference)
    hyp = normalize_vi(prediction["text"])
    words = edit_counts(ref.split(), hyp.split())
    chars = edit_counts(list(ref), list(hyp))
    word_errors = sum(words.values())
    char_errors = sum(chars.values())
    return {
        "id": row["id"], "reference": reference, "transcript": prediction["text"],
        "duration_s": prediction["duration_s"], "asr_latency_s": prediction["latency_s"],
        "asr_rtf": prediction["latency_s"] / prediction["duration_s"] if prediction["duration_s"] else None,
        "ref_words": len(ref.split()), "word_S": words["S"], "word_D": words["D"], "word_I": words["I"],
        "word_errors": word_errors, "wer": word_errors / len(ref.split()) if ref.split() else None,
        "ref_chars": len(ref), "char_errors": char_errors,
        "cer": char_errors / len(ref) if ref else None,
    }


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": percentile(values, .95), "max": max(values)}


def asr_summary(cases):
    total_words = sum(x["ref_words"] for x in cases)
    total_chars = sum(x["ref_chars"] for x in cases)
    return {
        "samples": len(cases),
        "corpus_wer": sum(x["word_errors"] for x in cases) / total_words if total_words else None,
        "corpus_cer": sum(x["char_errors"] for x in cases) / total_chars if total_chars else None,
        "word_S": sum(x["word_S"] for x in cases),
        "word_D": sum(x["word_D"] for x in cases),
        "word_I": sum(x["word_I"] for x in cases),
        "exact_normalized_transcripts": sum(x["word_errors"] == 0 for x in cases),
        "duration_s": describe([x["duration_s"] for x in cases]),
        "latency_s": describe([x["asr_latency_s"] for x in cases]),
        "rtf": describe([x["asr_rtf"] for x in cases]),
        "per_utterance_wer": describe([x["wer"] for x in cases]),
    }


def compact_history(history):
    turns = []
    for turn in history:
        if turn.get("role") == "user":
            turns.append({"role": "user", "text": turn.get("audio_text", "")})
        else:
            turns.append({"role": "assistant", "text": turn.get("text", ""),
                          "tool_calls": turn.get("tool_calls", []),
                          "tool_results": turn.get("tool_results", [])})
    return turns


def allowed_specs(row, catalog):
    names = {name for app in row["context"]["installed_apps"].values() for name in app["tools"]}
    names.add("sys.ask_missing_info")
    return [
        {"name": name, "kind": catalog[name].get("kind"),
         "description": catalog[name].get("description", ""),
         "parameters": catalog[name].get("parameters", {}),
         "returns": catalog[name].get("returns", "")}
        for name in sorted(names) if name in catalog
    ]


def folded_tokens(value):
    text = unicodedata.normalize("NFD", str(value).casefold()).replace("đ", "d")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return set(re.findall(r"[^\W_]+", text))


def fuzzy_f1(a, b):
    left, right = folded_tokens(a), folded_tokens(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    common = len(left & right)
    return 2 * common / (len(left) + len(right))


def same_call(pred, gold, catalog):
    if pred.get("name") != gold.get("name"):
        return False
    a, b = pred.get("arguments") or {}, gold.get("arguments") or {}
    if set(a) != set(b):
        return False
    spec = catalog.get(pred["name"], {}).get("parameters", {}).get("properties", {})
    for key in a:
        if spec.get(key, {}).get("x-match") == "fuzzy":
            if fuzzy_f1(a[key], b[key]) <= 0.6:
                return False
        elif a[key] != b[key]:
            return False
    return True


def max_matches(predicted, gold, catalog):
    edges = [[j for j, g in enumerate(gold) if same_call(p, g, catalog)] for p in predicted]
    owner = {}
    def add(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or add(owner[j], seen):
                owner[j] = i
                return True
        return False
    return sum(add(i, set()) for i in range(len(predicted)))


def ordered_matches(planned, gold):
    """LCS of tool names, including repeat occurrences; offline diagnostic only."""
    previous = [0] * (len(gold) + 1)
    for name in planned:
        current = [0]
        for j, target in enumerate(gold, 1):
            current.append(previous[j - 1] + 1 if name == target
                           else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def evaluate_agent_case(row, outcome, root, catalog, ToolEnv):
    gold_calls = row["label"]["tool_calls"]
    gold_env = ToolEnv("train", row["id"], root=root)
    for call in gold_calls:
        gold_env.call(call["name"], call.get("arguments") or {})
    gold_hash = gold_env.get_state_hash()
    predicted = outcome["tool_calls"]
    hash_match = outcome["state_hash"] == gold_hash
    matches = max_matches(predicted, gold_calls, catalog)
    if not gold_calls:
        proxy = 1.0 if hash_match and not predicted else 0.0
    elif not hash_match or not matches:
        proxy = 0.0
    else:
        precision = matches / len(predicted)
        recall = matches / len(gold_calls)
        proxy = 2 * precision * recall / (precision + recall)
    gold_names = Counter(c["name"] for c in gold_calls)
    selected = set(outcome['selected_tools'])
    gold_set = set(gold_names) - {'sys.ask_missing_info'}
    tool_set_recall = len(selected & gold_set) / len(gold_set) if gold_set else 1.0
    gold_order = [c['name'] for c in gold_calls if c['name'] != 'sys.ask_missing_info']
    tool_plan = outcome['tool_plan']
    order_recall = ordered_matches(tool_plan, gold_order) / len(gold_order) if gold_order else 1.0
    pred_names = Counter(c["name"] for c in predicted)
    missing_names = list((gold_names - pred_names).elements())
    extra_names = list((pred_names - gold_names).elements())
    flags = []
    if not hash_match:
        flags.append("state_hash_mismatch")
    if missing_names:
        flags.append("missing_tool")
    if extra_names:
        flags.append("extra_tool")
    if matches < min(len(predicted), len(gold_calls)) and not missing_names and not extra_names:
        flags.append("argument_mismatch")
    if outcome["stop_reason"] not in ("done", "ask_missing_info"):
        flags.append(outcome["stop_reason"])
    return {
        "id": row["id"], "reference": row["audio_text"], "transcript": outcome["transcript"],
        "gold_state_hash": gold_hash, "state_hash": outcome["state_hash"],
        "state_hash_match": hash_match, "gold_tool_count": len(gold_calls),
        "tool_count": len(predicted), "matched_calls_proxy": matches,
        "score_proxy": proxy, "missing_tool_names": missing_names, "extra_tool_names": extra_names,
        "error_flags": flags, "gold_tool_calls": gold_calls, "tool_calls": predicted,
        "stop_reason": outcome["stop_reason"], "decisions": outcome["decisions"],
        "selected_tools": outcome["selected_tools"], "selected_tool_count": len(selected),
        "tool_set_recall": tool_set_recall,
        "tool_plan": tool_plan, "plan_step_count": outcome["plan_step_count"],
        "processed_steps": outcome["processed_steps"], "skipped_steps": outcome["skipped_steps"],
        "tool_order_recall": order_recall, "tool_plan_exact_match": tool_plan == gold_order,
        "planner_calls": outcome["planner_calls"], "caller_calls": outcome["caller_calls"],
        "llm_calls": outcome["llm_calls"], "tool_iterations": outcome["tool_iterations"],
        "invalid_schema_proposals": outcome["invalid_schema_proposals"],
        "tool_failures": outcome.get("tool_failures", 0),
        "no_progress_events": outcome.get("no_progress_events", 0),
        "parse_errors": outcome["parse_errors"],
        "prompt_tokens_total": outcome["prompt_tokens_total"],
        "prompt_tokens_max": outcome["prompt_tokens_max"],
        "prompt_tokens_each": outcome["prompt_tokens_each"],
        "generation_tokens_total": outcome["generation_tokens_total"],
        "llm_latency_s": outcome["llm_latency_s"], "agent_latency_s": outcome["agent_latency_s"],
        "context_trimmed": outcome["context_trimmed"],
    }


def agent_summary(cases):
    if not cases:
        return {"samples": 0}
    return {
        "samples": len(cases),
        "mean_score_proxy": statistics.mean(x["score_proxy"] for x in cases),
        "score_proxy_positive": sum(x["score_proxy"] > 0 for x in cases),
        "state_hash_accuracy": statistics.mean(x["state_hash_match"] for x in cases),
        "perfect_proxy": sum(x["score_proxy"] == 1 for x in cases),
        "stop_reasons": dict(Counter(x["stop_reason"] for x in cases)),
        "error_flags": dict(Counter(flag for x in cases for flag in x["error_flags"])),
        "prompt_tokens_per_node_call": describe([n for x in cases for n in x["prompt_tokens_each"]]),
        "max_context_tokens_per_sample": describe([x["prompt_tokens_max"] for x in cases]),
        "prompt_tokens_total_per_sample": describe([x["prompt_tokens_total"] for x in cases]),
        "generation_tokens_per_sample": describe([x["generation_tokens_total"] for x in cases]),
        "decisions_per_sample": describe([x["decisions"] for x in cases]),
        "planner_calls_per_sample": describe([x["planner_calls"] for x in cases]),
        "caller_calls_per_sample": describe([x["caller_calls"] for x in cases]),
        "llm_calls_per_sample": describe([x["llm_calls"] for x in cases]),
        "selected_tools_per_sample": describe([x["selected_tool_count"] for x in cases]),
        "mean_tool_set_recall": statistics.mean(x["tool_set_recall"] for x in cases),
        "plan_steps_per_sample": describe([x["plan_step_count"] for x in cases]),
        "processed_steps_per_sample": describe([x["processed_steps"] for x in cases]),
        "skipped_steps_per_sample": describe([x["skipped_steps"] for x in cases]),
        "mean_tool_order_recall": statistics.mean(x["tool_order_recall"] for x in cases),
        "exact_tool_plans": sum(x["tool_plan_exact_match"] for x in cases),
        "tool_iterations_per_sample": describe([x["tool_iterations"] for x in cases]),
        "tool_failures_per_sample": describe([x.get("tool_failures", 0) for x in cases]),
        "no_progress_events_per_sample": describe([x.get("no_progress_events", 0) for x in cases]),
        "agent_latency_s": describe([x["agent_latency_s"] for x in cases]),
        "pipeline_latency_s": describe([x.get("pipeline_latency_s") for x in cases]),
        "pipeline_under_30s": (sum(x["pipeline_latency_s"] <= 30 for x in cases if x.get("pipeline_latency_s") is not None)
                               if any(x.get("pipeline_latency_s") is not None for x in cases) else None),
        "asr_wer_by_sample": describe([x.get("asr_wer") for x in cases]),
        "context_trimmed_samples": sum(x["context_trimmed"] for x in cases),
    }
