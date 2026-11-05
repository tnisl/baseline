"""Generated next-tool/stop metrics. Invalid generations always stay in denominators."""
from collections import Counter, defaultdict
import json
import math


def parse_decision(raw, allowed_tools):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        def reject_constant(value):
            raise ValueError(f"Non-JSON numeric constant: {value}")
        value = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (ValueError, TypeError):
        return {"json_valid": False, "contract_valid": False, "allowed_tool": False,
                "prediction": None, "error": "invalid_json"}
    valid = isinstance(value, dict) and (
        (set(value) == {"stop"} and value["stop"] is True) or
        (set(value) == {"tool"} and isinstance(value["tool"], str)))
    allowed = valid and ("stop" in value or value["tool"] in allowed_tools)
    return {"json_valid": True, "contract_valid": valid, "allowed_tool": bool(allowed),
            "prediction": value if valid else None,
            "error": None if allowed else "unavailable_tool" if valid else "invalid_contract"}


def grade(row, raw, **generation):
    parsed = parse_decision(raw, row["allowed_tools"])
    return {key: row[key] for key in ("key", "episode_id", "step", "position", "previous_kind",
                                     "episode_calls", "clarification_target", "target")} | {
        "raw": raw, **parsed, "correct": parsed["allowed_tool"] and parsed["prediction"] == row["target"],
        **generation,
    }


def ratio(a, b):
    return a / b if b else None


def accuracy_group(rows):
    return {"samples": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": ratio(sum(r["correct"] for r in rows), len(rows))}


def summarize(rows):
    if not rows or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty unique predictions")
    groups, episodes, per_tool = defaultdict(list), defaultdict(list), defaultdict(list)
    confusion = Counter()
    stop_tp = stop_fp = stop_fn = premature = unnecessary = 0
    for row in rows:
        gold_stop = "stop" in row["target"]
        prediction = row["prediction"]
        pred_stop = bool(prediction and prediction.get("stop") is True)
        pred_call = bool(prediction and "tool" in prediction and row["allowed_tool"])
        gold_route = "stop" if gold_stop else "call"
        pred_route = "stop" if pred_stop else "call" if pred_call else "invalid"
        confusion[(gold_route, pred_route)] += 1
        stop_tp += gold_stop and pred_stop
        stop_fp += not gold_stop and pred_stop
        stop_fn += gold_stop and not pred_stop
        premature += not gold_stop and pred_stop
        unnecessary += gold_stop and pred_call
        episodes[row["episode_id"]].append(row)
        for name in ("position:" + row["position"], "previous_kind:" + str(row["previous_kind"]),
                     "episode_calls:" + str(row["episode_calls"]),
                     "clarification:" + str(row["clarification_target"])):
            groups[name].append(row)
        if not gold_stop:
            per_tool[row["target"]["tool"]].append(row)
    calls = [r for r in rows if "tool" in r["target"]]
    stops = len(rows) - len(calls)
    latencies = sorted(r["latency_s"] for r in rows if "latency_s" in r)
    report = {
        **accuracy_group(rows), "episodes": len(episodes),
        "next_tool_accuracy": accuracy_group(calls),
        "call_stop_accuracy": ratio(sum(n for (g, p), n in confusion.items() if g == p), len(rows)),
        "stop_precision": ratio(stop_tp, stop_tp + stop_fp),
        "stop_recall": ratio(stop_tp, stop_tp + stop_fn),
        "stop_f1": ratio(2*stop_tp, 2*stop_tp + stop_fp + stop_fn),
        "premature_stop_rate": ratio(premature, len(calls)),
        "unnecessary_continuation_rate": ratio(unnecessary, stops),
        "json_validity": ratio(sum(r["json_valid"] for r in rows), len(rows)),
        "contract_validity": ratio(sum(r["contract_valid"] for r in rows), len(rows)),
        "permitted_decision_rate": ratio(sum(r["allowed_tool"] for r in rows), len(rows)),
        "unavailable_tool_count": sum(r["error"] == "unavailable_tool" for r in rows),
        "length_limited_count": sum(r.get("length_limited", False) for r in rows),
        "episode_macro_accuracy": sum(accuracy_group(g)["accuracy"] for g in episodes.values()) / len(episodes),
        "all_prefixes_correct_episode_rate": sum(all(r["correct"] for r in g) for g in episodes.values()) / len(episodes),
        "tool_macro_accuracy": sum(accuracy_group(g)["accuracy"] for g in per_tool.values()) / len(per_tool) if per_tool else None,
        "confusion": [{"gold": g, "prediction": p, "count": n} for (g, p), n in sorted(confusion.items())],
        "by_group": {name: accuracy_group(g) for name, g in sorted(groups.items())},
        "by_tool": {name: accuracy_group(g) for name, g in sorted(per_tool.items())},
        "errors": dict(Counter(r["error"] for r in rows if r["error"])),
        "metric_note": "Generated decisions on reference prefixes; not a Caller rollout or official contest score.",
    }
    apps = defaultdict(list)
    for tool, group in per_tool.items():
        apps[tool.split(".")[0]].extend(group)
    report["by_app"] = {app: accuracy_group(g) for app, g in sorted(apps.items())}
    if latencies:
        report["latency"] = {"mean_s": sum(latencies)/len(latencies),
                             "p95_s": latencies[max(0, math.ceil(.95*len(latencies))-1)],
                             "note": "Batch generation time divided by batch size; amortized, excludes model startup and ASR."}
    return report
