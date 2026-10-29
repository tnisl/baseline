"""Deterministic verification that argument source citations exist."""
from __future__ import annotations

import re

from source_contract import SOURCE_ORIGINS


def _path_parts(path):
    if not isinstance(path, str) or not path:
        raise ValueError("source path phải là chuỗi không rỗng")
    parts = []
    position = 0
    for match in re.finditer(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_-]*)|\[(\d+)\]", path):
        if match.start() != position:
            raise ValueError(f"source path không hợp lệ: {path}")
        parts.append(int(match[2]) if match[2] is not None else match[1])
        position = match.end()
    if position != len(path):
        raise ValueError(f"source path không hợp lệ: {path}")
    return parts


def _read_path(data, path):
    value = data
    for part in _path_parts(path):
        if isinstance(part, int) and isinstance(value, list) and 0 <= part < len(value):
            value = value[part]
        elif isinstance(part, str) and isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError(f"source path không tồn tại: {path}")
    return value


def _validate_source(source):
    if not isinstance(source, dict):
        raise ValueError("source phải là object")
    if set(source) != {"origin", "inputs", "rule"}:
        raise ValueError("source cần đúng origin, inputs và rule")
    origins, inputs = source["origin"], source["inputs"]
    if not isinstance(origins, list) or not origins or not isinstance(inputs, list) or len(origins) != len(inputs):
        raise ValueError("origin và inputs phải là hai mảng không rỗng cùng độ dài")
    if not isinstance(source["rule"], str) or not source["rule"].strip():
        raise ValueError("rule phải là chuỗi không rỗng")
    for index, (origin, citation) in enumerate(zip(origins, inputs)):
        if not isinstance(origin, str) or origin not in SOURCE_ORIGINS:
            raise ValueError(f"origin[{index}] không hợp lệ: {origin!r}")
        if not isinstance(citation, dict):
            raise ValueError(f"inputs[{index}] phải là object")
        fields = {
            "input": set(),
            "context": {"path"},
            "history": {"path"},
            "previous_tool_calls": {"index", "part", "path"},
            "schema": {"path"},
        }[origin]
        if set(citation) != fields:
            raise ValueError(f"inputs[{index}] của {origin} cần đúng các trường {sorted(fields)}")
        if origin == "previous_tool_calls" and citation["part"] not in ("result", "arguments"):
            raise ValueError("previous_tool_calls cần part=result hoặc arguments")
        if any(origin == origins[j] and citation == inputs[j] for j in range(index)):
            raise ValueError("Các trích dẫn trong inputs phải khác nhau")
    return source


def _citation_value(origin, citation, transcript, context, history, observations, spec):
    if origin == "input":
        return transcript
    if origin in ("context", "history", "schema"):
        root = {"context": context, "history": history, "schema": spec}[origin]
        return _read_path(root, citation["path"])
    index = citation["index"]
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(observations):
        raise ValueError("tool index không tồn tại")
    return _read_path(observations[index][citation["part"]], citation["path"])


def _citation_issue(key, source, transcript, context, history,
                    observations, spec):
    try:
        source = _validate_source(source)
        observed = []
        for origin, citation in zip(source["origin"], source["inputs"]):
            if origin == "schema":
                prop = ((spec.get("parameters") or {}).get("properties") or {}).get(key) or {}
                expected_path = f"parameters.properties.{key}.default"
                if "default" not in prop or citation["path"] != expected_path:
                    return {"code": "invalid_default", "field": key,
                            "message": "schema chỉ là nguồn cho default thật của argument; enum không phải default."}, None
            item = _citation_value(origin, citation, transcript, context, history,
                                   observations, spec)
            observed.append(item)
    except (ValueError, TypeError, KeyError) as error:
        return {"code": "invalid_source", "field": key, "message": str(error)}, None
    return None, {"inputs": observed, "rule": source["rule"]}


def check_sources(action, spec, transcript, context, history, observations):
    """Check one source object per argument."""
    args = action["arguments"]
    sources = action.get("sources")
    issues = []
    evidence = {}
    unverified = []
    if args and (not isinstance(sources, dict) or set(sources) != set(args)):
        source_keys = set(sources) if isinstance(sources, dict) else set()
        details = []
        if set(args) - source_keys:
            details.append("thiếu: " + ", ".join(sorted(set(args) - source_keys)))
        if source_keys - set(args):
            details.append("thừa: " + ", ".join(sorted(source_keys - set(args))))
        issues.append({"code": "missing_argument_sources", "field": None,
                       "message": "sources phải khớp arguments"
                                  + ("; " + "; ".join(details) if details else "") + "."})
    elif args:
        for key in args:
            issue, observed = _citation_issue(
                key, sources[key], transcript, context, history,
                observations, spec
            )
            if issue:
                issues.append(issue)
            else:
                evidence[key] = {"source": sources[key], "observed": observed}
                unverified.append(key)
    return {"ok": not issues, "issues": issues, "evidence": evidence,
            "unverified_fields": unverified}


def repair_source_paths(action, spec, transcript, context, history, observations):
    """Fix path prefixes only when the corrected citation exists."""
    repaired = dict(action.get("sources") or {})
    changes = []
    for key, value in action.get("arguments", {}).items():
        source = repaired.get(key)
        if not isinstance(source, dict) or not isinstance(source.get("origin"), list) or not isinstance(source.get("inputs"), list):
            continue
        candidate = dict(source)
        origins = list(source["origin"])
        inputs = []
        for index, item in enumerate(source["inputs"]):
            if not isinstance(item, dict):
                inputs.append(item)
                continue
            citation = dict(item)
            origin = origins[index] if index < len(origins) else None
            path = citation.get("path")
            if isinstance(path, str):
                if origin == "context" and path.startswith("context."):
                    citation["path"] = path[len("context."):]
                elif origin in ("context", "history") and path.startswith("history["):
                    origins[index] = "history"
                    citation["path"] = path[len("history"):]
                elif origin == "previous_tool_calls":
                    prefix = str(citation.get("part", "")) + "."
                    if path.startswith(prefix):
                        citation["path"] = path[len(prefix):]
            inputs.append(citation)
        candidate["origin"], candidate["inputs"] = origins, inputs
        if candidate != source:
            trial = {"arguments": {key: value}, "sources": {key: candidate}}
            if check_sources(trial, spec, transcript, context, history, observations)["ok"]:
                repaired[key] = candidate
                changes.append(key)
    return repaired, changes
