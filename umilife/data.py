from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from .core import *

def normalized_row(row, transcript=None):
    out = {"id": row["id"], "context": deepcopy(row["context"]), "history": []}
    for turn in row.get("history", []):
        copy = {k:deepcopy(v) for k,v in turn.items() if k not in ("audio", "audio_text")}
        if turn["role"] == "user":
            if transcript is None:
                copy["text"] = turn["audio_text"]
            else:
                copy["text"] = transcript[turn["audio"]]
        out["history"].append(copy)
    return out

def grouped_split(rows, seed=42, fraction=.15):
    groups = defaultdict(list)
    for row in rows:
        key = unicodedata.normalize("NFC", row["audio_text"]).lower()
        key = re.sub(r"[^\w\s]", " ", key)
        key = re.sub(r"\s+", " ", key).strip()
        groups[key].append(row["id"])
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    valid, target = set(), max(1, round(len(rows)*fraction))
    for key in keys:
        if len(valid) >= target: break
        valid.update(groups[key])
    if not valid or len(valid) == len(rows): raise ValueError("Invalid grouped split")
    return {r["id"]: "validation" if r["id"] in valid else "train" for r in rows}

def terminal(trajectory):
    if not trajectory:
        return {"thought": "Yêu cầu không cần hoặc không thể xử lý bằng các công cụ được cài.",
                "next": "give_up"}, "Không có thao tác nào được thực hiện."
    if trajectory[-1]["action"]["name"] == "sys.ask_missing_info":
        args = trajectory[-1]["action"]["arguments"]
        return {"thought": "Đã yêu cầu bổ sung thông tin còn thiếu, cần chờ người dùng trả lời.",
                "next": "summarizer"}, "Bạn vui lòng bổ sung: " + ", ".join(args.get("params", [])) + "."
    return {"thought": "Đã nhận kết quả cho các bước xử lý yêu cầu, chuyển sang tóm tắt.",
            "next": "summarizer"}, "Kết quả công cụ: " + dumps([{"tool":t["action"]["name"], "result":t["observation"]} for t in trajectory])

def annotation_tasks(rows, catalog, assignments):
    for row in rows:
        specs = allowed_specs(row, catalog)
        trajectory = []
        for step, (action, result) in enumerate(zip(row["label"]["tool_calls"], row["label"]["tool_results"])):
            if action["name"] not in specs: raise ValueError("Gold uses uninstalled tool")
            payload = {"context": {k:v for k,v in row["context"].items() if k != "installed_apps"},
                       "history": normalized_row(row)["history"], "request": row["audio_text"],
                       "previous_steps": deepcopy(trajectory), "next_action": action,
                       "tool_schema": specs[action["name"]]}
            # Current result and future results are deliberately unavailable to the annotator.
            key = hashlib.sha256(dumps(payload).encode()).hexdigest()
            prompt = [{"role":"system", "content":
                "Bạn tạo rationale huấn luyện cho Planner trong alpha-UMi. Giải thích ngắn "
                "vì sao cần next_action ở bước hiện tại dựa vào yêu cầu và previous_steps. "
                "Không giả định kết quả của next_action, không tuyên bố đã thực hiện. "
                "Chỉ xuất một câu tiếng Việt, tối đa 50 từ. Không xuất JSON hay lời gọi tool."},
                {"role":"user", "content":dumps(payload)}]
            yield {"id":row["id"], "step":step, "key":key, "partition":assignments[row["id"]], "messages":prompt}
            trajectory.append({"action": action, "observation": result})

def build(root, output, rationale_path, seed=42, limit=None):
    root, output = Path(root), Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(root/"train/train.jsonl")
    assignments = grouped_split(rows, seed)
    if limit:
        # Split on the complete train set first; deterministic subset for smoke only.
        chosen = []
        for part in ("train", "validation"):
            candidates = [r for r in rows if assignments[r["id"]]==part]
            random.Random(seed).shuffle(candidates)
            chosen += candidates[:max(2, limit if part=="train" else max(2,limit//5))]
        rows = chosen
    catalog = load_catalog(root, "train")
    tasks = {(x["id"],x["step"]):x for x in annotation_tasks(rows, catalog, assignments)}
    rationales = {r["key"]:r["thought"] for r in read_jsonl(rationale_path)}
    ToolEnv = import_toolenv(root, "train")
    handles = {(role,part):(output/f"{role}.{part}.jsonl").open("w",encoding="utf-8")
               for role in ("global",*ROLES) for part in ("train","validation")}
    counts, excluded, references = Counter(), [], []
    def add(role, row, specs, trajectory, target, plan=None):
        rec = {"id":row["id"], "role":role, "step":len(trajectory),
               "messages":messages(role, normalized_row(row), row["audio_text"], specs, trajectory, plan),
               "target":dumps(target)}
        part = assignments[row["id"]]
        handles[role,part].write(dumps(rec)+"\n"); counts[f"{role}.{part}"] += 1
    try:
        for row in rows:
            gold = row["label"]["tool_calls"]; results = row["label"]["tool_results"]
            if len(gold)!=len(results): raise ValueError("Gold call/result length mismatch")
            specs = allowed_specs(row,catalog)
            env = ToolEnv("train", row["id"], root=root)
            trajectory, pending = [], []
            valid = True
            for step, (action,result) in enumerate(zip(gold,results)):
                check = env.validate(action["name"],action["arguments"])
                if not check.get("ok"):
                    excluded.append({"id":row["id"],"reason":"invalid_gold","detail":check}); valid=False; break
                observed = env.call(action["name"],action["arguments"])
                if observed != result:
                    excluded.append({"id":row["id"],"reason":"gold_result_mismatch","step":step}); valid=False; break
                thought = rationales.get(tasks[row["id"],step]["key"])
                if not thought: raise ValueError(f"Missing rationale for {row['id']} step {step}")
                plan = {"thought":thought, "next":"caller", "tool":action["name"]}
                pending += [("planner",deepcopy(trajectory),plan,None),
                            ("caller",deepcopy(trajectory),action,plan),
                            ("global",deepcopy(trajectory),{**plan,"action":action},None)]
                trajectory.append({"thought":thought,"action":action,"observation":observed})
            if not valid: continue
            references.append({"id":row["id"],"tool_calls":gold,"state_hash":env.get_state_hash()})
            for role,prefix,target,plan in pending: add(role,row,specs,prefix,target,plan)
            plan, answer = terminal(trajectory)
            add("planner",row,specs,trajectory,plan)
            add("global",row,specs,trajectory,{**plan,"answer":answer})
            if plan["next"] == "summarizer":
                add("summarizer",row,specs,trajectory,{"answer":answer},plan)
    finally:
        for f in handles.values(): f.close()
    manifest = {"seed":seed, "model":"Qwen/Qwen3-4B-Instruct-2507", "split":assignments,
        "counts":dict(counts), "excluded":excluded, "selected_ids":[r["id"] for r in rows],
        "label_source":"train labels only; Qwen-generated planner rationales; grounded synthetic summaries",
        "instruction_reuse":True, "train_sha256":digest(root/"train/train.jsonl"),
        "rationales_sha256":digest(rationale_path),
        "prompts_sha256":hashlib.sha256(dumps(SYSTEMS).encode()).hexdigest(),
        "files_sha256":{f.name:digest(f) for f in output.glob("*.jsonl")}}
    write_json(output/"references.json",references)
    write_json(output/"manifest.json",manifest)
    print(dumps({"counts":dict(counts),"excluded":len(excluded)}),flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--rationales",required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--limit",type=int)
    a=p.parse_args();build(a.data_root,a.output,a.rationales,a.seed,a.limit)
if __name__ == "__main__": main()
