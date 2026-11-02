from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time

ROLES = ("planner", "caller", "summarizer")
RULES = """Bạn là trợ lý LifeMate tiếng Việt. Chỉ xử lý yêu cầu cuối; history đã được thực hiện.
Dùng context để tính ngày giờ, giữ timezone. Chỉ gọi tool đã cài. Đọc dữ liệu qua tool,
không đoán ID, số điện thoại, số tiền hay thông tin còn thiếu. Lấy ID từ kết quả tool,
history hoặc yêu cầu. Tôn trọng lời sửa đổi và điều kiện của người dùng. Chỉ hỏi qua
sys.ask_missing_info khi thiếu thông tin không thể tìm bằng tool. Không lặp thao tác ghi.
Mỗi lần chỉ quyết định một bước dựa trên observation hiện có. Chỉ xuất JSON, không markdown."""
SYSTEMS = {
 "planner": RULES + '\nLập kế hoạch bước tiếp theo. Xuất {"thought":"lý do ngắn gọn",'
            '"next":"caller|summarizer|give_up","tool":"tên tool khi next=caller"}. '
            'Chọn summarizer khi hoàn tất hoặc đã hỏi thiếu thông tin; give_up nếu ngoài phạm vi.',
 "caller": RULES + '\nTheo thought hiện tại, tạo đúng một lời gọi tool: '
           '{"name":"app.tool","arguments":{...}}. Dùng schema và observation, không tự tạo kết quả.',
 "summarizer": RULES + '\nTóm tắt kết quả quan sát được, không thực hiện thêm thao tác: '
               '{"answer":"câu trả lời tiếng Việt ngắn, trung thực"}.',
 "global": RULES + '\nThực hiện từng bước theo ReAct, xuất '
           '{"thought":"lý do ngắn","next":"caller|summarizer|give_up",'
           '"tool":"app.tool nếu gọi","action":{"name":"app.tool","arguments":{...}},'
           '"answer":"chỉ khi kết thúc"}. Không có action khi kết thúc.'
}

def dumps(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_catalog(root, split):
    raw = json.loads((Path(root)/split/f"{split}_tools_catalog.json").read_text())
    return {t["name"]: t for a in raw["apps"] for t in a["tools"]}

def allowed_specs(row, catalog):
    installed = {n for a in row["context"]["installed_apps"].values() for n in a["tools"]}
    installed.add("sys.ask_missing_info")
    return {n: catalog[n] for n in sorted(installed) if n in catalog}

def import_toolenv(root, split):
    # Only the organisers' runtime reads environment files.
    package_dir = (Path(root)/split).resolve()
    existing = sys.modules.get("zaic_tools")
    if existing and Path(existing.__file__).resolve().parent.parent != package_dir:
        for name in list(sys.modules):
            if name == "zaic_tools" or name.startswith("zaic_tools."):
                del sys.modules[name]
    sys.path.insert(0, str(package_dir))
    return importlib.import_module("zaic_tools").ToolEnv

def tool_index(specs):
    # Planner selects tools; caller receives the complete selected schema.
    return [{"name": n, "description": s["description"],
             "required": s["parameters"].get("required", [])} for n, s in specs.items()]

def signature_index(specs):
    fields = ("type", "enum", "format", "x-ref", "minimum", "maximum", "items")
    return [{"name":n, "description":s["description"], "parameters":{
        "properties":{k:{a:v for a,v in schema.items() if a in fields}
                      for k,schema in s["parameters"].get("properties",{}).items()},
        "required":s["parameters"].get("required",[]),
        **({"x-one-of":s["parameters"]["x-one-of"]} if "x-one-of" in s["parameters"] else {})
    }} for n,s in specs.items()]

def messages(role, row, text, specs, trajectory, plan=None, feedback=None):
    # Explicit field allowlist: no labels, final_state or future observations.
    content = {"context": {k:v for k,v in row["context"].items() if k != "installed_apps"}, "history": row.get("history", []),
               "request": text, "trajectory": trajectory}
    if role == "caller":
        name = (plan or {}).get("tool")
        content.update(plan=plan, tools=[specs[name]] if name in specs else tool_index(specs))
    elif role == "planner":
        content["tools"] = tool_index(specs)
    elif role == "global":
        content["tools"] = signature_index(specs)
    else:
        content["plan"] = plan
    if feedback: content["validation_feedback"] = feedback
    return [{"role": "system", "content": SYSTEMS[role]},
            {"role": "user", "content": dumps(content)}]

def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    value = json.loads(text)
    if not isinstance(value, dict): raise ValueError("Expected JSON object")
    return value

class Agent:
    def __init__(self, backend, max_calls=10, max_decisions=16, deadline_s=30, mode="multi"):
        self.backend, self.max_calls = backend, max_calls
        self.max_decisions, self.deadline_s, self.mode = max_decisions, deadline_s, mode

    def run(self, row, text, specs, env, started=None):
        start = time.monotonic() if started is None else started
        trajectory, trace, answer = [], [], None
        reason = "decision_limit"
        def generate(role, plan=None, feedback=None):
            remaining = self.deadline_s - (time.monotonic()-start)
            if remaining <= 0: raise TimeoutError("sample deadline")
            raw=self.backend.generate(role, messages(role, row, text, specs, trajectory, plan, feedback), remaining)
            trace.append({"role":role,"raw":raw})
            return raw
        for _ in range(self.max_decisions):
            try:
                role = "global" if self.mode == "single" else "planner"
                plan = parse_json(generate(role))
                trace.append({"role": role, "output": plan})
                route = plan.get("next")
                if route == "give_up": reason = "give_up"; break
                if route == "summarizer":
                    answer = plan.get("answer") if self.mode == "single" else parse_json(generate("summarizer", plan)).get("answer")
                    reason = "complete"; break
                if route != "caller": raise ValueError("invalid planner route")
                if len(env.get_tool_calls()) >= self.max_calls: reason = "call_limit"; break
                action = plan.get("action") if self.mode == "single" else parse_json(generate("caller", plan))
                for attempt in range(2):
                    if not isinstance(action, dict): raise ValueError("invalid action")
                    name, args = action.get("name"), action.get("arguments")
                    check = (env.validate(name, args) if name in specs and isinstance(args, dict)
                             else {"ok": False, "error": "Unknown tool or invalid arguments"})
                    trace.append({"role": "caller", "output": action, "validation": check})
                    if check.get("ok"): break
                    if attempt or self.mode == "single": raise ValueError("invalid tool schema")
                    action = parse_json(generate("caller", plan, {"action": action, "result": check}))
                result = env.call(name, args)
                trajectory.append({"thought": plan.get("thought", ""), "action": action, "observation": result})
            except TimeoutError as exc:
                reason = "timeout"; trace.append({"error": str(exc)}); break
            except (ValueError, KeyError, TypeError) as exc:
                reason = "format_error"; trace.append({"error": str(exc)}); break
        elapsed = time.monotonic()-start
        return {"id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}, {
            "id": row["id"], "reason": reason, "elapsed_s": elapsed,
            "over_deadline": elapsed > self.deadline_s, "answer": answer,
            "trace": trace, "trajectory": trajectory}
