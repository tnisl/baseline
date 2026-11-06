"""Single-model LifeMate baseline. Standard library only; no hidden state input."""
import copy
import json
import re
import time

SYSTEM = """You are LifeMate, a Vietnamese personal assistant.

Request and history:
- Execute only the current user request. Historical actions are already applied
  to the environment; never replay them just because they appear in history.
- Resolve references such as "that one", "the other one", and "same as before"
  from the conversation and actual tool results. Reuse established IDs and values
  when still applicable; look up again only when relevant information is missing,
  stale, or contradicted. An explicit new request to repeat an action is new work.
- Apply the user's latest correction to the affected task; keep other requested
  tasks. A withdrawn action must not execute. Never infer an unrequested action.

Conditions and selection:
- Separate independent tasks from tasks that depend on earlier results. Obtain
  evidence before evaluating a condition or carrying out a dependent action.
- Honor negations, exceptions, limits, and fallback choices. If a condition is
  false, skip only its dependent actions and continue other requested work.
- For "all except...", inspect enough data to apply every filter and exclusion,
  including relationships across objects or apps. Act on each matching object.
- Match candidates using all supplied attributes and history. Do not pick the
  first result arbitrarily. Resolve ambiguity with available evidence or lookup;
  if it remains unresolved, ask for the required identifying parameter.
- For "latest", "earliest", "cheapest", or similar comparisons, compare actual
  fields, timestamps, and constraints; do not assume list order. Derive amounts,
  addresses, or other values from observed data when the request refers to them.

Time and context:
- Use context.current_time and timezone, not the computer clock, for relative
  dates and times. Use location and user_profile for references to home or work.
- Compute offsets and intervals exactly, including crossing midnight. When moving
  an event, preserve its duration unless the user changes the duration or end.
- Respect schema formats: date YYYY-MM-DD, time HH:MM, date-time ISO-8601 with
  the context timezone where applicable. Do not invent a required time from a
  date alone. Use a recurrence parameter only if the tool supports it.

Tool contract:
- Use only supplied tools, including unfamiliar ones, according to their
  descriptions, Kind, Returns, and parameter schemas. read retrieves information;
  write changes app data; meta reports missing information.
- Include all required parameters. x-one-of requires at least one complete group
  in addition to required; it does not require every alternative group. Prefer a
  supported alternative already known, such as an explicit phone number over an
  unnecessary contact lookup. x-ref identifies the source of an ID: use a supplied
  or previously observed valid ID, or obtain it through the appropriate lookup.
  Never invent IDs or tool results.
- Follow types, enums, formats, bounds, and array item constraints. Numbers must
  be numeric values without units or thousands separators; phone numbers remain
  strings with leading zeroes. Omit unknown optional parameters rather than
  inventing values. Preserve supplied content and meaning; x-match: fuzzy does
  not permit adding facts or changing the requested message.
- For updates, change only requested fields and fields needed to keep the
  requested change consistent. Do not reset unrelated data or invent defaults.

Execution and completion:
- Choose exactly ONE native tool call per response, with no text after the call,
  then wait for its actual result. Read the result before choosing the next step.
- ok=true means the tool ran, not necessarily that the requested action succeeded.
  Check done, returned objects, and messages. Empty results, done=false, or an
  error are not success. Continue only with steps supported by observed evidence;
  never repeat a failed write unchanged or fabricate a successful outcome.
- Complete all independent supported tasks that can be resolved before asking
  about a blocked task. First use context, history, and relevant lookup tools to
  resolve missing values. Never ask for information already available there.
- If a supported task still lacks a required value or has an ambiguous target,
  call sys.ask_missing_info with the target tool_name and unresolved parameter
  names exactly as in its schema, including the chosen x-one-of alternative.
  Ask only for unresolved values. This meta call immediately ends the episode.
- Missing information MUST be reported by calling sys.ask_missing_info, not by
  asking a plain-text question and stopping. An empty lookup or multiple matches
  without a distinguishing fact leaves the target ID unresolved. Do not repeat
  the same lookup with unchanged evidence instead of reporting the missing ID.
- Before proposing a write, check required and x-one-of. If no alternative group
  is complete, resolve one supported alternative or report its missing field;
  do not emit a partial write. Do not ask for a contact ID if a valid supported
  phone-number alternative is already available.
- Example: if sms.send is available, message content is known, and lookup cannot
  identify the recipient, emit this call instead of asking who to message:
  <tool_call>
  <function=sys.ask_missing_info>
  <parameter=tool_name>
sms.send
</parameter>
  <parameter=params>
["contact_id"]
</parameter>
  </function>
  </tool_call>
  This is a format example; choose tool_name and params for the actual blocked task.
- An unavailable capability is unsupported, not a missing parameter of a made-up
  tool. Complete supported parts; do not call an unrelated tool as a substitute.
- Do not repeat completed actions or make unnecessary lookup or verification
  calls. Answer briefly without a tool call only after supported tasks are done
  and none remain blocked on required information.
  For a fully cancelled, unsupported, or no-action request, no tool call is needed.
"""

TIME_CONDITION_EXAMPLES = """Synthetic arithmetic and control examples (not tool definitions):
These are invented examples of interpreting constraints. Use actual context,
available tools, and observed data in the real task; never copy example values
or invent the capabilities mentioned here. Examples do not authorize new actions.

T1. Cách nói giờ: "mười hai giờ kém hai mươi trưa" = 11:40;
"một rưỡi chiều" = 13:30. "Kém" là trừ phút trước giờ được nói, không cộng
phút vào chính giờ đó; "rưỡi" là phút 30. Chọn ngày từ context và yêu cầu.

T2. Thiết bị cần sẵn sàng lúc 00:18 ngày 17/02/2034, khởi động mất 37 phút,
kiểm tra trước khởi động 8 phút. Các mốc đúng là khởi động 16/02 lúc 23:41,
kiểm tra 16/02 lúc 23:33. Tính từng mốc theo quan hệ phụ thuộc, giữ cả ngày.

T3. Context là 17/02/2034 lúc 22:05; tác vụ cần bắt đầu 18/02 lúc 01:20.
Nếu schema cần delay_minutes thì giá trị là 195, không phải 3 hoặc 315.
Đơn vị phải theo tên/schema tham số, kể cả khi khoảng chờ đi qua nửa đêm.

T4. Một khoảng từ 06:47 đến 07:23 được dời sớm 18 phút, thời lượng không đổi.
Khoảng mới là 06:29 đến 07:05, vẫn dài 36 phút. Dịch cả hai đầu khoảng;
không tự chọn thời lượng tròn một giờ hay sửa các trường không được yêu cầu.

C1. Quy tắc: "Nếu còn ít nhất 38 thẻ thì lấy 38 thẻ; nếu không thì in phần
còn thiếu." Quan sát còn 41 thẻ: chỉ lấy 38, không in. Nếu quan sát còn 30:
chỉ in 8, không lấy thêm 38. Hai nhánh nếu/không thì loại trừ nhau.

C2. Hai tệp cần 850 MB và 750 MB. Người dùng yêu cầu lưu cả hai trên SSD,
nếu SSD không đủ cho cả hai thì lưu cả hai trên HDD. SSD còn 1400 MB;
tổng cần 1600 MB, vì vậy cả hai dùng HDD. Không lưu tệp đầu trên SSD rồi mới
chuyển tệp sau sang HDD. Xét điều kiện của cả nhóm trước khi thực hiện từng phần.

C3. Chỉ đánh dấu lô có nhiệt độ > 63.4 và không thuộc nhóm đang hiệu chuẩn.
Quan sát: lô A=63.4 không hiệu chuẩn; B=64.1 đang hiệu chuẩn; C=63.9 không
hiệu chuẩn. Chỉ đánh dấu C: A bằng ngưỡng nên không đạt, B bị ngoại lệ loại.
Nếu không lô nào thỏa thì không thực hiện write; không hạ ngưỡng hoặc bỏ ngoại lệ.
"""
SYSTEM += "\n" + TIME_CONDITION_EXAMPLES

def tools_for(item, catalog):
    installed = {name for app in item["context"]["installed_apps"].values()
                 for name in app["tools"]}
    installed.add("sys.ask_missing_info")
    tools = {}
    for app in catalog["apps"]:
        for tool in app["tools"]:
            if tool["name"] in installed:
                tools[tool["name"]] = tool
    unknown = installed - tools.keys()
    if unknown:
        raise ValueError(f"Missing catalog tools: {sorted(unknown)}")
    return tools

def native_tools(tools):
    # Retain custom x-ref/x-one-of/x-match hints for the model and BTC validator.
    functions = []
    for tool in tools.values():
        description = tool["description"]
        for field, label in (("kind", "Kind"), ("returns", "Returns")):
            if tool.get(field):
                description += f"\n{label}: {tool[field]}"
        functions.append({"type": "function", "function": {
            "name": tool["name"], "description": description,
            "parameters": tool["parameters"]}})
    return functions

def transcript(message, mode, cache):
    if mode == "oracle":
        return message["audio_text"]
    text = cache[message["audio"]]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty/invalid transcript: {message['audio']}")
    return text

def call_message(call):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": copy.deepcopy(call)}]}

def result_message(result):
    return {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}

def messages_for(item, mode="oracle", cache=None):
    messages = [{"role": "system", "content": SYSTEM + "\nContext:\n" +
                 json.dumps(item["context"], ensure_ascii=False)}]
    for turn in item.get("history", []):
        if turn["role"] == "user":
            messages.append({"role": "user", "content": transcript(turn, mode, cache)})
        else:
            calls, results = turn.get("tool_calls", []), turn.get("tool_results", [])
            if len(calls) != len(results):
                raise ValueError("History tool calls/results are not aligned")
            for call, result in zip(calls, results):
                messages.extend([call_message(call), result_message(result)])
            if turn.get("text"):
                messages.append({"role": "assistant", "content": turn["text"]})
    messages.append({"role": "user", "content": transcript(item, mode, cache)})
    return messages

def parse_action(text, tools):
    """Extract the first native call outside thinking; validate its arguments strictly."""
    depth, block = 0, None
    for marker in re.finditer(r"<think>|</think>|<tool_call>", text):
        tag = marker.group()
        if tag == "<think>":
            depth += 1
        elif tag == "</think>":
            depth = max(0, depth - 1)
        elif depth == 0:
            end = text.find("</tool_call>", marker.end())
            if end < 0:
                raise ValueError("Incomplete tool call")
            block = text[marker.end():end]
            break
    if block is None:
        return None
    match = re.fullmatch(r"\s*<function=([^>\s]+)>(.*?)</function>\s*", block, re.S)
    if not match:
        raise ValueError("Malformed function")
    name, body = match.groups()
    if name not in tools:
        raise ValueError(f"Unavailable tool: {name}")
    properties = tools[name]["parameters"].get("properties", {})
    args = {}
    pattern = r"<parameter=([^>\s]+)>(.*?)</parameter>"
    if re.sub(pattern, "", body, flags=re.S).strip():
        raise ValueError("Malformed parameters")
    for key, raw in re.findall(pattern, body, re.S):
        if key in args or key not in properties:
            raise ValueError(f"Duplicate/unknown parameter: {key}")
        # Remove only the template's framing newlines for strings.
        value = raw.removeprefix("\n").removesuffix("\n")
        kind = properties[key].get("type")
        if kind == "boolean" and value.strip() in ("True", "False"):
            # Qwen's native template renders boolean arguments with Python casing.
            args[key] = value.strip() == "True"
        else:
            args[key] = value if kind == "string" else json.loads(value)
    return {"name": name, "arguments": args}

def _episode(item, catalog, env, mode, cache, max_steps):
    started = time.monotonic()
    tools = tools_for(item, catalog)
    messages = messages_for(item, mode, cache)
    trace, status = [], "step_limit"
    error = None
    for step in range(max_steps):
        entry = {"step": step}
        trace.append(entry)
        try:
            text = yield messages, native_tools(tools)
            if isinstance(text, Exception):
                raise text
            entry["raw"] = text
            call = parse_action(text, tools)
            if call is None:
                status = "stopped"
                break
            entry["action"] = call
            check = env.validate(call["name"], call["arguments"])
            if not check["ok"]:
                raise ValueError(f"Schema rejected proposal: {check}")
            result = env.call(call["name"], call["arguments"])
            entry["observation"] = result
            messages.extend([call_message(call), result_message(result)])
            if call["name"] == "sys.ask_missing_info":
                status = "missing_info"
                break
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
            entry["error"] = error
            break
    return {"id": item["id"], "tool_calls": env.get_tool_calls(),
            "state_hash": env.get_state_hash()}, {
                "id": item["id"], "status": status, "error": error, "trace": trace,
                "seconds": time.monotonic() - started}

def run_batch(items, catalog, envs, generate_batch, mode="oracle", cache=None, max_steps=8):
    """Advance independent episodes together, removing finished rows each round."""
    results = [None] * len(items)
    active = {}
    for i, (item, env) in enumerate(zip(items, envs, strict=True)):
        episode = _episode(item, catalog, env, mode, cache, max_steps)
        active[i] = (episode, next(episode))
    while active:
        try:
            outputs = generate_batch([request for episode, request in active.values()])
            if len(outputs) != len(active):
                raise ValueError("Generator returned the wrong number of batch outputs")
        except Exception as exc:
            outputs = [exc] * len(active)
        for i, output in zip(list(active), outputs):
            episode, request = active[i]
            try:
                active[i] = (episode, episode.send(output))
            except StopIteration as done:
                results[i] = done.value
                del active[i]
    return results

def run_episode(item, catalog, env, generate, mode="oracle", cache=None, max_steps=8):
    return run_batch([item], catalog, [env],
                     lambda requests: [generate(*request) for request in requests],
                     mode, cache, max_steps)[0]
