"""Single-model LifeMate baseline. Standard library only; no hidden state input."""
import copy
import json
import re
import time

SYSTEM = """You are LifeMate, a Vietnamese personal assistant.
Execute only the current user request. Earlier conversation is evidence, not new work.
Use the supplied context time/timezone, location and profile.
Choose exactly ONE tool call per response, then wait for its actual result.
Use only available tools. Obtain unknown IDs through lookup tools; never invent IDs.
Resolve references and corrections using history and observed results.
Read before writing when required data is unknown. Respect conditions and negations.
Complete all independent requested tasks that can be resolved.
For missing or ambiguous required values, use sys.ask_missing_info with the target
tool name and the unresolved schema parameter names. Do not ask for values that a
lookup can resolve. After this meta call the runtime ends this turn.
Do not repeat completed actions or make unnecessary verification calls.
When finished, unsupported, cancelled, or no action is appropriate, answer briefly
without a tool call. Never invent tool results.
"""

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
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"],
        "parameters": t["parameters"]}} for t in tools.values()]

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
    """Parse Qwen3.5 native XML. No repairs or guesses of malformed arguments."""
    text = text.strip()
    # Non-thinking prompting does not guarantee the model omits this wrapper.
    # Discard only one complete leading block; never execute calls inside it.
    if text.startswith("<think>"):
        thinking = re.match(r"<think>(.*?)</think>\s*", text, re.S)
        if not thinking or "<think>" in thinking.group(1):
            raise ValueError("Malformed or incomplete thinking block")
        text = text[thinking.end():].strip()
        if not text:
            raise ValueError("Empty answer after thinking block")
    if "<think>" in text or "</think>" in text:
        raise ValueError("Unexpected thinking marker outside leading block")
    if "<tool_call>" not in text:
        if any(tag in text for tag in ("tool_call", "<function=", "<parameter=")):
            raise ValueError("Malformed tool call")
        return None
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S)
    if len(blocks) != 1 or text.count("<tool_call>") != 1:
        raise ValueError("Expected exactly one complete tool call")
    prefix, suffix = text.split("<tool_call>", 1)
    suffix = suffix.split("</tool_call>", 1)[1]
    if suffix.strip():
        raise ValueError("Unexpected text after tool call")
    match = re.fullmatch(r"\s*<function=([^>\s]+)>(.*?)</function>\s*", blocks[0], re.S)
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
