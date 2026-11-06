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
- An unavailable capability is unsupported, not a missing parameter of a made-up
  tool. Complete supported parts; do not call an unrelated tool as a substitute.
- Do not repeat completed actions or make unnecessary lookup or verification
  calls. Once no executable task remains, answer briefly without a tool call.
  For a fully cancelled, unsupported, or no-action request, no tool call is needed.
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
