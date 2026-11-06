import copy
import json
import unittest
from agent import messages_for, native_tools, parse_action, run_batch, run_episode, tools_for

CATALOG = {"apps": [{"tools": [
    {"name": "contacts.search", "description": "Find", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "sms.send", "description": "Send", "parameters": {
        "type": "object", "properties": {"contact_id": {"type": "string"},
                                        "content": {"type": "string"}}}},
    {"name": "sys.ask_missing_info", "description": "Ask", "parameters": {
        "type": "object", "properties": {"tool_name": {"type": "string"},
                                        "params": {"type": "array"}}}}
]}]}
ITEM = {"id": "synthetic", "audio": "now.wav", "audio_text": "Nhắn Nam xin chào",
        "history": [], "context": {"installed_apps": {
            "contacts": {"tools": ["contacts.search"]},
            "sms": {"tools": ["sms.send"]}}}}
TOOLS = {t["name"]: t for t in CATALOG["apps"][0]["tools"]}

def xml(name, args):
    return "<tool_call>\n<function=" + name + ">\n" + "".join(
        "<parameter=" + k + ">\n" +
        (v if isinstance(v, str) else json.dumps(v)) + "\n</parameter>\n"
        for k, v in args.items()) + "</function>\n</tool_call>"

class Env:
    def __init__(self):
        self.calls = []
    def validate(self, name, args):
        return {"ok": not (name == "sms.send" and "contact_id" not in args)}
    def call(self, name, args):
        self.calls.append({"name": name, "arguments": args})
        return {"ok": True, "contacts": [{"id": "observed_42"}]}
    def get_tool_calls(self):
        return copy.deepcopy(self.calls)
    def get_state_hash(self):
        return "hash"

class AgentTests(unittest.TestCase):
    def test_scalar_booleans_accept_json_and_python_casing_only_for_boolean_schema(self):
        tools = {"demo.set": {"parameters": {"properties": {
            "enabled": {"type": "boolean"}, "text": {"type": "string"}}}}}
        for raw, expected in (("true", True), ("false", False), ("True", True), ("False", False)):
            action = parse_action(xml("demo.set", {"enabled": raw, "text": raw}), tools)
            self.assertIs(action["arguments"]["enabled"], expected)
            self.assertEqual(action["arguments"]["text"], raw)
        for invalid in ("yes", "FALSE", "True or False"):
            with self.subTest(invalid=invalid), self.assertRaises(json.JSONDecodeError):
                parse_action(xml("demo.set", {"enabled": invalid}), tools)

    def test_tool_metadata_and_complete_schema_survive_native_conversion(self):
        catalog = copy.deepcopy(CATALOG)
        tool = {"name": "new_app.dispatch", "kind": "write", "returns": "receipt_id",
                "description": "Dispatch to an observed target.", "parameters": {
                    "type": "object", "required": ["amount"],
                    "x-one-of": [["target_id"], ["phone"]], "properties": {
                        "target_id": {"type": "string", "x-ref": "new_app.targets"},
                        "phone": {"type": "string"},
                        "amount": {"type": "integer", "exclusiveMinimum": 0},
                        "note": {"type": "string", "x-match": "fuzzy"},
                        "date": {"type": "string", "format": "date"},
                        "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}}}}
        catalog["apps"].append({"tools": [tool]})
        item = copy.deepcopy(ITEM)
        item["context"]["installed_apps"] = {"new_app": {"tools": [tool["name"]]}}
        before = copy.deepcopy(catalog)
        functions = {t["function"]["name"]: t["function"]
                     for t in native_tools(tools_for(item, catalog))}
        self.assertEqual(set(functions), {tool["name"], "sys.ask_missing_info"})
        actual = functions[tool["name"]]
        self.assertEqual(actual["parameters"], tool["parameters"])
        self.assertEqual(actual["description"],
                         tool["description"] + "\nKind: write\nReturns: receipt_id")
        self.assertEqual(set(actual), {"name", "description", "parameters"})
        self.assertEqual(functions["sys.ask_missing_info"]["description"], "Ask")
        self.assertEqual(catalog, before)

    def test_independent_action_then_empty_lookup_then_terminal_missing_info(self):
        item = copy.deepcopy(ITEM)
        item["audio_text"] = "Nhắn Nam xin chào, rồi gọi Minh."
        item["history"] = [{"role": "assistant", "text": "Nam có ID known_nam."}]
        empty = {"ok": True, "done": False, "contacts": [], "message": "No match"}
        class EmptyLookupEnv(Env):
            def call(self, name, args):
                super().call(name, args)
                return copy.deepcopy(empty) if name == "contacts.search" else {"ok": True}
        env = EmptyLookupEnv()
        def generate(messages, tools):
            if not env.calls:
                self.assertIn("known_nam", str(messages))
                return xml("sms.send", {"contact_id": "known_nam", "content": "xin chào"})
            if len(env.calls) == 1:
                return xml("contacts.search", {"query": "Minh"})
            self.assertEqual(json.loads(messages[-1]["content"]), empty)
            return xml("sys.ask_missing_info", {"tool_name": "phone.call", "params": ["contact_id"]})
        # Supply the future target tool without requiring any hardcoded app list.
        catalog = copy.deepcopy(CATALOG)
        catalog["apps"].append({"tools": [{"name": "phone.call", "description": "Call",
            "parameters": {"type": "object", "properties": {"contact_id": {"type": "string"}}}}]})
        item["context"]["installed_apps"]["phone"] = {"tools": ["phone.call"]}
        prediction, detail = run_episode(item, catalog, env, generate)
        self.assertEqual([call["name"] for call in prediction["tool_calls"]],
                         ["sms.send", "contacts.search", "sys.ask_missing_info"])
        self.assertEqual(detail["status"], "missing_info")
        self.assertEqual(detail["trace"][1]["observation"], empty)

    def test_observation_drives_next_action(self):
        env = Env()
        def generate(messages, tools):
            self.assertNotIn("label", str(messages))
            if not env.calls:
                return xml("contacts.search", {"query": "Nam"})
            if len(env.calls) == 1:
                result = json.loads(messages[-1]["content"])
                return xml("sms.send", {"contact_id": result["contacts"][0]["id"],
                                        "content": "xin chào"})
            return "Đã gửi."
        pred, detail = run_episode(ITEM, CATALOG, env, generate)
        self.assertEqual(pred["tool_calls"][1]["arguments"]["contact_id"], "observed_42")
        self.assertEqual(detail["status"], "stopped")

    def test_history_is_evidence_not_replayed(self):
        item = copy.deepcopy(ITEM)
        item["history"] = [{"role": "user", "audio": "old.wav", "audio_text": "Tìm Nam"},
            {"role": "assistant", "text": "Đã tìm.", "tool_calls": [
                {"name": "contacts.search", "arguments": {"query": "Nam"}}],
             "tool_results": [{"ok": True, "contacts": [{"id": "old_id"}]}]}]
        env = Env()
        pred, _ = run_episode(item, CATALOG, env, lambda m, t: "Không cần thao tác.")
        self.assertEqual(pred["tool_calls"], [])
        self.assertIn("old_id", str(messages_for(item)))

    def test_asr_uses_history_and_current_without_oracle_fallback(self):
        item = copy.deepcopy(ITEM)
        item["history"] = [{"role": "user", "audio": "old.wav", "audio_text": "ORACLE"}]
        messages = messages_for(item, "asr", {"old.wav": "ASR old", "now.wav": "ASR now"})
        self.assertEqual([m["content"] for m in messages[1:]], ["ASR old", "ASR now"])
        with self.assertRaises(KeyError):
            messages_for(item, "asr", {"now.wav": "now"})

    def test_schema_rejection_does_not_execute(self):
        env = Env()
        pred, detail = run_episode(ITEM, CATALOG, env, lambda m, t: xml(
            "sms.send", {"content": "hi"}))
        self.assertEqual(pred["tool_calls"], [])
        self.assertEqual(detail["status"], "error")

    def test_missing_info_is_executed_and_terminal(self):
        env = Env()
        pred, detail = run_episode(ITEM, CATALOG, env, lambda m, t: xml(
            "sys.ask_missing_info", {"tool_name": "sms.send", "params": ["contact_id"]}))
        self.assertEqual(len(pred["tool_calls"]), 1)
        self.assertEqual(detail["status"], "missing_info")

    def test_reject_malformed_or_unknown_first_call(self):
        call = xml("contacts.search", {"query": "Nam"})
        for raw in [call[:-4],
                    xml("unknown.tool", {}), "<tool_call>broken</tool_call>"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_action(raw, TOOLS)

    def test_complete_leading_thinking_before_call_or_answer(self):
        call = xml("contacts.search", {"query": "Nam"})
        expected = {"name": "contacts.search", "arguments": {"query": "Nam"}}
        for block in ("<think>\n\n</think>", "<think>Chọn công cụ.\n</think>"):
            with self.subTest(block=block):
                self.assertEqual(parse_action(block + "\n" + call, TOOLS), expected)
                self.assertIsNone(parse_action(block + "\nĐã xong.", TOOLS))

    def test_calls_inside_thinking_are_never_executed(self):
        hidden = xml("sms.send", {"contact_id": "invented", "content": "hi"})
        visible = xml("contacts.search", {"query": "Nam"})
        env = Env()
        outputs = iter(["<think>" + hidden + "</think>\n" + visible, "Đã xong."])
        pred, detail = run_episode(ITEM, CATALOG, env, lambda m, t: next(outputs))
        self.assertEqual(pred["tool_calls"], [
            {"name": "contacts.search", "arguments": {"query": "Nam"}}])
        self.assertEqual(detail["status"], "stopped")
        self.assertIn(hidden, detail["trace"][0]["raw"])

    def test_thinking_is_ignored_wherever_it_occurs(self):
        call = xml("contacts.search", {"query": "Nam"})
        expected = {"name": "contacts.search", "arguments": {"query": "Nam"}}
        for raw in ("</think>" + call,
                    "<think>outer<think>inner</think></think>" + call,
                    "<think></think><think></think>" + call,
                    "prefix<think></think>" + call,
                    call + "<think></think>", call + "<think>unfinished"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_action(raw, TOOLS), expected)
        for raw in ("<think>" + call, "<think></think>", "<think>unfinished"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_action(raw, TOOLS))

    def test_first_call_only_then_wait_for_real_observation(self):
        first = xml("contacts.search", {"query": "Nam"})
        invented = xml("sms.send", {"contact_id": "invented", "content": "hi"})
        raw = first + "\nuser\n<think></think>\n" + invented + "\nassistant\nĐã gửi."
        env = Env()
        def generate(messages, tools):
            if not env.calls:
                return raw
            self.assertEqual(messages[-1]["role"], "tool")
            self.assertEqual(json.loads(messages[-1]["content"])["contacts"][0]["id"],
                             "observed_42")
            self.assertNotIn("invented", str(messages))
            return "Đã xong."
        pred, detail = run_episode(ITEM, CATALOG, env, generate)
        self.assertEqual(pred["tool_calls"], [
            {"name": "contacts.search", "arguments": {"query": "Nam"}}])
        self.assertEqual(detail["trace"][0]["raw"], raw)
        self.assertEqual(detail["status"], "stopped")

    def test_trailing_text_and_calls_are_ignored_without_editing_arguments(self):
        args = {"query": "literal <think>text</think>"}
        first = xml("contacts.search", args)
        for suffix in ("suffix", first, "<tool_call>incomplete"):
            with self.subTest(suffix=suffix):
                self.assertEqual(parse_action(first + suffix, TOOLS),
                                 {"name": "contacts.search", "arguments": args})

    def test_thinking_does_not_relax_tool_validation(self):
        call = xml("contacts.search", {"query": "Nam"})
        for raw in (call[:-4], xml("unknown.tool", {}) + call,
                    xml("contacts.search", {"unknown": "x"}) + call):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_action("<think>Chọn công cụ.</think>\n" + raw, TOOLS)

    def test_step_budget_preserves_executed_calls(self):
        env = Env()
        pred, detail = run_episode(ITEM, CATALOG, env, lambda m, t: xml(
            "contacts.search", {"query": "Nam"}), max_steps=2)
        self.assertEqual(len(pred["tool_calls"]), 2)
        self.assertEqual(detail["status"], "step_limit")

    def test_batch_shrinks_and_keeps_histories_tools_and_environments_separate(self):
        items = [copy.deepcopy(ITEM) for _ in range(4)]
        envs = [Env() for _ in items]
        for i, item in enumerate(items):
            item.update(id=f"row_{i}", audio_text=f"request_{i}")
        del items[3]["context"]["installed_apps"]["sms"]
        batch_sizes = []
        def generate(requests):
            batch_sizes.append(len(requests))
            outputs = []
            for messages, tools in requests:
                user = next(m["content"] for m in messages if m["role"] == "user")
                i = int(user.removeprefix("request_"))
                self.assertNotIn("label", str(messages))
                names = {t["function"]["name"] for t in tools}
                self.assertEqual("sms.send" in names, i != 3)
                if i == 0:
                    output = "Đã xong."
                elif i == 2:
                    output = xml("sys.ask_missing_info", {
                        "tool_name": "sms.send", "params": ["contact_id"]})
                elif not envs[i].calls:
                    output = xml("contacts.search", {"query": user})
                elif i == 3:
                    output = ValueError("row-specific generation failure")
                elif len(envs[i].calls) == 1:
                    self.assertEqual(json.loads(messages[-1]["content"])["contacts"][0]["id"],
                                     "observed_42")
                    self.assertEqual(envs[i].calls[0]["arguments"]["query"], user)
                    self.assertNotIn("request_3", str(messages))
                    output = xml("sms.send", {"contact_id": "observed_42", "content": "hi"})
                else:
                    output = "Đã gửi."
                outputs.append(output)
            return outputs
        results = run_batch(items, CATALOG, envs, generate)
        self.assertEqual(batch_sizes, [4, 2, 1])
        self.assertEqual([p["id"] for p, d in results], [item["id"] for item in items])
        self.assertEqual([d["status"] for p, d in results],
                         ["stopped", "stopped", "missing_info", "error"])
        self.assertEqual([len(p["tool_calls"]) for p, d in results], [0, 2, 1, 1])
        self.assertIn("row-specific", results[3][1]["error"])

    def test_batch_generation_failure_preserves_previously_executed_calls(self):
        envs = [Env(), Env()]
        def generate(requests):
            if not envs[0].calls:
                return [xml("contacts.search", {"query": "Nam"})] * len(requests)
            raise RuntimeError("batch failure")
        results = run_batch([ITEM, ITEM], CATALOG, envs, generate)
        for prediction, detail in results:
            self.assertEqual(len(prediction["tool_calls"]), 1)
            self.assertEqual(detail["status"], "error")
            self.assertIn("batch failure", detail["error"])

    def test_partial_batch_and_per_episode_step_budget(self):
        env = Env()
        sizes = []
        def generate(requests):
            sizes.append(len(requests))
            return [xml("contacts.search", {"query": "Nam"})] * len(requests)
        results = run_batch([ITEM], CATALOG, [env], generate, max_steps=2)
        self.assertEqual(sizes, [1, 1])
        self.assertEqual(len(results[0][0]["tool_calls"]), 2)
        self.assertEqual(results[0][1]["status"], "step_limit")

if __name__ == "__main__":
    unittest.main()
