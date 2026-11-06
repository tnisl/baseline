import copy
import json
import unittest
from agent import messages_for, parse_action, run_episode

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

    def test_reject_malformed_multiple_unknown_calls(self):
        call = xml("contacts.search", {"query": "Nam"})
        for raw in [call + call, call[:-4], call + "suffix",
                    xml("unknown.tool", {}), "<tool_call>broken</tool_call>"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_action(raw, TOOLS)

    def test_step_budget_preserves_executed_calls(self):
        env = Env()
        pred, detail = run_episode(ITEM, CATALOG, env, lambda m, t: xml(
            "contacts.search", {"query": "Nam"}), max_steps=2)
        self.assertEqual(len(pred["tool_calls"]), 2)
        self.assertEqual(detail["status"], "step_limit")

if __name__ == "__main__":
    unittest.main()
