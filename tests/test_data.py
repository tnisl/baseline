import copy
import json
from pathlib import Path
import tempfile
import unittest

from twonode.data import episode_examples, prepare
from twonode.io import write_json, write_jsonl


def fixture():
    catalog = {
        "contacts.search": {"name": "contacts.search", "kind": "read", "description": "Find a contact",
                            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        "sms.send": {"name": "sms.send", "kind": "write", "description": "Send a message",
                     "parameters": {"type": "object", "properties": {"contact_id": {"type": "string"},
                                     "content": {"type": "string"}}, "required": ["contact_id", "content"]}},
    }
    row = {"id": "train_0001", "audio_text": "Send Lan a message", "audio": "never-open.wav",
           "context": {"current_time": "2026-11-06T10:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
                       "installed_apps": {"contacts": {"tools": ["contacts.search"]}, "sms": {"tools": ["sms.send"]}}},
           "history": [{"role": "user", "audio_text": "Earlier request", "audio": "history.wav"},
                       {"role": "assistant", "text": "Previously completed", "tool_calls": [], "tool_results": []}],
           "label": {"tool_calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}},
                                     {"name": "sms.send", "arguments": {"contact_id": "c_observed", "content": "Hello"}}],
                     "tool_results": [{"ok": True, "contacts": [{"id": "c_observed", "name": "Lan"}]},
                                      {"ok": True, "done": True, "id": "future_response_marker"}],
                     "final_state": {"hidden_marker": "must not appear"}}}
    return row, catalog


class DataTests(unittest.TestCase):
    def test_only_past_evidence_and_complete_terminal(self):
        row, catalog = fixture()
        before = copy.deepcopy(row)
        examples = episode_examples(row, catalog)
        self.assertEqual([e["target"] for e in examples], [{"tool": "contacts.search"}, {"tool": "sms.send"}, {"stop": True}])
        first, second, last = [json.loads(e["messages"][1]["content"]) for e in examples]
        self.assertEqual(first["trajectory"], [])
        self.assertNotIn("c_observed", examples[0]["messages"][1]["content"])
        self.assertIn("c_observed", examples[1]["messages"][1]["content"])
        self.assertNotIn("future_response_marker", examples[1]["messages"][1]["content"])
        self.assertEqual(len(last["trajectory"]), 2)
        for example in examples:
            text = json.dumps(example["messages"])
            self.assertNotIn("hidden_marker", text)
            self.assertNotIn("never-open.wav", text)
            self.assertNotIn("history.wav", text)
            self.assertNotIn("train_0001", text)
        self.assertEqual(row, before)

    def test_clarification_maps_target_and_has_no_stop_after_meta(self):
        row, catalog = fixture()
        row["label"]["tool_calls"] = [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}]
        row["label"]["tool_results"] = [{"ok": True}]
        examples = episode_examples(row, catalog)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["target"], {"tool": "sms.send"})
        self.assertTrue(examples[0]["clarification_target"])
        self.assertNotIn("sys.ask_missing_info", examples[0]["allowed_tools"])

    def test_zero_actions_and_unavailable_target(self):
        row, catalog = fixture()
        row["label"]["tool_calls"], row["label"]["tool_results"] = [], []
        self.assertEqual(episode_examples(row, catalog)[0]["target"], {"stop": True})
        row, catalog = fixture()
        del row["context"]["installed_apps"]["sms"]
        with self.assertRaises(ValueError):
            episode_examples(row, catalog)

    def test_fixed_episode_split_and_quarantine(self):
        row, catalog = fixture()
        rows = []
        for i in range(202):
            example = copy.deepcopy(row)
            example["id"] = f"train_{i:04d}"
            rows.append(example)
        rows[-1]["label"]["tool_results"][-1] = {"ok": True, "done": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "train/train.jsonl", rows)
            write_json(root / "train/train_tools_catalog.json", {"apps": [{"tools": list(catalog.values())}]})
            split = root / "split.txt"
            split.write_text("\n".join(r["id"] for r in rows[:200]))
            report = prepare(root, split, root / "prepared")
            self.assertEqual(report["counts"]["validation"]["episodes"], 200)
            self.assertEqual(report["counts"]["train"]["episodes"], 1)
            self.assertEqual(report["counts"]["train"]["examples"], 3)
            self.assertEqual(report["quarantine"][0]["id"], "train_0201")
            self.assertFalse(set(report["counts"]["train"]["episode_ids"]) & set(report["counts"]["validation"]["episode_ids"]))
            with self.assertRaises(FileExistsError):
                prepare(root, split, root / "prepared")


if __name__ == "__main__":
    unittest.main()
