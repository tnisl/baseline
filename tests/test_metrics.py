import unittest
from twonode.metrics import grade, parse_decision, summarize


def example(key, target):
    return {"key": key, "episode_id": key.split(":")[0], "step": int(key.split(":")[1]),
            "position": "terminal" if "stop" in target else "initial", "previous_kind": None,
            "episode_calls": 1, "clarification_target": False,
            "target": target, "allowed_tools": ["a.read", "a.write"]}


class MetricsTests(unittest.TestCase):
    def test_strict_contract_no_repairs(self):
        for raw in ['```json\n{"stop":true}\n```', '{"stop":true,"tool":"a.read"}',
                    '{"stop":1}', '{"stop":false}', '{"tool":"a.read","tool":"a.write"}',
                    '<think>reason</think>{"stop":true}', '{"stop":NaN}']:
            result = parse_decision(raw, ["a.read"])
            self.assertFalse(result["contract_valid"], raw)
        result = parse_decision('{"tool":"unknown.tool"}', ["a.read"])
        self.assertTrue(result["contract_valid"])
        self.assertFalse(result["allowed_tool"])

    def test_invalid_outputs_count_and_episode_macro(self):
        rows = [grade(example("e1:0", {"tool": "a.read"}), '{"tool":"a.read"}'),
                grade(example("e1:1", {"stop": True}), '{"stop":true}'),
                grade(example("e2:0", {"tool": "a.write"}), '{"stop":true}'),
                grade(example("e2:1", {"stop": True}), 'bad'),
                grade(example("e3:0", {"stop": True}), '{"tool":"a.read"}')]
        report = summarize(rows)
        self.assertEqual(report["samples"], 5)
        self.assertAlmostEqual(report["accuracy"], .4)
        self.assertAlmostEqual(report["next_tool_accuracy"]["accuracy"], .5)
        self.assertAlmostEqual(report["premature_stop_rate"], .5)
        self.assertAlmostEqual(report["unnecessary_continuation_rate"], 1/3)
        self.assertAlmostEqual(report["stop_precision"], .5)
        self.assertAlmostEqual(report["stop_recall"], 1/3)
        self.assertAlmostEqual(report["stop_f1"], .4)
        self.assertAlmostEqual(report["episode_macro_accuracy"], 1/3)
        self.assertAlmostEqual(report["all_prefixes_correct_episode_rate"], 1/3)
        self.assertAlmostEqual(report["json_validity"], .8)
        with self.assertRaises(ValueError):
            summarize(rows + rows[:1])


if __name__ == "__main__":
    unittest.main()
