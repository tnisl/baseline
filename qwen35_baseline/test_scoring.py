import unittest
from scoring import matches, maximum_matches, score

CATALOG = {"sms.send": {"parameters": {"properties": {
    "contact_id": {"type": "string"},
    "content": {"type": "string", "x-match": "fuzzy"}}}}}

def call(content="xin chào", contact="c1"):
    return {"name": "sms.send", "arguments": {
        "contact_id": contact, "content": content}}

def prediction(calls, hash_value="correct"):
    return {"id": "synthetic", "tool_calls": calls, "state_hash": hash_value}

class ScoreTests(unittest.TestCase):
    def test_correct_calls_with_wrong_hash_score_zero(self):
        result = score(prediction([call()], "wrong"), prediction([call()]), CATALOG)
        self.assertEqual(result["tool_f1"], 1)
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["hash_match"])

    def test_duplicate_predictions_cannot_reuse_gold(self):
        result = score(prediction([call(), call()]), prediction([call()]), CATALOG)
        self.assertEqual(result["matched"], 1)
        self.assertAlmostEqual(result["score"], 2/3)

    def test_fuzzy_threshold_is_strict_and_ids_are_exact(self):
        self.assertFalse(matches(call("a b c d e"), call("a b c x y"), CATALOG))
        self.assertTrue(matches(call("Xin CHÀO"), call("xin chào"), CATALOG))
        self.assertFalse(matches(call(contact="c1"), call(contact="c2"), CATALOG))

    def test_order_independent_maximum_matching(self):
        pred = [call("a b c d e"), call("a b c d x")]
        gold = [call("a b c d x"), call("a b c e y")]
        self.assertEqual(maximum_matches(pred, gold, CATALOG), 2)

    def test_empty_calls_and_no_matches(self):
        self.assertEqual(score(prediction([]), prediction([]), CATALOG)["score"], 1)
        self.assertEqual(score(prediction([call()]), prediction([]), CATALOG)["score"], 0)
        self.assertEqual(score(prediction([]), prediction([call()]), CATALOG)["score"], 0)

if __name__ == "__main__":
    unittest.main()
