"""Focused v2 checks using the real train ToolEnv, without loading models."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "train+public_test"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "train"))

from agent import Agent, argument_proposal_error, argument_selection_error, reference_hints, valid_ask  # noqa: E402
from common import agent_summary, compact_history, evaluate_agent_case, load_catalog, result_succeeded  # noqa: E402
from grounding_qwen import check_sources as verify_sources, repair_source_paths as repair_paths  # noqa: E402
from node_reviews import SYSTEMS, review_node  # noqa: E402
from source_contract import SOURCE_FORMAT  # noqa: E402
from zaic_tools import ToolEnv  # noqa: E402


def source_array(source):
    """Express older test fixtures using the current origin-array contract."""
    if not isinstance(source, dict):
        return source
    if isinstance(source.get("origin"), list):
        inputs = [{} if origin == "input" else citation
                  for origin, citation in zip(source["origin"], source.get("inputs", []))]
        return {**source, "inputs": inputs,
                "rule": source.get("rule", "Sao chép nguyên giá trị từ trích dẫn.")}
    if not isinstance(source.get("origin"), str):
        return source
    origin = source["origin"]
    if origin == "multi_source":
        children = [source_array(item) for item in source["inputs"]]
        return {"origin": [kind for item in children for kind in item["origin"]],
                "inputs": [citation for item in children for citation in item["inputs"]],
                "rule": source["rule"]}
    if origin in ("input", "context", "history", "previous_tool_calls", "schema"):
        return {"origin": [origin],
                "inputs": ([{}] if origin == "input" else
                           [{key: value for key, value in source.items()
                             if key not in ("origin", "rule")}]),
                "rule": source.get("rule", "Sao chép nguyên giá trị từ trích dẫn.")}
    return source


def source_action(action):
    return {**action, "sources": {
        key: source_array(source) for key, source in (action.get("sources") or {}).items()
    }}


def check_sources(action, *args):
    return verify_sources(source_action(action), *args)


def repair_source_paths(action, *args):
    return repair_paths(source_action(action), *args)


def train_0015():
    with (ROOT / "train" / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] == "train_0015":
                return row
    raise AssertionError("train_0015 not found")


class ScriptedLLM:
    model_id = "scripted-test"

    def __init__(self, actions):
        self.actions = iter(actions)
        self.prompts = []
        self.pending_arguments = None

    def count(self, messages):
        return 100

    def complete(self, messages, max_new_tokens=256):
        self.prompts.append(messages)
        system = messages[0]["content"]
        if system.startswith("Bạn review"):
            payload = json.loads(messages[1]["content"])
            proposal = payload["action"]
            arguments = proposal.get("arguments") or {}
            name = proposal.get("name")
            if "THIẾU THAM SỐ" in system:
                missing = payload.get("missing_fields") or []
                answer = ({"reject_reason": "Thiếu " + ", ".join(missing)}
                          if missing else {})
            elif "CHỌN TOOL" in system:
                answer = ({"reject_reason": "sai mục tiêu"}
                          if name == "banking.open_savings" else {})
            elif "NGUỒN ARGUMENT" in system:
                source_map = proposal.get("sources") or {}
                answer = {"sources": source_map}
            elif "GIÁ TRỊ ARGUMENT" in system:
                reason = None
                if name == "banking.transfer" and arguments.get("from_account_id") == "acc_01":
                    reason = "số dư tài khoản không đủ"
                if (name == "banking.transfer"
                        and str(arguments.get("amount")) not in ("None", "218000")):
                    reason = "số tiền không thuộc hóa đơn internet"
                answer = {"reject_reason": reason} if reason else {}
            else:
                answer = {}
        elif system.startswith("Bạn chọn các trường ĐẦU VÀO"):
            proposal = self.pending_arguments
            schema = (json.loads(messages[1]["content"])["selected_tool"]
                      .get("parameters") or {})
            selected = dict.fromkeys(schema.get("required") or [])
            groups = schema.get("x-one-of") or []
            if groups:
                selected.update(dict.fromkeys(groups[0]))
            selected.update(dict.fromkeys((proposal.get("arguments") or {}).keys()))
            required = set(schema.get("required") or [])
            alternatives = {key for group in groups for key in group}
            answer = {"argument_reasons": {
                key: ("required" if key in required else
                      "Chọn phương án này cho tool"
                      if key in alternatives else
                      "Yêu cầu người dùng nêu rõ cần trường này")
                for key in selected}}
        elif system.startswith("Bạn điền GIÁ TRỊ"):
            proposal = self.pending_arguments
            self.pending_arguments = None
            arguments = proposal.get("arguments") or {}
            source_map = proposal.get("sources") or {}
            answer = {"arguments": arguments,
                      "sources": source_map,
                      "value_reasons": {key: "Giá trị phù hợp yêu cầu"
                                        for key in arguments}}
        else:
            answer = next(self.actions)
            if answer.get("type") == "call":
                self.pending_arguments = answer
                answer = {"type": "call", "name": answer["name"],
                          "tool_reason": "Bước cần thiết cho yêu cầu"}
        return {
            "text": json.dumps(answer, ensure_ascii=False),
            "prompt_tokens": 100, "generation_tokens": 20, "latency_s": 0.01,
        }


def result_source(index, path):
    return {"origin": ["previous_tool_calls"],
            "inputs": [{"index": index, "part": "result", "path": path}],
            "rule": "Sao chép nguyên giá trị từ kết quả tool."}


def call(name, sources=None, **arguments):
    return {"type": "call", "name": name, "arguments": arguments,
            "sources": {key: source_array(source) for key, source in
                        (sources if sources is not None else {}).items()}}


class GroundedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = train_0015()
        cls.catalog = load_catalog(ROOT)

    def run_actions(self, actions):
        agent = Agent(
            ScriptedLLM(actions), self.catalog, ToolEnv, ROOT,
            max_tool_calls=10, max_decisions=12, max_repairs=3,
        )
        return agent.run(self.row, self.row["audio_text"])

    def test_gold_path_replays_exactly(self):
        outcome = self.run_actions([
            call("bill_payment.list_bills"),
            call("banking.get_balance"),
            call("banking.list_beneficiaries"),
            call("banking.transfer", from_account_id="acc_02",
                 beneficiary_id="ben_02", amount=218000,
                 sources={"from_account_id": result_source(1, "accounts[1].id"),
                          "beneficiary_id": result_source(2, "beneficiaries[1].id"),
                          "amount": result_source(0, "bills[2].amount")}),
            {"type": "done"},
        ])
        case = evaluate_agent_case(
            self.row, outcome, ROOT, self.catalog, ToolEnv
        )
        self.assertTrue(case["state_hash_match"])
        self.assertEqual(
            agent_summary([case])["grounding_rejections_per_sample"]["max"], 0
        )
        self.assertEqual(case["score_proxy"], 1.0)
        self.assertEqual(outcome["grounding_rejections"], 0)
        transfer = outcome["trace"][3]
        self.assertEqual(list(transfer["node_reviews"]),
                         ["relevance", "schema", "argument_sources", "arguments"])
        self.assertEqual(
            transfer["grounding"]["reviewed_sources"]["amount"]["origin"],
            ["previous_tool_calls"],
        )

    def test_tool_selection_is_reviewed_before_arguments_are_generated(self):
        llm = ScriptedLLM([
            call("contacts.search", query="Phạm Hùng",
                 sources={"query": {"origin": "input", "quote": "Phạm Hùng"}}),
            {"type": "done"},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT,
                        max_decisions=3).run(self.row, self.row["audio_text"])
        first = outcome["trace"][0]
        self.assertEqual(first["tool_selection"]["name"], "contacts.search")
        self.assertNotIn("arguments", first["tool_selection"])
        relevance_prompt = next(messages for messages in llm.prompts
                                if "review bước CHỌN TOOL" in messages[0]["content"])
        reviewed_action = json.loads(relevance_prompt[1]["content"])["action"]
        self.assertNotIn("arguments", reviewed_action)
        self.assertEqual(first["action"]["value_reasons"]["query"],
                         "Giá trị phù hợp yêu cầu")
        review_payload = json.loads(relevance_prompt[1]["content"])
        self.assertEqual(review_payload["tool"]["name"], "contacts.search")
        self.assertIn("parameters", review_payload["tool"])
        self.assertNotIn("available_read_tools", review_payload)
        self.assertEqual(review_payload["history"], [])
        self.assertEqual(review_payload["tool_observations"], [])

    def test_reference_hints_identify_contact_from_history(self):
        with (ROOT / "train" / "train.jsonl").open(encoding="utf-8") as handle:
            row = next(json.loads(line) for line in handle
                       if json.loads(line)["id"] == "train_0063")
        hints = reference_hints(
            self.catalog["sms.send"], row["audio_text"],
            compact_history(row["history"]), [])
        self.assertEqual(hints[0]["value"], "c_10")
        self.assertEqual(hints[0]["source"]["origin"], ["history"])
        self.assertIn("tool_results[0].contacts[9].id",
                      hints[0]["source"]["inputs"][0]["path"])

    def test_repairs_relative_source_path_when_citation_exists(self):
        action = {"arguments": {"bill_id": "bill_02"},
                  "sources": {"bill_id": result_source(0, "result.bills[0].id")}}
        observations = [{"name": "bill_payment.list_bills", "arguments": {},
                         "result": {"bills": [{"id": "bill_02"}]}}]
        sources, changed = repair_source_paths(
            action, {}, "", {}, [], observations)
        self.assertEqual(changed, ["bill_id"])
        self.assertEqual(sources["bill_id"]["inputs"][0]["path"], "bills[0].id")
        action["arguments"]["bill_id"] = "bill_03"
        sources, changed = repair_source_paths(
            action, {}, "", {}, [], observations)
        self.assertEqual(changed, ["bill_id"])
        self.assertEqual(sources["bill_id"]["inputs"][0]["path"], "bills[0].id")

    def test_unknown_argument_is_rejected_before_toolenv(self):
        outcome = self.run_actions([
            call("bill_payment.list_bills", invented="tv",
                 sources={"invented": {"origin": "input", "quote": "tv"}}),
            {"type": "done"},
        ])
        self.assertEqual(outcome["trace"][0]["error"]["error"], "invalid_argument_selection")
        self.assertEqual(outcome["tool_calls"], [])

    def test_ask_accepts_semantically_needed_optional_value(self):
        with (ROOT / "train" / "train.jsonl").open(encoding="utf-8") as handle:
            row = next(json.loads(line) for line in handle
                       if json.loads(line)["id"] == "train_1715")
        specs = {"tv.control": self.catalog["tv.control"]}
        self.assertTrue(valid_ask("tv.control", ["volume"], specs))
        self.assertFalse(valid_ask("tv.control", ["invented"], specs))
        llm = ScriptedLLM([{"type": "ask", "tool_name": "tv.control",
                            "params": ["volume"],
                            "tool_reason": "Cần hỏi mức âm lượng vì yêu cầu không nêu số cụ thể."}])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT).run(
            row, row["audio_text"])
        self.assertEqual(outcome["tool_calls"], row["label"]["tool_calls"])
        self.assertIn("không nêu số", outcome["trace"][0]["action"]["tool_reason"])

    def test_ask_requires_tool_reason(self):
        llm = ScriptedLLM([
            {"type": "ask", "tool_name": "tv.control", "params": ["volume"]},
            {"type": "done"},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT).run(
            self.row, self.row["audio_text"])
        self.assertEqual(outcome["trace"][0]["error"]["error"], "missing_tool_reason")
        self.assertEqual(outcome["tool_calls"], [])

    def test_argument_fields_are_selected_before_values(self):
        llm = ScriptedLLM([
            call("contacts.search", query="Phạm Hùng",
                 sources={"query": {"origin": "input"}}),
            {"type": "done"},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT).run(
            self.row, self.row["audio_text"])
        event = outcome["trace"][0]
        self.assertEqual(set(event["argument_reasons"]), {"query"})
        self.assertIn("argument_selection_output", event)
        field_payload = next(json.loads(messages[1]["content"])
                             for messages in llm.prompts
                             if messages[0]["content"].startswith("Bạn chọn các trường ĐẦU VÀO"))
        self.assertEqual(set(field_payload), {
            "request", "context", "history", "tool_observations",
            "selected_tool", "tool_reason"})
        self.assertEqual(field_payload["selected_tool"]["name"], "contacts.search")
        self.assertEqual(field_payload["tool_reason"],
                         event["tool_selection"]["tool_reason"])
        fill_payload = next(json.loads(messages[1]["content"])
                            for messages in llm.prompts
                            if messages[0]["content"].startswith("Bạn điền GIÁ TRỊ"))
        self.assertEqual(fill_payload["selected_arguments"], ["query"])
        self.assertEqual(fill_payload["argument_reasons"], event["argument_reasons"])

    def test_planner_receives_history_and_latest_feedback(self):
        agent = Agent(ScriptedLLM([]), self.catalog, ToolEnv, ROOT)
        observations = [{"name": "bill_payment.list_bills", "kind": "read",
                         "arguments": {}, "result": {"ok": True, "bills": []}}]
        feedback = {"error": "invalid_argument_source", "tool": "bill_payment.pay"}
        messages, _, _ = agent.messages(
            self.row, self.row["audio_text"], observations, feedback)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(set(payload), {
            "request", "context", "history", "available_tools", "executed_calls",
            "tool_observations", "validation_feedback",
        })
        self.assertEqual(payload["history"], compact_history(self.row["history"]))
        self.assertEqual(payload["validation_feedback"], feedback)
        self.assertEqual(payload["executed_calls"], [
            {"name": "bill_payment.list_bills", "arguments": {}}])
        messages, _, _ = agent.messages(
            self.row, self.row["audio_text"], observations, None)
        self.assertIsNone(json.loads(messages[1]["content"])["validation_feedback"])

    def test_argument_selection_checks_required_fields_and_value_scope(self):
        spec = {"parameters": {"properties": {"content": {}, "repeat": {}},
                               "required": ["content"]}}
        self.assertIn("bắt buộc", argument_selection_error(
            {"argument_reasons": {"repeat": "Muốn lặp"}}, spec))
        self.assertIsNone(argument_selection_error(
            {"argument_reasons": {"content": "required"}}, spec))
        self.assertIn("'required'", argument_selection_error(
            {"argument_reasons": {"content": "Nội dung cần tạo"}}, spec))
        x_one_of_spec = {"parameters": {
            "properties": {"contact_id": {}, "phone_number": {}},
            "x-one-of": [["contact_id"], ["phone_number"]],
        }}
        self.assertIsNone(argument_selection_error(
            {"argument_reasons": {"contact_id": "Dùng định danh liên hệ"}},
            x_one_of_spec))
        self.assertIn("lý do cụ thể", argument_selection_error(
            {"argument_reasons": {"contact_id": "required"}},
            x_one_of_spec))
        self.assertIn("chưa chọn", argument_proposal_error(
            {"arguments": {"repeat": "once"}, "sources": {"repeat": {}},
             "value_reasons": {"repeat": "Nhắc một lần"}},
            selected_arguments={"content": "required"}))

    def test_fill_retries_when_json_hits_generation_limit(self):
        class TruncatingLLM(ScriptedLLM):
            truncated = False

            def complete(self, messages, max_new_tokens=256):
                if (messages[0]["content"].startswith("Bạn điền GIÁ TRỊ")
                        and not self.truncated):
                    self.truncated = True
                    return {"text": '{"arguments":', "prompt_tokens": 100,
                            "generation_tokens": max_new_tokens, "latency_s": 0.01}
                return super().complete(messages, max_new_tokens)

        llm = TruncatingLLM([call("bill_payment.list_bills"), {"type": "done"}])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT,
                        max_new_tokens=128).run(self.row, self.row["audio_text"])
        self.assertEqual(outcome["tool_calls"][0]["name"], "bill_payment.list_bills")
        self.assertEqual(outcome["trace"][0]["argument_retry"]["max_new_tokens"], 512)

    def test_fill_accepts_one_source_map_with_derived_origin(self):
        proposal = {
            "arguments": {"content": "nhớ mang tài liệu", "cycle": "quick"},
            "sources": {
                "content": {"origin": "input", "quote": "nhớ mang tài liệu"},
                "cycle": {"origin": "multi_source",
                          "inputs": [{"origin": "input", "quote": "giặt nhanh"}],
                          "rule": "giặt nhanh -> quick"},
            },
            "value_reasons": {
                "content": "Nội dung yêu cầu", "cycle": "Chế độ giặt nhanh",
            },
        }
        self.assertIsNone(argument_proposal_error(proposal))
        self.assertEqual(proposal["sources"]["cycle"]["origin"], "multi_source")
        proposal["sources"].pop("cycle")
        self.assertIn("Thiếu nguồn cho: cycle", argument_proposal_error(proposal))

    def test_repeated_read_reaches_tool_env(self):
        outcome = self.run_actions([
            call("bill_payment.list_bills"),
            call("bill_payment.list_bills"),
            {"type": "done"},
        ])
        self.assertEqual(len(outcome["tool_calls"]), 2)
        self.assertEqual(outcome["stop_reason"], "done")
        self.assertIn("tool_result", outcome["trace"][1])

    def test_gold_repeated_redeem_is_allowed(self):
        with (ROOT / "train" / "train.jsonl").open(encoding="utf-8") as handle:
            row = next(json.loads(line) for line in handle
                       if json.loads(line)["id"] == "train_0922")
        redeem = call(
            "loyalty.redeem", program="prog_815", reward_id="rw_02_1",
            sources={"program": result_source(0, "programs[0].id"),
                     "reward_id": result_source(0, "programs[0].rewards[0].id")},
        )
        actions = [
            call("loyalty.get_points", program="Cao Nguyên",
                 sources={"program": {"origin": "input", "quote": "Cao Nguyên"}}),
            redeem, redeem, {"type": "done"},
        ]
        outcome = Agent(ScriptedLLM(actions), self.catalog, ToolEnv, ROOT).run(
            row, row["audio_text"])
        self.assertEqual([item["name"] for item in outcome["tool_calls"]],
                         [call["name"] for call in row["label"]["tool_calls"]])
        self.assertEqual(outcome["stop_reason"], "done")

    def test_fill_feedback_identifies_extra_reason_key(self):
        error = argument_proposal_error({
            "arguments": {"type": "tv"},
            "sources": {"type": {"origin": "input", "quote": "tv"}},
            "value_reasons": {"type": "phù hợp", "because": "giải thích"},
        })
        self.assertIn("value_reasons thừa key: because", error)

    def test_enum_value_cannot_claim_schema_default_as_source(self):
        result = check_sources(
            {"arguments": {"type": "tv"},
             "sources": {"type": {"origin": "schema",
                                  "path": "properties/type/default"}}},
            {"parameters": {"properties": {
                "type": {"type": "string", "enum": ["tv", "water"]}}}},
            "truyền hình", {}, [], [],
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["issues"][0]["code"], "invalid_default")
        self.assertIn("enum", result["issues"][0]["message"])

    def test_fabricated_amount_is_blocked_before_tool_call(self):
        outcome = self.run_actions([
            call("banking.list_beneficiaries"),
            call("banking.get_balance"),
            call("banking.transfer", from_account_id="acc_01",
                 beneficiary_id="ben_02", amount=1000000,
                 sources={"from_account_id": result_source(1, "accounts[0].id"),
                          "beneficiary_id": result_source(0, "beneficiaries[1].id"),
                          "amount": {"origin": "input", "quote": "tiền mạng tháng này"}}),
            call("bill_payment.list_bills"),
            call("banking.transfer", from_account_id="acc_02",
                 beneficiary_id="ben_02", amount=218000,
                 sources={"from_account_id": result_source(1, "accounts[1].id"),
                          "beneficiary_id": result_source(0, "beneficiaries[1].id"),
                          "amount": result_source(2, "bills[2].amount")}),
            {"type": "done"},
        ])
        self.assertEqual(outcome["grounding_rejections"], 1)
        self.assertEqual(len(outcome["tool_calls"]), 4)
        blocked = [e for e in outcome["trace"]
                   if (e.get("action") or {}).get("name") == "banking.transfer"
                   and e["action"]["arguments"]["amount"] == 1000000]
        self.assertEqual(len(blocked), 1)
        self.assertFalse(blocked[0]["grounding"]["ok"])
        self.assertNotIn("tool_result", blocked[0])
        case = evaluate_agent_case(
            self.row, outcome, ROOT, self.catalog, ToolEnv
        )
        self.assertTrue(case["state_hash_match"])
        self.assertEqual(
            agent_summary([case])["grounding_rejections_per_sample"]["max"], 1
        )

    def test_open_savings_wrong_goal_is_rejected(self):
        outcome = self.run_actions([
            call("banking.get_balance"),
            call("banking.open_savings", from_account_id="acc_02",
                 amount=1000000, term_months=1,
                 sources={"from_account_id": result_source(0, "accounts[1].id"),
                          "amount": {"origin": "multi_source", "rule": "đề xuất gửi một triệu",
                                     "inputs": [{"origin": "input", "quote": "tiết kiệm"}]},
                          "term_months": {"origin": "multi_source", "rule": "kỳ hạn một tháng",
                                          "inputs": [{"origin": "input", "quote": "tháng này"}]}}),
            {"type": "done"},
        ])
        issues = outcome["trace"][1]["grounding"]["issues"]
        self.assertIn("qwen_rejected_call", [x["code"] for x in issues])
        self.assertEqual(len(outcome["tool_calls"]), 1)

    def test_existing_account_id_can_still_be_wrong_choice(self):
        outcome = self.run_actions([
            call("bill_payment.list_bills"),
            call("banking.get_balance"),
            call("banking.list_beneficiaries"),
            call("banking.transfer", from_account_id="acc_01",
                 beneficiary_id="ben_02", amount=218000,
                 sources={"from_account_id": result_source(1, "accounts[0].id"),
                          "beneficiary_id": result_source(2, "beneficiaries[1].id"),
                          "amount": result_source(0, "bills[2].amount")}),
            {"type": "done"},
        ])
        issues = outcome["trace"][3]["grounding"]["issues"]
        self.assertIn("qwen_rejected_argument", [x["code"] for x in issues])
        self.assertEqual(len(outcome["tool_calls"]), 3)

    def test_real_amount_from_wrong_bill_is_rejected_by_qwen(self):
        outcome = self.run_actions([
            call("bill_payment.list_bills"),
            call("banking.get_balance"),
            call("banking.list_beneficiaries"),
            call("banking.transfer", from_account_id="acc_02",
                 beneficiary_id="ben_02", amount=846000,
                 sources={"from_account_id": result_source(1, "accounts[1].id"),
                          "beneficiary_id": result_source(2, "beneficiaries[1].id"),
                          "amount": result_source(0, "bills[3].amount")}),
            call("banking.transfer", from_account_id="acc_02",
                 beneficiary_id="ben_02", amount=218000,
                 sources={"from_account_id": result_source(1, "accounts[1].id"),
                          "beneficiary_id": result_source(2, "beneficiaries[1].id"),
                          "amount": result_source(0, "bills[2].amount")}),
            {"type": "done"},
        ])
        wrong = outcome["trace"][3]
        self.assertEqual(wrong["grounding"]["stage"], "arguments")
        self.assertIn("qwen_rejected_argument",
                      [issue["code"] for issue in wrong["grounding"]["issues"]])
        self.assertNotIn("tool_result", wrong)
        self.assertTrue(wrong["validation"]["ok"])
        self.assertEqual(len(outcome["tool_calls"]), 4)

    def test_every_argument_requires_existing_source_path(self):
        outcome = self.run_actions([
            call("banking.get_balance"),
            call("banking.transfer", from_account_id="acc_02",
                 beneficiary_id="ben_02", amount=218000,
                 sources={"from_account_id": result_source(0, "accounts[99].id"),
                          "beneficiary_id": result_source(0, "accounts[1].id"),
                          "amount": result_source(0, "accounts[1].balance")}),
            {"type": "done"},
        ])
        issues = outcome["trace"][1]["grounding"]["issues"]
        self.assertIn("invalid_source", [issue["code"] for issue in issues])
        self.assertGreaterEqual(outcome["grounding_reviews"], 2)
        self.assertTrue(outcome["trace"][1]["validation"]["ok"])

    def test_missing_source_for_argument_blocks_call(self):
        outcome = self.run_actions([
            call("banking.get_balance"),
            call("banking.list_transactions", account_id="acc_02"),
            {"type": "done"},
        ])
        self.assertEqual(outcome["trace"][1]["error"]["error"],
                         "invalid_argument_proposal")
        self.assertEqual(len(outcome["tool_calls"]), 1)

    def test_no_argument_read_skips_sources_and_schema(self):
        outcome = self.run_actions([
            {"type": "call", "name": "banking.list_beneficiaries",
             "arguments": {}},
            {"type": "done"},
        ])
        event = outcome["trace"][0]
        self.assertTrue(event["grounding"]["ok"])
        self.assertEqual(event["grounding"]["source_check"], "skipped_no_arguments")
        self.assertEqual(event["validation"]["skipped"], "no_arguments")
        self.assertEqual(set(event["node_reviews"]), {"relevance"})
        self.assertEqual(len(outcome["tool_calls"]), 1)

    def test_read_lookup_can_precede_missing_transfer_details(self):
        outcome = self.run_actions([
            call("contacts.search", query="Phạm Hùng",
                 sources={"query": {"origin": "input", "quote": "Phạm Hùng"}}),
            {"type": "done"},
        ])
        event = outcome["trace"][0]
        self.assertTrue(event["grounding"]["ok"])
        self.assertEqual(event["grounding"]["review"], {})
        self.assertTrue(event["validation"]["ok"])
        self.assertEqual(outcome["tool_calls"][0]["name"], "contacts.search")

    def test_source_reviewer_can_reject_unsupported_value(self):
        class RejectUnsupportedValue(ScriptedLLM):
            def complete(self, messages, max_new_tokens=256):
                if "NGUỒN ARGUMENT" in messages[0]["content"]:
                    self.prompts.append(messages)
                    return {"text": '{"reject_reason":"Giá trị không theo rule từ trích dẫn."}',
                            "prompt_tokens": 100, "generation_tokens": 20,
                            "latency_s": 0.01}
                return super().complete(messages, max_new_tokens)

        outcome = Agent(RejectUnsupportedValue([
            call("contacts.search", query="Lê Minh",
                 sources={"query": {"origin": "input", "quote": "Phạm Hùng"}}),
            {"type": "done"},
        ]), self.catalog, ToolEnv, ROOT).run(self.row, self.row["audio_text"])
        event = outcome["trace"][0]
        self.assertEqual(event["grounding"]["stage"], "sources")
        self.assertIn("qwen_rejected_source",
                      [x["code"] for x in event["grounding"]["issues"]])
        self.assertTrue(event["validation"]["ok"])
        self.assertEqual(outcome["tool_calls"], [])

    def test_rule_is_required_for_both_copying_and_extraction(self):
        spec = {"parameters": {"properties": {"amount": {"type": "integer"}}}}
        missing_rule = verify_sources(
            {"arguments": {"amount": 2_000_000},
             "sources": {"amount": {"origin": ["input"],
                                    "inputs": [{}]}}},
            spec, "chuyển hai triệu", {}, [], [],
        )
        self.assertFalse(missing_rule["ok"])
        derived = check_sources(
            {"arguments": {"amount": 2_000_000},
             "sources": {"amount": {"origin": "multi_source",
                                    "inputs": [{"origin": "input", "quote": "hai triệu"}],
                                    "rule": "hai triệu -> 2000000"}}},
            spec, "chuyển hai triệu", {}, [], [],
        )
        self.assertTrue(derived["ok"])
        exact = verify_sources(
            {"arguments": {"amount": 2_000_000},
             "sources": {"amount": {"origin": ["context"],
                                    "inputs": [{"path": "amount"}],
                                    "rule": "Sao chép nguyên số từ context.amount."}}},
            spec, "", {"amount": 2_000_000}, [], [],
        )
        self.assertTrue(exact["ok"])

    def test_origin_array_combines_input_and_context(self):
        source = {
            "origin": ["input", "context"],
            "inputs": [{}, {"path": "current_time"}],
            "rule": "Ngày mai từ current_time, đặt giờ là 09:00.",
        }
        action = {"arguments": {"time": "2026-11-25T09:00:00+07:00"},
                  "sources": {"time": source}}
        result = verify_sources(
            action, {}, "nhắc tôi 9 giờ sáng mai",
            {"current_time": "2026-11-24T21:40:00+07:00"}, [], [],
        )
        self.assertTrue(result["ok"], result["issues"])
        self.assertEqual(result["unverified_fields"], ["time"])
        source_without_rule = {key: value for key, value in source.items()
                               if key != "rule"}
        action["sources"]["time"] = source_without_rule
        self.assertFalse(verify_sources(
            action, {}, "nhắc tôi 9 giờ sáng mai",
            {"current_time": "2026-11-24T21:40:00+07:00"}, [], [],
        )["ok"])

    def test_input_source_uses_entire_request_without_quote(self):
        request = "Nhắc tôi lấy áo lúc tám giờ."
        action = {
            "arguments": {"content": "lấy áo"},
            "sources": {"content": {
                "origin": ["input"], "inputs": [{}],
                "rule": "Rút cụm 'lấy áo' từ request.",
            }},
        }
        result = verify_sources(action, {}, request, {}, [], [])
        self.assertTrue(result["ok"], result["issues"])
        self.assertEqual(result["evidence"]["content"]["observed"]["inputs"],
                         [request])
        action["sources"]["content"]["inputs"] = [{"quote": "lấy áo"}]
        self.assertFalse(verify_sources(action, {}, request, {}, [], [])["ok"])

    def test_derived_source_records_a_translation_rule(self):
        check = check_sources(
            {"arguments": {"power": "off"},
             "sources": {"power": {"origin": "multi_source",
                                   "inputs": [{"origin": "input", "quote": "tắt đèn"}],
                                   "rule": "tắt đèn => power=off"}}},
            {"parameters": {"properties": {"power": {"type": "string"}}}},
            "tắt đèn", {}, [], [],
        )
        self.assertTrue(check["ok"])
        self.assertEqual(check["unverified_fields"], ["power"])

    def test_canonical_sources_cover_input_history_context_and_previous_calls(self):
        action = {
            "arguments": {"query": "Phạm Hùng", "account_id": "acc_02",
                          "nickname": "Sếp Dũng", "beneficiary_id": "ben_02"},
            "sources": {
                "query": {"origin": "input", "quote": "Phạm Hùng"},
                "account_id": {"origin": "context", "path": "account.id"},
                "nickname": {"origin": "history", "path": "[0].text"},
                "beneficiary_id": {"origin": "previous_tool_calls", "index": 0,
                                   "part": "result", "path": "beneficiaries[0].id"},
            },
        }
        result = check_sources(
            action, {}, "Chuyển cho Phạm Hùng",
            {"account": {"id": "acc_02"}},
            [{"text": "Sếp Dũng"}],
            [{"arguments": {}, "result": {"beneficiaries": [{"id": "ben_02"}]}}],
        )
        self.assertTrue(result["ok"], result["issues"])

    def test_legacy_origin_and_scalar_sources_are_rejected(self):
        for source in ({"origin": "request", "quote": "Phạm Hùng"}, "Phạm Hùng"):
            with self.subTest(source=source):
                result = check_sources(
                    {"arguments": {"query": "Phạm Hùng"},
                     "sources": {"query": source}},
                    {}, "Tìm Phạm Hùng", {}, [], [],
                )
                self.assertFalse(result["ok"])
                self.assertEqual(result["issues"][0]["code"], "invalid_source")

    def test_train_0063_uses_history_and_input_sources(self):
        with (ROOT / "train" / "train.jsonl").open(encoding="utf-8") as handle:
            row = next(json.loads(line) for line in handle
                       if json.loads(line)["id"] == "train_0063")
        llm = ScriptedLLM([
            call("sms.send", contact_id="c_10", content="nhớ mang tài liệu",
                 sources={
                     "contact_id": {"origin": "history",
                                    "path": "[1].tool_results[0].contacts[9].id"},
                     "content": {"origin": "input", "quote": "nhớ mang tài liệu"},
                 }),
            {"type": "done"},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT).run(row, row["audio_text"])
        case = evaluate_agent_case(row, outcome, ROOT, self.catalog, ToolEnv)
        self.assertTrue(case["state_hash_match"])
        self.assertEqual(case["score_proxy"], 1.0)
        self.assertEqual(outcome["trace"][0]["grounding"]["reviewed_sources"],
                         outcome["trace"][0]["action"]["sources"])

    def test_train_0063_full_request_with_extraction_rule_reaches_source_review(self):
        with (ROOT / "train" / "train.jsonl").open(encoding="utf-8") as handle:
            row = next(json.loads(line) for line in handle
                       if json.loads(line)["id"] == "train_0063")
        llm = ScriptedLLM([
            call("sms.send", contact_id="c_10", content="nhớ mang tài liệu",
                 sources={
                     "contact_id": {"origin": ["history"],
                                    "inputs": [{"path": "[1].tool_results[0].contacts[9].id"}],
                                    "rule": "Sao chép nguyên ID từ kết quả tìm liên hệ."},
                     "content": {"origin": ["input"],
                                 "inputs": [{}],
                                 "rule": "Lấy nội dung sau dấu hai chấm, bỏ khoảng trắng đầu và dấu chấm cuối."},
                 }),
            {"type": "done"},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT).run(row, row["audio_text"])
        case = evaluate_agent_case(row, outcome, ROOT, self.catalog, ToolEnv)
        self.assertTrue(case["state_hash_match"])
        self.assertEqual(case["score_proxy"], 1.0)
        self.assertIn("argument_sources", outcome["trace"][0]["node_reviews"])
        source_prompt = next(messages for messages in llm.prompts
                             if "NGUỒN ARGUMENT" in messages[0]["content"])
        self.assertIn("parameters", json.loads(source_prompt[1]["content"])["tool"])

    def test_train_0075_succeeds_with_source_for_every_argument(self):
        with (ROOT / "train" / "train.jsonl").open(encoding="utf-8") as handle:
            row = next(json.loads(line) for line in handle
                       if json.loads(line)["id"] == "train_0075")
        actions = [
            call("smart_home.get_status"),
            call("smart_home.light_set", device_id="light_living", power="on",
                 brightness=50, sources={
                     "device_id": result_source(0, "devices[0].id"),
                     "power": {"origin": "multi_source", "inputs": [
                         {"origin": "input", "quote": "Bật đèn phòng khách"}],
                         "rule": "bật đèn nghĩa là power=on"},
                     "brightness": {"origin": "multi_source", "inputs": [
                         {"origin": "input", "quote": "50 phần trăm"}],
                         "rule": "50 phần trăm -> số nguyên 50"},
                 }),
            call("smart_home.climate_set", device_id="ac_living", mode="cool",
                 temperature=26, sources={
                     "device_id": result_source(0, "devices[1].id"),
                     "mode": {"origin": "multi_source", "inputs": [
                         {"origin": "input", "quote": "bật điều hoà phòng khách 26 độ"}],
                         "rule": "bật điều hoà 26 độ nghĩa là mode=cool"},
                     "temperature": {"origin": "multi_source", "inputs": [
                         {"origin": "input", "quote": "26 độ"}],
                         "rule": "26 độ -> số nguyên 26"},
                 }),
            {"type": "done"},
        ]
        outcome = Agent(ScriptedLLM(actions), self.catalog, ToolEnv, ROOT).run(
            row, row["audio_text"])
        case = evaluate_agent_case(row, outcome, ROOT, self.catalog, ToolEnv)
        self.assertTrue(case["state_hash_match"])
        self.assertEqual(case["score_proxy"], 1.0)
        self.assertEqual(list(outcome["trace"][1]["node_reviews"]),
                         ["relevance", "schema", "argument_sources", "arguments"])
        self.assertEqual(len(outcome["trace"][1]["action"]["sources"]), 3)

    def test_missing_source_feedback_lists_fields(self):
        result = check_sources(
            {"arguments": {"device_id": "light_living", "power": "on",
                           "brightness": 50},
             "sources": {"device_id": result_source(0, "devices[0].id")}},
            {}, "Bật đèn phòng khách 50 phần trăm", {}, [], [],
        )
        self.assertFalse(result["ok"])
        self.assertIn("brightness", result["issues"][0]["message"])
        self.assertIn("power", result["issues"][0]["message"])

    def test_source_reviewer_requires_exact_argument_keys(self):
        class WrongKeys:
            def count(self, messages):
                return 100

            def complete(self, messages, max_new_tokens=256):
                return {"text": json.dumps({
                    "sources": {
                        "nhắn_tin_cho_minh": {"origin": "input", "quote": "Minh"}}}),
                    "prompt_tokens": 100, "generation_tokens": 20,
                    "latency_s": 0.01}

        verdict = review_node(
            WrongKeys(), "argument_sources",
            {"action": {"arguments": {"query": "Minh"}}}, 1000, 256,
        )
        self.assertFalse(verdict["ok"])
        self.assertIn("query", verdict["reason"])
        self.assertIn("nhắn_tin_cho_minh", verdict["reason"])

    def test_reviewer_origin_is_verified_before_tool_call(self):
        class FabricatedOriginLLM(ScriptedLLM):
            def complete(self, messages, max_new_tokens=256):
                if "NGUỒN ARGUMENT" in messages[0]["content"]:
                    self.prompts.append(messages)
                    return {
                        "text": json.dumps({"sources": {
                            "query": {"origin": "input", "quote": "Lê Minh"},
                        }}),
                        "prompt_tokens": 100, "generation_tokens": 20,
                        "latency_s": 0.01,
                    }
                return super().complete(messages, max_new_tokens)

        llm = FabricatedOriginLLM([
            call("contacts.search", query="Phạm Hùng",
                 sources={"query": {"origin": "input", "quote": "Phạm Hùng"}}),
            {"type": "done"},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT,
                        max_decisions=4).run(self.row, self.row["audio_text"])
        event = outcome["trace"][0]
        self.assertEqual(event["grounding"]["stage"], "sources")
        self.assertIn("invalid_source",
                      [issue["code"] for issue in event["grounding"]["issues"]])
        self.assertNotIn("tool_result", event)
        self.assertEqual(outcome["tool_calls"], [])

    def test_schema_review_rejection_returns_short_feedback(self):
        class RejectSchemaLLM(ScriptedLLM):
            def complete(self, messages, max_new_tokens=256):
                if "SCHEMA ARGUMENT" in messages[0]["content"]:
                    self.prompts.append(messages)
                    return {"text": '{"reject_reason":"query sai schema"}',
                            "prompt_tokens": 100, "generation_tokens": 20,
                            "latency_s": 0.01}
                return super().complete(messages, max_new_tokens)

        llm = RejectSchemaLLM([
            call("contacts.search", query="Phạm Hùng",
                 sources={"query": {"origin": "input", "quote": "Phạm Hùng"}}),
            {"type": "done"},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT,
                        max_decisions=4).run(self.row, self.row["audio_text"])
        event = outcome["trace"][0]
        self.assertEqual(event["grounding"]["stage"], "schema")
        self.assertEqual(event["error"]["reason"], "query sai schema")
        self.assertEqual(outcome["tool_calls"], [])

    def test_review_prompts_have_no_examples_and_planner_has_format_samples(self):
        self.assertEqual(set(SYSTEMS), {
            "missing_params", "relevance", "argument_sources",
            "arguments", "schema",
        })
        for node, prompt in SYSTEMS.items():
            with self.subTest(node=node):
                self.assertNotIn("Ví dụ", prompt)
                self.assertIn("reject_reason", prompt)
        self.assertIn(SOURCE_FORMAT, SYSTEMS["argument_sources"])
        from agent import SYSTEM, FILL_SYSTEM, LEGACY_FILL_SYSTEM
        self.assertIn(SOURCE_FORMAT, LEGACY_FILL_SYSTEM)
        for origin in ("input", "context", "history", "previous_tool_calls", "schema"):
            self.assertIn(origin, SOURCE_FORMAT)
        self.assertIn('"origin":["input"]', SOURCE_FORMAT)
        self.assertIn('input -> {}', SOURCE_FORMAT)
        self.assertIn("rule là chuỗi không rỗng kể cả khi sao chép chính xác", SOURCE_FORMAT)
        self.assertIn("không thêm result.", SOURCE_FORMAT)
        self.assertIn('{"type":"call","name":"<tool>"', SYSTEM)
        self.assertIn('{"type":"ask","tool_name":"<tool đích>"', SYSTEM)
        self.assertIn('{"type":"done","completed_goals"', SYSTEM)
        self.assertIn('{"type":"done","goals":[]}', SYSTEM)
        self.assertIn("`value_reasons`", FILL_SYSTEM)
        self.assertNotIn("set(arguments) == set(sources)", FILL_SYSTEM)

    def test_review_reject_reason_is_short(self):
        class LongReview:
            def count(self, messages):
                return 100

            def complete(self, messages, max_new_tokens=96):
                return {"text": json.dumps({"reject_reason": "sai " * 60}),
                        "prompt_tokens": 100, "generation_tokens": 40,
                        "latency_s": 0.01}

        verdict = review_node(
            LongReview(), "relevance", {"action": {"arguments": {}}},
            1000, 96,
        )
        self.assertFalse(verdict["ok"])
        self.assertLessEqual(len(verdict["reason"].split()), 20)

    def test_rejection_reason_enters_next_planner_context(self):
        llm = ScriptedLLM([
            call("contacts.search", query="Lê Minh",
                 sources={"query": {"origin": "context", "path": "missing.path"}}),
            {"type": "done"},
        ])
        agent = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=4)
        agent.run(self.row, self.row["audio_text"])
        next_input = json.loads(llm.prompts[-1][1]["content"])
        self.assertEqual(next_input["validation_feedback"]["error"],
                         "invalid_argument_source")
        self.assertIn("source path không tồn tại", next_input["validation_feedback"]["reason"])

    def test_empty_arguments_do_not_bypass_required_schema(self):
        outcome = self.run_actions([
            call("banking.transfer"),
            {"type": "done"},
        ])
        event = outcome["trace"][0]
        self.assertEqual(event["missing_fields"], ["amount", "beneficiary_id", "from_account_id"])
        self.assertEqual(event["error"]["error"], "missing_required_arguments")
        self.assertEqual(outcome["tool_calls"], [])

    def test_invalid_ask_for_argument_free_read_becomes_call(self):
        outcome = self.run_actions([
            {"type": "ask", "tool_name": "banking.list_beneficiaries",
             "params": ["beneficiary_name"],
             "tool_reason": "Cần hỏi tên người thụ hưởng để tra cứu."},
            {"type": "done"},
        ])
        event = outcome["trace"][0]
        self.assertEqual(event["action"]["type"], "call")
        self.assertEqual(event["action"]["arguments"], {})
        self.assertIn("auto_corrected_ask", event)
        self.assertEqual(outcome["tool_calls"][0]["name"],
                         "banking.list_beneficiaries")

    def test_schema_validation_runs_before_sources_and_relevance(self):
        outcome = self.run_actions([
            call("bill_payment.list_bills"),
            call("banking.get_balance"),
            call("banking.list_beneficiaries"),
            call("banking.transfer", from_account_id="acc_02",
                 beneficiary_id="ben_02", amount="218000",
                 sources={"from_account_id": result_source(1, "accounts[1].id"),
                          "beneficiary_id": result_source(2, "beneficiaries[1].id"),
                          "amount": result_source(0, "bills[2].amount")}),
            {"type": "done"},
        ])
        event = outcome["trace"][3]
        self.assertEqual(event["grounding"]["stage"], "schema")
        self.assertTrue(event["grounding"]["ok"])
        self.assertFalse(event["validation"]["ok"])
        self.assertEqual(event["error"]["error"], "invalid_schema")
        self.assertEqual(list(event["node_reviews"]), ["relevance", "schema"])

    def test_response_done_false_is_not_successful(self):
        self.assertFalse(result_succeeded({"ok": True, "done": False}))


if __name__ == "__main__":
    unittest.main()
