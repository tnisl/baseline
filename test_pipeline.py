"""Focused tests for the active three-generation pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "train+public_test"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "train"))

from agent_pipeline import (Agent, added_goals_error, available_lookup_for_ask,
                            done_evidence_error, select_argument_fields,
                            selector_issue, value_selection_error,
                            revised_goals_error)  # noqa: E402
from pipeline_rules import (known_answer_for_ask, literal_grounding_error,
                            optional_fields_for_intent,
                            repeated_write, spoken_phone_in_text,
                            tool_action_error, value_hints)  # noqa: E402
from common import evaluate_agent_case, load_catalog  # noqa: E402
from zaic_tools import ToolEnv  # noqa: E402


def sample(sample_id):
    with (ROOT / "train" / "train.jsonl").open(encoding="utf-8") as handle:
        return next(json.loads(line) for line in handle
                    if json.loads(line)["id"] == sample_id)


class ScriptedLLM:
    model_id = "scripted"
    seed = 42
    temperature = 0

    def __init__(self, answers):
        self.answers = iter(answers)
        self.prompts = []

    def count(self, messages):
        return 100

    def complete(self, messages, max_new_tokens=512):
        self.prompts.append(messages)
        return {"text": json.dumps(next(self.answers), ensure_ascii=False),
                "prompt_tokens": 100, "generation_tokens": 20,
                "latency_s": 0.01}


CALL = {"type": "call", "name": "appliances.start_cycle",
        "tool_reason": "Chạy chế độ giặt vừa được người dùng chọn.",
        "goals": [{"id": "wash", "task": "Chạy giặt nhanh cho máy đã chọn",
                   "when": "always", "kind": "write"}]}
DONE_WASH = {"type": "done", "completed_goals": {"wash": 0}, "skipped_goals": {}}
FIELDS = {"selected_arguments": ["appliance_id", "cycle"]}
VALUES = {
    "arguments": {"appliance_id": "appl_06", "cycle": "quick"},
    "value_reasons": {"appliance_id": "Máy được nhắc ở lượt trước",
                      "cycle": "Chế độ được chọn trong yêu cầu"},
}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = sample("train_0486")
        cls.catalog = load_catalog(ROOT)

    def run_agent(self, answers, row=None):
        row = row or self.row
        llm = ScriptedLLM(answers)
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT).run(row, row["audio_text"])
        return llm, outcome

    def test_required_fields_are_added_to_empty_qwen_selection(self):
        llm, outcome = self.run_agent([
            CALL, {"selected_arguments": []}, VALUES, DONE_WASH])
        case = evaluate_agent_case(self.row, outcome, ROOT, self.catalog, ToolEnv)
        self.assertEqual(case["score_proxy"], 1.0)
        self.assertEqual(outcome["grounding_reviews"], 0)
        self.assertEqual(outcome["unverified_arguments"], 2)
        self.assertEqual(len(llm.prompts), 4)
        self.assertFalse(any("Bạn review" in p[0]["content"] for p in llm.prompts))
        value_prompt = llm.prompts[2]
        payload = json.loads(value_prompt[1]["content"])
        self.assertEqual(payload["selected_arguments"], FIELDS["selected_arguments"])
        self.assertNotIn("argument_reasons", payload)
        self.assertNotIn("reference_hints", payload)
        self.assertNotIn("sources", value_prompt[0]["content"])

    def test_selector_issue_returns_to_planner_without_tool_call(self):
        row = sample("train_0236")
        call = {"type": "call", "name": "sms.send", "tool_reason": "Gửi SMS",
                "goals": [{"id": "sms", "task": "Gửi SMS cho sếp Dũng",
                           "when": "always", "kind": "write"}]}
        issue = {"issue": {"type": "unsupported_tool", "fields": [],
                           "reason": "Tool này không thể thực hiện mục tiêu"}}
        ask = {"type": "ask", "tool_name": "sms.send",
               "params": ["content"], "tool_reason": "Cần hỏi nội dung"}
        llm, outcome = self.run_agent([call, issue, ask], row)
        self.assertEqual(outcome["selector_issues"], 1)
        self.assertEqual(outcome["tool_calls"][0]["name"], "sys.ask_missing_info")
        feedback = json.loads(llm.prompts[2][1]["content"])["validation_feedback"]
        self.assertEqual(feedback["issue"], issue["issue"])

    def test_read_step_survives_whole_task_unsupported_issue(self):
        row = sample("train_0255")
        first = {"type": "call", "name": "gold_price.get_price",
                 "tool_reason": "Đọc giá vàng để chọn nhánh", "goals": [
                     {"id": "g1", "task": "Chuyển khoản nếu giá chưa đủ",
                      "when": "giá chưa đủ", "kind": "write"}]}
        llm = ScriptedLLM([first,
                           {"issue": {"type": "unsupported_tool", "fields": [],
                                      "reason": "Không thể chuyển khoản"}},
                           {"issue": {"type": "missing_data", "fields": ["type"],
                                      "reason": "Chưa biết loại vàng"}}])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=1).run(
            row, row["audio_text"])
        self.assertEqual(outcome["tool_calls"][0],
                         {"name": "gold_price.get_price", "arguments": {"type": "sjc"}})
        self.assertIn("ignored_read_issue", outcome["trace"][0])
        self.assertEqual(outcome["trace"][0]["used_value_hints"], ["type"])

    def test_missing_optional_read_filter_falls_back_to_full_list(self):
        row = sample("train_0075")
        first = {"type": "call", "name": "smart_home.get_status",
                 "tool_reason": "Tìm thiết bị phòng khách", "goals": [
                     {"id": "g1", "task": "Chỉnh đèn phòng khách", "when": "always",
                      "kind": "write"}]}
        llm = ScriptedLLM([first, {"selected_arguments": ["room"]},
                           {"issue": {"type": "missing_data", "fields": ["room"],
                                      "reason": "Không xác định được room"}}])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=1).run(
            row, row["audio_text"])
        self.assertEqual(outcome["tool_calls"][0],
                         {"name": "smart_home.get_status", "arguments": {}})
        self.assertEqual(outcome["trace"][0]["dropped_unavailable_read_filter"],
                         ["room"])

    def test_read_uses_complete_hints_when_value_selector_reports_missing(self):
        row = sample("train_1249")
        first = {"type": "call", "name": "lottery.check_ticket",
                 "tool_reason": "Dò tờ vé vừa nói", "goals": [
                     {"id": "g1", "task": "Dò vé số", "when": "always", "kind": "read"}]}
        llm = ScriptedLLM([first,
                           {"issue": {"type": "missing_data",
                                      "fields": ["province", "date", "number"],
                                      "reason": "Thiếu thông tin vé"}}])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=1).run(
            row, row["audio_text"])
        self.assertEqual(outcome["tool_calls"][0]["arguments"],
                         {"province": "Đồng Nai", "date": "2026-10-20",
                          "number": "332052"})
        self.assertEqual(set(outcome["trace"][0]["used_value_hints"]),
                         {"province", "date", "number"})

    def test_issue_contract_rejects_malformed_fields(self):
        spec = {"parameters": {"properties": {"amount": {}}}}
        _, error = selector_issue({"issue": {"type": "missing_data",
                                             "fields": ["not_a_field"],
                                             "reason": "unknown"}}, spec)
        self.assertIsNotNone(error)

    def test_value_issue_can_lead_planner_to_ask(self):
        issue = {"issue": {"type": "missing_data", "fields": ["cycle"],
                           "reason": "Chưa biết chế độ cần chạy"}}
        ask = {"type": "ask", "tool_name": "appliances.start_cycle",
               "params": ["cycle"], "tool_reason": "Cần hỏi chế độ máy giặt"}
        llm, outcome = self.run_agent([
            CALL, {"selected_arguments": []}, issue, ask])
        self.assertEqual(outcome["stop_reason"], "ask_missing_info")
        self.assertEqual(outcome["tool_calls"][0]["name"], "sys.ask_missing_info")
        feedback = json.loads(llm.prompts[3][1]["content"])["validation_feedback"]
        self.assertEqual(feedback["issue"], issue["issue"])

    def test_value_output_does_not_accept_sources(self):
        values = {**VALUES, "sources": {}}
        self.assertIsNotNone(value_selection_error(values, FIELDS["selected_arguments"]))
        llm, outcome = self.run_agent([
            CALL, {"selected_arguments": []}, values,
            {"type": "ask", "tool_name": "appliances.start_cycle",
             "params": ["cycle"], "tool_reason": "Cần hỏi chế độ"}])
        self.assertEqual(outcome["tool_calls"][0]["name"], "sys.ask_missing_info")
        feedback = json.loads(llm.prompts[3][1]["content"])["validation_feedback"]
        self.assertEqual(feedback["error"], "invalid_argument_proposal")

    def test_qwen_selects_optional_filter_without_reasons(self):
        row = {**sample("train_0236"), "audio_text": "Liệt kê hóa đơn truyền hình"}
        call = {"type": "call", "name": "bill_payment.list_bills",
                "tool_reason": "Tìm hóa đơn truyền hình",
                "goals": [{"id": "list", "task": "Xem hóa đơn truyền hình",
                           "when": "always", "kind": "read"}]}
        llm, outcome = self.run_agent([
            call, {"selected_arguments": ["type"]},
            {"arguments": {"type": "tv"},
             "value_reasons": {"type": "Lọc hóa đơn truyền hình"}},
            {"type": "done", "completed_goals": {"list": 0}, "skipped_goals": {}},
        ], row)
        self.assertEqual(outcome["tool_calls"][0]["arguments"], {"type": "tv"})
        self.assertEqual(outcome["trace"][0]["selected_arguments"], ["type"])
        fill_payload = json.loads(llm.prompts[2][1]["content"])
        self.assertEqual(fill_payload["selected_arguments"], ["type"])
        self.assertNotIn("argument_reasons", fill_payload)

    def test_required_fields_are_added_to_qwen_selection(self):
        spec = {"parameters": {"properties": {"bill_id": {}},
                               "required": ["bill_id"]}}
        selected, error = select_argument_fields({"selected_arguments": []}, spec)
        self.assertIsNone(error)
        self.assertEqual(selected, ["bill_id"])

    def test_x_one_of_requires_one_complete_group(self):
        spec = self.catalog["sms.send"]
        selected, error = select_argument_fields(
            {"selected_arguments": ["contact_id"]}, spec)
        self.assertIsNone(error)
        self.assertEqual(selected, ["content", "contact_id"])
        _, error = select_argument_fields({"selected_arguments": []}, spec)
        self.assertIn("x-one-of", error)

    def test_zero_argument_tool_skips_both_selectors(self):
        row = {**sample("train_0236"), "audio_text": "Xem số dư ví"}
        call = {"type": "call", "name": "ewallet.get_balance",
                "tool_reason": "Đọc số dư ví",
                "goals": [{"id": "balance", "task": "Xem số dư ví",
                           "when": "always", "kind": "read"}]}
        llm, outcome = self.run_agent([
            call, {"type": "done", "completed_goals": {"balance": 0}, "skipped_goals": {}}], row)
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(outcome["tool_calls"][0]["arguments"], {})
        self.assertEqual(outcome["trace"][0]["selected_arguments"], [])

    def test_required_only_tool_skips_argument_selector(self):
        row = sample("train_0063")
        call = {"type": "call", "name": "contacts.get",
                "tool_reason": "Đọc thông tin liên hệ",
                "goals": [{"id": "get", "task": "Xem thông tin liên hệ",
                           "when": "always", "kind": "read"}]}
        values = {"arguments": {"contact_id": "c_10"},
                  "value_reasons": {"contact_id": "Liên hệ cần đọc"}}
        llm, outcome = self.run_agent([
            call, values, {"type": "done", "completed_goals": {"get": 0},
                           "skipped_goals": {}}], row)
        self.assertEqual(len(llm.prompts), 3)
        self.assertEqual(outcome["trace"][0]["selected_arguments"], ["contact_id"])
        self.assertEqual(outcome["tool_calls"][0]["arguments"], {"contact_id": "c_10"})

    def test_train_0236_scripted_gold_path(self):
        row = sample("train_0236")
        def call(name):
            return {"type": "call", "name": name, "tool_reason": "Bước cần thực hiện"}
        first = {**call("bill_payment.list_bills"), "goals": [
            {"id": "pay_wallet", "task": "Trả tiền truyền hình bằng ví", "when": "ví đủ", "kind": "write"},
            {"id": "sms", "task": "Nhắn sếp Dũng đã đóng", "when": "ví đủ", "kind": "write"},
            {"id": "pay_bank", "task": "Trả tiền truyền hình bằng ngân hàng", "when": "ví thiếu", "kind": "write"},
            {"id": "remind", "task": "Nhắc nạp ví", "when": "ví thiếu", "kind": "write"},
        ]}
        llm, outcome = self.run_agent([
            first, {"selected_arguments": []},
            call("ewallet.get_balance"),
            call("bill_payment.pay"),
            {"arguments": {"bill_id": "bill_01", "source": "ewallet"},
             "value_reasons": {"bill_id": "Hóa đơn truyền hình",
                               "source": "Ví đủ tiền"}},
            call("contacts.search"), {"selected_arguments": ["query"]},
            {"arguments": {"query": "Sếp Dũng"},
             "value_reasons": {"query": "Người nhận cần tìm"}},
            call("sms.send"), {"selected_arguments": ["contact_id"]},
            {"arguments": {"content": "đã đóng tiền truyền hình rồi nhé",
                           "contact_id": "c_04"},
             "value_reasons": {"content": "Thông báo đã thanh toán",
                               "contact_id": "Liên hệ sếp Dũng"}},
            {"type": "done", "completed_goals": {"pay_wallet": 2, "sms": 4},
             "skipped_goals": {"pay_bank": "ví đủ", "remind": "ví đủ"}},
        ], row)
        case = evaluate_agent_case(row, outcome, ROOT, self.catalog, ToolEnv)
        self.assertEqual(case["score_proxy"], 1.0)
        self.assertEqual(outcome["stop_reason"], "done")
        self.assertEqual(len(llm.prompts), 12)

    def test_done_requires_evidence_for_each_goal(self):
        row = sample("train_0236")
        first = {"type": "call", "name": "bill_payment.list_bills",
                 "tool_reason": "Đọc hóa đơn", "goals": [
                     {"id": "pay", "task": "Trả hóa đơn truyền hình", "when": "always", "kind": "write"},
                     {"id": "sms", "task": "Nhắn sếp Dũng", "when": "always", "kind": "write"}]}
        _, outcome = self.run_agent([
            first, {"selected_arguments": []},
            {"type": "done", "completed_goals": {"pay": 0}, "skipped_goals": {}},
            {"type": "ask", "tool_name": "bill_payment.pay", "params": ["source"],
             "tool_reason": "Cần biết nguồn thanh toán"},
        ], row)
        self.assertEqual(outcome["trace"][1]["error"]["error"], "incomplete_request")
        self.assertEqual(outcome["stop_reason"], "ask_missing_info")

    def test_done_evidence_rejects_wrong_kind(self):
        goals = [{"id": "sms", "task": "Nhắn người nhận", "when": "always", "kind": "write"}]
        observations = [{"name": "contacts.search", "kind": "read", "result": {"ok": True}}]
        error = done_evidence_error(
            {"type": "done", "completed_goals": {"sms": 0}, "skipped_goals": {}},
            goals, observations)
        self.assertIsNotNone(error)

    def test_value_reason_cannot_disclaim_its_own_value(self):
        candidate = {"arguments": {"phone_number": "0912345678"},
                     "value_reasons": {"phone_number": "Không thể xác định số điện thoại"}}
        self.assertIsNotNone(value_selection_error(candidate, ["phone_number"]))

    def test_repeated_successful_write_is_rejected(self):
        repeated = {"type": "call", "name": "appliances.start_cycle",
                    "tool_reason": "Chạy lại máy giặt"}
        _, outcome = self.run_agent([
            CALL, {"selected_arguments": []}, VALUES,
            repeated, {"selected_arguments": []}, VALUES,
            DONE_WASH,
        ])
        self.assertEqual(outcome["trace"][1]["error"]["error"],
                         "duplicate_successful_write")
        self.assertEqual(len(outcome["tool_calls"]), 1)

    def test_done_allowed_after_last_tool_slot(self):
        llm = ScriptedLLM([
            CALL, {"selected_arguments": []}, VALUES, DONE_WASH,
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT,
                        max_tool_calls=1).run(self.row, self.row["audio_text"])
        self.assertEqual(outcome["stop_reason"], "done")
        self.assertEqual(len(outcome["tool_calls"]), 1)

    def test_ask_reference_uses_available_lookup_first(self):
        row = sample("train_0236")
        specs = {name: spec for name, spec in self.catalog.items()
                 if name in row["context"]["installed_apps"]["contacts"]["tools"]
                 or name == "sms.send"}
        lookup = available_lookup_for_ask(
            "sms.send", ["contact_id"], specs, [])
        self.assertEqual(lookup, "contacts.search")
        lookup = available_lookup_for_ask(
            "sms.send", ["contact_id"], specs,
            [{"name": "contacts.search"}])
        self.assertIsNone(lookup)
        self.assertEqual(available_lookup_for_ask(
            "email.create_draft", ["to"], self.catalog, []), "contacts.search")

    def test_planner_receives_compact_catalog_and_goal_plan(self):
        row = {**sample("train_0236"), "audio_text": "Cảm ơn"}
        llm, _ = self.run_agent([
            {"type": "done", "goals": []},
        ], row)
        payload = json.loads(llm.prompts[0][1]["content"])
        self.assertIsNone(payload["task_plan"])
        tool = next(spec for spec in payload["available_tools"]
                    if spec["name"] == "sms.send")
        self.assertEqual(tool["parameters"]["properties"],
                         ["contact_id", "phone_number", "content"])

    def test_outcome_goal_finishes_after_write_without_redundant_read(self):
        first = {**CALL, "goals": [
            {"id": "wash", "task": "Chạy giặt nhanh cho máy vừa nói tới",
             "when": "always", "kind": "write"}]}
        llm, outcome = self.run_agent([
            first, {"selected_arguments": []}, VALUES,
            {"type": "done", "completed_goals": {"wash": 0}},
        ])
        self.assertEqual(outcome["stop_reason"], "done")
        self.assertEqual(len(outcome["tool_calls"]), 1)
        self.assertEqual(evaluate_agent_case(self.row, outcome, ROOT, self.catalog,
                                             ToolEnv)["score_proxy"], 1.0)
        self.assertEqual(len(llm.prompts), 4)

    def test_planner_can_revise_wrong_read_goal_without_repeating_write(self):
        first = {**CALL, "goals": [
            {"id": "wash", "task": "Xem danh sách máy giặt", "when": "always", "kind": "read"}]}
        revised = {"type": "revise", "goals": [
            {"id": "wash", "task": "Chạy giặt nhanh cho máy đã chọn",
             "when": "always", "kind": "write"}],
            "removed_goals": {}, "reason": "Goal cũ nhầm bước đọc phụ thành kết quả cuối."}
        _, outcome = self.run_agent([
            first, {"selected_arguments": []}, VALUES, revised,
            {"type": "done", "completed_goals": {"wash": 0}},
        ])
        self.assertEqual(outcome["stop_reason"], "done")
        self.assertEqual(outcome["plan_revisions"], 1)
        self.assertEqual(len(outcome["tool_calls"]), 1)
        self.assertEqual(outcome["trace"][1]["revised_plan"], revised["goals"])

    def test_plan_revision_requires_reason_for_removed_goal(self):
        current = [{"id": "lookup", "task": "Tìm liên hệ", "when": "always", "kind": "read"},
                   {"id": "sms", "task": "Gửi SMS", "when": "always", "kind": "write"}]
        candidate = {"type": "revise", "goals": [current[1]],
                     "removed_goals": {}, "reason": "Lookup đã có từ history"}
        _, error = revised_goals_error(candidate, current)
        self.assertIsNotNone(error)
        candidate["removed_goals"] = {"lookup": "History đã có kết quả contacts.search"}
        _, error = revised_goals_error(candidate, current)
        self.assertIsNone(error)

    def test_later_call_adds_only_new_goal_and_done_covers_every_write(self):
        current = [{"id": "g1", "task": "Huỷ gói", "when": "always", "kind": "write"}]
        later = {"type": "call", "name": "email.create_draft",
                 "goals": [{"id": "g2", "task": "Soạn thư nháp", "when": "always",
                            "kind": "write"}]}
        merged, error = added_goals_error(later, current)
        self.assertIsNone(error)
        self.assertEqual([goal["id"] for goal in merged], ["g1", "g2"])
        observations = [
            {"name": "subscriptions.cancel", "kind": "write", "result": {"ok": True}},
            {"name": "email.create_draft", "kind": "write", "result": {"ok": True}},
        ]
        self.assertIn("Mỗi goal", done_evidence_error(
            {"type": "done", "completed_goals": {"g1": 0}}, merged, observations))
        self.assertIn("Write thành công", done_evidence_error(
            {"type": "done", "completed_goals": {"g1": 0}}, current, observations))
        self.assertIsNone(done_evidence_error(
            {"type": "done", "completed_goals": {"g1": 0, "g2": 1}},
            merged, observations))

    def test_missing_reference_feedback_names_untried_lookup(self):
        row = sample("train_1703")
        first = {"type": "call", "name": "medication.log_taken",
                 "tool_reason": "Ghi đã uống thuốc", "goals": [
                     {"id": "g1", "task": "Ghi đã uống thuốc", "when": "always",
                      "kind": "write"}]}
        llm = ScriptedLLM([first, {"selected_arguments": ["time"]},
                           {"arguments": {"schedule_id": "med_999", "time":
                                          "2026-12-13T07:00:00+07:00"},
                            "value_reasons": {"schedule_id": "Lịch thuốc",
                                              "time": "Lúc bảy giờ"}},
                           {"type": "done", "completed_goals": {"g1": 0}}])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=1).run(
            row, row["audio_text"])
        self.assertEqual(outcome["trace"][0]["error"]["suggested_lookup"],
                         "medication.list")

    def test_named_sms_uses_observed_contact_id(self):
        row = sample("train_0063")
        first = {"type": "call", "name": "sms.send", "tool_reason": "Nhắn Minh",
                 "goals": [{"id": "sms", "task": "Nhắn Minh nhớ mang tài liệu",
                            "when": "always", "kind": "write"}]}
        _, outcome = self.run_agent([
            first, {"selected_arguments": ["phone_number"]},
            {"arguments": {"content": "nhớ mang tài liệu", "phone_number": "09631116692"},
             "value_reasons": {"content": "Lời user hiện tại",
                               "phone_number": "Số Minh khách hàng trong contacts.search"}},
            {"type": "done", "completed_goals": {"sms": 0}},
        ], row)
        self.assertEqual(outcome["tool_calls"][0]["arguments"],
                         {"content": "nhớ mang tài liệu", "contact_id": "c_10"})
        self.assertEqual(outcome["trace"][0]["normalization"],
                         "phone_number -> contact_id")

    def test_fabricated_phone_and_unspecified_volume_never_execute(self):
        row = sample("train_0236")
        sms = {"type": "call", "name": "sms.send", "tool_reason": "Nhắn sếp Dũng",
               "goals": [{"id": "sms", "task": "Nhắn sếp Dũng",
                          "when": "always", "kind": "write"}]}
        llm = ScriptedLLM([
            sms, {"selected_arguments": ["phone_number"]},
            {"arguments": {"content": "đã đóng", "phone_number": "0912345678"},
             "value_reasons": {"content": "Lời user", "phone_number": "Số của sếp"}},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=1).run(
            row, row["audio_text"])
        self.assertEqual(outcome["trace"][0]["error"]["error"], "ungrounded_value")
        self.assertEqual(outcome["tool_calls"], [])

        row = sample("train_1715")
        tv = {"type": "call", "name": "tv.control", "tool_reason": "Giảm âm lượng",
              "goals": [{"id": "volume", "task": "Giảm âm lượng tivi",
                         "when": "always", "kind": "write"}]}
        llm = ScriptedLLM([
            tv, {"selected_arguments": ["volume"]},
            {"arguments": {"action": "volume", "volume": 1},
             "value_reasons": {"action": "Người dùng muốn chỉnh âm lượng",
                               "volume": "Mức thấp"}},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=1).run(
            row, row["audio_text"])
        self.assertEqual(outcome["trace"][0]["error"]["error"], "semantic_value")
        self.assertEqual(outcome["tool_calls"], [])

    def test_optional_arguments_do_not_cross_task_boundaries(self):
        fields, dropped = optional_fields_for_intent(
            "home_security.list_alerts", ["since"],
            "Xem cảnh báo mới nhất rồi chép vào ghi chú")
        self.assertEqual((fields, dropped), ([], ["since"]))
        fields, dropped = optional_fields_for_intent(
            "appliances.start_cycle", ["appliance_id", "cycle", "delay_minutes"],
            "Bật giặt nhanh rồi 10 giờ nhắc tôi lấy đồ")
        self.assertEqual(fields, ["appliance_id", "cycle"])
        self.assertEqual(dropped, ["delay_minutes"])
        fields, dropped = optional_fields_for_intent(
            "medication.add", ["name", "times", "days"],
            "Thêm thuốc uống lúc đèn phòng ngủ bật")
        self.assertEqual(fields, ["name", "times"])
        self.assertEqual(dropped, ["days"])
        fields, dropped = optional_fields_for_intent(
            "home_security.list_alerts", ["since"], "Xem cảnh báo hôm nay")
        self.assertEqual((fields, dropped), ([], ["since"]))
        fields, dropped = optional_fields_for_intent(
            "bill_payment.list_bills", ["type"],
            "Trả tất cả hóa đơn bằng ngân hàng")
        self.assertEqual((fields, dropped), ([], ["type"]))

    def test_known_values_prevent_false_missing_info(self):
        row = sample("train_0313")
        observations = [
            {"name": "ewallet.get_balance", "result": {"ok": True, "balance": 70011}},
            {"name": "bill_payment.list_bills", "result": {"ok": True,
             "bills": [{"id": "bill_02", "amount": 215000, "paid": False}]}},
        ]
        self.assertEqual(value_hints("bill_payment.pay", row["audio_text"], row,
                                     observations)["source"], "bank")
        self.assertEqual(known_answer_for_ask("bill_payment.pay", ["source"],
                                              row["audio_text"], row, observations),
                         {"source": "bank"})
        row = sample("train_1474")
        self.assertEqual(value_hints("medication.add", row["audio_text"], row,
                                     [])["times"], ["05:00"])
        row = sample("train_0063")
        hints = value_hints("sms.send", row["audio_text"], row, [])
        self.assertEqual(hints["contact_id"], "c_10")
        self.assertEqual(hints["phone_number"], "09631116692")
        row = sample("train_1249")
        hints = value_hints("lottery.check_ticket", row["audio_text"], row, [],
                            self.catalog["lottery.check_ticket"])
        self.assertEqual({key: hints[key] for key in ("province", "date", "number")},
                         {"province": "Đồng Nai", "date": "2026-10-20",
                          "number": "332052"})
        row = sample("train_1155")
        self.assertEqual(value_hints("podcast_radio.search", row["audio_text"],
                                     row, [])["query"], "gửi tiết kiệm")

    def test_history_time_hint_and_optional_frequency_reach_medication_tool(self):
        row = sample("train_1474")
        first = {"type": "call", "name": "medication.add", "tool_reason": "Thêm thuốc",
                 "goals": [{"id": "med", "task": "Thêm Omeprazole đúng giờ đèn bật",
                            "when": "always", "kind": "write"}]}
        llm = ScriptedLLM([
            first, {"selected_arguments": ["days"]},
            {"arguments": {"name": "Omeprazole", "times": ["05:00"]},
             "value_reasons": {"name": "Tên thuốc trong request",
                               "times": "Giờ bật đèn phòng ngủ trong history"}},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=1).run(
            row, row["audio_text"])
        self.assertEqual(outcome["tool_calls"][0]["arguments"],
                         {"name": "Omeprazole", "times": ["05:00"]})
        self.assertEqual(outcome["trace"][0]["dropped_optional_fields"], ["days"])
        self.assertEqual(json.loads(llm.prompts[2][1]["content"])["value_hints"],
                         {"times": ["05:00"]})

    def test_latest_alert_read_drops_unrequested_since(self):
        row = sample("train_0538")
        first = {"type": "call", "name": "home_security.list_alerts",
                 "tool_reason": "Lấy cảnh báo để tạo ghi chú",
                 "goals": [{"id": "note", "task": "Tạo ghi chú cảnh báo mới nhất",
                            "when": "always", "kind": "write"}]}
        llm = ScriptedLLM([first, {"selected_arguments": ["since"]}])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=1).run(
            row, row["audio_text"])
        self.assertEqual(outcome["tool_calls"][0]["arguments"], {})
        self.assertTrue(outcome["trace"][0]["tool_result"]["alerts"])
        self.assertEqual(len(llm.prompts), 2)

    def test_planner_cannot_ask_for_known_payment_source(self):
        row = sample("train_0313")
        first = {"type": "call", "name": "ewallet.get_balance",
                 "tool_reason": "Kiểm tra ví",
                 "goals": [{"id": "pay", "task": "Trả hóa đơn nước",
                            "when": "always", "kind": "write"}]}
        llm = ScriptedLLM([
            first,
            {"type": "call", "name": "bill_payment.list_bills",
             "tool_reason": "Tìm hóa đơn nước"},
            {"selected_arguments": ["type"]},
            {"arguments": {"type": "water"},
             "value_reasons": {"type": "Loại hóa đơn nước trong request"}},
            {"type": "ask", "tool_name": "bill_payment.pay", "params": ["source"],
             "tool_reason": "Cần nguồn thanh toán"},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=3).run(
            row, row["audio_text"])
        self.assertEqual(outcome["trace"][2]["error"]["error"], "answer_available")
        self.assertEqual(outcome["trace"][2]["error"]["known_values"], {"source": "bank"})
        self.assertNotIn("sys.ask_missing_info", [c["name"] for c in outcome["tool_calls"]])

    def test_same_read_can_run_again_after_successful_write(self):
        row = sample("train_0236")
        first = {"type": "call", "name": "bill_payment.list_bills",
                 "tool_reason": "Tìm hóa đơn", "goals": [
                     {"id": "pay", "task": "Trả tiền truyền hình", "when": "always", "kind": "write"}]}
        llm = ScriptedLLM([
            first, {"selected_arguments": []},
            {"type": "call", "name": "bill_payment.pay", "tool_reason": "Thanh toán"},
            {"arguments": {"bill_id": "bill_01", "source": "ewallet"},
             "value_reasons": {"bill_id": "Hóa đơn từ list_bills",
                               "source": "Ví đủ số dư"}},
            {"type": "call", "name": "bill_payment.list_bills",
             "tool_reason": "Kiểm tra trạng thái sau thanh toán"},
            {"selected_arguments": []},
        ])
        outcome = Agent(llm, self.catalog, ToolEnv, ROOT, max_decisions=3).run(
            row, row["audio_text"])
        self.assertEqual([c["name"] for c in outcome["tool_calls"]],
                         ["bill_payment.list_bills", "bill_payment.pay",
                          "bill_payment.list_bills"])

    def test_wrong_write_and_duplicate_event_are_blocked(self):
        row = sample("train_1703")
        self.assertIn("medication.log_taken",
                      tool_action_error("medication.add", row["audio_text"]))
        self.assertTrue(repeated_write(
            {"name": "medication.log_taken", "arguments": {
                "schedule_id": "med_04", "time": "2026-12-13T07:00:00+07:00"},
             "result": {"ok": True}},
            "medication.log_taken", {"schedule_id": "med_04"}))
        self.assertIsNone(value_selection_error(
            {"arguments": {"content": "Khách check-in"},
             "value_reasons": {"content": "Không có thông tin khác để thay đổi"}},
            ["content"]))
        self.assertIn("bill_payment.pay", tool_action_error(
            "ewallet.pay_qr", "Trả tiền nước bằng ví điện tử"))
        self.assertIn("notes.create", tool_action_error(
            "notes.append", "Tạo một ghi chú tên An ninh buổi sáng"))

    def test_invented_email_recipient_is_rejected(self):
        row = sample("train_0508")
        spec = self.catalog["email.create_draft"]
        field, _ = literal_grounding_error(
            "email.create_draft", {"to": ["vo.khoa@example.com"]}, spec,
            row["audio_text"], row, [])
        self.assertEqual(field, "to")
        field, _ = literal_grounding_error(
            "email.create_draft", {"to": ["lan.nguyen@example.test"]}, spec,
            row["audio_text"], row, [{"name": "contacts.search", "result":
                                    {"ok": True, "contacts": [{"email":
                                    "lan.nguyen@example.test"}]}}])
        self.assertIsNone(field)

    def test_spoken_phone_remains_usable_without_literal_digits(self):
        self.assertTrue(spoken_phone_in_text(
            "09836472519", "Gọi số không chín tám ba, sáu bốn bảy, hai năm một chín"))
        self.assertFalse(spoken_phone_in_text(
            "0912345678", "Nhắn sếp Dũng là đã đóng tiền"))


if __name__ == "__main__":
    unittest.main()
