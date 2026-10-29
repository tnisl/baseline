"""Three-generation LifeMate agent with deterministic checks and issue feedback."""
from __future__ import annotations

import time
import re

from agent import (
    Agent as PromptAgent, FILL_SYSTEM, SELECT_ARGUMENTS_SYSTEM,
    missing_fields, parse_action, parse_object,
    valid_ask,
)
from evidence_rules import read_invalidated
from common import allowed_specs, dumps, result_facts, result_succeeded
from pipeline_rules import (
    canonicalize_sms_destination, known_answer_for_ask,
    literal_grounding_error, optional_fields_for_intent, repeated_write,
    semantic_value_error, tool_action_error, value_hints,
)


ISSUE_TYPES = frozenset({
    "unsupported_tool", "missing_data", "ambiguous_target",
})

UNSUPPORTED_REASON = re.compile(
    r"không thể (?:xác định|điền|biết)|"
    r"(?:chưa|không) có (?:dữ liệu|thông tin) (?:để )?(?:xác định|điền)|"
    r"phải báo (?:lỗi|issue)", re.IGNORECASE)


def planner_goals_error(candidate):
    """Validate the outcome-oriented planner contract."""
    goals = candidate.get("goals")
    if not isinstance(goals, list) or not goals:
        return None, "Lượt planner đầu cần goals không rỗng."
    ids = set()
    for goal in goals:
        if not isinstance(goal, dict):
            return None, "Mỗi goal phải là object."
        valid = (set(goal) == {"id", "task", "when", "kind"}
                 and all(isinstance(goal[k], str) and goal[k].strip()
                         for k in ("id", "task", "when"))
                 and goal["kind"] in ("read", "write"))
        if not valid or goal["id"] in ids:
            return None, "Mỗi goal cần id duy nhất, task/kind/when hợp lệ."
        ids.add(goal["id"])
    return goals, None


def revised_goals_error(candidate, current):
    """Allow explicit, auditable correction of a mistaken outcome plan."""
    if set(candidate) != {"type", "goals", "removed_goals", "reason"}:
        return None, "revise cần goals, removed_goals và reason."
    goals, error = planner_goals_error(candidate)
    if error:
        return None, error
    removed = candidate["removed_goals"]
    old = {goal["id"]: goal for goal in current}
    new = {goal["id"]: goal for goal in goals}
    if (not isinstance(removed, dict) or set(removed) != set(old) - set(new)
            or any(not isinstance(reason, str) or not reason.strip()
                   for reason in removed.values())):
        return None, "Mỗi goal bị bỏ cần lý do trong removed_goals."
    if not isinstance(candidate["reason"], str) or not candidate["reason"].strip():
        return None, "revise cần lý do sửa plan."
    if goals == current:
        return None, "Plan không thay đổi."
    return goals, None


def added_goals_error(candidate, current):
    """Accept a later call that explicitly supplies only new outcome goals.

    Small models sometimes discover a second requested action while choosing its
    tool.  Preserve the original goals and record that addition instead of
    silently discarding the new goal.
    """
    proposed, error = planner_goals_error(candidate)
    if error:
        return None, error
    old = {goal["id"]: goal for goal in current}
    for goal in proposed:
        if goal["id"] in old and old[goal["id"]] != goal:
            return None, "Goal cũ thay đổi; dùng revise để sửa hoặc bỏ goal."
    added = [goal for goal in proposed if goal["id"] not in old]
    return [*current, *added], None


def done_evidence_error(candidate, goals, observations):
    """Require one successful, appropriately typed result per outcome goal."""
    if set(candidate) not in ({"type", "completed_goals"},
                              {"type", "completed_goals", "skipped_goals"}):
        return "done cần completed_goals; skipped_goals là object tùy chọn."
    completed = candidate["completed_goals"]
    skipped = candidate.get("skipped_goals", {})
    if not isinstance(completed, dict) or not isinstance(skipped, dict):
        return "Hai trường goal của done phải là object JSON."
    goal_map = {goal["id"]: goal for goal in goals}
    if set(completed) & set(skipped) or set(completed) | set(skipped) != set(goal_map):
        return "Mỗi goal phải được hoàn tất hoặc bỏ qua đúng một lần."
    for goal_id, index in completed.items():
        if (not isinstance(index, int) or isinstance(index, bool)
                or index < 0 or index >= len(observations)
                or not result_succeeded(observations[index]["result"])):
            return "Goal hoàn tất cần index của kết quả tool thành công."
        goal = goal_map[goal_id]
        observation = observations[index]
        if observation.get("kind") != goal["kind"]:
            return "Goal cần kết quả tool đúng loại read/write."
    if len(set(completed.values())) != len(completed):
        return "Mỗi goal độc lập cần kết quả tool riêng."
    completed_indices = set(completed.values())
    unassigned_writes = [index for index, item in enumerate(observations)
                         if item.get("kind") == "write"
                         and result_succeeded(item["result"])
                         and index not in completed_indices]
    if unassigned_writes:
        return ("Write thành công chưa được gắn với goal: "
                + ", ".join(map(str, unassigned_writes))
                + ". Bổ sung goal bị thiếu rồi done.")
    for goal_id, reason in skipped.items():
        if (goal_map[goal_id]["when"] == "always"
                or not isinstance(reason, str) or not reason.strip()):
            return "Chỉ được bỏ qua goal có điều kiện, kèm lý do."
    return None


def available_lookup_for_ask(name, fields, specs, observations):
    """Find an untried read tool in the app named by a requested reference field."""
    properties = (specs[name].get("parameters") or {}).get("properties") or {}
    called = {item["name"] for item in observations}
    if (((name == "sms.send" and "phone_number" in fields)
         or (name.startswith("email.") and set(fields) & {"to", "cc", "bcc"}))
            and "contacts.search" in specs and "contacts.search" not in called):
        return "contacts.search"
    for field in fields:
        prop = properties.get(field, {})
        explicit = (re.findall(r"[a-z_]+\.[a-z_]+", prop.get("description", ""))
                    if prop.get("x-ref") else [])
        if name == "food_delivery.place_order" and field == "items":
            explicit = ["food_delivery.get_menu"]
        for tool in explicit:
            if tool in specs and specs[tool].get("kind") == "read" and tool not in called:
                return tool
        ref = prop.get("x-ref", "")
        if "." not in ref:
            continue
        app, collection = ref.split(".", 1)
        options = [tool for tool, candidate in specs.items()
                   if tool.startswith(app + ".") and candidate.get("kind") == "read"
                   and collection in str(candidate.get("returns", "")) and tool not in called
                   and not candidate.get("parameters", {}).get("required")]
        if options:
            return sorted(options)[0]
    return None


def select_argument_fields(candidate, spec):
    """Add required fields in code and check Qwen's extra field selection."""
    params = spec.get("parameters") or {}
    properties = params.get("properties") or {}
    required = list(dict.fromkeys(params.get("required") or []))
    if not isinstance(candidate, dict) or set(candidate) != {"selected_arguments"}:
        return None, "Argument selector cần đúng object selected_arguments."
    extra = candidate["selected_arguments"]
    if not isinstance(extra, list) or any(not isinstance(field, str) for field in extra):
        return None, "selected_arguments phải là mảng tên field."
    if len(set(extra)) != len(extra):
        return None, "selected_arguments có field trùng lặp."
    unknown = set(extra) - set(properties)
    if unknown:
        return None, "Argument không có trong schema: " + ", ".join(sorted(unknown))
    selected = list(dict.fromkeys([*required, *extra]))
    groups = params.get("x-one-of") or []
    if groups and not any(set(group) <= set(selected) for group in groups):
        return None, "Chưa chọn đủ một nhóm x-one-of."
    return selected, None


def value_selection_error(candidate, selected_arguments):
    """Validate value-only output without requiring source citations."""
    if not isinstance(candidate, dict) or set(candidate) != {"arguments", "value_reasons"}:
        return "Value selector cần đúng arguments và value_reasons."
    args, reasons = candidate["arguments"], candidate["value_reasons"]
    if not isinstance(args, dict) or not isinstance(reasons, dict):
        return "arguments và value_reasons phải là object JSON."
    if set(args) != set(reasons) or not set(args) <= set(selected_arguments):
        return "Hai object phải có cùng field và chỉ dùng field đã chọn."
    if any(not isinstance(reason, str) or not reason.strip() for reason in reasons.values()):
        return "Mỗi value cần value_reasons không rỗng."
    if any(UNSUPPORTED_REASON.search(reason) for reason in reasons.values()):
        return "value_reasons thừa nhận thiếu dữ liệu; hãy trả issue thay vì đoán value."
    return None


def selector_issue(candidate, spec, selected_arguments=None):
    """Accept only a bounded, field-specific issue object."""
    if not isinstance(candidate, dict) or "issue" not in candidate:
        return None, None
    if set(candidate) != {"issue"}:
        return None, "issue phải là đầu ra duy nhất của selector"
    issue = candidate["issue"]
    if not isinstance(issue, dict) or set(issue) != {"type", "fields", "reason"}:
        return None, "issue cần đúng type, fields và reason"
    fields = issue["fields"]
    allowed = set(((spec.get("parameters") or {}).get("properties") or {}))
    if selected_arguments is not None:
        allowed &= set(selected_arguments)
    if (not isinstance(issue["type"], str) or issue["type"] not in ISSUE_TYPES
            or not isinstance(fields, list)
            or not all(isinstance(x, str) for x in fields)
            or not set(fields) <= allowed or len(set(fields)) != len(fields)
            or (issue["type"] != "unsupported_tool" and not fields)
            or not isinstance(issue["reason"], str) or not issue["reason"].strip()):
        return None, "issue có type, fields hoặc reason không hợp lệ"
    return {"type": issue["type"], "fields": fields,
            "reason": issue["reason"].strip()[:240]}, None


class Agent(PromptAgent):
    """Planner, field selector, value selector; no Qwen reviewer calls."""

    value_system = FILL_SYSTEM
    arguments_system = SELECT_ARGUMENTS_SYSTEM
    include_reference_hints = False
    include_task_plan = True
    compact_planner_specs = True

    def run(self, row, transcript):
        env = self.ToolEnv("train", row["id"], root=self.root)
        specs = {spec["name"]: spec for spec in allowed_specs(row, self.catalog)}
        observations, trace, prompt_tokens_each = [], [], []
        generated_total = llm_latency = 0
        feedback = None
        goals = None
        repair_streak = parse_total = repairs_total = invalid_total = 0
        plan_revisions = 0
        issue_count = tool_failures = no_progress_total = 0
        grounding_rejections = semantic_rejections = 0
        unverified_total = 0
        no_progress_streak = 0
        last_failure = None
        known_facts = set()
        any_trim = False
        stop_reason = "decision_limit"
        start = time.perf_counter()

        def generate(phase="select", **kwargs):
            nonlocal generated_total, llm_latency, any_trim
            messages, _, trimmed = self.messages(
                row, transcript, observations, feedback, phase=phase,
                planner_goals=goals, **kwargs)
            any_trim |= trimmed
            output = self.llm.complete(messages, max_new_tokens=self.max_new_tokens)
            prompt_tokens_each.append(output["prompt_tokens"])
            generated_total += output["generation_tokens"]
            llm_latency += output["latency_s"]
            return output

        def reject(event, stage, error, *, issue=None, extra=None):
            nonlocal feedback, repair_streak, repairs_total, stop_reason
            nonlocal last_failure, issue_count
            nonlocal grounding_rejections, semantic_rejections
            if issue is not None:
                issue_count += 1
            if error == "ungrounded_value":
                grounding_rejections += 1
            if error in ("wrong_tool_for_intent", "semantic_value",
                         "unnecessary_optional_ask", "answer_available"):
                semantic_rejections += 1
            feedback = {"error": error, "stage": stage,
                        "tool": event.get("tool_selection", {}).get("name"),
                        **({"issue": issue} if issue else {}), **(extra or {})}
            if issue and issue.get("type") in ("missing_data", "ambiguous_target"):
                tool = feedback["tool"]
                if tool in specs:
                    lookup = available_lookup_for_ask(
                        tool, issue.get("fields", []), specs, observations)
                    if lookup:
                        feedback["suggested_lookup"] = lookup
            if error in ("read_already_observed", "duplicate_successful_write"):
                tool = feedback["tool"]
                args = (event.get("action") or {}).get("arguments")
                for index, item in enumerate(observations):
                    if item["name"] == tool and item["arguments"] == args:
                        feedback["observed_result_index"] = index
                        break
            event["error"] = feedback
            event["issue"] = issue
            trace.append(event)
            repairs_total += 1
            repair_streak += 1
            fingerprint = (stage, event.get("tool_selection", {}).get("name"),
                           dumps(issue or extra or error), len(known_facts))
            repeated = fingerprint == last_failure
            last_failure = fingerprint
            event["no_new_information"] = repeated
            if repeated or repair_streak >= self.max_repairs:
                stop_reason = "issue_no_progress" if issue else "validation_no_progress"
                return True
            return False

        for decision in range(1, self.max_decisions + 1):
            try:
                output = generate()
            except ValueError as exc:
                trace.append({"decision": decision, "error": str(exc)})
                stop_reason = "context_overflow"
                break
            event = {"decision": decision, "raw_output": output["text"],
                     "prompt_tokens": output["prompt_tokens"],
                     "generation_tokens": output["generation_tokens"],
                     "llm_latency_s": output["latency_s"]}
            candidate = parse_object(output["text"])
            choice = (candidate if isinstance(candidate, dict)
                      and candidate.get("type") == "revise"
                      else parse_action(output["text"]))
            event["action"] = choice
            if choice is None:
                parse_total += 1
                if reject(event, "planner", "invalid_json",
                          extra={"reason": "Planner cần JSON call, ask hoặc done"}):
                    stop_reason = "parse_error"
                    break
                continue
            if (choice.get("type") == "call" and feedback
                    and choice.get("name") == feedback.get("tool")
                    and not feedback.get("suggested_lookup")):
                blocked = feedback.get("issue") or {}
                fields = blocked.get("fields", [])
                if (blocked.get("type") == "ambiguous_target"
                        or (choice["name"] == "tv.control" and fields == ["volume"])):
                    event["recovery_override"] = {"requested": choice["name"], "ask": fields}
                    choice = {"type": "ask", "tool_name": choice["name"], "params": fields,
                              "tool_reason": blocked.get("reason", "Cần làm rõ đối tượng")}
                    event["action"] = choice
            if choice["type"] == "revise":
                if goals is None or plan_revisions >= 2:
                    if reject(event, "planner", "invalid_plan_revision",
                              extra={"reason": "Chỉ sửa plan đã có, tối đa hai lần."}):
                        break
                    continue
                new_goals, plan_error = revised_goals_error(choice, goals)
                if plan_error:
                    if reject(event, "planner", "invalid_plan_revision",
                              extra={"reason": plan_error}):
                        break
                    continue
                goals = new_goals
                plan_revisions += 1
                event["revised_plan"] = goals
                trace.append(event)
                feedback = None
                repair_streak, last_failure = 0, None
                continue
            if goals is None and choice["type"] != "done":
                proposed_goals, plan_error = planner_goals_error(choice)
                if plan_error:
                    if reject(event, "planner", "invalid_goal_plan",
                              extra={"reason": plan_error}):
                        break
                    continue
                goals = proposed_goals
            elif goals is None and choice["type"] == "done":
                if set(choice) != {"type", "goals"} or choice["goals"] != []:
                    if reject(event, "planner", "invalid_goal_plan",
                              extra={"reason": "done đầu tiên cần goals=[] khi không có việc phải làm."}):
                        break
                    continue
                goals = []
            elif goals is not None and choice["type"] in ("call", "ask") and "goals" in choice:
                new_goals, plan_error = added_goals_error(choice, goals)
                if plan_error:
                    if reject(event, "planner", "invalid_goal_plan",
                              extra={"reason": plan_error}):
                        break
                    continue
                if new_goals != goals:
                    if plan_revisions >= 2:
                        if reject(event, "planner", "invalid_plan_revision",
                                  extra={"reason": "Đã sửa plan hai lần."}):
                            break
                        continue
                    goals = new_goals
                    plan_revisions += 1
                    event["added_goals"] = goals
            if choice["type"] == "done":
                if goals:
                    done_error = done_evidence_error(choice, goals, observations)
                    if done_error:
                        if reject(event, "planner", "premature_done",
                                  extra={"reason": done_error, "evidence_options": [
                                      {"index": i, "name": item["name"], "kind": item.get("kind"),
                                       "succeeded": result_succeeded(item["result"])}
                                      for i, item in enumerate(observations)]}):
                            break
                        continue
                trace.append(event)
                stop_reason = "done"
                break
            if choice["type"] == "ask":
                if len(env.get_tool_calls()) >= self.max_tool_calls:
                    stop_reason = "tool_limit"
                    trace.append(event)
                    break
                name, params = choice.get("tool_name"), choice.get("params")
                if (not valid_ask(name, params, specs)
                        or not isinstance(choice.get("tool_reason"), str)
                        or not choice["tool_reason"].strip()):
                    if reject(event, "planner", "invalid_ask",
                              extra={"reason": "ask cần tool đích, field và lý do hợp lệ"}):
                        break
                    continue
                known = known_answer_for_ask(
                    name, params, transcript, row, observations, specs[name])
                if known:
                    if reject(event, "planner", "answer_available",
                              extra={"reason": "Một hoặc nhiều field đã xác định được; tiếp tục call.",
                                     "known_values": known}):
                        break
                    continue
                _, unnecessary = optional_fields_for_intent(name, params, transcript)
                if unnecessary:
                    if reject(event, "planner", "unnecessary_optional_ask",
                              extra={"reason": "Field optional không thuộc ý định: " + ", ".join(unnecessary)}):
                        break
                    continue
                lookup = available_lookup_for_ask(name, params, specs, observations)
                if lookup:
                    if reject(event, "planner", "lookup_before_ask",
                              extra={"reason": "Còn tool read chưa thử: " + lookup}):
                        break
                    continue
                event["tool_result"] = env.call("sys.ask_missing_info", {
                    "tool_name": name, "params": params})
                trace.append(event)
                stop_reason = "ask_missing_info"
                break

            name = choice.get("name")
            if feedback and name == feedback.get("tool") and feedback.get("suggested_lookup") in specs:
                lookup = feedback["suggested_lookup"]
                event["recovery_override"] = {"requested": name, "lookup": lookup}
                choice = {**choice, "name": lookup,
                          "tool_reason": "Tra dữ liệu còn thiếu cho " + str(name)}
                name = lookup
            if len(env.get_tool_calls()) >= self.max_tool_calls:
                stop_reason = "tool_limit"
                trace.append(event)
                break
            if (not isinstance(name, str) or name not in specs
                    or name == "sys.ask_missing_info"
                    or not isinstance(choice.get("tool_reason"), str)
                    or not choice["tool_reason"].strip()):
                if reject(event, "planner", "invalid_tool_selection",
                          extra={"reason": "Chọn tool khả dụng kèm tool_reason"}):
                    break
                continue
            spec = specs[name]
            selection = {"type": "call", "name": name,
                         "tool_reason": choice["tool_reason"]}
            event["tool_selection"] = selection
            action_error = tool_action_error(name, transcript)
            if action_error:
                if reject(event, "planner", "wrong_tool_for_intent",
                          extra={"reason": action_error}):
                    break
                continue
            params = spec.get("parameters") or {}
            selectable = set(params.get("properties") or {}) - set(params.get("required") or [])
            if selectable:
                try:
                    fields_output = generate("select_arguments", selected_tool=spec,
                                             tool_reason=choice["tool_reason"])
                except ValueError as exc:
                    event["error"] = str(exc)
                    trace.append(event)
                    stop_reason = "context_overflow"
                    break
                event["argument_selection_output"] = fields_output["text"]
                fields = parse_object(fields_output["text"])
                issue, issue_error = selector_issue(fields, spec)
                if (issue and issue["type"] == "unsupported_tool"
                        and spec.get("kind") == "read"
                        and not params.get("required") and not params.get("x-one-of")):
                    # A broad read is valid even if the model mistakes it for
                    # the eventual write. Prefer any unambiguous enum hint.
                    hinted = value_hints(name, transcript, row, observations, spec)
                    extra_fields = [key for key in hinted if key in selectable]
                    fields = {"selected_arguments": extra_fields}
                    event["ignored_read_issue"] = issue
                    issue = None
                if issue or issue_error:
                    if reject(event, "argument_selection", "selector_issue" if issue else "invalid_issue",
                              issue=issue, extra={"reason": issue_error} if issue_error else None):
                        break
                    continue
            else:
                fields = {"selected_arguments": []}
            selected_arguments, field_error = select_argument_fields(fields, spec)
            if field_error:
                if reject(event, "argument_selection", "invalid_argument_selection",
                          extra={"reason": field_error}):
                    break
                continue
            selected_arguments, dropped = optional_fields_for_intent(
                name, selected_arguments, transcript)
            if dropped:
                event["dropped_optional_fields"] = dropped
            event["selected_arguments"] = selected_arguments
            hints = value_hints(name, transcript, row, observations, spec)
            if selected_arguments:
                try:
                    value_output = generate("fill", selected_tool=spec,
                                            tool_reason=choice["tool_reason"],
                                            selected_arguments=selected_arguments,
                                            value_hints=hints)
                except ValueError as exc:
                    event["error"] = str(exc)
                    trace.append(event)
                    stop_reason = "context_overflow"
                    break
                event["argument_output"] = value_output["text"]
                values = parse_object(value_output["text"])
                issue, issue_error = selector_issue(values, spec, selected_arguments)
                if issue or issue_error:
                    safe_broad_read = (spec.get("kind") == "read"
                                       and not params.get("required")
                                       and not params.get("x-one-of"))
                    if (issue and issue["type"] == "missing_data"
                            and spec.get("kind") == "read"
                            and set(issue["fields"]) <= set(selected_arguments)
                            and set(selected_arguments) <= set(hints)):
                        values = {"arguments": {field: hints[field] for field in selected_arguments},
                                  "value_reasons": {field: "Giá trị đã xác định từ dữ liệu hiện có."
                                                    for field in selected_arguments}}
                        event["used_value_hints"] = list(selected_arguments)
                    elif (issue and issue["type"] == "missing_data"
                            and safe_broad_read
                            and set(issue["fields"]) <= set(selected_arguments)
                            and not any(field in hints for field in issue["fields"])):
                        # The optional filter is unavailable. Read the full
                        # collection and let the planner use its actual rows.
                        event["dropped_unavailable_read_filter"] = issue["fields"]
                        selected_arguments = []
                        event["selected_arguments"] = []
                        values = {"arguments": {}, "value_reasons": {}}
                    else:
                        known = {field: hints[field] for field in issue["fields"] if field in hints} if issue else {}
                        if known:
                            if reject(event, "value_selection", "answer_available",
                                      extra={"reason": "Giá trị đã có trong value_hints.",
                                             "known_values": known}):
                                break
                            continue
                        if reject(event, "value_selection", "selector_issue" if issue else "invalid_issue",
                                  issue=issue, extra={"reason": issue_error} if issue_error else None):
                            break
                        continue
            else:
                values = {"arguments": {}, "value_reasons": {}}
            value_error = value_selection_error(values, selected_arguments)
            if value_error:
                if reject(event, "value_selection", "invalid_argument_proposal",
                          extra={"reason": value_error}):
                    break
                continue
            selected_arguments, values, normalized = canonicalize_sms_destination(
                name, transcript, row, observations, selected_arguments, values)
            if normalized:
                event["normalization"] = normalized
                event["selected_arguments"] = selected_arguments
            action = {**selection, "selected_arguments": selected_arguments, **values}
            event["action"] = action
            args = action["arguments"]
            missing = missing_fields(spec, args, selected_arguments)
            if missing:
                issue = {"type": "missing_data", "fields": missing,
                         "reason": "Thiếu giá trị bắt buộc hoặc một nhóm x-one-of"}
                if reject(event, "value_selection", "selector_issue", issue=issue):
                    break
                continue
            field, reason = literal_grounding_error(
                name, args, spec, transcript, row, observations)
            if field:
                issue = {"type": "ambiguous_target" if "ambiguous_target" in reason else "missing_data",
                         "fields": [field], "reason": reason}
                if reject(event, "value_selection", "ungrounded_value", issue=issue):
                    break
                continue
            if name == "bill_payment.pay" and "source" in hints and "source" in args:
                if args["source"] != hints["source"]:
                    event["normalized_source"] = {"from": args["source"], "to": hints["source"]}
                    args["source"] = hints["source"]
                    action["value_reasons"]["source"] = "So sánh số dư với số tiền của hóa đơn đã xác định."
            field, reason = semantic_value_error(name, args, transcript, hints)
            if field:
                issue = {"type": "missing_data", "fields": [field], "reason": reason}
                if reject(event, "value_selection", "semantic_value", issue=issue):
                    break
                continue
            validation = env.validate(name, args)
            event["validation"] = validation
            if not validation["ok"]:
                invalid_total += 1
                if reject(event, "validation", "invalid_schema",
                          extra={"reason": validation["error"]}):
                    break
                continue
            if (spec.get("kind") == "write" and any(
                    repeated_write(prior, name, args) for prior in observations)):
                if reject(event, "validation", "duplicate_successful_write",
                          extra={"reason": "Cùng hành động trên cùng đối tượng đã thành công trong request."}):
                    break
                continue
            if (spec.get("kind") == "read" and any(
                    prior["name"] == name and prior["arguments"] == args
                    and result_succeeded(prior["result"])
                    and not any(read_invalidated(name, later)
                                for later in observations[index + 1:])
                    for index, prior in enumerate(observations))):
                if reject(event, "validation", "read_already_observed",
                          extra={"reason": "Kết quả read cùng arguments đã có và chưa có write thay đổi trạng thái."}):
                    break
                continue
            unverified_total += len(args)
            result = env.call(name, args)
            event["tool_result"] = result
            event["tool_succeeded"] = result_succeeded(result)
            observations.append({"name": name, "kind": spec.get("kind"),
                                 "arguments": args, "result": result})
            trace.append(event)
            repair_streak, last_failure = 0, None
            new_facts = result_facts(result) - known_facts
            known_facts.update(new_facts)
            if not result_succeeded(result):
                tool_failures += 1
                no_progress_streak += 1
                feedback = {"error": "tool_not_done", "tool": name,
                            "result": result}
            elif spec.get("kind") == "read" and not new_facts:
                no_progress_streak += 1
                no_progress_total += 1
                feedback = {"error": "read_no_new_information", "tool": name}
            else:
                no_progress_streak = 0
                feedback = None
            if no_progress_streak >= self.max_no_progress:
                stop_reason = "no_progress"
                break
        else:
            stop_reason = "decision_limit"

        return {
            "id": row["id"], "transcript": transcript,
            "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
            "stop_reason": stop_reason, "decisions": len(trace),
            "tool_iterations": len(env.get_tool_calls()),
            "repair_attempts": repairs_total, "invalid_schema_proposals": invalid_total,
            "grounding_rejections": grounding_rejections,
            "semantic_rejections": semantic_rejections,
            "unverified_arguments": unverified_total, "tool_failures": tool_failures,
            "no_progress_events": no_progress_total, "parse_errors": parse_total,
            "selector_issues": issue_count, "prompt_tokens_each": prompt_tokens_each,
            "plan_revisions": plan_revisions,
            "task_plan": goals or [],
            "grounding_prompt_tokens_each": [], "grounding_reviews": 0,
            "node_review_counts": {}, "node_rejections": {},
            "grounding_generation_tokens_total": 0, "grounding_llm_latency_s": 0.0,
            "prompt_tokens_total": sum(prompt_tokens_each),
            "prompt_tokens_max": max(prompt_tokens_each, default=0),
            "generation_tokens_total": generated_total,
            "llm_latency_s": llm_latency,
            "agent_latency_s": time.perf_counter() - start,
            "context_trimmed": any_trim, "trace": trace,
        }
