"""LifeMate v2 agent: schema, evidence and tool-result checks in one ToolEnv."""
from __future__ import annotations

import json
import time
from collections import Counter

from evidence_rules import reference_candidates
from intent_contract import request_requirements, intent_fields
from common import allowed_specs, compact_history, dumps, folded_tokens, leaves, result_facts, result_succeeded
from grounding_qwen import check_sources, repair_source_paths
from node_reviews import review_node
from source_contract import SOURCE_FORMAT


SYSTEM = """#Context#
Bạn là PLANNER, chọn bước tiếp theo trong planner → argument selector → value selector → code → tool.
Input JSON:
- request: yêu cầu hiện tại; context: thời gian, múi giờ, hồ sơ hiện có.
- history: lời user và kết quả tool trước đây để giải tham chiếu. Tác vụ cũ không tự trở thành tác vụ mới.
- available_tools: các tool khả dụng, kind read/write và tên input. returns là output.
- tool_observations: các kết quả hiện tại, mỗi kết quả có index 0-based, kind và succeeded. index KHÔNG phải số decision.
- execution_status: bảng ngắn index/name/kind/succeeded để chọn bằng chứng done.
- task_plan: null lượt đầu, sau đó là các kết quả cuối cùng cần đạt. call/ask chỉ thêm goal mới; nếu vô tình viết lại ID cũ, code giữ goal cũ và ghi cảnh báo, vẫn kiểm call. Không dùng lookup làm goal.
- validation_feedback: lỗi bước trước; suggested_lookup là tool đọc đúng nguồn field còn thiếu; evidence_options là các index có thể dùng.

#Objective#
Trước khi chọn action, kiểm lần lượt: các kết quả user yêu cầu; phần đã xong; điều kiện bước tiếp; dữ kiện còn thiếu. Chỉ xuất JSON, không viết chuỗi suy luận dài.
Chọn MỘT bước và hoàn thành đủ yêu cầu. request_requirements là checklist từ code cho một số hành động rõ; active phải hoàn thành, inactive không thực hiện, unknown cần làm rõ. Checklist có thể chưa đủ, vẫn phải đọc toàn bộ request.
1. Đọc toàn bộ request, giữ câu sửa cuối và kênh giao tiếp được chỉ định. Phân biệt tra cứu, tạo mới, cập nhật bản ghi, ghi nhận việc đã diễn ra. Phủ định là ràng buộc, không phải một hành động phải gọi tool. Không kéo việc cũ từ history vào goal mới.
2. Lượt đầu lập đủ goal cho TỪNG kết quả độc lập. Mỗi hành động trong từng nhánh có goal riêng. when là điều kiện bằng lời từ user; không tự đặt ngưỡng, không ghi always cho hai nhánh loại trừ nhau. Tra ID/số dư/menu để phục vụ write là bước phụ, không phải goal. Tính toán/đọc thông tin là read, không phải write.
3. Xác định điều kiện bằng dữ kiện đúng đối tượng. Đủ tiền nghĩa là số dư >= khoản phải trả; nhiều khoản phải so tổng. Giờ đến, giờ đón và giờ nhắc là các mốc khác nhau; tính ngược đủ thời gian di chuyển và khoảng cách user yêu cầu. Với ngày mai/thứ trong tuần dùng current_time và timezone.
4. Khi thiếu dữ kiện, đọc đúng nguồn: nguồn field được ghi trong schema. Đọc hóa đơn để lấy số tiền; chuyển tiền cho người thụ hưởng dùng banking.transfer, trả hóa đơn trực tiếp dùng bill_payment.pay. ID contacts không phải chat_id. Chưa đọc menu thì chưa đặt món. Chưa có kết quả thanh toán thành công thì chưa gửi tin “đã đóng”.
5. Feedback thiếu ID: chọn suggested_lookup trước. Nếu tool vừa thất bại, sửa giá trị từ kết quả mới, không lặp giá trị cũ. Read đã có đủ dữ liệu: dùng dữ liệu đó. Write đã thành công: chuyển goal tiếp theo, không gọi lại để xác minh. Với “chưa làm thì làm”, kiểm kết quả thành công trong history trước; đã làm thì không lặp. Bằng chứng history được đánh dấu origin=history, không phải tool vừa chạy. Đúng ID nhưng nhiều bản ghi phù hợp: hỏi người dùng thay vì tự chọn. Có hai người cùng tên không cho phép tự ưu tiên bạn/đồng nghiệp; user chọn một mối không được gửi cho mọi địa chỉ.
6. ask khi field cần thiết chưa rõ và không còn lookup hữu ích; hỏi tool đích và field thật. Không hỏi thứ đã có trong request/context/history. Hướng thay đổi không xác định mức đích bắt buộc thì hỏi mức đích.
7. done sau khi rà từng mệnh đề request: mọi goal áp dụng có kết quả thành công ĐÚNG Ý NGHĨA. Map mỗi goal vào index của chính kết quả đó, không dùng index read cho write hay cùng index cho hai goal độc lập. Chỉ skip nhánh có điều kiện sai. Tool thành công với value sai chưa đáp ứng goal. Nếu plan thiếu, revise thêm; không thêm goal để hợp thức hóa một hành động làm sai.

#Style#
JSON ngắn, tên tool/key giữ nguyên; tool_reason tiếng Việt nêu mục tiêu bước đang gọi.
#Tone#
Chính xác, không đoán dữ kiện hoặc làm thêm việc.
#Audience#
Code kiểm action, lưu plan và đưa call sang hai selector. Không có reviewer.
#Response#
Chỉ MỘT JSON object, không markdown. Lượt đầu call/ask BẮT BUỘC có goals:
{"type":"call","name":"<tool>","tool_reason":"<mục tiêu bước>","goals":[{"id":"g1","task":"<kết quả cuối>","when":"always hoặc điều kiện từ user","kind":"read hoặc write"}]}
{"type":"ask","tool_name":"<tool đích>","params":["<field>"],"tool_reason":"<thiếu gì>","goals":[...]}
Khi task_plan đã có:
{"type":"call","name":"<tool>","tool_reason":"<mục tiêu bước>"}
{"type":"ask","tool_name":"<tool đích>","params":["<field>"],"tool_reason":"<thiếu gì>"}
{"type":"revise","goals":[<toàn bộ goal sau sửa>],"removed_goals":{"<id bỏ>":"<lý do>"},"reason":"<vì sao sửa>"}
{"type":"done","completed_goals":{"<goal_id>":<index tool_observations>},"skipped_goals":{"<goal_id>":"<điều kiện không áp dụng>"}}
call/ask có thể kèm goals chỉ chứa goal mới; không đổi/xóa goal cũ. Thêm goal mới không tính vào giới hạn sửa plan. Sửa/xóa goal bằng revise tối đa hai lần; removed_goals={} khi không bỏ goal. done có thể bỏ skipped_goals khi không skip. Mỗi goal có id duy nhất, task/when không rỗng, kind chỉ read/write. Nếu request không cần hành động nào, done lượt đầu: {"type":"done","goals":[]}.
"""

LEGACY_SELECT_ARGUMENTS_SYSTEM = """Bạn chọn các trường ĐẦU VÀO cho selected_tool. Chỉ xuất JSON {"argument_reasons":{"<field>":"<lý do>"}}. Chọn mọi required với lý do chính xác "required", đủ một nhóm x-one-of và optional theo ý định. Không lấy returns làm input, không điền value. Không có field thì {"argument_reasons":{}}."""

SELECT_ARGUMENTS_SYSTEM = """#Context#
Bạn là ARGUMENT SELECTOR cho MỘT bước do planner chọn.
Input JSON: request (yêu cầu hiện tại), context (thời gian/hồ sơ), history (lời cũ và tool results), tool_observations (kết quả hiện tại), selected_tool (name/kind/description/parameters/returns), tool_reason (mục tiêu bước).
CHỈ parameters.properties chứa tên input. required là field code tự thêm. x-one-of là mảng các nhóm thay thế. returns là output, không được chọn.
#Objective#
Chọn field BỔ SUNG cần cho chính bước selected_tool. Không điền value hoặc lý do từng field.
- Không chọn lại required. Với x-one-of, chọn đủ phần còn thiếu của MỘT nhóm: OR giữa nhóm, AND trong nhóm. Người nhận gọi bằng tên/quan hệ: ưu tiên contact_id để tra danh bạ; phone_number chỉ khi biết số đúng người.
- Optional chỉ chọn khi user yêu cầu thuộc tính của chính hành động. Không chép giá trị trạng thái đang có vào optional. Bật đèn cần power, chỉnh mờ cần brightness; không tự thêm brightness khi chỉ bật. Giờ lời nhắc không phải delay thiết bị. Số lượng user nói rõ phải được chọn, dù tool có default.
- Read/list có thể truyền {} và lọc từ kết quả; tránh filter làm mất bản ghi cần cho bước khác. Search query lấy tên/chủ đề user nêu, không đòi biết kết quả trước khi tìm. Nguồn tiền không phải loại hóa đơn.
- Đánh giá BƯỚC đang chọn, không đòi read/search phải thực hiện cả yêu cầu write. Thiếu value chuyển value selector; không gọi unsupported_tool vì chưa biết ID. intent_required_fields phải được chọn dù optional trong schema. validation_feedback.argument_selection_error yêu cầu xem lại lựa chọn field cho cùng tool.
#Style#
Tên key/field chính xác theo schema; JSON ngắn.
#Tone#
Chính xác, không đoán.
#Audience#
Code ghép required và kiểm x-one-of trước khi đưa sang value selector.
#Response#
Chỉ MỘT JSON object:
{"selected_arguments":["<field optional hoặc field nhóm x-one-of>"]}
Không có field bổ sung: {"selected_arguments":[]}. Không có field thì không có reason.
Chỉ khi schema không thể biểu diễn chính bước đang chọn: {"issue":{"type":"unsupported_tool","fields":[],"reason":"<vì sao bước không thể thực hiện>"}}.
Không thêm arguments, argument_reasons, value hoặc key khác. Mảng không trùng và chỉ chứa tên input thật.
"""

LEGACY_FILL_SYSTEM = """Bạn điền GIÁ TRỊ cho selected_arguments của selected_tool. Chỉ xuất JSON có arguments, sources, value_reasons cùng tập key. Value đúng schema và đúng bản ghi từ request/context/history/tool results; không đoán. Thiếu dữ kiện thì chỉ issue. Mỗi sources[field] gồm origin, inputs và rule, theo hợp đồng nguồn bên dưới.""" + "\n" + SOURCE_FORMAT

FILL_SYSTEM = """#Context#
Bạn là VALUE SELECTOR cho MỘT lời gọi selected_tool.
Input JSON:
- request: yêu cầu hiện tại, câu sửa cuối thắng. context: thời gian/múi giờ/hồ sơ có thật.
- history: lời user cũ và tool results để giải tham chiếu; lời assistant không tạo dữ kiện hoặc yêu cầu mới.
- tool_observations: name/arguments/result/index; chỉ result thành công là bằng chứng đã làm.
- selected_tool: schema input trong parameters.properties; required và x-one-of là ràng buộc. returns là output.
- selected_arguments: danh sách ĐÓNG các field được điền. intent_required_fields là thuộc tính cần có để đáp ứng ý định dù optional trong schema.
- reference_candidates: bản ghi đã thấy của đúng x-ref, nhóm theo field; có thể rỗng. Phải chọn đúng loại, người/vật và bản ghi, không chọn chỉ vì ID xuất hiện.
- value_hints: gợi ý từ dữ kiện hiển thị, phải kiểm lại quan hệ và tính duy nhất. Chỉ trusted_value_hints mới được code dùng làm ràng buộc; không coi mọi hint là đáp án.
- tool_reason: mục tiêu bước, không phải bằng chứng. validation_feedback: lỗi cũ để sửa, không phải nguồn value. executed_calls không chứng minh thành công.
#Objective#
Điền đúng value cho field đã chọn, hoặc báo issue.
1. Đọc type/enum/format/x-ref của từng field. Enum và ví dụ schema không phải default hoặc dữ liệu thật. Không dựng ID/email/số điện thoại từ tên người hay quan hệ.
2. Tra dữ liệu của đúng app/collection và đúng bản ghi. Không dùng contact_id làm chat_id, ID đèn cho điều hòa, ID món từ quán khác. items[].item_id phải từ menu đúng restaurant_id. Nhiều bản ghi cùng thỏa mà user chưa chọn: ambiguous_target.
3. Thông tin có ngay trong request được rút trích trực tiếp; query không đòi có kết quả tìm kiếm. Không báo thiếu tên/chủ đề/địa phương đã được nêu. Với dữ kiện còn thiếu cần thiết, chỉ issue; không xuất value đoán rồi giải thích rằng không biết.
4. Lấy đúng số và thời điểm của bản ghi được tham chiếu. Tính cộng/trừ từ dữ kiện, không chép giờ deadline làm giờ sự kiện trước deadline. Giờ đến khác giờ xuất phát. Ngày mai và thứ trong tuần tính từ context, không lấy ngày hiện tại. So số dư với đúng số tiền hóa đơn để chọn nguồn.
5. content/body giữ đủ ý người dùng muốn truyền, đúng kênh, giờ đã sửa và số liệu cuối. Không thêm nhà cung cấp/ngày/chi tiết không được yêu cầu. “Chép nguyên” phải bằng chính content bản ghi đúng, không phải bản ghi mới nhất của mọi đối tượng. Kết quả calculator là giá trị đưa vào tin nhắn; không tự đổi số lần nữa. Không gửi “đã thanh toán” trước khi có kết quả thanh toán thành công.
6. Required và nhóm x-one-of đã chọn phải có value. Optional không có căn cứ thì bỏ nếu không làm mất ý định; không điền rỗng/default tự đặt. Nếu optional cần để đáp ứng hành động nhưng chưa rõ thì issue. Hai object arguments/`value_reasons` phải cùng key; reason phải mô tả CHÍNH value đã xuất.
#Style#
JSON ngắn. value_reasons/issue.reason tiếng Việt, mỗi reason một câu ngắn; không diễn giải dài.
#Tone#
Chính xác, không tự giả định.
#Audience#
Code kiểm contract, ID, schema và một số ngữ nghĩa; planner nhận issue để lookup hoặc hỏi.
#Response#
Chỉ MỘT trong ba JSON object, không markdown:
Điền được: {"arguments":{"<field>":<giá trị JSON đúng type>},"value_reasons":{"<field>":"<căn cứ chọn đúng value>"}}
Không cần field nào: {"arguments":{},"value_reasons":{}}. Chỉ khi không có required/x-one-of và {} vẫn đáp ứng bước.
Thiếu dữ liệu: {"issue":{"type":"missing_data","fields":["<field>"],"reason":"<cần tra dữ kiện gì>"}}
Nếu thiếu tên field cần thiết trong selected_arguments, chỉ trả {"issue":{"type":"argument_selection_error","fields":["<tên field thật trong schema>"],"reason":"<vì sao cần field>"}} để sửa ở argument selector, không tự thêm field vào arguments.
Nhiều ứng viên: dùng cùng dạng issue với type="ambiguous_target". Với missing_data/ambiguous_target, fields không rỗng, không trùng, thuộc selected_arguments; riêng argument_selection_error chỉ dùng field thật trong schema chưa được chọn. Không ghép issue với arguments/value_reasons. Nếu không xác định được value cần thiết thì chỉ issue.
"""

def parse_action(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") in ("call", "ask", "done"):
            return obj
    return None


def parse_object(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def argument_selection_error(candidate, spec):
    """Validate the field list and one reason per selected field."""
    if not isinstance(candidate, dict) or set(candidate) != {"argument_reasons"}:
        return "Bước chọn argument cần đúng object argument_reasons."
    reasons = candidate["argument_reasons"]
    if not isinstance(reasons, dict):
        return "argument_reasons phải là object JSON."
    properties = ((spec.get("parameters") or {}).get("properties") or {})
    unknown = set(reasons) - set(properties)
    if unknown:
        return "Argument không có trong schema: " + ", ".join(sorted(unknown))
    invalid = [key for key, value in reasons.items()
               if not isinstance(value, str) or not value.strip()]
    if invalid:
        return "argument_reasons cần lý do cho: " + ", ".join(sorted(invalid))
    params = spec.get("parameters") or {}
    missing = set(params.get("required") or []) - set(reasons)
    if missing:
        return "Chưa chọn argument bắt buộc: " + ", ".join(sorted(missing))
    groups = params.get("x-one-of") or []
    if groups and not any(set(group) <= set(reasons) for group in groups):
        return "Chưa chọn đủ một nhóm x-one-of."
    required = set(params.get("required") or [])
    wrong_required = [key for key in required if reasons[key] != "required"]
    if wrong_required:
        return "Lý do của argument bắt buộc phải là 'required': " + ", ".join(sorted(wrong_required))
    wrong_nonrequired = [key for key in set(reasons) - required
                         if reasons[key].strip() == "required"]
    if wrong_nonrequired:
        return "Field không bắt buộc cần lý do cụ thể: " + ", ".join(sorted(wrong_nonrequired))
    return None


def argument_proposal_error(candidate, selected_arguments=None):
    """Explain malformed fill output."""
    if not isinstance(candidate, dict):
        return "Output điền arguments không phải JSON object."
    args = candidate.get("arguments")
    if not isinstance(args, dict):
        return "arguments phải là object JSON."
    expected = set(args)
    parts = []
    if selected_arguments is not None:
        unselected = expected - set(selected_arguments)
        if unselected:
            parts.append("arguments có field chưa chọn: " + ", ".join(sorted(unselected)))
    sources = candidate.get("sources")
    if not isinstance(sources, dict):
        parts.append("sources phải là object JSON")
    else:
        if expected - set(sources):
            parts.append("Thiếu nguồn cho: " + ", ".join(sorted(expected - set(sources))))
        if set(sources) - expected:
            parts.append("Nguồn thừa: " + ", ".join(sorted(set(sources) - expected)))
    reasons = candidate.get("value_reasons")
    if not isinstance(reasons, dict):
        parts.append("value_reasons phải là object JSON")
    else:
        missing, extra = expected - set(reasons), set(reasons) - expected
        if missing:
            parts.append("value_reasons thiếu key: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("value_reasons thừa key: " + ", ".join(sorted(extra)))
        bad = [key for key in expected & set(reasons)
               if not isinstance(reasons[key], str) or not reasons[key].strip()]
        if bad:
            parts.append("value_reasons cần chuỗi lý do: " + ", ".join(sorted(bad)))
    return "; ".join(parts) if parts else None


def missing_fields(spec, args, selected=None):
    args = args if isinstance(args, dict) else {}
    params = spec.get("parameters") or {}
    empty = (None, "", [], {})
    missing = [key for key in params.get("required", []) if args.get(key) in empty]
    groups = params.get("x-one-of") or []
    if groups and not any(all(args.get(key) not in empty for key in group) for group in groups):
        selected_groups = [group for group in groups
                           if selected is not None and set(group) <= set(selected)]
        group = selected_groups[0] if selected_groups else groups[0]
        missing.extend(key for key in group if args.get(key) in empty)
    return sorted(set(missing))


def valid_ask(name, params, specs):
    if name not in specs or name == "sys.ask_missing_info":
        return False
    if not isinstance(params, list) or not params or not all(isinstance(p, str) for p in params):
        return False
    schema = specs[name].get("parameters") or {}
    return set(params) <= set((schema.get("properties") or {}))


def argument_free_call(spec, args):
    """True when an empty call has no required or one-of arguments."""
    params = spec.get("parameters") or {}
    return not args and not params.get("required") and not params.get("x-one-of")


def reference_hints(spec, transcript, history, observations, limit=8):
    """Expose exact IDs and citation paths already visible to the agent."""
    properties = (spec.get("parameters") or {}).get("properties") or {}
    refs = {key: prop["x-ref"].split(".", 1)[0]
            for key, prop in properties.items() if prop.get("x-ref")}
    if not refs:
        return []
    words = folded_tokens(" ".join([transcript] +
                               [turn.get("text", "") for turn in history]))
    hints = []

    def collect(field, result, source_for):
        for path, value, record in leaves(result):
            if path.rsplit(".", 1)[-1] != "id" or not isinstance(value, str):
                continue
            details = {key: record[key] for key in
                       ("name", "title", "relation", "type", "phone", "location", "time")
                       if key in record and isinstance(record[key], (str, int, float))}
            score = len(words & folded_tokens(" ".join(map(str, details.values()))))
            citation = source_for(path)
            origin = citation.pop("origin")
            hints.append((field, score, {"argument": field, "value": value,
                                         "source": {"origin": [origin], "inputs": [citation],
                                                    "rule": "Sao chép nguyên ID từ trích dẫn này."},
                                         "record": details}))

    for field, app in refs.items():
        for turn_index, turn in enumerate(history):
            calls = turn.get("tool_calls") or []
            results = turn.get("tool_results") or []
            for call_index, (call, result) in enumerate(zip(calls, results)):
                if call.get("name", "").startswith(app + "."):
                    collect(field, result, lambda path, i=turn_index, j=call_index:
                            {"origin": "history", "path": f"[{i}].tool_results[{j}].{path}"})
        for index, item in enumerate(observations):
            if item["name"].startswith(app + "."):
                collect(field, item["result"], lambda path, i=index:
                        {"origin": "previous_tool_calls", "index": i,
                         "part": "result", "path": path})
    selected = []
    for field in refs:
        selected.extend(item for _, _, item in sorted(
            (entry for entry in hints if entry[0] == field),
            key=lambda entry: -entry[1])[:limit])
    return selected


class Agent:
    def __init__(self, llm, catalog, ToolEnv, root, max_tool_calls=10,
                 max_decisions=24, max_repairs=3, max_prompt_tokens=14500,
                 max_new_tokens=512, max_no_progress=2):
        self.llm = llm
        self.catalog = catalog
        self.ToolEnv = ToolEnv
        self.root = root
        self.max_tool_calls = max_tool_calls
        self.max_decisions = max_decisions
        self.max_repairs = max_repairs
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.max_no_progress = max_no_progress

    def messages(self, row, transcript, observations, feedback,
                 phase="select", selected_tool=None, tool_reason=None,
                 argument_reasons=None, selected_arguments=None,
                 planner_goals=None, value_hints=None):
        context = {key: value for key, value in row["context"].items() if key != "installed_apps"}
        specs = allowed_specs(row, self.catalog)
        if phase == "select" and getattr(self, "compact_planner_specs", False):
            specs = [{
                "name": spec["name"], "kind": spec["kind"],
                "description": spec["description"],
                "parameters": {
                    "properties": list((spec["parameters"] or {}).get("properties") or {}),
                    "required": (spec["parameters"] or {}).get("required") or [],
                    "x-one-of": (spec["parameters"] or {}).get("x-one-of") or [],
                },
                "returns": spec["returns"],
            } for spec in specs]
        history = compact_history(row.get("history") or [])
        observed = []
        if getattr(self, "include_task_plan", False) and len(observations) > 8:
            for index, item in enumerate(observations[:-8]):
                observed.append({
                    "index": index, "name": item["name"],
                    "arguments": item["arguments"],
                    "result": {"succeeded": result_succeeded(item["result"])},
                })
        for index, item in enumerate(observations[-8:], max(0, len(observations) - 8)):
            serialized = dumps(item["result"])
            observed.append({
                "index": index, "kind": item.get("kind"), "origin": item.get("origin", "current"),
                "succeeded": result_succeeded(item["result"]),
                "effect_verified": item.get("effect_verified"),
                "name": item["name"], "arguments": item["arguments"],
                "result": (item["result"] if len(serialized) <= 12000
                           else serialized[:12000] + " [truncated]"),
            })
        trimmed = False
        while True:
            payload = {
                "request": transcript, "context": context,
                "history": history, "available_tools": specs,
                "executed_calls": [
                    {"name": item["name"], "arguments": item["arguments"]}
                    for item in observations if item.get("origin") != "history"
                ],
                "tool_observations": observed,
                "validation_feedback": feedback,
                "remaining_tool_calls": self.max_tool_calls - sum(item.get("origin") != "history" for item in observations),
            }
            if phase == "select":
                payload = {key: payload[key] for key in
                           ("request", "context", "history", "available_tools",
                            "executed_calls", "tool_observations",
                            "validation_feedback")}
                if getattr(self, "include_task_plan", False):
                    payload["task_plan"] = planner_goals
                    payload["request_requirements"] = request_requirements(transcript,
                        {item["name"]: item for item in specs}, observations, row)
                    payload["execution_status"] = [
                        {"index": i, "name": item["name"], "kind": item.get("kind"),
                         "succeeded": result_succeeded(item["result"]),
                         "effect_verified": item.get("effect_verified")}
                        for i, item in enumerate(observations)]
            elif phase == "select_arguments":
                payload = {key: payload[key] for key in
                           ("request", "context", "history", "tool_observations")}
                payload["selected_tool"] = selected_tool
                payload["tool_reason"] = tool_reason
                if getattr(self, "include_task_plan", False):
                    payload["validation_feedback"] = feedback
                    payload["selected_arguments"] = selected_arguments
                    payload["intent_required_fields"] = intent_fields(selected_tool["name"], transcript, selected_tool)
            elif phase == "fill":
                payload.pop("available_tools")
                payload["selected_tool"] = selected_tool
                payload["tool_reason"] = tool_reason
                if getattr(self, "include_task_plan", False):
                    payload["reference_candidates"] = reference_candidates(selected_tool, row, observations)
                    payload["intent_required_fields"] = intent_fields(selected_tool["name"], transcript, selected_tool)
                if value_hints:
                    payload["value_hints"] = value_hints
                    if getattr(self, "include_task_plan", False):
                        from pipeline_rules import trusted_value_hints
                        payload["trusted_value_hints"] = trusted_value_hints(selected_tool["name"], value_hints)
                if argument_reasons is not None:
                    payload["argument_reasons"] = argument_reasons
                payload["selected_arguments"] = (
                    list(selected_arguments) if selected_arguments is not None
                    else list(argument_reasons or {}))
                if getattr(self, "include_reference_hints", True):
                    payload["reference_hints"] = reference_hints(
                        selected_tool, transcript, history, observations)
            messages = [
                {"role": "system", "content": (
                    getattr(self, "value_system", LEGACY_FILL_SYSTEM) if phase == "fill" else
                    getattr(self, "arguments_system", LEGACY_SELECT_ARGUMENTS_SYSTEM)
                    if phase == "select_arguments" else SYSTEM)},
                {"role": "user", "content": dumps(payload)},
            ]
            tokens = self.llm.count(messages)
            if tokens <= self.max_prompt_tokens:
                return messages, tokens, trimmed
            trimmed = True
            if history:
                history.pop(0)
            elif len(observed) > 1:
                observed.pop(0)
            elif observed and len(dumps(observed[-1]["result"])) > 1500:
                observed[-1]["result"] = dumps(observed[-1]["result"])[:1500] + " [truncated]"
            else:
                raise ValueError(
                    f"Prompt needs {tokens} tokens after safe trimming; tool schemas cannot be dropped"
                )

    def run(self, row, transcript):
        env = self.ToolEnv("train", row["id"], root=self.root)
        specs = {spec["name"]: spec for spec in allowed_specs(row, self.catalog)}
        observations = []
        trace = []
        feedback = None
        repair_streak = 0
        parse_streak = 0
        no_progress_streak = 0
        last_rejection = None
        last_missing_fact_count = None
        known_facts = set()
        prompt_tokens_each = []
        generated_total = 0
        llm_latency = 0.0
        grounding_prompt_tokens_each = []
        grounding_generated_total = 0
        grounding_latency = 0.0
        repairs_total = 0
        invalid_total = 0
        grounding_total = 0
        semantic_total = 0
        unverified_total = 0
        tool_failures = 0
        no_progress_total = 0
        parse_total = 0
        any_trim = False
        stop_reason = "decision_limit"
        start = time.perf_counter()
        context = {key: value for key, value in row["context"].items()
                   if key != "installed_apps"}
        history = compact_history(row.get("history") or [])

        def node_payload(action, spec, **extra):
            return {
                "request": transcript, "context": context, "history": history,
                "tool_observations": [
                    {"index": i, **item} for i, item in enumerate(observations)
                ],
                "tool": spec, "action": action, **extra,
            }

        def run_node(node, payload, event, label=None):
            nonlocal generated_total, grounding_generated_total
            nonlocal llm_latency, grounding_latency
            verdict = review_node(
                self.llm, node, payload, self.max_prompt_tokens,
                self.max_new_tokens,
            )
            verdict["node"] = node
            event.setdefault("node_reviews", {})[label or node] = verdict
            usage = verdict.get("usage") or {}
            if usage.get("prompt_tokens"):
                grounding_prompt_tokens_each.append(usage["prompt_tokens"])
                generated_total += usage["generation_tokens"]
                grounding_generated_total += usage["generation_tokens"]
                llm_latency += usage["latency_s"]
                grounding_latency += usage["latency_s"]
            return verdict

        def reject_call(name, args, stage, code, reason, event, grounding):
            nonlocal grounding_total, semantic_total, repairs_total
            nonlocal repair_streak, last_rejection, feedback, stop_reason
            grounding["ok"] = False
            grounding["stage"] = stage
            grounding["issues"].append({
                "code": code, "field": None, "message": reason,
            })
            grounding_total += 1
            if stage in ("relevance", "arguments"):
                semantic_total += 1
            repairs_total += 1
            repair_streak += 1
            rejection = (name, dumps(args),
                         dumps((event.get("action") or {}).get("sources")),
                         stage, code, len(known_facts))
            repeated = rejection == last_rejection
            last_rejection = rejection
            event["no_new_information"] = repeated
            feedback = {
                "error": "review_rejected" if code.startswith("qwen_")
                         else "invalid_argument_source",
                "tool": name, "submitted_arguments": args,
                "stage": stage, "reason": reason,
                "issues": grounding["issues"],
                "instruction": "Đọc lý do, sửa action hoặc nguồn; read/search có thể là bước tra cứu trước write.",
            }
            event["error"] = feedback
            trace.append(event)
            if repeated or repair_streak >= self.max_repairs:
                stop_reason = "grounding_no_progress"
                return True
            return False

        for decision_number in range(1, self.max_decisions + 1):
            if len(env.get_tool_calls()) >= self.max_tool_calls:
                stop_reason = "tool_limit"
                break
            try:
                messages, _, trimmed = self.messages(
                    row, transcript, observations, feedback
                )
            except ValueError as error:
                trace.append({
                    "decision": decision_number, "event": "context_overflow",
                    "error": str(error),
                })
                stop_reason = "context_overflow"
                break
            any_trim |= trimmed
            output = self.llm.complete(messages, max_new_tokens=self.max_new_tokens)
            prompt_tokens_each.append(output["prompt_tokens"])
            generated_total += output["generation_tokens"]
            llm_latency += output["latency_s"]
            event = {
                "decision": decision_number,
                "prompt_tokens": output["prompt_tokens"],
                "generation_tokens": output["generation_tokens"],
                "llm_latency_s": output["latency_s"],
                "context_trimmed": trimmed,
                "raw_output": output["text"],
            }
            action = parse_action(output["text"])
            event["action"] = action
            if action is None:
                parse_total += 1
                repairs_total += 1
                parse_streak += 1
                feedback = "Không parse được JSON. Chỉ xuất một object call, ask hoặc done."
                event["error"] = feedback
                trace.append(event)
                if parse_streak >= self.max_repairs:
                    stop_reason = "parse_error"
                    break
                continue
            parse_streak = 0

            if action["type"] == "done":
                trace.append(event)
                stop_reason = "done"
                break

            if action["type"] == "ask":
                name, params = action.get("tool_name"), action.get("params")
                ask_reason = action.get("tool_reason")
                if not isinstance(ask_reason, str) or not ask_reason.strip():
                    feedback = {
                        "error": "missing_tool_reason", "tool": name,
                        "instruction": "Ask cần tool_reason nêu vì sao phải hỏi và không thể tra được giá trị.",
                    }
                    event["error"] = feedback
                    trace.append(event)
                    repairs_total += 1
                    repair_streak += 1
                    if repair_streak >= self.max_repairs:
                        stop_reason = "missing_tool_reason"
                        break
                    continue
                if not valid_ask(name, params, specs):
                    target = specs.get(name)
                    if (target and target.get("kind") == "read"
                            and argument_free_call(target, {})):
                        # An ask for a zero-argument read tool is a schema
                        # misunderstanding. Execute its read step after the
                        # same relevance review as any other call.
                        event["auto_corrected_ask"] = {"tool_name": name,
                                                         "invalid_params": params}
                        action = {"type": "call", "name": name,
                                  "arguments": {}, "sources": {}}
                        event["action"] = action
                    else:
                        schema = (target or {}).get("parameters") or {}
                        feedback = {
                            "error": "invalid_ask", "tool": name,
                            "invalid_params": params,
                            "required": schema.get("required", []),
                            "x-one-of": schema.get("x-one-of", []),
                            "instruction": "Chỉ ask required/x-one-of của tool đích. Nếu read tool gọi được với {}, hãy call ngay.",
                        }
                        event["error"] = feedback
                        trace.append(event)
                        repairs_total += 1
                        repair_streak += 1
                        if repair_streak >= self.max_repairs:
                            stop_reason = "invalid_ask"
                            break
                        continue
                else:
                    ask_review = run_node(
                        "missing_params",
                        node_payload(
                            action, specs[name],
                            available_read_tools=[
                                {"name": tool_name,
                                 "description": tool_spec.get("description", "")}
                                for tool_name, tool_spec in specs.items()
                                if tool_spec.get("kind") == "read"
                            ],
                        ), event,
                    )
                    if not ask_review["ok"]:
                        feedback = {
                            "error": "premature_ask", "tool": name,
                            "params": params, "reason": ask_review["reason"],
                            "instruction": "Tra cứu nếu còn cách lấy dữ liệu; chỉ ask khi thực sự không thể tra.",
                        }
                        event["error"] = feedback
                        trace.append(event)
                        repairs_total += 1
                        repair_streak += 1
                        if repair_streak >= self.max_repairs:
                            stop_reason = "ask_no_progress"
                            break
                        continue
                    result = env.call("sys.ask_missing_info", {
                        "tool_name": name, "params": params,
                    })
                    event["tool_result"] = result
                    trace.append(event)
                    stop_reason = "ask_missing_info"
                    break

            # Tool selection is reviewed before argument generation. The
            # reviewer sees the purpose of this step, not candidate values.
            name = action.get("name")
            if (not isinstance(name, str) or name not in specs
                    or name == "sys.ask_missing_info"):
                feedback = {"error": "unavailable_tool",
                            "instruction": "Chọn một tool trong available_tools."}
                event["error"] = feedback
                trace.append(event)
                repairs_total += 1
                repair_streak += 1
                if repair_streak >= self.max_repairs:
                    stop_reason = "unavailable_tool"
                    break
                continue
            spec = specs[name]
            tool_reason = action.get("tool_reason")
            if event.get("auto_corrected_ask"):
                tool_reason = "Tra cứu thông tin còn thiếu để thực hiện yêu cầu."
            if not isinstance(tool_reason, str) or not tool_reason.strip():
                feedback = {"error": "missing_tool_reason",
                            "instruction": "Chọn tool kèm tool_reason giải thích ích lợi của bước này."}
                event["error"] = feedback
                trace.append(event)
                repairs_total += 1
                repair_streak += 1
                if repair_streak >= self.max_repairs:
                    stop_reason = "missing_tool_reason"
                    break
                continue
            selection = {"type": "call", "name": name,
                         "tool_reason": tool_reason}
            event["tool_selection"] = selection
            relevance = run_node(
                "relevance",
                {
                    "request": transcript,
                    "tool": {key: spec.get(key) for key in
                             ("name", "kind", "description", "returns", "parameters")},
                    "action": selection,
                },
                event,
            )
            if not relevance["ok"]:
                grounding = {"ok": True, "stage": "relevance", "issues": []}
                event["grounding"] = grounding
                if reject_call(name, {}, "relevance", "qwen_rejected_call",
                               relevance["reason"], event, grounding):
                    break
                continue

            if not event.get("auto_corrected_ask"):
                try:
                    field_messages, _, field_trimmed = self.messages(
                        row, transcript, observations, feedback,
                        phase="select_arguments", selected_tool=spec,
                        tool_reason=tool_reason,
                    )
                except ValueError as error:
                    event["error"] = str(error)
                    trace.append(event)
                    stop_reason = "context_overflow"
                    break
                any_trim |= field_trimmed
                fields_output = self.llm.complete(
                    field_messages, max_new_tokens=self.max_new_tokens)
                prompt_tokens_each.append(fields_output["prompt_tokens"])
                generated_total += fields_output["generation_tokens"]
                llm_latency += fields_output["latency_s"]
                event["argument_selection_output"] = fields_output["text"]
                event["argument_selection_prompt_tokens"] = fields_output["prompt_tokens"]
                event["argument_selection_generation_tokens"] = fields_output["generation_tokens"]
                event["argument_selection_latency_s"] = fields_output["latency_s"]
                field_candidate = parse_object(fields_output["text"])
                selection_error = argument_selection_error(field_candidate, spec)
                if selection_error:
                    feedback = {
                        "error": "invalid_argument_selection", "tool": name,
                        "reason": selection_error,
                        "instruction": "Chọn đúng field theo schema và nêu lý do cho từng field; chưa điền value.",
                    }
                    event["error"] = feedback
                    trace.append(event)
                    repairs_total += 1
                    repair_streak += 1
                    if repair_streak >= self.max_repairs:
                        stop_reason = "invalid_argument_selection"
                        break
                    continue
                argument_reasons = field_candidate["argument_reasons"]
                event["argument_reasons"] = argument_reasons
                try:
                    fill_messages, _, fill_trimmed = self.messages(
                        row, transcript, observations, feedback,
                        phase="fill", selected_tool=spec,
                        tool_reason=tool_reason,
                        argument_reasons=argument_reasons,
                    )
                except ValueError as error:
                    event["error"] = str(error)
                    trace.append(event)
                    stop_reason = "context_overflow"
                    break
                any_trim |= fill_trimmed
                filled = self.llm.complete(fill_messages,
                                           max_new_tokens=self.max_new_tokens)
                prompt_tokens_each.append(filled["prompt_tokens"])
                generated_total += filled["generation_tokens"]
                llm_latency += filled["latency_s"]
                event["argument_output"] = filled["text"]
                event["argument_prompt_tokens"] = filled["prompt_tokens"]
                event["argument_generation_tokens"] = filled["generation_tokens"]
                event["argument_latency_s"] = filled["latency_s"]
                candidate = parse_object(filled["text"])
                if (candidate is None and filled["generation_tokens"] >= self.max_new_tokens
                        and self.max_new_tokens < 512):
                    remaining = getattr(self.llm, "max_model_len", 16000) - filled["prompt_tokens"]
                    retry_tokens = min(512, remaining)
                    if retry_tokens > self.max_new_tokens:
                        retried = self.llm.complete(fill_messages,
                                                    max_new_tokens=retry_tokens)
                        prompt_tokens_each.append(retried["prompt_tokens"])
                        generated_total += retried["generation_tokens"]
                        llm_latency += retried["latency_s"]
                        event["argument_retry"] = {
                            "max_new_tokens": retry_tokens,
                            "initial_output": filled["text"],
                        }
                        filled = retried
                        event["argument_output"] = filled["text"]
                        event["argument_prompt_tokens"] += filled["prompt_tokens"]
                        event["argument_generation_tokens"] += filled["generation_tokens"]
                        event["argument_latency_s"] += filled["latency_s"]
                        candidate = parse_object(filled["text"])
                args = candidate.get("arguments") if candidate else None
                sources = candidate.get("sources") if candidate else None
                reasons = candidate.get("value_reasons") if candidate else None
                proposal_error = argument_proposal_error(
                    candidate, selected_arguments=argument_reasons)
                if proposal_error:
                    feedback = {
                        "error": "invalid_argument_proposal",
                        "tool": name, "reason": proposal_error,
                        "instruction": "Sửa đúng key trong arguments, sources và value_reasons; giữ tool đã chọn nếu phù hợp.",
                    }
                    event["error"] = feedback
                    trace.append(event)
                    repairs_total += 1
                    repair_streak += 1
                    if repair_streak >= self.max_repairs:
                        stop_reason = "invalid_argument_proposal"
                        break
                    continue
                action = {**selection, "argument_reasons": argument_reasons,
                          "arguments": args, "sources": sources,
                          "value_reasons": reasons}
                fixed_sources, fixes = repair_source_paths(
                    action, spec, transcript, context, history, observations)
                if fixes:
                    action["sources"] = fixed_sources
                    event["source_path_repairs"] = fixes
                event["action"] = action
            else:
                action = {**selection, "argument_reasons": {},
                          "arguments": {}, "sources": {},
                          "value_reasons": {}}
                event["action"] = action

            name, args = action.get("name"), action.get("arguments")
            if not isinstance(name, str) or not isinstance(args, dict):
                feedback = {"error": "invalid_call_json",
                            "instruction": "call cần name:string và arguments:object."}
                event["error"] = feedback
                trace.append(event)
                repairs_total += 1
                repair_streak += 1
                if repair_streak >= self.max_repairs:
                    stop_reason = "invalid_call_json"
                    break
                continue
            if name not in specs or name == "sys.ask_missing_info":
                feedback = {
                    "error": "unavailable_tool",
                    "instruction": f"{name!r} không được phép gọi trực tiếp. Chọn available_tools hoặc action ask.",
                }
                event["error"] = feedback
                trace.append(event)
                repairs_total += 1
                repair_streak += 1
                if repair_streak >= self.max_repairs:
                    stop_reason = "unavailable_tool"
                    break
                continue

            properties = (spec.get("parameters") or {}).get("properties") or {}
            unknown = sorted(set(args) - set(properties))
            if unknown:
                feedback = {"error": "unknown_argument_fields", "tool": name,
                            "fields": unknown,
                            "instruction": "Bỏ các argument không có trong selected_tool.parameters.properties."}
                event["error"] = feedback
                trace.append(event)
                repairs_total += 1
                repair_streak += 1
                if repair_streak >= self.max_repairs:
                    stop_reason = "unknown_argument_fields"
                    break
                continue

            no_arguments = argument_free_call(spec, args)
            missing = [] if no_arguments else missing_fields(spec, args)
            if missing:
                event["missing_fields"] = missing
                event["no_new_information"] = (
                    last_missing_fact_count == len(known_facts)
                )
                last_missing_fact_count = len(known_facts)
                missing_review = run_node(
                    "missing_params",
                    node_payload(
                        action, spec, missing_fields=missing,
                        available_read_tools=[
                            {"name": tool_name,
                             "description": tool_spec.get("description", "")}
                            for tool_name, tool_spec in specs.items()
                            if tool_spec.get("kind") == "read"
                        ],
                    ), event,
                )
                reason = missing_review["reason"] or (
                    "Thiếu tham số bắt buộc: " + ", ".join(missing)
                )
                feedback = {
                    "error": "missing_required_arguments", "tool": name,
                    "missing_fields": missing, "reason": reason,
                    "no_new_information": event["no_new_information"],
                    "instruction": "Tra read/search nếu còn cách; chỉ ask khi không thể lấy tham số bắt buộc.",
                }
                event["error"] = feedback
                repairs_total += 1
                repair_streak += 1
                if repair_streak >= self.max_repairs:
                    ask_action = {"type": "ask", "tool_name": name,
                                  "params": missing,
                                  "tool_reason": (
                                      "Cần hỏi người dùng giá trị "
                                      + ", ".join(missing)
                                      + " để gọi tool; chưa tìm được từ request, context, history hoặc tra cứu."
                                  )}
                    ask_review = run_node(
                        "missing_params",
                        node_payload(
                            ask_action, spec,
                            available_read_tools=[
                                {"name": tool_name,
                                 "description": tool_spec.get("description", "")}
                                for tool_name, tool_spec in specs.items()
                                if tool_spec.get("kind") == "read"
                            ],
                        ), event, label="missing_params_auto_ask",
                    )
                    if valid_ask(name, missing, specs) and ask_review["ok"]:
                        event["auto_ask"] = ask_action
                        event["tool_result"] = env.call(
                            "sys.ask_missing_info",
                            {"tool_name": name, "params": missing},
                        )
                        trace.append(event)
                        stop_reason = "ask_missing_info"
                        break
                    event["ask_review_reject_reason"] = ask_review["reason"]
                    trace.append(event)
                    stop_reason = "missing_no_progress"
                    break
                trace.append(event)
                continue

            grounding = {
                "ok": True, "stage": "complete" if no_arguments else "schema",
                "issues": [],
                "evidence": {}, "unverified_fields": [],
                "review": relevance["review"], "reviewed_sources": None,
                "usage": relevance["usage"],
            }
            event["grounding"] = grounding
            if no_arguments:
                grounding["source_check"] = "skipped_no_arguments"
                event["validation"] = {"ok": True, "error": None,
                                       "skipped": "no_arguments"}
            else:
                schema_review = run_node(
                    "schema",
                    node_payload(action, {"name": name,
                                          "parameters": spec.get("parameters") or {}}),
                    event,
                )
                if not schema_review["ok"]:
                    invalid_total += 1
                    if reject_call(name, args, "schema", "qwen_rejected_schema",
                                   schema_review["reason"], event, grounding):
                        break
                    continue
                validation = env.validate(name, args)
                event["validation"] = validation
                if not validation["ok"]:
                    invalid_total += 1
                    repairs_total += 1
                    repair_streak += 1
                    feedback = {
                        "error": "invalid_schema", "tool": name,
                        "submitted_arguments": args,
                        "schema_error": validation["error"],
                        "missing_fields": missing_fields(spec, args),
                        "instruction": "Sửa arguments theo schema; không đoán giá trị.",
                    }
                    event["error"] = feedback
                    trace.append(event)
                    if repair_streak >= self.max_repairs:
                        stop_reason = "validation_no_progress"
                        break
                    continue

                grounding["stage"] = "sources"
                planner_source = check_sources(
                    action, spec, transcript, context, history, observations,
                )
                grounding["evidence"]["planner"] = planner_source["evidence"]
                grounding["issues"].extend(planner_source["issues"])
                if not planner_source["ok"]:
                    if reject_call(name, args, "sources", "invalid_planner_source",
                                   planner_source["issues"][0]["message"],
                                   event, grounding):
                        break
                    continue

                source_review = run_node(
                    "argument_sources",
                    node_payload(action, {key: spec.get(key) for key in
                                          ("name", "kind", "description", "parameters")}),
                    event,
                )
                grounding["reviewed_sources"] = source_review["sources"]
                if not source_review["ok"]:
                    if reject_call(name, args, "sources", "qwen_rejected_source",
                                   source_review["reason"], event, grounding):
                        break
                    continue
                reviewed_action = {**action, "sources": source_review["sources"]}
                reviewed_source = check_sources(
                    reviewed_action, spec, transcript, context, history,
                    observations,
                )
                grounding["evidence"]["reviewer"] = reviewed_source["evidence"]
                grounding["unverified_fields"] = sorted(set(
                    planner_source["unverified_fields"]
                    + reviewed_source["unverified_fields"]
                ))
                grounding["issues"].extend(reviewed_source["issues"])
                if not reviewed_source["ok"]:
                    if reject_call(name, args, "sources", "invalid_reviewed_source",
                                   reviewed_source["issues"][0]["message"],
                                   event, grounding):
                        break
                    continue

            if not no_arguments:
                grounding["stage"] = "arguments"
                argument_review = run_node(
                    "arguments",
                    node_payload(
                        action, {key: spec.get(key) for key in
                                 ("name", "kind", "description", "returns")},
                        verified_sources=grounding["evidence"],
                    ), event,
                )
                if not argument_review["ok"]:
                    if reject_call(name, args, "arguments", "qwen_rejected_argument",
                                   argument_review["reason"], event, grounding):
                        break
                    continue
            grounding["stage"] = "complete"


            unverified_total += len(grounding["unverified_fields"])
            result = env.call(name, args)
            event["tool_result"] = result
            succeeded = result_succeeded(result)
            event["tool_succeeded"] = succeeded
            new_facts = result_facts(result) - known_facts
            known_facts.update(new_facts)
            event["new_fact_count"] = len(new_facts)
            observations.append({
                "name": name, "kind": spec.get("kind"),
                "arguments": args, "result": result,
            })
            trace.append(event)
            repair_streak = 0
            last_rejection = None
            last_missing_fact_count = None
            if not succeeded:
                tool_failures += 1
                no_progress_streak += 1
                feedback = {
                    "error": "tool_not_done",
                    "tool": name, "arguments": args, "result": result,
                    "instruction": "Tool chưa hoàn thành. Dùng lý do thất bại để lập bước khác; không lặp call.",
                }
            elif not new_facts and specs[name].get("kind") == "read":
                no_progress_streak += 1
                no_progress_total += 1
                feedback = {
                    "error": "read_no_new_information",
                    "tool": name, "result": result,
                    "instruction": "Read tool không cung cấp dữ liệu mới. Chọn cách tra khác, ask nếu thiếu tham số bắt buộc không thể tra, hoặc done.",
                }
            else:
                no_progress_streak = 0
                feedback = None
            if no_progress_streak >= self.max_no_progress:
                stop_reason = "no_progress"
                break
        else:
            stop_reason = "decision_limit"

        calls = env.get_tool_calls()
        review_counts = Counter()
        rejection_counts = Counter()
        for event in trace:
            for review in (event.get("node_reviews") or {}).values():
                node = review.get("node")
                if node:
                    review_counts[node] += 1
                    if not review.get("ok"):
                        rejection_counts[node] += 1
        return {
            "id": row["id"], "transcript": transcript,
            "tool_calls": calls, "state_hash": env.get_state_hash(),
            "stop_reason": stop_reason, "decisions": len(trace),
            "tool_iterations": len(calls), "repair_attempts": repairs_total,
            "invalid_schema_proposals": invalid_total,
            "grounding_rejections": grounding_total,
            "semantic_rejections": semantic_total,
            "unverified_arguments": unverified_total,
            "tool_failures": tool_failures,
            "no_progress_events": no_progress_total,
            "parse_errors": parse_total,
            "prompt_tokens_each": prompt_tokens_each,
            "grounding_prompt_tokens_each": grounding_prompt_tokens_each,
            "grounding_reviews": len(grounding_prompt_tokens_each),
            "node_review_counts": dict(review_counts),
            "node_rejections": dict(rejection_counts),
            "grounding_generation_tokens_total": grounding_generated_total,
            "grounding_llm_latency_s": grounding_latency,
            "prompt_tokens_total": sum(prompt_tokens_each) + sum(grounding_prompt_tokens_each),
            "prompt_tokens_max": max(prompt_tokens_each + grounding_prompt_tokens_each, default=0),
            "generation_tokens_total": generated_total,
            "llm_latency_s": llm_latency,
            "agent_latency_s": time.perf_counter() - start,
            "context_trimmed": any_trim,
            "trace": trace,
        }
