"""Small deterministic checks for facts that schema validation cannot express."""
from __future__ import annotations

import re

from common import dumps, folded_tokens, result_succeeded


def optional_fields_for_intent(tool_name, fields, request):
    """Remove optional controls whose effect belongs to a different user task."""
    text = request.casefold()
    dropped = []
    kept = []
    for field in fields:
        unsupported = (
            tool_name == "home_security.list_alerts" and field == "since"
            and not re.search(r"\b(từ|sau|kể từ|trong vòng|trong \d+)\b", text)
        ) or (
            tool_name == "bill_payment.list_bills" and field == "type"
            and not re.search(r"truyền hình|tivi|\btv\b|internet|\bmạng\b|\bnước\b|\bđiện\b|\bgas\b|điện thoại|cáp", text)
        ) or (
            tool_name == "appliances.start_cycle" and field == "delay_minutes"
            and not re.search(r"hẹn.*(?:chạy|bắt đầu)|(?:chạy|bắt đầu|khởi động|bật)\s+(?:sau|lúc)|bật.*\blúc\b", text)
        ) or (
            tool_name == "medication.add" and field == "days"
            and not re.search(r"hàng ngày|hằng ngày|mỗi ngày|mỗi sáng|mỗi tối|ngày thường|các ngày trong tuần|mỗi tuần|thứ (?:[2-7]|hai|ba|tư|năm|sáu|bảy|chủ nhật)", text)
        )
        (dropped if unsupported else kept).append(field)
    return kept, dropped


def tool_action_error(tool_name, request):
    """Reject a few clearly different, state-changing actions before selectors run."""
    text = request.casefold()
    if (tool_name == "medication.add" and "uống" in text
            and ("đã" in text or "rồi" in text)
            and any(word in text for word in ("ghi lại", "ghi nhận", "đánh dấu"))
            and not any(word in text for word in ("thêm thuốc", "tạo lịch", "thêm vào lịch"))):
        return "Yêu cầu ghi nhận liều đã uống; tìm lịch thuốc rồi dùng medication.log_taken."
    if (tool_name == "home_security.acknowledge_alert" and "ghi chú" in text
            and not any(word in text for word in ("xác nhận", "đánh dấu", "đã xem", "đã xử lý", "ack"))):
        return "Người dùng không yêu cầu xác nhận hoặc đánh dấu cảnh báo."
    if (tool_name == "ewallet.pay_qr" and "qr" not in text
            and re.search(r"hóa đơn|hoá đơn|tiền (?:nước|mạng|truyền hình|điện|internet)", text)):
        return "Thanh toán hóa đơn qua bill_payment.pay; ewallet.pay_qr dùng cho mã QR merchant."
    if (tool_name == "notes.append" and re.search(r"(?:tạo|thêm|lập|một)\s+(?:\w+\s+)?ghi chú", text)
            and not re.search(r"ghi chú (?:đã có|cũ|trước đó)", text)):
        return "Yêu cầu tạo ghi chú mới; dùng notes.create thay vì append vào ghi chú cũ."
    return None


def _history_results(row):
    for turn in row.get("history") or []:
        for result in turn.get("tool_results") or []:
            if result_succeeded(result):
                yield result


def _all_results(row, observations):
    yield from _history_results(row)
    for item in observations:
        if result_succeeded(item["result"]):
            yield item["result"]


def _referenced_contact(request, row, observations):
    contacts = [contact for result in _all_results(row, observations)
                for contact in result.get("contacts", [])
                if contact.get("id") and contact.get("name")]
    words = folded_tokens(request)
    matches = [contact for contact in contacts
               if folded_tokens(contact["name"]) <= words]
    ids = {contact["id"] for contact in matches}
    if len(ids) == 1:
        return matches[-1]
    if not matches:
        return None
    last_user = next((turn.get("audio_text", "") for turn in reversed(row.get("history") or [])
                      if turn.get("role") == "user"), "")
    history_words = folded_tokens(last_user)
    resolved = [contact for contact in matches
                if folded_tokens(contact.get("relation", ""))
                and folded_tokens(contact["name"]) <= history_words
                and folded_tokens(contact["relation"]) <= history_words]
    return resolved[-1] if len({contact["id"] for contact in resolved}) == 1 else None


def _requested_bill_type(request):
    text = request.casefold()
    candidates = {
        "tv": ("truyền hình", "tiền tv", "hóa đơn tv", "hoá đơn tv"),
        "internet": ("tiền mạng", "hóa đơn mạng", "hoá đơn mạng", "internet"),
        "water": ("tiền nước", "hóa đơn nước", "hoá đơn nước"),
        "electricity": ("tiền điện", "hóa đơn điện", "hoá đơn điện"),
    }
    matches = [kind for kind, phrases in candidates.items()
               if any(phrase in text for phrase in phrases)]
    return matches[0] if len(matches) == 1 else None


def value_hints(tool_name, request, row, observations, spec=None):
    """Only facts with an unambiguous deterministic derivation are hinted."""
    hints = {}
    text = request.casefold()
    if tool_name == "bill_payment.pay":
        wanted_type = _requested_bill_type(request)
        bills = [bill for item in observations if item["name"] == "bill_payment.list_bills"
                 and result_succeeded(item["result"])
                 for bill in item["result"].get("bills", []) if not bill.get("paid")]
        if wanted_type and any(bill.get("type") for bill in bills):
            bills = [bill for bill in bills if bill.get("type") == wanted_type]
        if wanted_type and len({bill.get("id") for bill in bills}) == 1:
            hints["bill_id"] = bills[-1]["id"]
        if "ví" in text and "ngân hàng" in text:
            balances = [item["result"].get("balance") for item in observations
                        if item["name"] == "ewallet.get_balance" and result_succeeded(item["result"])]
            if balances and len(bills) == 1 and isinstance(balances[-1], (int, float)):
                hints["source"] = "ewallet" if balances[-1] >= bills[0]["amount"] else "bank"
        elif "ngân hàng" in text:
            hints["source"] = "bank"
        elif "ví" in text:
            hints["source"] = "ewallet"
    if tool_name == "gold_price.get_price":
        if "sjc" in text:
            hints["type"] = "sjc"
        elif "vàng nhẫn" in text:
            hints["type"] = "ring"
    if tool_name == "lottery.check_ticket":
        province = re.search(
            r"(?i:vé(?: số)?|đài)\s+([A-ZÀ-ỸĐ][\w-]*(?:\s+[A-ZÀ-ỸĐ][\w-]*){0,2})",
            request)
        if province:
            hints["province"] = province.group(1)
        user_turns = [turn.get("audio_text", "") for turn in row.get("history") or []
                      if turn.get("role") == "user"]
        if any(word in (request + " " + (user_turns[-1] if user_turns else "")).casefold()
               for word in ("hôm nay", "bữa nay")):
            current_time = row.get("context", {}).get("current_time", "")
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", current_time[:10]):
                hints["date"] = current_time[:10]
        digit_words = r"(?:không|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín)"
        ticket_numbers = []
        for utterance in [request, *user_turns[-2:]]:
            if "vé" not in utterance.casefold():
                continue
            ticket_numbers.extend(re.findall(r"(?<!\d)\d{5,6}(?!\d)", utterance))
            spoken = re.search(rf"\b{digit_words}(?:\s+{digit_words}){{5}}\b",
                               utterance.casefold())
            if spoken:
                ticket_numbers.append("".join(_DIGIT_WORDS[word]
                                              for word in spoken.group().split()))
        if len(set(ticket_numbers)) == 1:
            hints["number"] = ticket_numbers[-1]
    if tool_name == "podcast_radio.search":
        topic = re.search(r"(?:nói|bàn|chia sẻ)\s+(?:về\s+)?(.+?)(?:\s+(?:không|hông|ko)\b|[?.!,]|$)",
                          text)
        if topic and topic.group(1).strip():
            hints["query"] = topic.group(1).strip()
    if tool_name == "sms.send" and not re.search(r"\b0\d{8,10}\b", request):
        contact = _referenced_contact(request, row, observations)
        if contact:
            hints["contact_id"] = contact["id"]
            if contact.get("phone"):
                hints["phone_number"] = contact["phone"]
    if tool_name == "medication.add" and "đèn" in text:
        matches = []
        request_words = folded_tokens(request)
        for result in _all_results(row, observations):
            for rule in [*result.get("rules", []), *([result["rule"]] if isinstance(result.get("rule"), dict) else [])]:
                title_words = folded_tokens(rule.get("title", ""))
                if (rule.get("action") == "on" and rule.get("time")
                        and len(request_words & title_words) >= 3):
                    matches.append(rule["time"])
        if matches:
            hints["times"] = [matches[-1]]
    properties = ((spec or {}).get("parameters") or {}).get("properties") or {}
    for field, property_spec in properties.items():
        options = property_spec.get("enum") or []
        matched = [value for value in options
                   if isinstance(value, str) and len(value) >= 3
                   and re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", text)]
        if len(matched) == 1 and field not in hints:
            hints[field] = matched[0]
    return hints


def known_answer_for_ask(tool_name, fields, request, row, observations, spec=None):
    hints = value_hints(tool_name, request, row, observations, spec)
    return {field: hints[field] for field in fields if field in hints}


_DIGIT_WORDS = {
    "không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4",
    "tư": "4", "năm": "5", "lăm": "5", "sáu": "6", "bảy": "7",
    "tám": "8", "chín": "9",
}
_PHONE_TOKEN = re.compile(r"\b(?:không|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín|\d+)\b", re.IGNORECASE)


def spoken_phone_in_text(phone, text):
    """Recognize phone digits spoken one by one in a user utterance."""
    if not isinstance(phone, str) or not phone.isdigit() or len(phone) < 9:
        return False
    previous = None
    run = ""
    for token in _PHONE_TOKEN.finditer(text):
        if previous is not None and re.fullmatch(r"[\s,.;:-]*", text[previous:token.start()]):
            run += _DIGIT_WORDS.get(token.group().casefold(), token.group())
        else:
            run = _DIGIT_WORDS.get(token.group().casefold(), token.group())
        if phone in run:
            return True
        previous = token.end()
    return False


def literal_grounding_error(tool_name, arguments, spec, request, row, observations):
    """Check that identifiers and phone numbers actually occur in visible evidence."""
    properties = (spec.get("parameters") or {}).get("properties") or {}
    evidence = [request, {key: value for key, value in row.get("context", {}).items()
                          if key != "installed_apps"}]
    for turn in row.get("history") or []:
        if turn.get("role") == "user":
            evidence.append(turn.get("audio_text", ""))
        evidence.extend(result for result in turn.get("tool_results") or []
                        if result_succeeded(result))
    evidence.extend(item["result"] for item in observations if result_succeeded(item["result"]))
    corpus = dumps(evidence)
    user_texts = [request, *(turn.get("audio_text", "") for turn in row.get("history") or []
                             if turn.get("role") == "user")]
    for field, value in arguments.items():
        if field in ("to", "cc", "bcc") and isinstance(value, list):
            for address in value:
                if (isinstance(address, str) and "@" in address
                        and not re.search(r"(?<![\w@.])" + re.escape(address)
                                          + r"(?![\w@.])", corpus, re.IGNORECASE)):
                    return field, f"{field} chứa email chưa thấy trong dữ liệu: {address}."
        if (properties.get(field, {}).get("x-ref") or field == "phone_number") and isinstance(value, str):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
            if not re.search(pattern, corpus) and not (
                    field == "phone_number" and any(spoken_phone_in_text(value, text)
                                                     for text in user_texts)):
                return field, f"{field}={value} không có trong request, history hoặc tool result."
    return None, None


def canonicalize_sms_destination(tool_name, request, row, observations, selected, values):
    """Prefer an observed contact ID when a named recipient's phone matches it."""
    if (tool_name != "sms.send" or "phone_number" not in values["arguments"]
            or "contact_id" in values["arguments"]):
        return selected, values, None
    phone = values["arguments"]["phone_number"]
    if not isinstance(phone, str) or phone in request:
        return selected, values, None
    contacts = [contact for result in _all_results(row, observations)
                for contact in result.get("contacts", [])
                if contact.get("phone") == phone and contact.get("id")]
    ids = {contact["id"] for contact in contacts}
    if len(ids) != 1:
        return selected, values, None
    contact = contacts[-1]
    if not folded_tokens(contact.get("name", "")) <= folded_tokens(request):
        return selected, values, None
    args = dict(values["arguments"])
    reasons = dict(values["value_reasons"])
    del args["phone_number"]
    del reasons["phone_number"]
    args["contact_id"] = contact["id"]
    reasons["contact_id"] = f"Số đã chọn khớp liên hệ {contact.get('name', '')} trong kết quả contacts."
    selected = ["contact_id" if field == "phone_number" else field for field in selected]
    return selected, {"arguments": args, "value_reasons": reasons}, "phone_number -> contact_id"


def semantic_value_error(tool_name, arguments, request, hints):
    if tool_name == "tv.control" and arguments.get("action") == "volume":
        target = re.search(r"(?:âm lượng|volume|mức|lên|xuống|về)\s*(?:mức\s*)?(\d+)\b", request.casefold())
        if not target:
            return "volume", "Người dùng chưa chỉ định mức âm lượng đích."
        if arguments.get("volume") != int(target.group(1)):
            return "volume", "Mức âm lượng không khớp giá trị người dùng chỉ định."
    for field, expected in hints.items():
        if field in arguments and arguments[field] != expected:
            return field, f"{field} phải là {expected} theo dữ kiện đã quan sát."
    return None, None


def repeated_write(prior, name, arguments):
    """Recognize the same user action even if an incidental argument changed."""
    if prior["name"] != name or not result_succeeded(prior["result"]):
        return False
    old = prior["arguments"]
    if old == arguments:
        return True
    identity = {
        "appliances.start_cycle": ("appliance_id",),
        "medication.add": ("name",),
        "bill_payment.pay": ("bill_id",),
    }.get(name)
    if name == "medication.log_taken":
        return (old.get("schedule_id") == arguments.get("schedule_id")
                and (old.get("time") == arguments.get("time")
                     or not old.get("time") or not arguments.get("time")))
    return bool(identity and all(old.get(key) == arguments.get(key) and old.get(key) is not None
                                 for key in identity))
