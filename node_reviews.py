"""Small, explicit Qwen reviews for each decision node in the workflow.

Each reviewer may reject with one short reason. The source reviewer additionally
returns a provenance object for every argument; code verifies those citations.
"""
from __future__ import annotations

import json

from common import dumps
from source_contract import SOURCE_FORMAT


MISSING_SYSTEM = """Bạn review node THIẾU THAM SỐ của tool.
Xét action được đề xuất và schema tool. Nếu là call, mọi required/x-one-of phải có giá trị; nếu thiếu, bác bỏ và nêu đúng trường thiếu. Nếu là ask, đọc tool_reason để kiểm vì sao cần hỏi các params; chỉ chấp nhận trường có trong properties và thật sự cần cho hành động cụ thể, kể cả field optional khi yêu cầu cần nó; phải không thể lấy từ read/search, request, context, history hay tool result. Đừng bắt read tool có arguments={} đi hỏi người dùng.
Chỉ trả JSON {} nếu hợp lệ; nếu bác bỏ, chỉ trả {"reject_reason":"lý do cụ thể, tối đa 20 từ"}. Không thêm trường khác.
"""

RELEVANCE_SYSTEM = """Bạn review bước CHỌN TOOL, trước khi arguments được tạo. Nhiệm vụ DUY NHẤT: kiểm tra `action.tool_reason` có giải thích công dụng hợp lý của `action.name` đối với request hay không. Dùng mô tả, returns và schema của tool đã chọn để kiểm tra lý do có khớp khả năng của tool.

Với read/search, lý do có thể nêu dữ liệu tool sẽ tra để phục vụ yêu cầu. Việc dữ liệu đó hiện chưa có chính là lý do tra cứu, không phải lý do bác. Với write, lý do phải gắn với hành động người dùng yêu cầu. Không xét ở đây liệu đã đủ ID, số dư, trạng thái, thời gian hay giá trị argument để gọi tool; các bước điền và kiểm argument xử lý điều đó.
Không chọn tool khác, không đổi thứ tự các bước, không lập kế hoạch, không suy đoán arguments hoặc kết quả tool. Chỉ bác khi lý do không liên quan request, gán cho tool khả năng/kiểu kết quả mà tool không có, hoặc mâu thuẫn rõ với mục tiêu người dùng. Nếu chưa đủ căn cứ để bác, chấp nhận và để các bước sau kiểm tra tiếp.
Chỉ trả JSON {} nếu lý do hợp lý; nếu bác, chỉ trả {"reject_reason":"nêu mâu thuẫn cụ thể giữa lý do, request và khả năng tool; tối đa 20 từ"}. Không thêm trường khác.
"""

SOURCE_SYSTEM = """Bạn review node NGUỒN ARGUMENT. Nhiệm vụ DUY NHẤT: với MỖI action.arguments[field], kiểm tra các trích dẫn trong action.sources[field] có thật và giá trị argument có được tạo ra trung thực từ các trích dẫn đó theo rule hay không. Tự đọc dữ liệu gốc; không chấp nhận rule chỉ vì người điền arguments đã viết ra.

1. Kiểm từng cặp origin[i], inputs[i]: input phải có inputs[i]={} và dùng toàn bộ request làm nguồn trực tiếp, không cần quote; path của context, history, previous_tool_calls và schema phải tồn tại; schema chỉ được dẫn tới default thật. history[i].text là lời người dùng chỉ khi history[i].role="user". Nếu nguồn không tồn tại hoặc sai định dạng, bác.
2. rule là bắt buộc cho mọi argument. Thực hiện phép sao chép, rút trích, chuyển đổi, tính toán hoặc kết hợp được mô tả trong rule trên chính các giá trị đã dẫn, rồi so với action.arguments[field]. Với origin=input, đọc toàn bộ request và kiểm phần được rút trích hoặc suy ra; không đòi argument bằng cả câu. Sao chép chính xác cũng phải có rule và phải giữ nguyên giá trị, kể cả kiểu dữ liệu. Bác nếu rule mơ hồ, chỉ nhắc lại kết quả, thêm thông tin không có trong nguồn, hoặc cho ra giá trị khác argument.
3. KHÔNG đánh giá trích dẫn có liên quan đến request hay mục tiêu người dùng không, có phải bản ghi cần chọn không, hoặc tool và thứ tự bước có hợp lý không. Nếu nguồn có thật và rule tạo ra đúng giá trị thì chấp nhận; review GIÁ TRỊ ARGUMENT xử lý việc chọn đúng đối tượng, bản ghi, điều kiện và mục tiêu.

Khi chấp nhận, chỉ trả {"sources":{"<field>":<source_object>,...}} với đúng một source object cho mỗi argument. Khi bác, chỉ trả {"reject_reason":"nêu nguồn không tồn tại hoặc phép tạo giá trị sai"}. Không trả thêm key.

MẪU GIẢ ĐỊNH (không phải dữ liệu thực): context.sample.code là "ZXQ-741", action.arguments.code là "ZXQ-741". Có thể trả:
{"sources":{"code":{"origin":["context"],"inputs":[{"path":"sample.code"}],"rule":"Sao chép nguyên chuỗi từ context.sample.code."}}}
Phân tích trước khi trả: kiểm path sample.code tồn tại và đọc ra "ZXQ-741"; rule sao chép nguyên chuỗi nên kết quả bằng action.arguments.code. Không dùng request để quyết định code này có phải mã cần chọn hay không.
""" + "\n" + SOURCE_FORMAT

ARGUMENT_SYSTEM = """Bạn review node GIÁ TRỊ ARGUMENT sau khi nguồn đã được kiểm tra. Với từng argument, đối chiếu `argument_reasons` (vì sao cần trường) và `value_reasons` (vì sao chọn giá trị) với request và kết quả tool: trường có cần cho hành động này không; giá trị có đúng đối tượng, đúng bản ghi và điều kiện không? Với trường optional, phải có lý do rõ vì sao cần truyền thay vì bỏ trường. Nguồn tồn tại chưa đủ để chứng minh chọn đúng. Read/search query gần đúng tên người hoặc chủ đề có thể hữu ích; đừng đòi read tool hoàn thành mục tiêu cuối. Chỉ xét arguments của tool đã chọn, không review lại việc chọn tool.
Chỉ trả JSON {} nếu hợp lệ; nếu bác bỏ, chỉ trả {"reject_reason":"lý do cụ thể, tối đa 20 từ"}. Không thêm trường khác.
"""

SCHEMA_SYSTEM = """Bạn review node SCHEMA ARGUMENT. Chỉ xét arguments theo schema của tool: required, x-one-of, type, enum, format và trường không được phép. Không xét ý nghĩa nghiệp vụ hoặc nguồn. Code sẽ gọi env.validate() sau review này.
Chỉ trả JSON {} nếu hợp lệ; nếu bác bỏ, chỉ trả {"reject_reason":"lý do cụ thể, tối đa 20 từ"}. Không thêm trường khác.
"""

SYSTEMS = {
    "missing_params": MISSING_SYSTEM,
    "relevance": RELEVANCE_SYSTEM,
    "argument_sources": SOURCE_SYSTEM,
    "arguments": ARGUMENT_SYSTEM,
    "schema": SCHEMA_SYSTEM,
}


def _first_json(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _short_reason(value):
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.split()[:20])[:180]


def review_node(llm, node, payload, max_prompt_tokens, max_new_tokens):
    """Run one Qwen node. Source approval uses the shared sources contract."""
    if node not in SYSTEMS:
        raise ValueError(f"Unknown review node: {node}")
    data = dict(payload)
    data["history"] = list(data.get("history") or [])
    data["tool_observations"] = list(data.get("tool_observations") or [])
    budget = min(max_new_tokens, 256 if node == "argument_sources" else 96)
    while True:
        messages = [
            {"role": "system", "content": SYSTEMS[node]},
            {"role": "user", "content": dumps(data)},
        ]
        if llm.count(messages) + budget <= max_prompt_tokens:
            break
        if data["history"]:
            data["history"].pop(0)
        elif data["tool_observations"]:
            data["tool_observations"].pop(0)
        else:
            return {"ok": False, "reason": "Review vượt giới hạn context.",
                    "review": None, "sources": None,
                    "usage": {"prompt_tokens": 0, "generation_tokens": 0,
                              "latency_s": 0.0}}
    output = llm.complete(messages, max_new_tokens=budget)
    usage = {key: output[key] for key in
             ("prompt_tokens", "generation_tokens", "latency_s")}
    verdict = _first_json(output["text"])
    sources = None
    if verdict is None:
        reason = "Review không trả JSON object hợp lệ."
    elif node == "argument_sources":
        extra = set(verdict) - {"sources", "reject_reason"}
        reason = _short_reason(verdict.get("reject_reason"))
        source_map = verdict.get("sources")
        arguments = (data.get("action") or {}).get("arguments") or {}
        if extra or ("reject_reason" in verdict and reason is None):
            reason = "Review nguồn trả sai cấu trúc."
        elif reason is None:
            if not isinstance(source_map, dict):
                reason = "Review nguồn cần sources là object."
            else:
                keys = set(source_map)
                if keys != set(arguments):
                    expected = ", ".join(sorted(arguments))
                    received = ", ".join(sorted(keys))
                    reason = f"Nguồn cần đúng key [{expected}]; đã nhận [{received}]."
                else:
                    sources = source_map
    else:
        reason = _short_reason(verdict.get("reject_reason"))
        if set(verdict) - {"reject_reason"} or ("reject_reason" in verdict and reason is None):
            reason = "Review trả sai cấu trúc."
    return {"ok": reason is None, "reason": reason, "review": verdict,
            "sources": sources, "usage": usage}
