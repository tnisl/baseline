"""Citation contract shared by argument filling and source review."""

SOURCE_ORIGINS = frozenset({
    "input", "context", "history", "previous_tool_calls", "schema",
})

SOURCE_FORMAT = """ĐỊNH DẠNG ARGUMENT VÀ SOURCE:
arguments là object field -> giá trị truyền cho tool. sources là object có đúng cùng các field. Mỗi sources[field] là MỘT object thống nhất:
{"origin":["<loại nguồn 1>",...],"inputs":[<trích dẫn 1>,...],"rule":"<cách lấy giá trị argument từ các trích dẫn>"}
Mỗi source BẮT BUỘC có đúng ba key origin, inputs, rule; rule là chuỗi không rỗng kể cả khi sao chép chính xác. origin luôn là MẢNG không rỗng. Mỗi origin[i] ứng với inputs[i]; hai mảng phải cùng độ dài. Mỗi phần tử origin chỉ được là input, context, history, previous_tool_calls hoặc schema. Nếu có nhiều trích dẫn, chúng phải khác nhau. Không dùng origin="multi_source" hay origin là chuỗi.

Format của inputs[i] theo origin[i], không đặt thêm origin trong inputs[i]:
- input -> {}; toàn bộ request là giá trị nguồn, không cần và không được tự tạo quote.
- context -> {"path":"<đường dẫn từ context>"}
- history -> {"path":"[<turn_index>].text"} cho lời nói trong một lượt user, hoặc {"path":"[<turn_index>].tool_results[<call_index>].<field>"} cho kết quả tool trong history.
- previous_tool_calls -> {"index":<số nguyên từ 0>,"part":"result","path":"<đường dẫn trong result>"}; part cũng có thể là "arguments".
- schema -> {"path":"parameters.properties.<field>.default"}; chỉ khi default được khai báo.

Với input, reviewer đọc trực tiếp toàn bộ request để kiểm giá trị argument theo rule; Qwen không phải sao chép một đoạn request vào source. Mọi path khác phải tồn tại; schema chỉ được dẫn tới default thật, enum không phải default. Giá trị argument không cần bằng toàn bộ request hoặc giá trị tại path: nó phải được nguồn hỗ trợ theo rule. Với sao chép chính xác, rule nói rõ sao chép nguyên giá trị từ đâu. Với rút trích, chuyển đổi, tính toán hoặc kết hợp nguồn, rule nói rõ phần nào được lấy và cách tạo ra giá trị. Không tự đặt giá trị hoặc trích dẫn. Reviewer kiểm tra nguồn có thật và rule có tạo ra đúng giá trị argument; việc nguồn có phù hợp với yêu cầu hay đúng bản ghi cần chọn thuộc review giá trị argument.
Path dùng dấu chấm cho key và [i] cho phần tử list. history bắt đầu từ list history; previous_tool_calls.index là thứ tự trong tool_observations hiện tại. Path của previous_tool_calls bắt đầu từ object của part, không thêm result. hoặc arguments. ở đầu.
MẪU GIẢ ĐỊNH, KHÔNG PHẢI DỮ LIỆU THỰC:
- Khớp chính xác: nếu context.sample.code là "ZXQ-741" và arguments.code là "ZXQ-741", source là {"origin":["context"],"inputs":[{"path":"sample.code"}],"rule":"Sao chép nguyên chuỗi từ context.sample.code."}
- Rút trích: nếu request là "Đặt nhãn cho hộp: Mưa trên Sao Mộc" và arguments.label là "Mưa trên Sao Mộc", source là {"origin":["input"],"inputs":[{}],"rule":"Lấy phần văn bản sau dấu hai chấm và bỏ khoảng trắng đầu trong request."}
"""
