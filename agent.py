"""LifeMate v2 agent: schema, evidence and tool-result checks in one ToolEnv."""
from __future__ import annotations

import json
import time
from collections import Counter

from common import allowed_specs, compact_history, dumps, folded_tokens, leaves, result_facts, result_succeeded
from grounding_qwen import check_sources, repair_source_paths
from node_reviews import review_node
from source_contract import SOURCE_FORMAT


SYSTEM = """#Context#
Bạn là PLANNER trong workflow planner → argument selector → value selector → tool. Input JSON:
- `request`: yêu cầu hiện tại. `context`: dữ liệu hiện có, gồm thời gian/múi giờ nếu có.
- `history`: lượt cũ; dùng lời user và kết quả tool để giải tham chiếu. Lời assistant không tạo yêu cầu mới.
- `available_tools`: tool khả dụng. `parameters.properties` chỉ là TÊN INPUT; `returns` là OUTPUT.
- `tool_observations`: `{index,name,arguments,result}` của request hiện tại; chỉ result thành công mới chứng minh việc đã làm. `executed_calls` thiếu result nên không chứng minh thành công.
- `validation_feedback`: lỗi hoặc issue của bước trước, không phải dữ kiện nghiệp vụ. `suggested_lookup` (nếu có) là tool read còn khả dụng để tìm field thiếu; `observed_result_index` trỏ tới kết quả đã có khi gọi lặp.
- `task_plan`: null ở lượt đầu, sau đó là danh sách goal kết quả người dùng cần.

#Objective#
Chọn MỘT bước tiếp theo và theo dõi đủ mục tiêu người dùng.
1. Lượt đầu: chia request thành các KẾT QUẢ CUỐI CÙNG độc lập. Mỗi hành động nối bằng “và/rồi” là một goal; mỗi nhánh điều kiện có goal riêng. Goal mô tả việc người dùng muốn đạt, KHÔNG phải tool phụ trợ. `kind="write"` khi cần tạo/gửi/thanh toán/điều khiển/ghi log; `kind="read"` chỉ khi đọc thông tin là kết quả người dùng thực sự muốn. Không đặt goal cho việc tra số dư, tìm ID, đọc danh sách chỉ để phục vụ write. Không đặt goal cho hành động không được yêu cầu, như xác nhận cảnh báo, tạo lịch mới, chuyển khoản riêng thay cho thanh toán hóa đơn.
   Giữ các hành động trong từng nhánh thành goal riêng: thanh toán và nhắn tin là hai kết quả; thanh toán và nhắc nạp ví cũng là hai kết quả. `when` phải là điều kiện thật của từng nhánh, không ghi `always` cho hai nhánh loại trừ nhau. Trước khi xuất lượt đầu, rà lại từng mệnh đề hành động của request và history mà request đang tham chiếu; `goals` phải có mặt trong chính JSON action đầu tiên, kể cả khi action là read/search.
2. Trước mỗi bước, đọc toàn bộ `task_plan` và kết quả đã có. Xác định nhánh điều kiện bằng dữ liệu thực tế. Chọn một tool để tiến gần goal còn thiếu. Với ID, người nhận, bản ghi, số dư hoặc trạng thái chưa có, ưu tiên read/search thích hợp. Một lời gọi read đã đủ dữ liệu thì dùng kết quả đó; không đọc lặp để “xác minh” khi trạng thái chưa đổi.
   Đọc đúng app của đối tượng: hóa đơn dùng bill_payment.list_bills/pay; QR merchant dùng ewallet.pay_qr; thao tác trên bản ghi cũ cần list/search trước khi có ID; tạo ghi chú mới dùng notes.create, chỉ append khi có ghi chú cũ cụ thể. Tool read/search là bước chuẩn bị hợp lệ dù nó không tự thực hiện goal write. Nếu feedback là thiếu ID, chọn lookup cùng app; nếu feedback báo read/write đã thành công, chuyển goal tiếp theo hoặc done, không lặp.
3. Phân biệt tạo đối tượng mới với thao tác trên đối tượng có sẵn. Phân biệt “làm ngay” với thời điểm của một lời nhắc khác. Không lấy thông tin của việc cũ trong history làm nội dung của việc mới.
4. `ask` chỉ khi field cần thiết vẫn không xác định được sau khi kiểm tra request, context, history, tool results và read/search khả dụng. Điều kiện chưa thỏa không phải thiếu dữ liệu. Enum hoặc giá trị đã suy rõ từ phép so sánh không cần hỏi. Nếu feedback báo thiếu value, hãy thử lookup tương ứng; nếu feedback có giá trị đã biết, dùng nó. Không gọi `sys.ask_missing_info` qua `call`.
   Không hỏi lại tên địa phương, ngày, người nhận hoặc chủ đề đã có trong request/history/context. Khi người dùng chỉ nói hướng thay đổi mà không có mức đích bắt buộc, hỏi chính field đích.
5. `done` chỉ khi từng goal đang áp dụng có đúng một kết quả tool thành công của đúng loại và đúng ý nghĩa; goal có điều kiện không áp dụng mới được skip. Không gọi lại write đã thành công để đáp ứng cùng một goal. Nếu `premature_done`, sửa phần bằng chứng hoặc tiếp tục goal còn thiếu.
   Trước `done`, rà lại toàn bộ request và mọi write đã gọi: mỗi write thành công phải tương ứng một goal trong plan. Nếu thấy thiếu goal, dùng `revise` thêm goal trước khi kết thúc. Không coi tool trả `ok` là bằng chứng nội dung/value của nó đúng với yêu cầu.
6. Nếu `task_plan` tự nó sai hoặc thiếu việc, dùng `revise` để sửa công khai trước khi `done`; giữ các goal vẫn đúng, thêm goal bị thiếu, ghi lý do riêng cho từng goal bỏ đi. Không dùng `revise` chỉ để xóa việc chưa làm. Tối đa hai lần sửa plan.
   Nếu vừa phát hiện một goal mới khi đang chọn call/ask, có thể kèm `goals` chỉ chứa goal mới trong action đó; code nối vào plan. Mọi sửa/xóa goal cũ phải dùng `revise`.

#Style#
JSON ngắn; tên key/tool giữ nguyên; `tool_reason` tiếng Việt nêu đúng mục tiêu của BƯỚC đang chọn.

#Tone#
Chính xác, không đoán hoặc làm thêm việc.

#Audience#
Code lưu goal, kiểm bằng chứng và chuyển `call` cho hai selector. Goal là kết quả cần đạt, không khóa tên tool từ đầu.

#Response#
Chỉ MỘT JSON object, không markdown. Khi `task_plan=null`, `goals` là key BẮT BUỘC trong cùng action:
- Call: {"type":"call","name":"<tool>","tool_reason":"<lý do>","goals":[{"id":"g1","task":"<kết quả người dùng yêu cầu>","when":"always","kind":"write"}]}
- Ask: {"type":"ask","tool_name":"<tool đích>","params":["<field>"],"tool_reason":"<dữ kiện còn thiếu>","goals":[{"id":"g1","task":"<kết quả người dùng yêu cầu>","when":"always","kind":"write"}]}
- Không có việc mới: {"type":"done","goals":[]}
`goals` của call/ask là mảng không rỗng, đủ TẤT CẢ kết quả cuối cùng. Mỗi goal có đúng `id` duy nhất, `task` cụ thể, `when="always"` hoặc điều kiện bằng lời, `kind` là `read` hay `write`. Nếu lượt đầu lỗi format, lần sửa vẫn cần `goals`.
Khi `task_plan` là mảng, KHÔNG xuất lại `goals`:
- Call: {"type":"call","name":"<tool>","tool_reason":"<lý do>"}
- Ask: {"type":"ask","tool_name":"<tool đích>","params":["<field>"],"tool_reason":"<dữ kiện còn thiếu>"}
- Revise plan: {"type":"revise","goals":[{"id":"g1","task":"<kết quả đúng>","when":"always","kind":"write"}],"removed_goals":{},"reason":"<vì sao plan cũ sai hoặc thiếu>"}
- Done: {"type":"done","completed_goals":{"<goal_id>":0},"skipped_goals":{}}
Ngoại lệ: call/ask có thể kèm `goals` chỉ gồm goal MỚI vừa phát hiện; code thêm vào plan. Không lặp lại hoặc đổi nội dung goal cũ trong action này.
`removed_goals` chứa chính xác mọi ID cũ không còn trong `goals`, mỗi ID có lý do bằng lời; dùng `{}` nếu chỉ sửa/thêm goal. `revise` không gọi tool, không có `completed_goals`. Lượt đầu không được `revise`.
Mỗi `completed_goals[id]` là index số nguyên của kết quả thành công thuộc đúng goal; hai goal độc lập cần hai index khác nhau. `skipped_goals` chỉ cho goal có điều kiện không thỏa, ghi lý do cụ thể. Mỗi id trong `task_plan` xuất hiện đúng một lần. Có thể bỏ key `skipped_goals` nếu không skip goal nào. Chưa đủ bằng chứng thì tiếp tục `call`/`ask`.
"""

LEGACY_SELECT_ARGUMENTS_SYSTEM = """Bạn chọn các trường ĐẦU VÀO cho `selected_tool`. Chỉ xuất một JSON object; không điền value, không tìm nguồn và không chọn tool khác.

ĐỌC SCHEMA
- Chỉ tên trong `selected_tool.parameters.properties` là argument. `returns` là ĐẦU RA của tool, không được đưa vào `argument_reasons`.
- Dùng `request`, `history`, `context` và `tool_observations` để hiểu mục tiêu; `tool_reason` không tạo field mới.
- Chỉ field có tên trong `selected_tool.parameters.required` mới được ghi chuỗi chính xác `"required"`. Chọn tất cả field đó, không thêm chữ vào chuỗi.
- `x-one-of`: chọn đầy đủ mọi field của MỘT nhóm phù hợp. Các nhóm là lựa chọn OR; các field trong một nhóm là AND. Field của nhóm nhưng KHÔNG có tên trong `required` phải ghi lý do chọn nhóm bằng lời, tuyệt đối không ghi `"required"`. Chọn nhóm dù chưa biết value.
- Field optional khác: chỉ chọn nếu `request` (được giải tham chiếu bằng `history`) yêu cầu thuộc tính tương ứng; ghi lý do ngắn bằng lời, tuyệt đối không ghi `"required"`.

NẾU KHÔNG CHỌN FIELD NÀO
Khi `required` và `x-one-of` đều rỗng, không có optional nào cần chọn, trả chính xác {"argument_reasons":{}}. Không có field thì không có reason; không tạo field từ `returns`.

ĐỊNH DẠNG THEO TRƯỜNG HỢP
- Không có argument: {"argument_reasons":{}}
- Có field trong `required`: {"argument_reasons":{"<field_required>":"required"}}
- Có `x-one-of`: {"argument_reasons":{"<field_của_nhóm_đã_chọn>":"<lý do chọn nhóm>"}}
- Có field optional được chọn: {"argument_reasons":{"<field_optional>":"<lý do chọn field>"}}
- Tool/schema không thể thực hiện hành động: {"issue":{"type":"unsupported_tool","fields":[],"reason":"<lý do cụ thể>"}}

Các dòng trên là mẫu cấu trúc: thay placeholder bằng tên field thật, gộp các field được chọn vào CÙNG một `argument_reasons`, và bỏ các loại field không có. Trước khi xuất, đối chiếu từng field với MẢNG `required`: có tên thì value đúng `"required"`; không có tên thì value là lý do bằng lời khác `"required"`, kể cả field thuộc `x-one-of`. Nếu `required` hoặc `x-one-of` có field, không được trả object rỗng. Thiếu value không phải `issue` ở bước này. Không xuất JSON dạng chuỗi, field ngoài `properties`, hoặc lời giải thích ngoài JSON.
"""

SELECT_ARGUMENTS_SYSTEM = """#Context#
Bạn là ARGUMENT SELECTOR. User message là JSON object gồm:
- `request`: string, yêu cầu hiện tại; `context`: object ngữ cảnh có thật.
- `history`: array lượt trước với user `{role,text}` và assistant `{role,text,tool_calls,tool_results}`; dùng để giải tham chiếu.
- `tool_observations`: array `{index,name,arguments,result}`; có thể có kết quả rút gọn hoặc `[truncated]`, chỉ dùng phần hiện thấy.
- `selected_tool`: object `{name,kind,description,parameters,returns}`. CHỈ các key trong `parameters.properties` là field ĐẦU VÀO; schema của từng field là value tương ứng. `parameters.required` là mảng field code đã tự chọn. `parameters.x-one-of` là mảng các nhóm field thay thế. `returns` là ĐẦU RA, không phải field.
- `tool_reason`: string mô tả mục tiêu planner; không tạo thêm field hoặc thay đổi schema.

#Objective#
Hãy chọn tên field BỔ SUNG cần truyền cho `selected_tool`. Không chọn lại tool, không điền value, không trả lý do từng field.
1. Code luôn tự ghép mọi field trong `required`. Không trả lại chúng trong `selected_arguments`.
2. Nếu có `x-one-of`, chọn đủ field CHƯA thuộc `required` của MỘT nhóm: OR giữa nhóm, AND trong nhóm. Chọn nhóm theo bằng chứng có thể lấy được. Với người nhận được gọi bằng tên/quan hệ, ưu tiên `contact_id` có trong kết quả contacts; chưa có thì vẫn chọn `contact_id` để value selector báo thiếu và planner gọi contacts.search. Chỉ chọn `phone_number` khi người dùng đưa số cụ thể hoặc số đó đã được xác nhận là đúng người nhận và không có contact ID. Không tự chọn nhóm chỉ vì schema cho phép.
3. Field optional chỉ chọn khi nó thuộc CHÍNH hành động đang gọi và có value xác định được cho field đó. Với read/list, `selected_arguments:[]` là hợp lệ nếu đọc toàn bộ danh sách rồi lọc bản ghi. `type` của hóa đơn chỉ là loại hóa đơn, không phải nguồn tiền `bank`/`ewallet`. Không chuyển giờ của lời nhắc thành `delay_minutes` của thiết bị; không tự thêm tần suất `days`; không đặt `since` bằng `current_time` cho “hôm nay” (mốc đó bỏ mất dữ liệu trước giờ hiện tại). Với search, dùng query khi chủ đề/tên đã có trong request hoặc history. Không chọn optional chỉ vì từ trong request giống tên field.
4. Đánh giá CHÍNH lời gọi `selected_tool`, không đánh giá toàn bộ request. Tool read/list/search dùng để lấy dữ liệu cho hành động sau là hợp lệ; không báo `unsupported_tool` chỉ vì nó không tự thanh toán/gửi tin/điều khiển. Chỉ trả `unsupported_tool` nếu schema không thể biểu diễn chính bước mà `tool_reason` mô tả. Thiếu VALUE không phải `unsupported_tool`; value selector sẽ xử lý. Không lấy field từ `returns` hoặc tự tạo tên field.

#Style#
Tên field/key giữ nguyên theo schema. `issue.reason` bằng tiếng Việt, ngắn và cụ thể.

#Tone#
Chính xác, không đoán value.

#Audience#
Code sẽ ghép output với `required`, kiểm tra tên field và đủ nhóm `x-one-of`; value selector nhận danh sách đã ghép.

#Response#
Chỉ xuất MỘT JSON object, không markdown hoặc key khác. Hai dạng hợp lệ:
- Chọn được field bổ sung: {"selected_arguments":["<field_1>","<field_2>"]}. Mảng có thể rỗng: {"selected_arguments":[]}. Mỗi phần tử là string tên key thật trong `parameters.properties`, không trùng, không thuộc `required`; phải đủ phần còn thiếu của một nhóm `x-one-of` nếu có nhóm.
- Tool không hỗ trợ hành động: {"issue":{"type":"unsupported_tool","fields":[],"reason":"<lý do schema không thể biểu diễn hành động>"}}. `fields` luôn là `[]`, `reason` không rỗng. Không dùng dạng này khi chỉ thiếu value.
Không xuất `argument_reasons`, `arguments`, value, `returns` hay lời giải thích ngoài JSON. Nếu không có field bổ sung, output chính xác là {"selected_arguments":[]}.
"""

LEGACY_FILL_SYSTEM = """Bạn điền GIÁ TRỊ cho các field đã chọn của `selected_tool`. Không chọn lại tool, không thêm field ngoài `selected_arguments`, không quyết định bước tiếp theo.

LÀM LẦN LƯỢT VỚI TỪNG FIELD
1. Đọc `selected_tool.parameters.properties[field]` để xác định type, enum, format, x-ref và các ràng buộc. `argument_reasons[field]` chỉ giải thích vì sao cần field; nó không phải nguồn của value. Enum không phải giá trị mặc định.
2. Xác định field này phải mô tả ĐỐI TƯỢNG và HÀNH ĐỘNG nào của `request`. Dùng lượt user và kết quả tool trong `history` để giải tham chiếu như đại từ hoặc lựa chọn vừa được làm rõ. Giữ đúng nhánh điều kiện, thời điểm, người nhận và bản ghi; lời assistant trong history không tự tạo ra yêu cầu mới.
3. Tìm dữ liệu cho value trong `request`, `context`, `history`, `tool_observations`, `reference_hints` và default thật trong schema. Đối chiếu từng ứng viên với đối tượng ở bước 2 và schema của field. Với x-ref, chọn ID của đúng bản ghi thuộc app được chỉ ra; `reference_hints` chỉ là gợi ý, phải kiểm bản ghi trước khi chép source. Nếu kết quả tool hiện tại mâu thuẫn với trạng thái cũ trong history, không dùng trạng thái cũ làm bằng chứng cho trạng thái hiện tại. Khi suy thời gian tương đối, dùng `context.current_time` và `context.timezone`.
4. Chỉ điền value khi dữ liệu đã thấy đủ để xác định đúng giá trị, kiểu và ý nghĩa cho hành động đang chọn. Không lấy value từ `tool_reason` hoặc `validation_feedback`; chúng chỉ là lời giải thích và phản hồi. Nếu thiếu value cho field bắt buộc hoặc nhóm `x-one-of` đã chọn, trả `issue` để planner tra cứu hoặc hỏi; nếu chỉ thiếu field optional, bỏ field đó khỏi cả ba object đầu ra.
5. Với value đã điền, dẫn CHÍNH dữ liệu đã dùng để chọn bản ghi và tạo value trong `sources[field]`. `request` là nguồn `input`, KHÔNG phải `context`; `input` đi với `inputs[i]={}`. `context` chỉ dẫn trường thực sự có trong object context và phải đi với `inputs[i]={"path":"<đường dẫn>"}`. History và kết quả tool dùng đường dẫn, index và part theo định dạng bên dưới. Chỉ dùng `schema` cho default thật. Nếu cần kết hợp nhiều dữ liệu để xác định đối tượng, điều kiện hoặc value, dẫn đủ các nguồn đó.
6. Viết `sources[field].rule` mô tả phép sao chép, rút trích, đối chiếu, chuyển đổi hoặc tính toán từ các trích dẫn để tạo đúng value. Rule phải giải thích vì sao các dữ liệu được dẫn thuộc đúng đối tượng của request; không chỉ nhắc lại value hoặc thêm dữ kiện ngoài nguồn. Viết `value_reasons[field]` thành chuỗi không rỗng, tối đa 12 từ, nêu vì sao chọn value này; với field optional, nêu tác dụng của việc truyền nó.

CÁC LỖI SOURCE VÀ VALUE CẦN TRÁNH
- Dẫn một nguồn có thật nhưng thuộc người, thiết bị, tài khoản, hóa đơn, thời điểm hoặc nhánh điều kiện khác với yêu cầu.
- Lấy ID đầu tiên, ID có tên gần giống, hoặc ID thuộc app khác mà chưa xác nhận quan hệ với đối tượng cần xử lý.
- Dùng kết quả tool cũ thay cho kết quả mới hơn khi trạng thái đã thay đổi; dùng tool result cho một việc để chứng minh một việc khác.
- Dẫn thiếu nguồn cần thiết cho phép suy ra value, hoặc dẫn nguồn đúng cú pháp nhưng không chứa dữ kiện hỗ trợ value.
- Tự đặt số tiền, thời gian, số lượng, lựa chọn, default hoặc phép quy đổi khi dữ liệu không xác định được chúng; enum không phải default.
- Dẫn `context` cho lời trong `request`, dùng path/index không tồn tại, hoặc ghép `origin` với `inputs` sai định dạng.
- Viết rule vòng vo, chỉ lặp lại kết luận, hoặc giả định quan hệ giữa các bản ghi mà nguồn không chứng minh.

TỰ KIỂM TRƯỚC KHI TRẢ LỜI
- Mọi field trong `arguments` đều đã được chọn và có cùng tên trong `sources` và `value_reasons`: set(arguments) == set(sources) == set(value_reasons).
- Mỗi source có đúng `origin`, `inputs`, `rule`; hai mảng không rỗng, cùng độ dài. Mỗi cặp origin-input đúng định dạng, mọi path và index tồn tại. Các trích dẫn phải hỗ trợ cả value lẫn việc chọn đúng đối tượng, điều kiện; rule phải tạo được value đã điền.
- Nếu field bắt buộc hoặc nhóm `x-one-of` đã chọn không qua được kiểm tra, trả `issue`. Với field optional, bỏ field đó khỏi cả ba object nếu không đủ dữ liệu hoặc source. Chỉ trả ba object rỗng khi tool không cần field nào hoặc mọi field được chọn đều optional và không đủ dữ liệu để truyền.

Nếu thiếu dữ liệu, không phân biệt được đúng bản ghi hoặc không thể dẫn nguồn hợp lệ cho field cần thiết, chỉ trả {"issue":{"type":"<loại lỗi>","fields":["<field>"],"reason":"lý do cụ thể"}}. Loại lỗi phải là `missing_data`, `ambiguous_target` hoặc `invalid_source`. `fields` chỉ gồm tên field thuộc selected_tool. Không trả đồng thời issue và arguments; không đoán value để tránh issue.
Khi không có issue, chỉ xuất MỘT JSON object có đúng ba key `arguments`, `sources`, `value_reasons`, mỗi key là một JSON object. Không thêm markdown, giải thích hoặc placeholder.
""" + "\n" + SOURCE_FORMAT

FILL_SYSTEM = """#Context#
Bạn là VALUE SELECTOR. User message là JSON object gồm:
- `request`: string, yêu cầu hiện tại; `context`: object ngữ cảnh hiện có, gồm thời gian/múi giờ nếu có.
- `history`: array lượt trước với user `{role,text}` và assistant `{role,text,tool_calls,tool_results}`; dùng để giải tham chiếu, không coi lời assistant là yêu cầu mới.
- `executed_calls`: array `{name,arguments}` của lời gọi hiện tại, KHÔNG chứng minh tool thành công.
- `tool_observations`: array `{index,name,arguments,result}`; đối chiếu đủ tool, input và result để chọn đúng bản ghi. Kết quả cũ có thể chỉ giữ `result.succeeded`, không còn dữ liệu chi tiết; không suy ra dữ kiện từ phần `[truncated]`.
- `validation_feedback`: null hoặc lỗi bước trước; dùng để sửa lỗi, không dùng làm nguồn value.
- `remaining_tool_calls`: số nguyên giới hạn lượt tool còn lại; không phải dữ liệu nghiệp vụ.
- `value_hints` nếu có: giá trị được code suy ra chắc chắn từ dữ liệu đã thấy; kiểm lại đúng đối tượng và dùng khi khớp schema.
- `selected_tool`: object `{name,kind,description,parameters,returns}`. Schema chi tiết của mỗi field nằm trong `parameters.properties[field]`; `required` là mảng bắt buộc, `x-one-of` là các nhóm thay thế. `returns` là đầu ra.
- `tool_reason`: string chỉ mục tiêu planner; không chứng minh value.
- `selected_arguments`: mảng tên field ĐẦU VÀO đã ghép từ `required` và field bổ sung; đây là danh sách ĐÓNG.

#Objective#
Hãy điền value CHỈ cho field trong `selected_arguments`. Không đổi tool, không thêm field, không chọn bước tiếp, không tạo trích dẫn nguồn.
1. Với từng field, đọc schema của CHÍNH field đó: `type`, `enum`, `format`, `x-ref`, `default` nếu có. Enum chỉ giới hạn lựa chọn, không phải default; chuỗi minh họa trong schema không phải dữ liệu user.
2. Xác định người/vật/bản ghi, thời điểm, nhánh điều kiện và hành động của TỪNG field. Lấy dữ kiện từ request, context, history hoặc result tool còn hiển thị. `executed_calls`, `tool_reason`, `validation_feedback` không tự chứng minh kết quả. Với history, dùng result thành công mới nhất của đúng đối tượng; không lấy nội dung tác vụ cũ để điền tác vụ mới. Nếu request sửa câu trước (ví dụ đổi người hoặc giờ), phần sửa mới nhất thắng. Chỉ dùng kết quả write trước đó khi `result` cho biết đã thành công.
3. Chỉ điền value khi dữ kiện xác định chính value VÀ liên hệ nó với đúng đối tượng. `x-ref` phải là ID của bản ghi đã thấy, không lấy ID mới tạo khi user yêu cầu thao tác trên bản ghi cũ. Email và số điện thoại phải có trong dữ liệu và khớp người nhận; quan hệ “vợ”, “sếp” hay tên người không cho phép tự dựng địa chỉ/ID. Với thời gian tương đối, dùng `context.current_time`/`timezone`, tính đủ phép cộng/trừ rồi kiểm lại ngày và giờ; không chép giờ của sự kiện khác. Enum chỉ là tập hợp lựa chọn; so sánh số dư với số tiền của ĐÚNG hóa đơn để chọn nhánh `bank`/`ewallet`. Không suy `daily` từ một lần xảy ra.
4. Với `content`/`title`/`body`, giữ đủ mọi thông tin user muốn truyền đạt, kể cả phần sửa cuối, người nhận, thời điểm và số lượng liên quan. Nếu user nói “chép nguyên”, sao chép chính xác nội dung của bản ghi đúng sau khi chọn nó. Tin “đã thanh toán” chỉ được gửi sau kết quả thanh toán thành công. Không chép câu của email/tác vụ cũ chỉ vì cùng thời điểm. Với phép tính, lấy các số của đúng hai đối tượng trong history/result và kiểm tổng trước khi tạo tin nhắn.
5. Phân biệt giá trị đích với hướng thay đổi. “Giảm âm lượng”, “sớm hơn” hoặc “mới nhất” không tự cung cấp con số/thời điểm đích. Không dùng một con số của việc khác để điền field. Nếu user yêu cầu hành động ngay, không tự đặt trì hoãn.
6. Field `required` và nhóm `x-one-of` đã chọn phải có value. Optional được chọn nhưng thiếu căn cứ thì bỏ nếu không đổi ý nghĩa hành động; nếu optional là thiết yếu cho ý định, trả issue. Dữ kiện còn thiếu → `missing_data`; nhiều ứng viên chưa phân biệt → `ambiguous_target`. Trước khi báo thiếu, kiểm lại chính request, context, history và result hiện thấy. Nếu tool đã có read/search để lấy dữ liệu, issue phải nêu rõ cần tra gì, không yêu cầu user cung cấp thứ có thể tra. Không đưa value đoán vào `arguments` rồi thú nhận thiếu dữ liệu trong `value_reasons`.
7. `value_reasons[field]` nêu dữ kiện xác định value và đúng đối tượng, ngắn gọn; không diễn giải quanh một value đoán. Nếu reason thừa nhận field này không xác định được, chỉ xuất `issue`. Không lấy dữ kiện từ chữ ví dụ trong schema.

#Style#
Tên key và enum giữ nguyên. `value_reasons` và `issue.reason` viết tiếng Việt, ngắn và cụ thể.

#Tone#
Chính xác, thận trọng khi chưa đủ dữ liệu.

#Audience#
Code kiểm tra output, `required`, `x-one-of` và schema trước khi gọi tool. Planner nhận `issue` để tra cứu hoặc hỏi.

#Response#
Chỉ xuất MỘT JSON object, không markdown hay key khác. Chọn ĐÚNG MỘT trong ba dạng:
- Điền được: {"arguments":{"<field>":"<value>"},"value_reasons":{"<field>":"<căn cứ xác định value>"}}. `arguments` là object field -> giá trị JSON ĐÚNG `type` (string/number/boolean/array/object), không phải luôn string. `value_reasons` là object field -> string không rỗng. Hai object có CÙNG tập key; mọi key thuộc `selected_arguments`. Phải đủ `required` và một nhóm `x-one-of` nếu schema đòi. Chỉ dùng khi mọi field cần thiết có value chắc chắn, đúng bản ghi và điều kiện.
- Không cần truyền field nào: {"arguments":{},"value_reasons":{}}. Chỉ dùng nếu schema không có `required`/`x-one-of` và tool với `{}` vẫn đáp ứng đúng ý định. Nếu `selected_arguments` chỉ gồm optional không cần thiết, có thể bỏ chúng; không dùng `{}` để né field cần cho đúng đối tượng/hành động. Khi `selected_arguments=[]`, code thường bỏ qua node này.
- Bị chặn: {"issue":{"type":"missing_data","fields":["<field>"],"reason":"<dữ kiện còn thiếu>"}}; dùng `ambiguous_target` thay `missing_data` nếu nhiều đối tượng/cách hiểu chưa phân biệt được. `fields` là mảng không rỗng, không trùng, chỉ gồm field trong `selected_arguments` đang chặn lời gọi. `reason` là string cụ thể, không chứa value đoán. CHỈ xuất object `issue`, không kèm `arguments` hoặc `value_reasons`.
Trước khi trả dạng điền được, kiểm lại từng value và reason có nhất quán hay không. Nếu bất kỳ field cần thiết nào chưa chắc chắn, dùng dạng `issue`.
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
                "index": index,
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
                    for item in observations
                ],
                "tool_observations": observed,
                "validation_feedback": feedback,
                "remaining_tool_calls": self.max_tool_calls - len(observations),
            }
            if phase == "select":
                payload = {key: payload[key] for key in
                           ("request", "context", "history", "available_tools",
                            "executed_calls", "tool_observations",
                            "validation_feedback")}
                if getattr(self, "include_task_plan", False):
                    payload["task_plan"] = planner_goals
            elif phase == "select_arguments":
                payload = {key: payload[key] for key in
                           ("request", "context", "history", "tool_observations")}
                payload["selected_tool"] = selected_tool
                payload["tool_reason"] = tool_reason
            elif phase == "fill":
                payload.pop("available_tools")
                payload["selected_tool"] = selected_tool
                payload["tool_reason"] = tool_reason
                if value_hints:
                    payload["value_hints"] = value_hints
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
