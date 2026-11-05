# LifeMate — Planner / Caller fine-tune độc lập

Nhánh này đặc tả một phương pháp mới cho LifeMate với đúng hai node LLM: **Planner** và **Caller**. Hiện nhánh chỉ có README; chưa có mã nguồn, dữ liệu sinh, notebook, checkpoint hoặc kết quả đánh giá.

Tài liệu là hợp đồng thiết kế để triển khai và kiểm tra sau này. Các ví dụ JSON minh họa giao diện, không phải dự đoán đã chạy. Không kế thừa mặc định kiến trúc, prompt, model, adapter, vòng sửa lỗi hay cấu hình huấn luyện từ phương pháp trước.

## 1. Phạm vi đã thống nhất

1. Planner chọn một app.tool tiếp theo hoặc quyết định dừng.
2. Caller nhận tool đã chọn và điền arguments, values dựa trên yêu cầu và bằng chứng hiện có.
3. Nếu đủ dữ kiện, runtime kiểm tra rồi thực thi đúng tool Planner đã chọn.
4. Action và tool response thực tế được thêm vào trajectory để Planner quyết định bước tiếp theo.
5. Nếu Caller thiếu dữ kiện để điền arguments, Caller trả missing_info. Runtime chuyển thành lời gọi sys.ask_missing_info và kết thúc lượt hiện tại.
6. Không có vòng revise, retry, tự sửa JSON, sửa arguments, tự sửa tool hay phản hồi lỗi cho node sinh lại.
7. Hai node được fine-tune riêng trên hai bộ mẫu nội suy từ training_data. Không truyền gradient, rationale, hidden state hoặc checkpoint đã fine-tune từ node này sang node kia.

**Độc lập khi huấn luyện không có nghĩa là độc lập về bằng chứng khi inference.** Cả hai node nhìn thấy cùng request, history, context và trajectory thực tế; chúng có trách nhiệm và đầu ra khác nhau.

## 2. Bài toán và dữ liệu cuộc thi

Nguồn dữ liệu local là thư mục ../train+public_test/. “training_data” trong tài liệu này chỉ bộ train có nhãn của BTC; không phải tên một file đã được tạo trên nhánh.

| Đặc điểm | Train | Public test |
|---|---:|---:|
| Số mẫu | 1.545 | 634 |
| Mẫu có history | 1.018 | 419 |
| Audio user trong history | 3.491 | 1.299 |
| Số tool trong catalog, gồm meta tool | 147 | 122 |

Train có 3.541 tool call trong nhãn, 217 mẫu có sys.ask_missing_info và 111 mẫu không gọi tool. Chuỗi gold dài nhất có 6 call. Những con số này mô tả bản dữ liệu v3.2.0, không phải cấu hình giới hạn của phương pháp.

Public có 34 tên tool không nằm trong catalog train. Vì vậy đầu vào phải mang mô tả/schema của tool ở split đang chạy, không dùng bộ phân loại đóng chỉ biết các tên tool từng làm target trong train.

### 2.1. Dữ liệu của một mẫu

- **request**: yêu cầu user ở lượt cuối, là lượt cần xử lý và được chấm.
- **history**: hội thoại trước lượt cuối; các thao tác của assistant trong history đã được môi trường áp dụng.
- **context**: current_time, timezone, location, user_profile và installed_apps.
- **catalog**: mô tả app, tool, loại read/write/meta, parameters và returns.
- **label**, chỉ có trong train: tool_calls, tool_results và final_state.

History giúp giải tham chiếu, lời sửa và thông tin đã biết. Không replay history vào ToolEnv của lượt hiện tại.

### 2.2. Audio và transcript

Khi dựng dữ liệu SFT ban đầu, dùng audio_text của train cho request và các lượt user trong history. Khi chạy end-to-end, dùng ASR cho cả request hiện tại và audio user trong history; giữ nguyên text/tool_calls/tool_results assistant được BTC cung cấp.

ASR là thành phần tiền xử lý, không phải node LLM thứ ba. Không có transcript oracle của public/private. Backbone, ASR, huấn luyện ASR và tăng cường nhiễu ASR chưa được chốt trong thiết kế này.

## 3. Kiến trúc tổng thể

~~~text
Audio request + audio user history
                  |
            ASR / chuẩn hóa input
                  |
        request + history + context
                  |
                  v
          Planner, một lần sinh
             /            \
         stop           app.tool
          |                 |
      kết thúc              v
                    Caller, một lần sinh
                       /             \
                missing_info       arguments
                     |                 |
             validate meta call   validate call
                     |                 |
          sys.ask_missing_info     env.call
                     |                 |
                  kết thúc       append action/response
                                   /           \
                             thành công       lỗi
                                 |             |
                              Planner       kết thúc
~~~

Runtime là bộ điều phối thông thường, không phải một model hoặc node suy luận bổ sung. Nó quản lý phiên môi trường, kiểm tra contract, ghép tên tool, thực thi, lưu trajectory, giới hạn tài nguyên và xuất submission.

Không có Summarizer, Global, Critic, Reviser hoặc một model phân xử thành công. Không yêu cầu câu trả lời tự nhiên ở cuối vì đầu ra nộp là tool_calls và state_hash.

## 4. Thông tin dùng chung

### 4.1. Request

Giữ đầy đủ yêu cầu hiện tại, bao gồm nhiều mục tiêu, điều kiện, phủ định, số lượng, lời sửa và nội dung cần gửi/lưu. Không thay request gốc bằng bản diễn giải của Planner trước khi đưa cho Caller.

### 4.2. History

Giữ thứ tự hội thoại và thông tin được cung cấp. Các lượt user được chuyển từ audio thành text theo chế độ train hoặc inference. Các lượt assistant giữ text và các tool_calls/tool_results sẵn có.

History là bằng chứng trước lượt hiện tại. Một thao tác trong history không phải một call mới của lượt đang chấm. Tuy nhiên user có thể yêu cầu làm lại một thao tác; không được cấm mọi thao tác giống history một cách máy móc.

### 4.3. Context

Cung cấp thời gian, timezone, vị trí và profile mà mẫu có. “Hôm nay”, “mai”, “tối nay” được hiểu theo context của mẫu, không theo đồng hồ máy chạy.

Installed_apps dùng để xác định tool được phép. Runtime ghép danh sách tên tool được cài với catalog tương ứng của split để tạo mô tả/schema đầu vào; không lấy schema từ state ẩn.

### 4.4. Trajectory: action/observation của lượt hiện tại

Trajectory bắt đầu rỗng. Mỗi phần tử chứa đúng action đã thực thi và observation thực tế tương ứng:

~~~json
{
  "action": {
    "name": "contacts.search",
    "arguments": {"query": "Sếp Dũng"}
  },
  "observation": {
    "ok": true,
    "contacts": [{"id": "c_example", "name": "Sếp Dũng"}]
  }
}
~~~

Đây là ví dụ giao diện; tên trường trong response thực tế do tool BTC quyết định.

- Giữ thứ tự call và cấu trúc kết quả; cả hai node dùng cùng một nguồn trajectory.
- Proposal chưa chạy không được ghi thành action đã thực thi.
- Không tự tạo observation, ID hay trạng thái hoàn thành.
- Không trộn history với trajectory hiện tại.
- Không đưa kết quả của bước đang dự đoán hoặc bước tương lai vào input train.
- Không đưa label.final_state, state ẩn hoặc nhãn gold vào input inference.
- Không thiết kế thêm trường rationale hoặc bảng mục tiêu do một node khác cung cấp trong baseline này.

## 5. Contract của Planner

### 5.1. Đầu vào

| Trường | Nội dung | Mục đích |
|---|---|---|
| request | Yêu cầu hiện tại | Xác định việc cần làm và điều kiện |
| history | Hội thoại trước lượt hiện tại | Giải tham chiếu, thông tin cũ và lời sửa |
| context | Thời gian, timezone, vị trí, profile | Diễn giải yêu cầu theo bối cảnh |
| available_tools | Tool được cài, kèm mô tả và schema | Biết khả năng và phụ thuộc tham số |
| trajectory | Action/observation đã có | Biết việc đã làm, kết quả và bước còn lại |

Planner phải có đủ mô tả/schema để biết một thao tác cần lookup nào trước. Cách biểu diễn schema cụ thể chưa chốt; mọi biểu diễn phải giữ thông tin bắt buộc, lựa chọn tham số và nguồn ID.

Trong baseline này, sys.ask_missing_info được dành cho nhánh missing_info của runtime. Planner chọn các tool tác vụ được phép hoặc stop; không chọn trực tiếp meta tool này. Việc loại nó khỏi tập lựa chọn Planner là quyết định thiết kế, không phải hạn chế của thư viện BTC.

### 5.2. Đầu ra

Chọn đúng một tool:

~~~json
{"tool": "contacts.search"}
~~~

Hoặc dừng:

~~~json
{"stop": true}
~~~

Hai dạng loại trừ nhau. Không sinh arguments, rationale, thought, câu trả lời hoặc danh sách nhiều bước. Không xuất cả tool và stop; stop=false không phải đầu ra hợp lệ.

### 5.3. Trách nhiệm

- Chọn bước tiếp theo, không sinh cả chuỗi workflow trước khi quan sát.
- Chọn lookup trước khi write nếu thông tin cần thiết chưa có.
- Xử lý thứ tự phụ thuộc, điều kiện và yêu cầu nhiều việc.
- Dùng observation để biết còn mục tiêu nào chưa được đáp ứng.
- Không coi một lookup hoặc một write thành công là toàn bộ yêu cầu đã hoàn tất.
- Không gọi tool chỉ để tạo câu trả lời cuối hoặc xác minh lại một hiệu ứng đã đủ bằng chứng.

### 5.4. Ý nghĩa của stop

Stop nghĩa là không có hành động công cụ tiếp theo phù hợp với yêu cầu và khả năng hiện tại, bao gồm:

- Các việc được yêu cầu đã hoàn tất.
- Điều kiện của người dùng không thỏa nên không được thực hiện nhánh phụ thuộc.
- Người dùng hủy hoặc yêu cầu không thao tác.
- History đã đủ để đáp ứng yêu cầu hiện tại mà không cần call mới.
- Tool được cài không hỗ trợ yêu cầu.

Stop không đồng nghĩa mọi mục tiêu đều đã thực hiện thành công. Lý do dừng không nằm trong output tối thiểu này và không có nhãn suy luận tự nhiên bổ sung.

## 6. Contract của Caller

### 6.1. Đầu vào

| Trường | Nội dung | Mục đích |
|---|---|---|
| request | Cùng request gốc Planner nhận | Giữ đúng ý định và các giá trị |
| history | Cùng history | Lấy thông tin và tham chiếu cũ |
| context | Cùng context | Tính thời gian và giá trị theo bối cảnh |
| selected_tool | app.tool Planner đã chọn | Cố định thao tác cần điền |
| selected_tool_schema | Schema đầy đủ của selected_tool | Kiểu, trường bắt buộc, enum, ràng buộc |
| trajectory | Cùng action/observation đã có | Lấy ID và giá trị từ các lần call trước |

Caller không nhận rationale, hidden state hay dự đoán các bước tương lai của Planner. Caller không cần toàn bộ danh sách tool khác vì không có quyền tự chọn lookup hoặc thay đổi tool.

Schema cần giữ description, properties, required, kiểu lồng nhau, items, enum, format, minimum/maximum, x-one-of, x-ref và x-match khi có. “Optional theo API” không có nghĩa được bỏ qua một trường người dùng đã yêu cầu rõ.

### 6.2. Đầu ra đủ dữ kiện

~~~json
{"arguments": {"query": "Sếp Dũng"}}
~~~

Runtime tạo action bằng selected_tool và arguments. Caller không sinh lại tên tool, không thể thay tool và không trực tiếp điều khiển executor.

Arguments rỗng là hợp lệ nếu tool cho phép và không cần tham số:

~~~json
{"arguments": {}}
~~~

### 6.3. Đầu ra thiếu dữ kiện

~~~json
{"missing_info": ["contact_id"]}
~~~

- Danh sách phải không rỗng, không trùng và gồm tên trường của selected_tool.
- Chỉ nêu trường thực sự cần để thực hiện ý định mà chưa thể xác định từ bằng chứng hiện có.
- Không đưa null, chuỗi rỗng hay ID đoán vào arguments để thay missing_info.
- Không trả đồng thời arguments và missing_info.
- Với x-one-of, nếu một nhóm hợp lệ đã có đủ dữ kiện thì không hỏi thêm nhóm thay thế. Nếu chưa có nhóm nào đủ, chọn một phương án phù hợp để hỏi, không hỏi tất cả nhóm một cách máy móc.
- Thiếu giá trị optional mà người dùng không yêu cầu thì bỏ trường đó, không hỏi thêm.
- Nếu có nhiều thực thể phù hợp và chưa có căn cứ chọn, không tự lấy kết quả đầu tiên.

Thiếu trường bắt buộc trong một output arguments là lỗi contract; runtime không tự chuyển lỗi đó thành missing_info.

### 6.4. Nguồn giá trị được phép

Giá trị lấy từ request, history, context, observation hoặc suy ra có căn cứ từ các nguồn này, ví dụ đổi “mai lúc chín giờ tối” sang timestamp theo context.

ID phải được cung cấp hoặc quan sát trước đó; không học thuộc ID theo sample. Nội dung gửi/lưu giữ đúng các sự kiện người dùng yêu cầu. Với cấu trúc như items[].item_id, từng ID bên trong cũng cần có bằng chứng.

### 6.5. Lookup trước hay missing_info

Planner chịu trách nhiệm chọn lookup trước khi chọn tool cần dữ kiện đó. Caller chỉ làm việc với bằng chứng hiện có và selected_tool.

Nếu Planner chọn sms.send trước khi tìm được người nhận, Caller có thể trả missing_info cho contact_id. Baseline chấp nhận đây là lỗi lập kế hoạch dẫn đến hỏi sớm; không có need_lookup, need_replan hoặc vòng trả lại Planner để sửa lựa chọn.

## 7. Runtime và chính sách không revise

### 7.1. Một bước thông thường

1. Kiểm tra deadline/ngân sách trước khi sinh.
2. Gọi Planner đúng một lần.
3. Nếu stop hợp lệ, kết thúc.
4. Nếu chọn tool, kiểm tra tool thuộc danh sách được phép.
5. Gọi Caller đúng một lần với tool và schema đã cố định.
6. Kiểm tra output Caller.
7. Nếu arguments hợp lệ, kiểm tra bằng env.validate rồi thực thi env.call.
8. Append action/response thực tế vào trajectory.
9. Nếu tool trả kết quả thực thi bình thường, quay lại Planner cho bước mới; nếu lỗi, kết thúc.

Một bước mới sau observation thành công là hoạt động bình thường của agent, không phải revise.

### 7.2. Nhánh missing_info

Runtime chuyển output Caller thành:

~~~json
{
  "name": "sys.ask_missing_info",
  "arguments": {
    "tool_name": "sms.send",
    "params": ["contact_id"]
  }
}
~~~

Tool_name lấy từ selected_tool, params lấy từ missing_info. Runtime kiểm tra meta call, gọi đúng một lần, ghi nhận action/response rồi kết thúc. Không gọi lại Planner, không chờ một lượt user mới trong cùng mẫu và không thực hiện tool đích khi chưa đủ dữ kiện.

Đây chỉ là ánh xạ giao diện nội bộ sang tool BTC, không phải node LLM thứ ba.

### 7.3. Phân biệt kết quả bình thường và lỗi

Không dùng một điều kiện “response có done=false ở bất kỳ đâu” để xác định thất bại:

- ok=false ở cấp response là lỗi executor/schema.
- ok=true không đảm bảo tác vụ nghiệp vụ đã thực hiện.
- done=false ở cấp kết quả thao tác kèm thông báo không tìm thấy ID hoặc không đủ số dư có thể là thất bại nghiệp vụ.
- done=false bên trong một task có thể chỉ là trạng thái task chưa hoàn thành.
- Một read trả danh sách rỗng vẫn có thể là kết quả bình thường và cần đưa lại Planner.

Phải diễn giải theo contract tool, không đoán một schema response chung cho mọi app. Chi tiết phân loại response cần đối chiếu thư viện BTC khi triển khai.

Baseline ghi lại response lỗi và kết thúc, không yêu cầu Planner/Caller sửa hành động đã thất bại. Kết quả đọc rỗng bình thường vẫn quay lại Planner để chọn bước mới từ bằng chứng vừa có.

### 7.4. Các tình huống kết thúc

| Tình huống | Hành vi |
|---|---|
| Planner trả stop hợp lệ | Kết thúc bình thường |
| Caller trả missing_info hợp lệ | Gọi meta tool rồi kết thúc |
| JSON không parse được hoặc sai contract node | Kết thúc, không sinh lại |
| Planner chọn tool không được phép | Kết thúc, không gọi Caller/executor |
| Arguments không qua env.validate | Kết thúc, không thực thi proposal |
| Tool thực thi nhưng báo thất bại | Lưu action/response rồi kết thúc |
| Tool read trả rỗng nhưng hợp lệ | Append rồi quay lại Planner |
| Response thành công bình thường | Append rồi quay lại Planner |
| Hết thời gian, token context hoặc ngân sách bước | Kết thúc và ghi rõ nguyên nhân |

Không có completion review, repair prompt, JSON fixer, thử tool thay thế sau lỗi, speculative tool call, rollback hoặc reset ToolEnv. Nếu model sinh sai, không gọi thêm một model hoặc cùng node để “chuẩn hóa” đầu ra.

Parser và schema validator chỉ chấp nhận hoặc từ chối; không thêm tham số, đổi kiểu, thay ID hoặc sửa tên tool. Không có bộ suy luận nghiệp vụ ẩn trong runtime để thay quyết định của node.

Ngân sách bước/call cụ thể chưa chốt. Cần giới hạn hữu hạn để tránh lặp, nhưng không kế thừa tự động con số 7/7 của hệ cũ. Không mặc định dùng token-level grammar/enforcer; nếu chọn sau này phải ghi rõ vì nó thay đổi chế độ decoding.

## 8. Dựng hai bộ SFT độc lập từ training_data

### 8.1. Chia tập trước

Chia train/validation ở cấp episode trước khi nội suy thành mẫu từng bước. Mọi prefix của cùng episode phải ở cùng partition.

Nhóm các request trùng hoặc cùng template khi có thể để giảm rò rỉ; không chia ngẫu nhiên các dòng Planner/Caller sau khi sinh. Split ID, seed và hash phải được lưu khi triển khai. Nhánh này chưa chọn split và không mặc định dùng split cũ.

Không dùng public/private có nhãn tự tạo để train hoặc tune. Không đưa validation vào optimizer update hoặc tạo luật prompt riêng theo ID.

### 8.2. Mẫu Planner cho chuỗi tool thông thường

Gọi action gold thứ t là A_t, observation gold tương ứng là O_t. Prefix P_t chỉ chứa các cặp (A_i, O_i) với i < t.

Với một episode bình thường có T action:

- T mẫu Planner nhận request/history/context/available_tools/P_t, target là tên tool của A_t.
- Một mẫu Planner cuối nhận toàn bộ P_T, target là stop.
- Không đưa arguments gold hiện tại vào input Planner.
- Không đưa O_t vào input của bước t.

Planner học next-tool và stopping trực tiếp; không cần sinh hoặc gán rationale làm target.

### 8.3. Mẫu Caller cho chuỗi tool thông thường

Với mỗi action A_t:

- Input: request/history/context, tên tool gold của A_t, schema của tool đó và P_t.
- Target: arguments gold của A_t.
- Không đưa arguments gold hiện tại vào input.
- Không đưa observation hiện tại/tương lai vào input.
- Không gọi hoặc dùng output của Planner đã fine-tune để tạo input train Caller.

Với episode bình thường có T action, có T mẫu Caller. Caller không có mẫu stop ở cuối vì việc dừng thuộc Planner.

### 8.4. Chuyển gold sys.ask_missing_info

Gold BTC có thể chứa:

~~~json
{
  "name": "sys.ask_missing_info",
  "arguments": {
    "tool_name": "sms.send",
    "params": ["contact_id"]
  }
}
~~~

Tại prefix trước call này:

- Target Planner là tool đích: {"tool": "sms.send"}.
- Input Caller có selected_tool=sms.send và schema sms.send.
- Target Caller là {"missing_info": ["contact_id"]}.
- Không huấn luyện Planner chọn trực tiếp sys.ask_missing_info.
- Không huấn luyện Caller sinh arguments của sys.ask_missing_info.
- Không tạo mẫu Planner stop sau meta call vì runtime đã kết thúc nhánh này.

Nếu một episode có K call thông thường rồi một meta call kết thúc, nó tạo K+1 mẫu Planner và K+1 mẫu Caller. Nhãn ở bước meta cần kiểm tra tool đích có được phép và params có thuộc schema tương ứng hay không.

Các trường hợp meta call không kết thúc trajectory hoặc không ánh xạ được sang contract mới phải được ghi audit, không âm thầm bỏ hoặc chế nhãn.

### 8.5. Episode không có tool call

Tạo một mẫu Planner với trajectory rỗng và target stop; không tạo mẫu Caller.

Không tự gán tất cả các episode này là ngoài phạm vi. Output stop chung không yêu cầu nhãn nguyên nhân, nhưng khi audit vẫn cần phân biệt no-op hợp lệ và nhãn có vấn đề.

### 8.6. Kiểm tra chất lượng nhãn

- Đối chiếu tool, arguments, số lượng call/result và replay bằng thư viện BTC ở bước chuẩn bị dữ liệu.
- Replay khớp nhãn không chứng minh tác vụ thành công: gold vẫn có thể chứa thất bại nghiệp vụ.
- Không tạo target stop mang ý nghĩa “hoàn tất thành công” từ một chuỗi đã thất bại.
- Các episode train không tương thích chính sách dừng khi lỗi hoặc có ID không đủ bằng chứng phải được audit/quarantine và báo số lượng.
- Validation giữ nguyên reference; không sửa reference để tăng điểm. Báo riêng các trường hợp không tương thích khi phân tích node metrics.
- Không lấy giá trị từ state ẩn để chữa nhãn.
- Việc replay phục vụ annotation/scoring phải tách khỏi phiên agent inference, không trở thành cơ chế thử rồi reset trong một lần giải mẫu.

### 8.7. Độc lập giữa hai quá trình fine-tune

Hai bộ SFT có thể dùng cùng episode IDs và cùng gold prefix, nhưng có input/target và quá trình tối ưu riêng.

- Planner chỉ tối ưu nhiệm vụ chọn tool hoặc stop.
- Caller chỉ tối ưu nhiệm vụ arguments hoặc missing_info.
- Không có Global warm-start từ một model đã fine-tune chung.
- Không dùng adapter Planner để khởi tạo Caller hoặc ngược lại.
- Không joint loss, joint optimizer, mutual distillation hay huấn luyện qua output dự đoán của node kia.
- Có thể dùng cùng một pretrained backbone gốc với hai adapter độc lập, hoặc hai bản sao khởi tạo độc lập; lựa chọn chưa chốt.
- Nếu dùng chung backbone đóng băng, “độc lập” vẫn yêu cầu adapter và gradient update tách riêng, không gộp hai dataset vào cùng adapter.
- Mỗi node chọn checkpoint theo đánh giá nhiệm vụ riêng; sau đó đánh giá cặp checkpoint bằng workflow thực tế.

## 9. Ví dụ luồng tối thiểu

Yêu cầu minh họa: “Nhắn Sếp Dũng là tôi đến muộn mười phút.”

1. Planner thấy chưa có ID người nhận, chọn contacts.search.
2. Caller điền query=Sếp Dũng từ request.
3. Runtime thực thi lookup, append action và response có ID người nhận.
4. Planner nhận trajectory mới, chọn sms.send.
5. Caller lấy contact_id từ response và content từ request.
6. Runtime gửi tin, append kết quả.
7. Planner nhận bằng chứng gửi thành công và trả stop.

Nếu bước 4 xảy ra trước lookup, Caller không được đoán contact_id. Nó trả missing_info; runtime hỏi qua sys.ask_missing_info rồi kết thúc. Không sửa lại quyết định Planner.

Với yêu cầu nhiều việc, ví dụ “gửi tin rồi tạo nhắc việc”, Planner phải tiếp tục sau response gửi tin thành công. Nó chỉ stop khi không còn thao tác cần làm. Ví dụ này minh họa trách nhiệm của Planner, không thêm node quản lý mục tiêu.

## 10. Đánh giá

### 10.1. Planner riêng

Trên reference prefix, đo:

- Tỷ lệ chọn đúng next-tool.
- Tỷ lệ stop đúng và stop quá sớm.
- Tỷ lệ tiếp tục khi reference đã kết thúc.
- JSON/contract hợp lệ và tool có được phép.
- Phân nhóm bước đầu, trung gian, sau read, sau write, nhiều mục tiêu và tool ít/chưa có target train.

Các nhãn meta được chuyển sang tool đích theo mục 8.4. Không coi accuracy từng bước là điểm workflow thực tế.

### 10.2. Caller riêng

Cấp gold selected_tool và reference prefix để tách khỏi lỗi Planner. Đo:

- Arguments đúng toàn bộ và chất lượng từng trường.
- Đúng ID, thời gian, số lượng, giá trị enum và các trường lồng nhau.
- Giữ trường optional cần thiết cho ý định người dùng.
- Phân biệt arguments với missing_info và độ chính xác tập trường thiếu.
- JSON/contract/schema hợp lệ.

Phải báo các trường hợp missing_info trong validation, không chỉ đánh giá trên action thực thi được.

### 10.3. Workflow hai node

Dùng output thực của Planner, Caller và ToolEnv; không dùng gold action/observation trong input rollout. Báo:

- Điểm contest mô phỏng, final-state/hash accuracy và tool-call F1.
- Tỷ lệ hỏi thiếu thông tin, dừng sớm, lỗi định dạng, schema và nghiệp vụ.
- Số call, số lượt sinh, latency trung bình, p95 và số mẫu quá deadline.
- Kết quả theo độ dài chuỗi, nhiều app, history và tool mới.
- Mọi mẫu lỗi/timeout vẫn thuộc mẫu số; không chỉ tính trên các mẫu hoàn thành.

Đánh giá transcript sạch để đo năng lực tool-use và audio ASR để đo end-to-end phải tách riêng. Loss teacher-forced chỉ là chỉ số chẩn đoán.

### 10.4. Không che lỗi bằng revise

Mỗi lần Planner hoặc Caller được gọi chỉ có một generation. Tỷ lệ parse/schema sai phải được giữ trong báo cáo. Không bổ sung repair vào evaluator để làm đẹp kết quả của baseline.

Số lần sinh bình thường trên episode T call không-meta thành công rồi stop là T+1 lần Planner và T lần Caller. Runtime validation không phải generation; meta call và ASR vẫn tiêu tốn thời gian thực tế.

## 11. Ràng buộc môi trường và submission

### 11.1. Quy định trong gói dữ liệu

Nguồn: ../train+public_test/README.md và README của zaic_tools trong split.

- Mỗi mẫu dùng đúng một ToolEnv cho lần giải đó.
- Chỉ gọi các tool được phép theo installed_apps và catalog split.
- Agent không mở, đọc, copy hoặc parse file trong các thư mục *_environment/.
- Không dùng env.get_state() làm observation cho Planner/Caller; chỉ dùng response của các tool tác vụ.
- Không sửa thư viện BTC hoặc environment.
- Không reset môi trường, thử nhiều trajectory rồi chọn lại.
- Mọi call đã thực thi, gồm meta call và call bị executor từ chối nếu có, lấy nguyên từ env.get_tool_calls().
- Hash lấy từ env.get_state_hash(); không tự dựng hash từ điều agent nghĩ đã xảy ra.
- Dữ liệu cuộc thi không được phân phối lại trong repo.

Danh sách tool cho model là metadata được cung cấp qua context/catalog và giao diện công khai của runtime, không phải dữ liệu app trong state ẩn.

### 11.2. Giới hạn kỹ thuật cần bảo đảm khi triển khai

Bản quy định BTC lưu local được đọc trong đợt khảo sát nêu: tổng hệ thống dưới 6B tham số, LLM dưới 5B, context tối đa 16.384 token, offline inference, tối đa 30 giây/mẫu và phần cứng chấm một NVIDIA A30. Bản này còn nêu cutoff tài nguyên công khai trước 26/10/2026 và yêu cầu seed/decoding tái lập.

Các giới hạn trên không được ghi đầy đủ trong README gói dữ liệu; cần đối chiếu bản quy định và hướng dẫn Docker chính thức khi đóng gói. Chúng chưa được xác minh lại online cho nhánh mới này.

Tính ngân sách trên toàn hệ thống thực sự load, gồm ASR, backbone và adapter. Không mặc định hai bản backbone đều vừa giới hạn chỉ vì mỗi bản riêng lẻ dưới 5B. Không coi chia sẻ backbone đóng băng là chia sẻ quá trình fine-tune.

Đo thời gian end-to-end gồm xử lý audio hiện tại/history, mọi generation, validation và tool execution. Tách thời gian nạp model để báo minh bạch và đối chiếu cách BTC tính thời gian; không loại ASR khỏi latency mỗi mẫu. Không ghi over_deadline=false cố định hoặc coi số bước hữu hạn là bằng chứng đã đạt 30 giây.

### 11.3. Định dạng kết quả

Mỗi mẫu đúng một dòng JSON UTF-8:

~~~json
{
  "id": "sample_id",
  "tool_calls": [
    {"name": "app.tool", "arguments": {}}
  ],
  "state_hash": "<env.get_state_hash()>"
}
~~~

ID và hash ở trên là placeholder minh họa. Stop ngay từ đầu cho tool_calls rỗng. Thiếu thông tin cho tool_calls chứa sys.ask_missing_info cùng các call trước đó nếu có. Mẫu kết thúc do lỗi vẫn xuất mọi call đã thực thi và hash hiện tại nếu phiên môi trường còn truy cập được; trạng thái lỗi nằm trong trace đánh giá, không thêm trường tùy ý vào submission.

## 12. Các lựa chọn chưa chốt

| Hạng mục | Trạng thái |
|---|---|
| Hai node, trách nhiệm và không revise | Đã chốt |
| Planner tool/stop; Caller arguments/missing_info | Contract tối thiểu cho baseline |
| Missing_info thành meta call rồi kết thúc | Chính sách baseline |
| Lỗi thực thi kết thúc, read rỗng hợp lệ tiếp tục | Chính sách baseline |
| Backbone và cách load hai node | Chưa chọn |
| ASR và có fine-tune ASR hay không | Chưa chọn |
| JSON Schema/TypeScript hay biểu diễn schema khác | Chưa chọn |
| Ngôn ngữ prompt | Chưa chọn |
| Full FT/LoRA/QLoRA, rank, LR, batch, epoch | Chưa chọn |
| Split, seed, checkpoint selection chi tiết | Chưa chọn |
| Độ dài context/output và ngân sách bước/call | Chưa chọn, phải đáp ứng giới hạn BTC |
| Token-level grammar/enforcer | Chưa chọn; không mặc định áp dụng |
| Dữ liệu tăng cường hoặc negative examples Caller | Chưa có trong baseline đầu tiên |
| Mã nguồn, notebook, Docker, training/evaluation runs | Chưa triển khai |

## 13. Tiêu chí nghiệm thu triển khai sau này

1. Dựng hai bộ SFT từ train theo đúng prefix và mapping missing_info; không rò rỉ action/result hiện tại vào input.
2. Train hai node độc lập, ghi rõ backbone khởi tạo, split, prompt/schema contract và lineage checkpoint.
3. Mỗi generation chỉ một lần; mọi lỗi được ghi nhận, không có đường repair ngầm.
4. Planner và Caller nhận đủ bằng chứng như đã đặc tả; Caller không chọn lại tool.
5. Agent lấy dữ liệu app chỉ qua tool và chỉ có một ToolEnv cho mỗi mẫu trong một lần inference.
6. Chạy đủ validation với reference-prefix node metrics và workflow thực tế, tính cả lỗi/timeout.
7. Báo riêng transcript sạch và ASR end-to-end; xác nhận ngân sách tham số, context, latency và khả năng chạy offline trước khi coi là solution contest hoàn chỉnh.

Hiện chưa có các kết quả nghiệm thu này. Nội dung nhánh là đặc tả thiết kế, không phải tuyên bố một hệ thống đã huấn luyện hoặc đã đạt giới hạn cuộc thi.
