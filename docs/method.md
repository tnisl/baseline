# Đối chiếu paper và LifeMate

Paper: Shen et al., *Small LLMs Are Weak Tool Learners: A Multi-LLM Agent*, EMNLP 2024, trang 16658–16680.
Bản cung cấp: ../paper.pdf. [ACL Anthology](https://aclanthology.org/2024.emnlp-main.929/), [mã tác giả](https://github.com/X-PLUG/Multi-LLM-Agent).

## Phần giữ lại

α-UMi (mục 3.2) dùng Planner để sinh rationale và chọn Caller / Summarizer / Give up.
Caller nhận rationale, schema tool và quỹ đạo trước đó, xuất một tool call.
Observation thực tế quay lại Planner; Summarizer chỉ tạo câu trả lời cuối.

GLPFT (mục 3.3): giai đoạn global học đồng thời thought, action và answer trong định dạng ReAct.
Ba bản sao của kết quả global tiếp tục học riêng Planner, Caller và Summarizer trên **cùng tập instruction train**.
Triển khai dùng PeftModel.from_pretrained(global_adapter, is_trainable=True) để khởi tạo từng nhánh local.
SHA256 toàn bộ tensor adapter phải giống kết quả global trước khi local chạy. Mỗi nhánh có optimizer mới,
không dùng trọng số từ nhánh local khác. Adapter local đã bao gồm biến đổi global, không cộng adapter global thêm lần nữa.

Lịch full lấy theo mục 4.3: global 2 epoch, LR 5e-5; Planner và Caller 1 epoch, Summarizer 2 epoch, LR 1e-5.
Cấu hình T4: NF4 double quantization, LoRA rank 16/alpha 32/dropout 0.05 trên q/k/v/o và gate/up/down projections,
micro-batch 1, accumulation 8 (batch hiệu dụng 8; paper dùng 48), gradient checkpointing.
Loss cross entropy chỉ tính token completion thuộc nhiệm vụ, mask toàn bộ system/user/prefix observation.
Decoding greedy, seed 42. Context mặc định 8192, không vượt 16384; mẫu quá dài được đếm và loại công khai khi train,
inference từ chối cắt mất chứng cứ.

## Điều chỉnh bắt buộc

| Vấn đề | Paper | LifeMate / triển khai |
|---|---|---|
| Backbone | LLaMA-2-chat 7B/13B, ba mô hình | Qwen3-4B-Instruct-2507 dùng chung backbone, ba adapter độc lập |
| Fine-tuning | Chủ yếu full FT; có thí nghiệm LoRA | QLoRA NF4 trên Kaggle T4, global rồi tiếp tục local |
| Ngân sách | Ba bản sao backbone | Một backbone 4B + ba adapter rank 16 + Whisper small 244M; kiểm tra tham số lúc load |
| Đầu vào | Instruction văn bản | Audio tiếng Việt, context, history; ASR lượt cuối và các lượt user trong history |
| Nhãn rationale/answer | Có trong trajectory gốc | Tự sinh rationale bằng Qwen; summary tổng hợp từ observation |
| Tool | API ngoài có thể lỗi mạng | zaic_tools local, state thay đổi trong RAM |
| Thiếu dữ kiện | Có thể give up | Caller gọi sys.ask_missing_info đúng schema |
| Chấm điểm | Plan ACC, Action EM, Arg F1, ROUGE, ToolEval | State hash đúng mới tính tool-call F1, thêm giới hạn thời gian |
| Thời gian | Không cùng ràng buộc | 30 giây/mẫu, tính cả ASR history; ghi timeout và latency |

QLoRA và chia sẻ backbone là **biến thể áp dụng**, không phải tái lập full-parameter GLPFT của paper.
Paper cho thấy full FT tốt hơn LoRA trong thí nghiệm của họ; chưa thể suy ra mức tăng tương tự trên LifeMate.
Các hệ số học của paper được giữ cho cấu hình full nhưng cần đánh giá độ phù hợp với QLoRA bằng validation.

## Sinh rationale và chống rò rỉ

Module umilife.rationale chạy Qwen3-4B gốc ở greedy decoding để chú giải train.
Mỗi tác vụ annotation có request, context, history đã thực hiện, các action/observation **trước bước hiện tại**,
schema và **gold action của bước hiện tại**. Gold action chỉ hướng dẫn tạo nhãn huấn luyện, không xuất hiện trong prompt inference.
Không cung cấp observation hiện tại, observation tương lai hoặc final_state cho annotator.
Rationale lưu kèm hash input, model/revision, id và bước, có thể tiếp tục từ cache.

Rationale bước cuối là quy tắc tổng hợp cho hoàn tất / chờ thông tin / ngoài phạm vi.
Do train không có final answer gold, nhãn Summarizer là bản biểu diễn kết quả observation bằng văn bản và JSON;
không tự gán thành công nếu tool trả done=false. Đây là giới hạn của dữ liệu huấn luyện.
Cần đọc rationale sinh ra và validation trước khi kết luận chất lượng.
Câu trả lời cuối không nằm trong submission nhưng Summarizer vẫn được huấn luyện và chạy để giữ kiến trúc paper.

Chia train/validation trước khi sinh dữ liệu; request giống nhau sau chuẩn hóa nằm cùng partition.
Đây chỉ chống trùng request chính xác, không đảm bảo tách toàn bộ template ngữ nghĩa.
Cả hai giai đoạn tái sử dụng đúng train IDs; không dùng validation làm optimizer update.
Public test chỉ dùng audio/input/schema cho inference, không dùng pseudo-label để fit hoặc chọn mô hình.
Agent tạo đúng một ToolEnv/mẫu. Mọi call đã chạy lấy từ get_tool_calls, hash từ get_state_hash.
Không mã agent nào mở hoặc parse file trong các thư mục environment.

## Đánh giá

So sánh global single-LLM với hệ ba adapter local trên cùng validation:
(1) oracle transcript từ train để đo tool learning, (2) từ audio để đo end-to-end.
Bộ chấm local dùng matching một-một tối đa, fuzzy token F1 > 0.6, hash gate và timeout 30 giây.
Đây là mô phỏng theo mô tả công bố, không phải scorer chính thức; tokenization fuzzy có thể khác BTC.
Public test không có nhãn nên chỉ tạo submission và trace, không báo accuracy public tự suy đoán.

- smoke: 12 train episodes + một phần validation, 3 update/mỗi stage.
- pilot: 160 train episodes + một phần validation, 80 global + 40 update/mỗi local, accumulation 4.
- full: toàn bộ train partition và lịch epoch nêu trên.

Pilot xác nhận thí nghiệm hữu hạn, không phải mô hình đã hội tụ.
T4 không phải A30 chấm thi; thời gian trên Kaggle chỉ tham khảo, cần benchmark offline trên A30.
