# Trạng thái kiểm tra

Ngày thực hiện: 2026-11-02.

## Đã kiểm tra local

Bộ 8 kiểm thử đều qua trên Python 3.12.14, PyTorch CPU 2.8.0, Transformers 4.57.6 và PEFT 0.18.0.
Thời gian chạy bộ kiểm thử cuối: khoảng 6,3 giây.

- Audit 1.545 train, 634 public; train có 3.541 tool call, 217 mẫu hỏi thiếu thông tin, 111 mẫu không gọi tool.
- Replay tất cả 1.545 chuỗi gold qua đúng zaic_tools: kết quả khớp label, không có lỗi schema.
- Chia tập theo nhóm request chuẩn hóa: 1.313 train, 232 validation; request trùng không vượt partition.
- Kiểm thử loss mask, scorer matching một-một/hash gate/fuzzy F1, giới hạn gold annotation prefix,
  không đưa gold/final_state vào prompt inference, không thực thi proposal sai schema.
- Kiểm chứng thực tế trên Qwen3 ngẫu nhiên kích thước nhỏ + PyTorch CPU 2.8.0:
  completion loss hữu hạn, global LoRA cập nhật; ba nhánh local khởi tạo đúng hash tensor global,
  cập nhật độc lập và không thay đổi adapter global.
- Kiểm tra tokenizer thật Qwen3-4B: xem tokenization_audit.json.
  Đây dùng rationale ngắn thay thế để kiểm tra cấu trúc; dữ liệu sau annotation phải đo lại.
- Các module Python và code cell notebook đều parse thành công.

ResourceWarning về file handle của zaic_tools đến từ thư viện BTC, không sửa thư viện này.

## Kaggle — kiểm tra lại MCP

Lỗi quyền save_notebook trước đó đã được gỡ trong lần kiểm tra tiếp theo ngày 2026-11-02.

- Đọc quota và notebook của tài khoản lctrnnguynkhi thành công.
- Còn khoảng 26,8 giờ GPU trong tuần tại thời điểm kiểm tra.
- Đã tạo và chạy notebook riêng tư thử nghiệm; bản đầu hoàn tất và xác nhận 2 Tesla T4.
- Lỗi gắn dataset của MCP vẫn còn: cả datasetDataSources và biến thể Setter đều không gắn nguồn.
  Notebook chạy với thư mục input rỗng, metadata không có data sources.
  Hiện tượng trùng [Kaggle issue #1140](https://github.com/Kaggle/kaggle-cli/issues/1140).
- download_dataset qua kết nối hiện có trả được link tải tạm thời cho dataset version 48.
  Notebook thử nghiệm GLPFT dùng link này để tải vào thư mục tạm, không đưa link hoặc dataset vào Git.

Notebook kiểm tra:
https://www.kaggle.com/code/lctrnnguynkhi/lifemate-paper-qwen3-environment-probe

Notebook smoke Qwen3-4B đã gửi chạy:
https://www.kaggle.com/code/lctrnnguynkhi/lifemate-paper-qwen3-glpft-smoke

Chưa kết luận fine-tune thành công cho đến khi đọc được status, lineage và metrics của notebook smoke.
Không coi replay gold, tokenizer audit hay test Qwen3 nhỏ là chất lượng solution đã fine-tune.
Dockerfile và offline bundle chưa được build/benchmark A30 trong phiên này.
