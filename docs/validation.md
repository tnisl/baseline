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

## Kaggle

Lệnh save_notebook qua MCP bị từ chối trước khi tạo notebook:
"MCP tool call requires approval, but approval policy is never".

Môi trường local không có Kaggle CLI hay cấu hình KAGGLE_API_TOKEN/KAGGLE_USERNAME/KAGGLE_KEY.
Không truy xuất hoặc xuất credential của connector.
Chưa có notebook chạy, chưa có rationale sinh bằng Qwen4B, adapter 4B hoặc kết quả fine-tune/validation GPU.
Không coi replay gold, tokenizer audit hay test Qwen3 nhỏ là chất lượng của solution đã fine-tune.

Notebook sẵn sàng trong notebooks/kaggle_glpft.ipynb, metadata để private.
Cần cho phép Kaggle MCP save_notebook hoặc chạy notebook bằng tài khoản Kaggle đã được cấp quyền dữ liệu.
Sau khi notebook hoàn tất, lấy các file *.metrics.json, lineage.json và status.json để cập nhật báo cáo.
Dockerfile và offline bundle là cấu hình triển khai chưa được build/benchmark A30 trong phiên này.
