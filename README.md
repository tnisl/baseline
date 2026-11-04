# LifeMate — Planner / Caller với Qwen3-4B

Repo giữ các notebook train, validation và chẩn đoán đang dùng. Prompt tiếng Anh, tool schema TypeScript; Planner chọn bước tiếp theo, Caller điền arguments, tool result quay lại Planner. Route `summarizer` kết thúc workflow mà không cần sinh bằng một model Summarizer.

| Mục đích | Mã nguồn | Kaggle |
|---|---|---|
| Finetune Planner + Caller | [planner_caller_compiled](notebooks/planner_caller_compiled/README.md) | [Training](https://www.kaggle.com/code/anartt/lifemate-planner-caller-compiled) |
| Validation đầy đủ, 245 mẫu | [concise_validation_fast](notebooks/concise_validation_fast/README.md) | [Validation](https://www.kaggle.com/code/anartt/lifemate-concise-validation-fast) |
| Kiểm tra 4 mẫu từng lỗi | [unguided_failure_subset](notebooks/unguided_failure_subset/README.md) | [Diagnostic](https://www.kaggle.com/code/anartt/lifemate-unguided-failure-subset) |
| So sánh base với adapter | [base_adapter_comparison](notebooks/base_adapter_comparison/README.md) | [Comparison](https://www.kaggle.com/code/anartt/lifemate-base-adapter-comparison) |

Notebook được tạo bằng `make_notebook.py` trong từng thư mục và nhúng runtime đi kèm. Core được chốt ở commit cũ để tái lập; runtime nhúng là phần ghi đè có hiệu lực. Sửa `runtime/` rồi build lại notebook, không chỉ sửa payload trong `.ipynb`.

## Cấu hình hiện tại

- Backbone: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Train: Unsloth QLoRA rank16/alpha32, FP16 trên 2 T4; micro-batch train/eval4 mỗi GPU, batch hiệu dụng8. Compile bật, cosine LR/warmup, early-stopping patience3, checkpoint mỗi25 updates. Mặc định train local Planner và Caller; Global vẫn là tùy chọn.
- Split: 1300 train /245 valid, giữ nguyên ID trong các runtime; không dùng nhãn valid làm đầu vào model.
- Inference: vLLM, batch4 mỗi GPU; ASR Zipformer được cung cấp chạy cho audio hiện tại và history. Không train ASR.
- **Không dùng token-level enforcer ở Planner/Caller**, kể cả backend Transformers dự phòng. Giữ JSON/schema validation sau sinh, kiểm tra tool arguments và tối đa một lượt sửa mỗi node trong mỗi quyết định.
- Mỗi mẫu tối đa **7 quyết định /7 tool calls thực thi**. Đọc trùng bị chặn vẫn tiêu tốn một quyết định. Giới hạn này chặn vòng lặp kéo dài, không sửa lỗi suy luận nghiệp vụ.

Các lock vLLM vẫn chứa `lm-format-enforcer` vì đó là dependency gián tiếp của phiên bản vLLM đã chốt. Package không được gắn vào sampler hoặc dùng để lọc token; xóa riêng khỏi lock sẽ làm môi trường thiếu dependency. `requirements.txt` của backend Transformers đã bỏ dependency trực tiếp này.

## Cấu trúc

```text
notebooks/
  planner_caller_compiled/    # Train + checkpoint/resume
  concise_validation_fast/   # Full valid, hai worker GPU
  unguided_failure_subset/   # Chẩn đoán bốn mẫu
  base_adapter_comparison/   # Cùng đầu vào, base vs adapter
umilife/                     # Core, backend Transformers, train/evaluate helpers
tests/                      # CPU contracts và kiểm tra tích hợp tùy dependency
docs/                       # Đối chiếu paper, audit và provenance rationale
outputs/codex_rationales/   # Rationale cũ được giữ nguyên, không commit Git
```

Các script gốc `finetune.py`, `make_notebook.py`, `run.py` và Dockerfile thuộc luồng cũ; giữ để tương thích và bảo toàn thay đổi local. Luồng đang dùng được chỉ rõ trong bảng notebook phía trên. Không dùng adapter mới với prompt cũ chỉ vì chúng cùng backbone.

## Kiểm tra

```bash
python -B -m unittest discover -s tests -v
python notebooks/planner_caller_compiled/check_contracts.py
python notebooks/concise_validation_fast/checks.py
python notebooks/unguided_failure_subset/checks.py
```

Các kiểm tra nhẹ cần `jsonschema`; kiểm tra tensor chỉ chạy khi có Torch/Transformers/PEFT. CPU checks không xác nhận chất lượng hay tốc độ trên GPU. Loss valid teacher-forced không thay thế đánh giá workflow.

## Lưu trữ bản cũ

Các notebook thử nghiệm đã thay thế và output tải về trước đợt dọn này được chuyển ra `../paper_solution_archive/cleanup_20261105/`. Có `moved_paths.json` liệt kê đường dẫn; log và checkpoint không bị xóa. Rationale trong `outputs/codex_rationales/` và các thay đổi local chưa commit được giữ nguyên. Mã cũ đã commit cũng có thể tra lại trong lịch sử Git.

Xem thêm [đối chiếu paper](docs/method.md) và [hướng dẫn nhãn hiện tại](notebooks/planner_caller_compiled/README.md). Tài liệu audit cũ mô tả các phiên lịch sử; bảng notebook và cấu hình ở README này mô tả luồng hiện tại.
