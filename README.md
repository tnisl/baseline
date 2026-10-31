# baseline_v8_finetune

Ba LoRA độc lập cho workflow v8: **planner chọn danh sách tool + thứ tự → caller chọn arg/value → summarizer quyết định caller/retry/done/give_up/wait**. Mỗi adapter bắt đầu từ `Qwen/Qwen3-4B-Instruct-2507` gốc. Không có global fine-tune.

Notebook: [notebooks/kaggle_finetune.ipynb](notebooks/kaggle_finetune.ipynb). Notebook clone GitHub, in chính xác commit đang chạy, cài dependencies và gọi cùng CLI với local. Không cần chép nhiều cell train thủ công hoặc attach source vào dataset.

## Chạy thử trên Kaggle

1. Attach dataset `cymeu4l0t/zaloai-dataset` hoặc dataset chứa `train/train.jsonl` cùng catalog và environment.
2. Bật GPU và Internet; mở notebook, giữ `PROFILE="smoke"`.
3. Run all. Mặc định: 20 episode, hai optimizer step/role, batch 1, accumulation 1, validation và lưu đủ ba adapter.
4. Cell cuối kiểm cả ba role hoàn tất, trọng số LoRA hữu hạn và đã thay đổi. Output `lora_adapters.zip` chỉ chứa adapters/tokenizer/metadata, không chứa base weights.

`PROFILE="train"` dùng toàn bộ tập ngoài holdout, một epoch, accumulation 16. Hai profile dùng thư mục output khác nhau, tránh dùng nhầm adapter chạy thử. Muốn tái lập phiên bản đã chạy, đặt `GIT_REF` bằng commit SHA thay vì tên nhánh.

Notebook mặc định không bật `CUDA_DEBUG`; chỉ bật khi cần xác định kernel gây lỗi. Cả ba job chạy tuần tự trong process riêng và **chỉ nhìn thấy một GPU**, kể cả khi Kaggle cấp T4 x2. `GPU_INDEX=0` chọn GPU đầu trong CUDA_VISIBLE_DEVICES kế thừa; không tự bật DataParallel/DDP.

## CLI thống nhất

```bash
python finetune.py --profile smoke --data-root ../train+public_test --output-dir /tmp/v8_lora_smoke
python finetune.py --profile train --data-root ../train+public_test --output-dir /tmp/v8_lora_train
```

Có thể ghi đè tham số, ví dụ:

```bash
python finetune.py --profile train --output-dir /tmp/v8_lora_epoch2 \
  --epochs 2 --learning-rate 1e-4 --negative-weight 0.1
```

`--roles caller` chỉ chạy caller. `--cuda-debug` bật CUDA_LAUNCH_BLOCKING trong process train. `--full-precision-base` tắt lượng tử hóa 4-bit nhưng vẫn chỉ train LoRA. `--overwrite-adapters` cho phép ghi đè; mặc định CLI từ chối output đã có adapter, không tự bỏ qua job khi hyperparameter đã đổi. Tiếp tục checkpoint từng role bằng `train_role.py --resume-from-checkpoint PATH`.

| Tham số | Smoke | Train |
|---|---:|---:|
| Data limit | 20 | 0 = toàn bộ |
| Max optimizer steps | 2 | -1 = theo epoch |
| Epochs | 1 | 1 |
| Microbatch | 1 | 1 |
| Gradient accumulation | 1 | 16 |
| Learning rate | 1e-4 | 1e-4 |
| Max sequence length | 8192 | 8192 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| Ranking weight | 0.1 | 0.1 |
| Ranking beta / margin | 1.0 / 0.2 | 1.0 / 0.2 |
| Logging interval | 1 | 5 |
| Save/eval interval | 50 | 50 |

Các giá trị là cấu hình khởi đầu, chưa chứng minh tối ưu điểm số. QLoRA mặc định dùng frozen NF4 base + double quant; compute FP16 trên T4, BF16 khi GPU hỗ trợ. LoRA áp dụng q/k/v/o và gate/up/down projections. Chỉ tham số LoRA được train; code từ chối base weights trainable.

## Nhãn và mẫu âm

- Planner SFT chuỗi tool gold theo thứ tự, giữ bước lặp. `sys.ask_missing_info` chuyển thành slot tool nghiệp vụ cho caller học ask.
- Caller học completion đúng gồm type/arguments hoặc ask/blocked. Rejected là output sai có cấu trúc, không dùng làm target dương.
- Counterfactual chạy trong ToolEnv độc lập. Chỉ tạo recovery khi lỗi thật không làm đổi state và call sửa đúng được kiểm chứng. Caller thấy lỗi trước đó, không thấy response tương lai của chính target sửa.
- Summarizer học `next`; terminal `answer` bị mask vì dataset không có nhãn câu trả lời cuối. Runtime vẫn giữ contract answer của v8.
- Summarizer negatives gồm lỗi recoverable → retry, write sai effect nhưng env success → give_up, plan thiếu write cuối → give_up. Rejected chỉ chọn trong các route contract cho phép ở trạng thái đó.
- Generator có bộ lọc missing write dependency, nhưng dữ liệu hiện tại không sinh được case thỏa lọc này; không coi đây là hành vi đã được train hiệu quả.

```text
loss = SFT(chosen) + weight * softplus(margin - beta * (score_chosen - score_rejected))
score = mean log-probability trên các token được supervise
```

Đây là SFT + **reference-free pairwise ranking**, không phải DPO. Planner luôn weight=0. Mỗi row chọn một rejected theo seed từ tối đa hai candidate lưu trong dataset. Đặt weight=0 bỏ ranking nhưng vẫn giữ positive recovery/counterfactual rows.

Prompt, padding và answer của summarizer không tính loss. Dùng chat template thật và kiểm assistant prefix; không cắt nửa JSON hoặc bỏ mất dữ kiện để vừa context. Row quá dài bị loại và thống kê. Logits chỉ materialize ở các vị trí cần supervise để giảm bộ nhớ.

## Chia dữ liệu và giới hạn nhãn

Split theo episode ID **trước augmentation**. Mặc định giữ riêng 100 ID từ report_v8, rồi chia 10% phần còn lại làm validation. Train/validation/holdout không chung ID. Đây chưa phải split theo scenario/template family.

Dry run toàn bộ dữ liệu hiện tại: 1.300 episode train, 144 validation, 100 holdout; một episode gold bị loại vì chat_id không tồn tại. Planner có 1.300/144 row train/validation, caller 3.718/436 và summarizer 5.165/619. Xem [manifest](docs/data_manifest_validation.json). Smoke chỉ dùng subset; không dùng kết quả smoke để kết luận chất lượng.

Các episode gold rỗng chỉ supervise planner tools=[], không tự gán done/give_up cho summarizer vì nhãn không phân biệt casual/no-op/unsupported. Prompt model chỉ có request/context/history/catalog và feedback đã xảy ra; gold/final_state/verifier metadata không được đưa vào input. State hash chỉ phục vụ verifier offline, chưa kiểm hết free-text semantics. Synthetic negatives vẫn cần kiểm chất lượng.

Planner label là **nhánh gold thực thi**, không phải plan chứa đầy đủ mọi nhánh điều kiện hoặc foreach chưa biết trước lookup. Fine-tune không tự khắc phục giới hạn static plan của v8. Đánh giá workflow trên validation để chọn cấu hình; dùng holdout cho đánh giá cuối.

## Output và chẩn đoán

```text
output/
  run_config.json / run_status.json / metrics.json
  logs/prepare_data.console.log
  data/                          # role JSONL, manifest, split/holdout/excluded IDs
  local_lora/
    adapters/planner|caller|summarizer/
    checkpoints/planner|caller|summarizer/
    training_runs/<role>.console.log
    training_runs/<role>.runtime.json
    training_runs/<role>.config.json
  lora_adapters.zip
```

Manifest lưu hash từng JSONL; trainer kiểm trước khi dùng. `run_status.json` ghi preparing/training/failed/complete và role đã hoàn tất. Traceback gốc được lưu đầy đủ. Runtime ghi GPU, compute capability, CUDA/PyTorch/bitsandbytes và mapping một GPU. Adapter hash trước/sau được so sánh; nếu không cập nhật hoặc có NaN/Inf, job không được báo thành công. Dùng ít nhất hai optimizer step cho smoke khi warmup bắt đầu từ LR zero.

Tokenization chạy CPU trước model GPU; MAX_STEPS không giới hạn số row tokenize. Console có loading/loaded tokenizer, số row xử lý, model load và bắt đầu train. Ba microbatch đầu có start/returned; debug ghi thêm mỗi 10 microbatch. Accumulation=16 cần 16 microbatch trước optimizer step đầu; logging_first_step bật để không chờ nhiều optimizer step mới thấy loss.

Cảnh báo PAD/BOS/EOS aligned là đồng bộ config ở đầu Trainer.train(), không phải lỗi tokenize. Cảnh báo multiple devices trong Transformers 4.57.6 có thể xuất hiện chỉ vì hf_device_map; xem runtime report để biết GPU thực tế. Khi illegal memory access xảy ra, kết thúc process CUDA lỗi trước lần thử mới; không dùng empty_cache để cố phục hồi process đã lỗi. CUDA_DEBUG chỉ chẩn đoán, không tự sửa kernel.

## Inference và kiểm thử

```bash
python run.py --mode workflow --data-root ../train+public_test \
  --adapter-root /tmp/v8_lora_train/local_lora/adapters \
  --sample-ids-file /tmp/v8_lora_train/data/holdout_ids.json \
  --llm-backend transformers --inference-workers 1

python -m unittest discover -s tests -p 'test_*.py' -v
```

Một inference base dùng ba adapter theo role. Transformers switch adapter dưới lock; vLLM gắn LoRARequest riêng cho từng request trong batch. Metadata kiểm role/base/prompt hash. Không adapter-root thì chạy base đối chứng. [Workflow inference](docs/inference_workflow.md), [sơ đồ training](docs/finetune_workflow.mmd), [validation evidence](docs/validation.json).

Source API tham khảo: [PEFT quantization](https://huggingface.co/docs/peft/v0.18.0/developer_guides/quantization), [Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer), [vLLM multi-LoRA](https://docs.vllm.ai/en/stable/features/lora/). CPU tests không thay thế kiểm thử CUDA; kết quả GPU thực tế được ghi riêng khi job Kaggle hoàn tất.
