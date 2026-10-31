# baseline_v8_finetune — local LoRA cho planner, caller và summarizer

Notebook chính: **[notebooks/kaggle_finetune.ipynb](notebooks/kaggle_finetune.ipynb)**.

Sơ đồ: [local fine-tune](docs/finetune_workflow.mmd) và [workflow inference](docs/workflow.mmd).

Giữ workflow v8: planner chọn danh sách tool + thứ tự một lần → caller chọn arg/value cho từng slot → summarizer quyết định tiếp tục, retry cùng slot hoặc kết luận. Setup này **bỏ hoàn toàn global fine-tune**, train ba LoRA độc lập từ cùng checkpoint `Qwen/Qwen3-4B-Instruct-2507` gốc.

| Role | Target được train | Mẫu âm / phản hồi |
|---|---|---|
| Planner | `{"tools":[...]}` gồm thứ tự và phần tử lặp | Chỉ SFT gold plan; không train reason hoặc task description |
| Caller | call arguments, ask params hoặc blocked | Chosen/rejected values sai; tình huống lỗi thật → target sửa đúng, không học hành động sai |
| Summarizer | `next` trong caller/retry/done/give_up/wait | Wrong effect nhưng env success, plan thiếu bước, lỗi recoverable; chosen/rejected decision |

Prompt/contract và quyền điều phối không đổi so với v8. Không thêm reflector, model lớn để sinh nhãn hoặc vòng tự huấn luyện. Đây là **supervised local fine-tuning với nhãn tạo tự động từ gold và verifier**, không gọi là unsupervised.

## Chạy Kaggle

1. Attach `train+public_test` và toàn bộ thư mục `baseline_v8_finetune`.
2. Mở `notebooks/kaggle_finetune.ipynb`, bật GPU/Internet.
3. Chạy cell dependencies, tạo data, xem manifest và train ba role.
4. Download `local_lora/adapters/` hoặc ZIP adapter cuối notebook.
5. Dùng notebook inference/CLI với `--adapter-root` để chấm holdout.

Defaults phù hợp để bắt đầu thử T4, chưa phải thông số đã tối ưu bằng thực nghiệm:

```python
USE_4BIT = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 1
MAX_STEPS = -1
MAX_SEQ_LENGTH = 8192
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 2e-4
NEGATIVE_WEIGHT = 0.2
```

`USE_4BIT=True` là **4-bit frozen base + LoRA (QLoRA)**; vẫn chỉ cập nhật LoRA. False dùng base FP16/BF16 + LoRA và cần nhiều VRAM hơn. T4 dùng FP16 compute; GPU hỗ trợ BF16 sẽ tự chọn BF16. Không có bước train toàn bộ base weights.

Nên chạy `MAX_STEPS=5–10` trước để kiểm VRAM. Sau smoke test, chọn WORK_DIR mới hoặc overwrite có chủ đích trước khi train đầy đủ. `SKIP_COMPLETED_ROLES=True` sẽ tái sử dụng adapter đã hoàn tất nếu role/base/prompt/data manifest khớp; nó không tự train thêm khi đổi hyperparameters. `RESUME_FROM[role]` có thể trỏ checkpoint để tiếp tục job bị ngắt. Ba role chạy tuần tự trong subprocess riêng để giải phóng GPU giữa các job.

Đây là notebook **training**, chỉ cài `requirements/finetune.txt`, giữ PyTorch CUDA sẵn có của Kaggle. Notebook inference `kaggle_runner.ipynb` có thể cài vLLM trong session riêng. Không tự upload model/data lên Hub.

## Tạo dữ liệu

```bash
python baseline_v8_finetune/prepare_data.py \
  --data-root train+public_test \
  --output-dir /tmp/lifemate_role_data \
  --holdout-ids baseline_v8_finetune/configs/holdout_ids_seed42.json
```

Input model lấy bằng chính `Agent.messages()` của inference. Gold calls là target; replay ToolEnv tạo các response đã xảy ra. Model không nhận label, gold tool list, final_state, future observations hoặc metadata verifier.

- Planner thấy request/context/history/catalog trước khi tương tác tool.
- Caller chỉ thấy schema tool tại slot hiện tại và feedback trước call đó.
- Summarizer thấy response validate/call thật cùng con trỏ đã tăng sau success, giữ nguyên sau failure/ask.
- Gold sys.ask_missing_info chuyển thành slot tool nghiệp vụ của `tool_name`; caller học ask cho params thật, summarizer học wait.

Split theo **episode ID trước mọi augmentation**, mặc định 10% validation. `configs/holdout_ids_seed42.json` chứa 100 ID đã dùng trong report_v8; những ID này không vào train hoặc validation. Tất cả biến thể dương/âm cùng case ở cùng split. Split theo ID chưa phải split theo template/scenario family.

Các file sinh ra:

```text
planner.train.jsonl / planner.validation.jsonl
caller.train.jsonl / caller.validation.jsonl
summarizer.train.jsonl / summarizer.validation.jsonl
manifest.json
split_ids.json / holdout_ids.json
excluded_episodes.json
```

Mỗi dòng gồm `case_id`, `role`, `messages`, `target`, `rejected`, `source`, `loss_scope`, metadata verifier. Chỉ `messages` vào prompt; target/rejected chỉ dùng tính loss.

Có hai trường hợp không tự dựng nhãn:

- `gold tool_calls=[]` không phân biệt được trò chuyện/no-op với hành động không được hỗ trợ. Planner vẫn học tools=[], nhưng bỏ supervision summarizer cho các episode này thay vì gán done/give_up tùy tiện.
- Gold validation/call thất bại thì loại cả episode và ghi lỗi. Không biến gold lỗi thành hành vi đúng.

Dry run trên dữ liệu hiện có: 100 holdout, 144 validation, 1.300 episode train sử dụng; một episode gold bị loại do chat_id không tồn tại. Số lượng chi tiết được ghi trong docs/data_manifest_validation.json và có thể thay đổi khi đổi split/negative settings.

| Role | Mẫu train | Mẫu validation |
|---|---:|---:|
| Planner | 1.300 | 144 |
| Caller | 3.718 | 436 |
| Summarizer | 5.165 | 619 |

Manifest lưu cấu hình augmentation và checksum từng JSONL. Trainer từ chối dữ liệu bị sửa sau khi tạo manifest; notebook kiểm manifest trước khi tái sử dụng adapter đã hoàn tất.

## Mẫu âm được dùng thế nào?

Không đưa một hành động sai vào assistant target rồi gọi đó là SFT mẫu âm.

### Caller

Target dương là arguments/values từ gold. Rejected gồm giá trị sai có cấu trúc: ID khác bản ghi/namespace từ dữ kiện đã có, số tiền/số lượng bị đổi, enum sai nguồn, thời gian lệch hoặc skip không đúng. Các trường hợp gold ask có blocked làm rejected.

Ngoài preference pair, generator chạy counterfactual call trong ToolEnv độc lập. Nếu call sai thất bại, không làm thay đổi state, và chạy call đúng sau đó thực sự khôi phục được state mong đợi, tạo:

1. Summarizer input có response lỗi thật → target retry.
2. Caller input tại cùng slot, có response lỗi thật → target call đúng, rejected là call sai trước đó.

Không dùng kết quả của call sửa đúng làm observation trong input của chính caller đang được train. Các kiểm chứng bằng state hash chỉ nằm trong verifier/metadata offline.

### Summarizer

Các nguồn negative-context:

- **Wrong effect:** đổi value của write cuối, env chấp nhận nhưng state khác call gold ở cùng prefix. Target give_up, rejected done. Không coi ok=true là đủ.
- **Incomplete plan:** bỏ write cuối cần thiết, plan prefix đã hết nhưng state chưa đạt gold. Target give_up, rejected done.
- **Recoverable failure:** lỗi env thật, correction đã được verifier kiểm chứng. Target retry, rejected give_up.
- **Missing write dependency:** nếu bỏ write trước làm bước sau thất bại và không thay đổi state, tạo context thiếu dependency để học give_up/blocked. Chỉ thêm khi dữ liệu thực sự tạo được tình huống này; dry run hiện tại không có episode thỏa bộ lọc.
- Gold trajectory đúng: tiếp tục caller khi còn slot; done khi hết plan; wait sau ask. Rejected chỉ là các route khác mà contract cho phép trong trạng thái đó, nên không chỉ học né JSON sai.

State hash không kiểm hết free-text semantics. Generator không dùng đổi câu văn làm negative sai nghiệp vụ chỉ vì khác cách diễn đạt. Synthetic negatives vẫn cần kiểm chất lượng bằng validation và holdout.

## Loss và LoRA

Mỗi role được load từ `--model` gốc và gọi `get_peft_model()` mới. Chỉ q/k/v/o và gate/up/down projection LoRA trainable; code từ chối nếu có base parameter trainable. Không khởi tạo caller từ adapter planner, cũng không khởi tạo summarizer từ caller.

Loss:

```text
L = L_SFT(chosen) + weight × softplus(margin - beta × (score(chosen) - score(rejected)))
score = mean log-probability trên các token được supervise
```

Đây là **SFT + reference-free pairwise ranking**, không phải DPO và không có reference model. Planner dùng weight=0. Caller/summarizer mặc định 0.2; đặt 0 để ablate ranking nhưng vẫn giữ các positive recovery/counterfactual rows trong SFT.

Prompt và padding luôn có label -100. Planner/caller học completion đầy đủ. Summarizer chỉ học `next` và cấu trúc prefix liên quan:

- Dataset không có nhãn câu kết luận cuối, nên không train SFT vào một answer do code bịa ra.
- Terminal answer trong serialization là placeholder trung tính, **mọi answer token bị mask**.
- Runtime vẫn giữ output answer như v8; nội dung answer dựa vào khả năng base model, chưa được setup này huấn luyện/đánh giá riêng.

Dùng tokenizer chat template thật và kiểm prefix trước khi tạo mask. Không cắt target hoặc mất một phần JSON để vừa context; ví dụ quá max_seq_length bị loại và thống kê theo role/split. Tăng context nếu GPU đủ VRAM thay vì cắt các dữ kiện cần thiết rồi giữ nguyên target.

Loss chỉ materialize logits ở vị trí prediction cần supervise qua `logits_to_keep`, tránh tạo ma trận logits cho toàn bộ prompt dài. Stack Transformers được pin để hỗ trợ Qwen3 và API này. Validation ghi SFT loss, ranking loss và preference accuracy; các metric đó chưa phải accuracy tool-use của workflow.

## CLI train từng role

```bash
python baseline_v8_finetune/train_role.py \
  --role planner --data-dir /tmp/lifemate_role_data \
  --output-dir /tmp/lifemate_lora --max-seq-length 8192

python baseline_v8_finetune/train_role.py \
  --role caller --data-dir /tmp/lifemate_role_data \
  --output-dir /tmp/lifemate_lora --negative-weight 0.2

python baseline_v8_finetune/train_role.py \
  --role summarizer --data-dir /tmp/lifemate_role_data \
  --output-dir /tmp/lifemate_lora --negative-weight 0.2
```

Output:

```text
local_lora/
├── adapters/
│   ├── planner/      # adapter config/weights, tokenizer, role_metadata, metrics
│   ├── caller/
│   └── summarizer/
├── checkpoints/      # optimizer/checkpoint để resume, không phải ba base models
└── training_runs/    # config, GPU, dtype, trainable params và tokenization stats
```

`--full-precision-base` tắt 4-bit. `--resume-from-checkpoint PATH` tiếp tục role tương ứng. `--overwrite-adapter` cho phép thay adapter đã hoàn tất. Không cần/không có global stage.

Nếu gặp `CUDA error: an illegal memory access was encountered`, dòng dropout trong traceback có thể chỉ là nơi lỗi bất đồng bộ được báo lại. Chưa đủ dữ kiện để quy lỗi cho dropout, 4-bit hoặc phiên bản Python.

- Notebook có `CUDA_DEBUG=True`, tương ứng CLI `--cuda-debug`; đặt CUDA_LAUNCH_BLOCKING=1 trước khi process train import PyTorch. Chỉ bật để chẩn đoán, sẽ làm chậm.
- Mỗi job ghi GPU/compute capability, PyTorch/CUDA và phiên bản dependencies vào `training_runs/<role>.runtime.json` trước khi load model. Notebook lưu toàn bộ stdout/stderr vào `<role>.console.log`.
- Sau illegal memory access, process CUDA bị lỗi phải kết thúc trước khi thử lại; nếu kernel notebook cũng bị ảnh hưởng, restart session. Không cố phục hồi bằng empty_cache trong process đã lỗi.
- Chạy lại role lỗi với MAX_STEPS=5 trong thư mục output mới, giữ nguyên dropout/quantization/negative loss để lấy lỗi gốc. Không thay nhiều tham số cùng lúc hoặc cài lại PyTorch/vLLM khi chưa có full trace.
- Diagnostics không thay workflow, loss hoặc sửa tự động kernel. Lỗi CUDA thực tế vẫn cần xác định bằng log từ GPU; test CPU không chứng minh đã sửa lỗi đó.

## Inference với ba adapters

```bash
python baseline_v8_finetune/run.py --mode workflow \
  --data-root train+public_test \
  --adapter-root /tmp/lifemate_lora/adapters \
  --sample-ids-file /tmp/lifemate_role_data/holdout_ids.json \
  --llm-backend transformers --inference-workers 1
```

- Transformers load **một base + ba PEFT adapters**, chọn adapter bằng system prompt của node, khóa việc switch/generate. Không load ba bản Qwen4B.
- vLLM bật multi-LoRA và gắn **LoRARequest riêng cho từng request trong batch**, kể cả batch trộn planner/caller/summarizer. Không dùng một adapter chung cho cả batch.
- Runtime kiểm metadata role, base model và prompt hash. Ba role phải có đủ adapters. Không truyền adapter-root thì chạy base đối chứng và config ghi finetune_method=none.
- sample-ids-file chọn đúng ID holdout, không phụ thuộc seed/limit khác khiến vô tình đánh giá lại train.

Notebook training có cell eval Transformers tùy chọn, mặc định chưa chạy. Với vLLM, dùng [notebooks/kaggle_runner.ipynb](notebooks/kaggle_runner.ipynb), đặt ADAPTER_ROOT và SAMPLE_IDS_FILE rồi chạy trong session inference. Giữ batch/scheduler/contract v8; [docs/inference_workflow.md](docs/inference_workflow.md) mô tả workflow đầy đủ.

## Giới hạn nhãn planner

Planner target hiện là **chuỗi tool thực thi trong gold**, có chuyển ask thành slot nghiệp vụ. Đây là khởi đầu SFT có nhãn hợp lệ, nhưng không biến thành một plan bao phủ mọi nhánh runtime:

- Nếu nhánh chỉ biết sau lookup, gold chỉ thể hiện nhánh đã chạy, không phải đầy đủ cả hai nhánh.
- Không có foreach khi số đối tượng chỉ biết sau lookup, guard hoặc target binding có cấu trúc trong plan.
- Fine-tune không tự gỡ giới hạn plan cố định, quyền retry cùng slot hoặc bổ sung tool thiếu của v8.

Đây là giới hạn thiết kế cần đánh giá tiếp, không giải quyết bằng cách đưa hidden state vào prompt. Không thay đổi workflow để che vấn đề đó trong bản setup train này.

## Kiểm tra implementation

```bash
python -m unittest discover -s baseline_v8_finetune/tests -p 'test_*.py' -v
```

Ngoài các test v8: kiểm split/holdout isolation, label/response đúng thời điểm, rejected không làm target dương, exact loss mask, padding, gradient đẩy rejected xuống, training step thật trên tiny random Qwen3 + PEFT LoRA, save adapter, và mixed-role adapter routing. CPU smoke model nhỏ chỉ kiểm implementation; chưa chạy train Qwen4B trên GPU hoặc chứng minh tăng điểm.

Kết quả kiểm tra hiện tại: **46/46 test pass**. Toàn bộ sáu tập role/split đã tokenize bằng tokenizer Qwen thật, không mẫu nào vượt 8.192 token; các answer token của summarizer đều bị mask. Xem [docs/validation.json](docs/validation.json) và [docs/tokenization_validation.json](docs/tokenization_validation.json).

Tài liệu API dùng khi triển khai: [PEFT quantization/LoRA](https://huggingface.co/docs/peft/v0.18.0/developer_guides/quantization), [Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer), [vLLM LoRA requests](https://docs.vllm.ai/en/stable/features/lora/).
