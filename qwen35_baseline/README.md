# Qwen3.5-4B LifeMate baseline — vLLM

Một model Qwen/Qwen3.5-4B chưa fine-tune, native XML tool calling, non-thinking,
greedy FP16. vLLM offline LLM.generate, không cần API server/OpenAI SDK.

## Kaggle

[Mở notebook riêng tư](https://www.kaggle.com/code/cymeu4l0t/lifemate-qwen35-4b-random200).
File local: kaggle_qwen35_4b_random200.ipynb. Giữ cấu hình user hiện tại: 100 mẫu
ngẫu nhiên, seed=42, BATCH_SIZE=4, MAX_INPUT=16384. Đặt SAMPLE_SIZE=200 nếu cần.
Dùng audio_text cho yêu cầu hiện tại và history; không chạy ASR, không dùng label
trong prompt. Dataset riêng tư được tìm trong Input hoặc nạp qua KaggleHub.

Dừng session HF cũ và bắt đầu session mới, bật Internet + GPU T4 x2, rồi Run All.
Notebook nhúng source, không cần GitHub token. vLLM được pin 0.33.0; pip giải
quyết phiên bản torch/CUDA theo package, không dùng lại môi trường training Qwen3.
Chưa chạy GPU để xác nhận kernel/driver Qwen3.5 trên Kaggle T4 trong lần chuyển này.

## Cấu hình bộ nhớ

- tensor_parallel_size=2: chia trọng số trong từng layer trên hai GPU, thay cho
  device_map chia layer; không tạo hai replica.
- dtype=float16; language_model_only=True, bỏ tải vision encoder cho transcript.
- gpu_memory_utilization=0.80 mỗi GPU, chừa headroom; không bảo đảm hết OOM.
- enable_chunked_prefill=True, max_num_batched_tokens=1024: xử lý prefill theo
  từng phần, không cắt prompt hay giảm context xuống 1024.
- max_num_seqs=2: batch 4 được scheduler xếp lịch, tối đa 2 sequence đồng thời.
- enforce_eager=True: tắt CUDA graphs/compile để giảm overhead bộ nhớ.
  Đây là tùy chọn CUDA execution, khác với format enforcer.
- disable_custom_all_reduce=True: dùng collective chuẩn thay custom all-reduce.
- enable_prefix_caching=False: khởi đầu không dùng prefix cache cho hybrid state.
- max_model_len=MAX_INPUT+MAX_NEW_TOKENS (17152 mặc định). Input vượt MAX_INPUT
  báo lỗi, không truncate. Không giảm output budget âm thầm.

GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS chỉnh ở config cell.
Nếu cần giảm concurrency, dùng MAX_NUM_SEQS=1. Ghi nhận engine_settings và package
versions trong config.json. observed_gpu_used_gib_per_gpu lấy từ driver sau load
và các lượt generate hoàn tất, gồm bộ nhớ worker; không phải peak tức thời.
Không dùng torch.max_memory_allocated ở process cha để đại diện worker vLLM.

## Không có format enforcer

SamplingParams(structured_outputs=None), không grammar/JSON schema/guided decoder,
không tool_choice required và không gọi LLM.chat để tự ép structured tool output.
Tools vẫn được đưa đầy đủ vào chat template; model sinh text tự do. Prompt vẫn
hướng dẫn XML, nhưng không khóa token bằng schema. Stop token im_end kết thúc lượt
chat; không sửa XML hoặc tự điền arguments.

Parser lấy tool_call đầu tiên ngoài thinking, bỏ phần tự sinh tiếp. Mỗi call được
kiểm schema bằng env.validate rồi thực thi trên ToolEnv riêng của mẫu. Call không
hợp lệ báo lỗi, không tìm call khác thay thế. Kết quả JSON thật được thêm với
role=tool cho lượt sau. Call bên trong thinking và observation model tự bịa không
được thực thi hoặc thêm vào history. Raw output vẫn lưu trong traces.jsonl.
Generation hết token budget mà chưa stop được ghi lỗi, không chạy action cắt dở.
Tham số schema boolean nhận true/false và True/False; chỉ chuẩn hóa Python casing
cho boolean đơn, không thay nội dung string và không dùng eval.

## Nghiệp vụ và scoring

SYSTEM trong agent.py giữ quy tắc history/sửa lời, tham chiếu, điều kiện/ngoại lệ,
chọn đối tượng, ngày giờ theo context, giữ thời lượng lịch, required/x-one-of/x-ref,
phân biệt ok và done, làm phần độc lập trước khi hỏi thiếu thông tin. Kind/Returns
được bổ sung vào description; parameters giữ nguyên. Tool mới của public không bị
lọc theo nhãn train. Không đọc state trực tiếp cho inference, không replay history.

Prompt yêu cầu gọi meta tool thay vì hỏi bằng lời rồi dừng, có ví dụ XML minh họa.
Lookup rỗng/nhiều ứng viên chưa phân biệt được phải báo ID thiếu sau khi làm phần
độc lập; không lặp lookup y nguyên và không gửi write thiếu required/x-one-of.
Đây là thay đổi chỉ dẫn, cần chạy model để đo lại; runtime không tự đoán missing-info.
sys.ask_missing_info kết thúc episode. Mỗi mẫu một ToolEnv, tối đa MAX_STEPS bước,
không reset/retry ngầm. Mẫu lỗi vẫn tính trong mẫu số SAMPLE_SIZE.
score_100 là F1 tool call gated bởi state hash; hash_accuracy báo riêng. Scorer local
matching một-một, fuzzy token F1 > 0.6 có thể khác tokenizer BTC; không latency gate.
Trọng số model không lượng tử hóa trong lần chuyển backend này; chất lượng vLLM
cần đo lại, không suy ra từ kết quả HF.

Artifacts: config.json, submission.jsonl, traces.jsonl, report.json, selected_ids.txt,
selection.json và ZIP. Không công khai output có dữ liệu cuộc thi.

## CLI và kiểm tra

```bash
pip install -r qwen35_baseline/requirements.txt
python qwen35_baseline/run.py --data-root ../train+public_test --split train --mode oracle --num-gpus 2 --batch-size 4 --max-input 16384 --gpu-memory-utilization 0.80 --max-num-seqs 2 --max-num-batched-tokens 1024 --limit 8 --output outputs/qwen35_vllm_smoke
python -B -m unittest discover -s qwen35_baseline -p 'test_*.py'
python -B qwen35_baseline/make_notebook.py
```

PROFILE=smoke chạy 8 mẫu; validation dùng 245 IDs cố định; random200 dùng
SAMPLE_SIZE mẫu; public chạy toàn bộ public_test và bắt buộc MODE=asr.
ASR dùng model Zipformer ONNX BTC qua transcribe.py trên CPU; đặt ASR_MODEL_DIR
chứa encoder.onnx, decoder.onnx, joiner.onnx và tokens.txt. CLI --transcripts nhận
JSON audio-path -> text, không dùng oracle fallback.

Tests CPU mock vLLM để kiểm cấu hình TP, scheduler, không format enforcer, input
không padding, lỗi từng mẫu, observation và scoring; không xác nhận VRAM/kernel.

Nguồn: https://docs.vllm.ai/en/latest/configuration/optimization/
https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md
