# Qwen3.5-4B LifeMate baseline

Một model Qwen/Qwen3.5-4B chưa fine-tune chọn tool và arguments từng bước.
Native Qwen3.5 XML tool calls, non-thinking, greedy, FP16 trên GPU0.
Không phụ thuộc các thay đổi local hay runtime Planner/Caller của repo.

## Kaggle

Notebook riêng cho kiểm tra transcript: `kaggle_qwen35_4b_random200.ipynb`.
[Mở trên Kaggle](https://www.kaggle.com/code/cymeu4l0t/lifemate-qwen35-4b-random200).
Mặc định chọn ngẫu nhiên 200 mẫu từ toàn bộ train, không thay thế, seed=42.
Đọc audio_text cho user hiện tại và history; không cần ASR. Tìm dataset
cymeu4l0t/zaloai-dataset trong Input hoặc nạp qua KaggleHub nếu chưa gắn.
Bấm Run All sau khi bật GPU và Internet trên bản Kaggle riêng tư.
Đổi SEED để lấy tập khác; selected_ids.txt, selection.json lưu ID và SHA256 dữ liệu.
Mọi mẫu lỗi vẫn nằm trong mẫu số 200; không loại mẫu theo kết quả model.

Upload kaggle_qwen35_4b.ipynb vào notebook **riêng tư**, bật Internet và GPU T4.
Notebook tự nhúng source, không cần clone repo hay GitHub token.
Gắn dataset cuộc thi riêng tư (thư mục hoặc train+public_test.zip).
Đổi DATA_ROOT/DATA_ZIP nếu có nhiều dataset trùng cấu trúc.

1. PROFILE="smoke", MODE="oracle": chạy 8 mẫu train để kiểm tra tải model và runtime.
2. PROFILE="validation", MODE="oracle": đo agent trên 245 IDs của validation hiện có.
3. PROFILE="validation", MODE="asr": thêm ASR_MODEL_DIR, đo cả audio hiện tại và history.
4. PROFILE="public", MODE="asr": chạy toàn bộ public và xuất submission.jsonl.

ASR_MODEL_DIR chứa encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt của BTC.
ASR CPU sherpa-onnx, không tải model ASR ngoài; không dùng oracle fallback.
Muốn dùng transcript có sẵn, chạy CLI với --transcripts (JSON audio-path -> text).
T4 x2 chỉ dùng GPU0; đây là baseline tuần tự, chưa tối ưu throughput.
Không cài dependency của baseline vào môi trường Transformers 4.x đang train Qwen3.
Không cần flash-attn/FLA: dùng SDPA và đường fallback Gated DeltaNet của Transformers.
Chưa chạy GPU Kaggle trong lần triển khai; smoke là bước kiểm chứng tương thích/VRAM.
Version packages, resolved HF revision, GPU, token counts, thời gian và peak VRAM
được lưu để tái lập. Có thể cố định REVISION bằng commit HF hoặc dùng snapshot local.

## Local CLI

Chạy trong paper_solution với Python >=3.11 và CUDA, môi trường riêng:

~~~bash
pip install -r qwen35_baseline/requirements.txt
python qwen35_baseline/run.py --data-root ../train+public_test --split train --ids qwen35_baseline/validation_ids.txt --mode oracle --limit 8 --output outputs/qwen35_smoke_01
python qwen35_baseline/transcribe.py --data-root ../train+public_test --split train --ids qwen35_baseline/validation_ids.txt --model-dir ../asr_model --output outputs/qwen35_asr.json
python qwen35_baseline/run.py --data-root ../train+public_test --split train --ids qwen35_baseline/validation_ids.txt --mode asr --transcripts outputs/qwen35_asr.json --output outputs/qwen35_validation_asr_01
~~~

## Contract và đánh giá

- Schema từ catalog đúng split, lọc installed_apps; sys.ask_missing_info luôn có.
- History được đưa vào prompt nhưng không replay. Chỉ một ToolEnv/mẫu.
- Không đưa label, final_state hoặc file state vào prompt. State train chỉ đọc
  sau inference để tính diagnostic hash của state gold đầy đủ.
- Mỗi generation một call, tối đa 8 quyết định; mỗi bước đọc observation thật.
- env.validate kiểm schema trước execution. Proposal lỗi được log, không chạy.
  Không repair/retry/reset, không giấu các call đã thực thi.
- Dừng khi model không sinh call, khi gọi sys.ask_missing_info, lỗi hoặc hết budget.
- Không truncate input; prompt quá dài và output thiếu EOS được ghi là lỗi.
- Mọi mẫu vẫn có dòng submission, kể cả mẫu lỗi; lỗi có trace/status riêng.
- score_100 = 100 × trung bình tool-call F1 khi hash đúng (hash sai thì điểm 0).
  Matching một-một tối đa, không xét thứ tự; fuzzy token F1 > 0.6 với Unicode NFC
  và lowercase. Hai danh sách call rỗng có F1=1; không có match thì F1=0.
  Scorer local theo công thức công bố, tokenizer fuzzy có thể khác BTC.
  Không áp latency gate; đây là baseline transcript, không phải điểm end-to-end audio.
- hash_accuracy / hash_accuracy_percent: tỷ lệ hash đúng toàn bộ state canonical.
  tool_call_f1 báo riêng, chưa gate bởi hash; exact_call_sequence_match là diagnostic.
- Không giới hạn agent ở các tool từng có nhãn train: public có tool mới.
- Output mới phải dùng thư mục mới. Baseline không tự resume và không reset mẫu.
- Artifact gồm config.json, submission.jsonl, traces.jsonl, report.json.
  Không commit dữ liệu cuộc thi, transcript, prompt thực tế hoặc model weights.

## Kiểm tra và tạo lại notebook

~~~bash
python -B qwen35_baseline/test_agent.py
python -B qwen35_baseline/test_scoring.py
python -B qwen35_baseline/make_notebook.py
~~~

Tests dùng fake generator/env để kiểm observation, history, ASR, schema rejection,
missing-info, malformed calls và step budget. Không chứng minh chất lượng model/GPU.

Nguồn giao thức:
https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja
https://huggingface.co/docs/transformers/model_doc/qwen3_5
https://challenge.zalo.ai/portal/lifemate (Evaluation)
