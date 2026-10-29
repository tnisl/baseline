# LifeMate baseline v2

This folder is an independent experiment. It keeps PhoWhisper-medium, Qwen3-4B-Instruct-2507, vLLM/Transformers adapters, and the three v1 experiment modes. It writes to a separate output folder, so v1 results remain available for comparison.

## Workflow đang chạy

`run.py` dùng [agent_pipeline.py](agent_pipeline.py). Cốt lõi vẫn là **planner → argument selector → value selector → kiểm bằng code → gọi tool**; tối đa ba lượt Qwen cho mỗi tool call. Không gọi reviewer. Ảnh mới: [workflow.png](workflow.png), [workflow.svg](workflow.svg); tạo lại bằng `generate_workflow.py`.

1. **Planner:** lần đầu lập goal theo **kết quả cuối cùng người dùng muốn**, mỗi goal `{id,task,when,kind}` với `kind` là `read` hoặc `write`. Các hành động và nhánh điều kiện độc lập phải có goal riêng; read/search chỉ để chuẩn bị write không thành goal. Code lưu `task_plan`; planner chọn `call`, `ask`, `revise` hoặc `done`. Nếu planner phát hiện goal mới trong call/ask sau đó, code nối goal mới vào plan; sửa/xóa goal cũ vẫn phải dùng `revise`. `done` còn kiểm mọi write thành công đã được gắn với một goal. Catalog planner được rút gọn, selector nhận schema đầy đủ.
2. **Argument selector:** code thêm mọi `required`, Qwen chỉ chọn nhóm `x-one-of` và optional. Selector chỉ đánh giá *bước đang chọn*, nên tool read/list/search phục vụ write không bị coi là `unsupported_tool`. Với read không có field bắt buộc, code có thể bỏ issue `unsupported_tool` sai và đọc rộng; optional filter không có value thì cũng đọc rộng. Code bỏ `since` khi user chỉ nói “hôm nay”, bỏ `type` của `list_bills` khi request không chỉ định loại hóa đơn, và tiếp tục bỏ `delay_minutes`/`days` không thuộc tác vụ. Bỏ qua Qwen node nếu tool không có field bổ sung.
3. **Value selector:** nhận `selected_arguments` và `value_hints` xác định từ dữ liệu đã thấy, điền `arguments` + `value_reasons` hoặc báo `issue`. Hint hiện bao gồm ID/nguồn của một hóa đơn khớp loại và số dư, loại giá vàng, liên hệ đã thấy, dữ kiện vé số đang được giải tham chiếu, chủ đề podcast rõ trong request và giờ lịch thuốc khi có quy tắc đèn phù hợp. Với read filter optional, code có thể dùng hint chắc chắn hoặc bỏ filter để đọc toàn bộ khi model báo thiếu. Bỏ qua node nếu không có field.
4. **Kiểm bằng code:** kiểm contract, required, x-one-of, schema và các ngữ nghĩa trong [pipeline_rules.py](pipeline_rules.py): ID/số điện thoại/email người nhận phải xuất hiện trong dữ liệu request/context/history/tool result; `tv.control(action=volume)` cần mức đích cụ thể; nguồn thanh toán phải khớp nhánh đã xác định. Chặn `ewallet.pay_qr` khi request là trả hóa đơn và không nói QR, chặn `notes.append` khi yêu cầu tạo ghi chú mới. Số điện thoại liên hệ đã thấy được đổi sang `contact_id` khi phù hợp. Chặn read trùng arguments và write lặp. Feedback thiếu tham chiếu có `suggested_lookup`; feedback gọi lặp có `observed_result_index` nếu tìm thấy. Mọi lỗi quay về planner trước `env.call`.
5. **Hỏi/kết thúc:** `ask` bị bác khi giá trị đã xác định, field optional không cần hỏi, hoặc vẫn có read/search cho tham chiếu liên hệ. `done` phải map mỗi goal sang một `tool_observations[index]` thành công và đúng `kind`; `skipped_goals` chỉ dành cho nhánh điều kiện không áp dụng, và có thể bỏ key này khi không skip. Sau khi dùng hết lượt tool, planner vẫn có thể `done`.

Planner output lượt đầu phải có `goals` trong cùng action. Các lượt sau dùng `task_plan`; call/ask có thể kèm `goals` chỉ gồm goal mới, còn `revise` có format `{"type":"revise","goals":[...],"removed_goals":{},"reason":"..."}`. `done` có dạng `{"type":"done","completed_goals":{"<id>":<tool_result_index>},"skipped_goals":{}}`; `done` ngay lượt đầu chỉ có dạng `{"type":"done","goals":[]}` khi không có việc cần làm. `task_plan` được ghi trong `traces.jsonl`; `workflow_sha256` của prompt và code active được ghi trong `run_config.json` và summary để so các lần chạy. Kết quả tool cũ hơn tám lượt giữ index, tên, arguments và trạng thái thành công dạng rút gọn.

**Giới hạn:** code không thể tự chứng minh planner đã liệt kê đủ mọi goal hay kết quả tool có đúng *ý nghĩa* goal chỉ từ tên và `kind`. Một goal chưa từng được planner nhắc đến vẫn có thể bị bỏ sót; điều kiện skip hoặc lý do xóa goal khi `revise` cũng do planner diễn giải. Literal grounding chứng minh ID/email/số đã xuất hiện, chưa chứng minh đầy đủ quan hệ đúng người trong mọi trường hợp; phép tính, diễn đạt tin nhắn và thời gian suy luận cũng chưa được kiểm ngữ nghĩa tổng quát. Read rộng khi optional filter thiếu value có thể tăng kích thước tool result. Không có Qwen reviewer; `unverified_arguments` vẫn là số field chưa được xác minh ngữ nghĩa toàn diện. Legacy reviewer trong `agent.py`/`node_reviews.py` không nằm trong pipeline active.

## Kết quả thực nghiệm làm mốc (trước các chốt mới)

Batch 30 mẫu trước đó (đã bị batch 10 mẫu ghi đè trong `lifemate_baseline_v2/workflow_gold`) có proxy trung bình **10,28%**, hash đúng **9/30**, proxy dương **5/30**, không có mẫu proxy hoàn hảo. Có **17/30** mẫu thiếu tool, **23/30** có tool thừa; agent mất trung bình **55,1 giây/mẫu**, p95 **115,5 giây** khi dùng transcript chuẩn, chưa tính ASR. Không còn artifact của batch 30 ở đường dẫn này và các file cũ không ghi mã phiên bản prompt/code; **không dùng chúng để kết luận hiệu quả của thay đổi mới**. Cần chạy lại cùng sample IDs, seed, giới hạn và phần cứng, rồi so `score_proxy`, hash, lỗi, thời gian và trace.

Batch 10 mẫu ngày 2026-10-29 sau khi thêm `task_plan` có **0/10 tool call** và **10/10 `validation_no_progress`**: Qwen luôn xuất `call` thiếu `goals`. Batch kế tiếp đạt **proxy trung bình 10%, 1/10 mẫu hoàn hảo, 2/10 hash đúng**, agent trung bình **68,2 giây/mẫu**. Các artifact này đã bị batch mới ghi đè.

Batch 30 mẫu mới nhất trong `lifemate_baseline_v2/workflow_gold` (mốc **trước các sửa đổi trong mục Workflow đang chạy ở trên**) có **proxy trung bình 22,78%, 5/30 hoàn hảo, 12/30 hash đúng**, 21/30 thiếu tool; agent trung bình **47,6 giây**, p95 **96,1 giây**, chưa tính ASR. Có 12 `issue_no_progress`, 10 `validation_no_progress`, 7 `done`, 1 `ask_missing_info`; trace cho thấy selector đánh giá sai read phụ trợ, plan thiếu goal, Qwen bịa ID/email, và lặp read/write đã xong. Batch 30 dùng `workflow_sha256=c2ef07384cb868351c0a5892b9131638559fba0aa3c7a0fa0d1bec74e865504d`. **Chưa có batch Qwen mới cho code vừa sửa**; test scripted không thay thế kết quả thực nghiệm. Khi chạy lại, so cùng 30 ID/seed/model/phần cứng và hash mới.

## Experiments

- `asr`: audio → PhoWhisper transcript; WER/CER and ASR timing.
- `workflow`: gold `audio_text` → Qwen agent and ToolEnv.
- `baseline`: PhoWhisper transcript → the same agent and ToolEnv.
- `all`: run all three on identical sampled train IDs.

## Kaggle

Attach the extracted `train+public_test` folder and `baseline_v2` folder as Kaggle inputs. Enable GPU and Internet for initial model downloads. Open [kaggle_runner.ipynb](kaggle_runner.ipynb), set `DATA_ROOT` if autodetection cannot find your data, and run the cells. Use `MODE="workflow"` to test the agent without ASR first, then `MODE="all"`. `LIMIT=0` means all train samples. Set `SAMPLE_ID="train_1703"` to run only that ID with the gold transcript; the notebook uses `run_one.py` and writes to a separate `single/train_1703` folder.

The notebook installs `requirements-vllm.txt` for the vLLM backend. Its default `TRITON_ATTN` and `float16` settings support Kaggle Tesla T4 and avoid the FlashInfer linker/BF16 errors seen in v1. PhoWhisper reads PCM16 WAV directly rather than using the Transformers ASR pipeline, avoiding a TorchCodec dependency. ASR and Qwen are loaded sequentially to limit peak VRAM.

Command line equivalent:

```bash
python -m pip install -r baseline_v2/requirements-vllm.txt
python baseline_v2/run.py --mode workflow --data-root train+public_test --limit 50 --output-dir /kaggle/working/lifemate_baseline_v2
python baseline_v2/run.py --mode all --data-root train+public_test --limit 50 --output-dir /kaggle/working/lifemate_baseline_v2
python baseline_v2/run_one.py train_1703 --data-root train+public_test
```

`run_one.py` runs one train ID with the gold transcript and writes to `lifemate_baseline_v2/single/<ID>/workflow_gold/` by default. Use `--output-dir` for another location; the same model and limit flags as `run.py` are accepted.

Useful limits: `--max-tool-calls 10`, `--max-decisions 24`, `--max-repairs 3`, `--max-no-progress 2`, `--max-prompt-tokens 11500`, `--max-new-tokens 512`. The default vLLM model context is 12288 tokens. Use `--llm-backend transformers` with `requirements.txt` for a separate backend comparison.

Qwen defaults to `--llm-temperature 0.7` on both backends, with `top_p=0.8` and `top_k=20`. Set `--llm-temperature 0` to use greedy decoding. `--seed 42` controls sample selection, Python hashing, Python/NumPy/PyTorch RNGs, and the Qwen sampler. PhoWhisper uses non-sampling decoding. The runner sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disables cuDNN benchmarking, requests deterministic PyTorch algorithms where available, and disables vLLM V1 multiprocessing for deterministic offline scheduling. Direct `run.py` invocation restarts once if the process hash seed is not already correct; the Kaggle notebook passes the environment explicitly. Cached ASR predictions are reused only for the same model, batch size and seed.

The seed, settings, and package versions are recorded in `run_config.json`; workflow and ASR summaries also record the seed. Compare outputs on the same hardware with the same package versions, model revision, data, mode and inference settings. A fixed seed with temperature 0.7 makes sampling repeatable under those conditions, but does not guarantee identical results across different GPUs or software versions. Timing measurements will still vary. PyTorch's deterministic setting uses warning mode so unsupported operations do not stop a Kaggle run; treat any such warning as a reproducibility limit.

## Results

The output directory defaults to `/kaggle/working/lifemate_baseline_v2` on Kaggle. It contains:

- `asr/`: predictions, WER/CER cases, summary, report, chart.
- `workflow_gold/` and `baseline_asr/`: predictions, per-sample cases, traces, summary, report, chart.
- `comparison.csv`, `comparison_summary.json`, and `run_config.json`.

The active pipeline reports `selector_issues`, `plan_revisions`, grounding/semantic rejections, tool failures, and no-progress events. Legacy reviewer columns remain zero for output compatibility. `score_proxy` is a local approximation; `state_hash_match` compares exact ToolEnv hashes. Neither is the official leaderboard score. Compare runs only with the same seed, limit, model settings and `workflow_sha256`.

Run the focused local checks with `python -m unittest discover -s baseline_v2 -p 'test_*.py'`. They cover the active pipeline, issue handoff, and legacy behavior without loading Qwen weights. They do not measure real Qwen accuracy or GPU reproducibility. Evaluate those by running `workflow` or `baseline` twice on the same Kaggle GPU and comparing `predictions.jsonl` and `traces.jsonl` after excluding latency fields.
