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

## Sửa sau batch seed 1121

Giữ nguyên planner → argument selector → value selector → checker → tool. Các thay đổi xử lý ba nhóm lỗi trong `report_v2_newseed.zip`:

1. **Contract goal:** inline `call/ask.goals` chỉ thêm goal mới; không dùng quota sửa/xóa. Nếu model viết lại ID cũ, code giữ goal canonical và ghi `ignored_goal_updates`; không bác call chỉ vì phần plan phụ. Inline goals sai format được ghi `goal_warning`, call vẫn phải qua mọi kiểm tra tool/value. `revise` sửa/xóa goal vẫn cần lý do và giới hạn hai lần; revise chỉ nối thêm goal không tốn quota. `plan_revisions` từ phiên bản này đếm sửa/xóa, không đếm thêm goal, nên không so trực tiếp counter này với batch cũ.
2. **Độ chắc chắn của hint:** `value_hints` là gợi ý, `trusted_value_hints` là tập nhỏ đủ điều kiện cho kiểm cưỡng chế/điền từ hint khi model báo thiếu. Query và suy ghép ngữ nghĩa không được dùng như đáp án chắc chắn. Province vé số chỉ rút trích khi có tên địa phương trong từ vựng bảo thủ sau dấu hiệu vé/đài/tỉnh; tên người viết hoa không đủ. Từ vựng không phủ mọi địa phương/cách viết. Forecast chỉ suy ngày mai với dấu hiệu thời gian rõ, không coi tên người Mai là ngày mai. Nếu vé số tham chiếu một người có nhiều chuyến công tác mà chưa rõ ngày, code báo `ambiguous_target` cho date.
3. **Ambiguity và history trước write:** chặn tự chọn giữa các liên hệ cùng tên khi chưa có quan hệ/số/ID đủ phân biệt; cho phép giải tham chiếu theo lượt user trước đã chỉ định quan hệ. Chặn mở rộng một người nhận thành nhiều địa chỉ khi user chỉ chọn một mối. Khi yêu cầu chuyển khoản có điều kiện “chưa chuyển thì chuyển”, đối chiếu giao dịch thành công trong history theo tài khoản, người nhận, số tiền; không chuyển lại. Bằng chứng cũ được đưa vào observations với `origin=history`, không ghi thành executed call mới, không dùng tool slot; planner có thể dùng index đó để done.

Trace bổ sung `goal_warning`, `ignored_goal_updates`, `history_evidence`; bằng chứng history giữ result thành công, không tự suy từ lời assistant. Chưa có bộ giải ambiguity tổng quát: các guard trên chỉ phủ quan hệ/liên hệ, một người nhận, ngày công tác và giao dịch có điều kiện đã xác định; không tự cấm mọi giao dịch lặp có chủ ý.

Kiểm tra: **106 test qua**; replay checker trên **100 mẫu / 230 gold call** của seed 42 và 1121 không chặn gold call. Thông tin phiên bản trong [validation_after_seed1121.json](validation_after_seed1121.json). Label chỉ dùng trong kiểm tra, không đưa vào inference. Chưa chạy lại Qwen trên Kaggle; tính đủ goal và suy giá trị ngữ nghĩa vẫn cần đo thực nghiệm. Batch mới phải có fingerprint mới; kết quả zip seed 1121 vẫn thuộc code cũ `4e1ae5…`, proxy 24,8%, agent trung bình 51,9s.

## Sửa sau batch 50 mẫu (2026-10-30)

Cốt lõi vẫn gồm ba node Qwen và checker bằng Python. Không đọc label/final_state trong inference, không thêm reviewer, không đổi score_proxy.

- **Ba prompt:** viết lại theo Context/Objective/Style/Tone/Audience/Response. Planner phân biệt kết quả cuối với lookup, điều kiện thật với ngưỡng tự đoán, câu sửa cuối với tác vụ cũ; giữ đúng email/Zalo, ghi nhận quá khứ/tạo lịch. Argument selector chọn đủ thuộc tính user yêu cầu và tránh optional chỉ lấy từ trạng thái. Value selector giữ đúng bản ghi, thời gian tương đối, số liệu và nội dung cần chép.
- **Lookup:** ưu tiên tên tool trong description của field x-ref; nếu không có thì tìm read trả đúng collection. Tài khoản lấy từ `banking.get_balance`, thiết bị từ `smart_home.get_status`, món từ menu đúng nhà hàng. Không gợi ý một read cần ID chưa có chỉ vì tên app giống nhau.
- **Bằng chứng:** module mới `evidence_rules.py` dựng `reference_candidates` từ tool results thành công trong history/observations. Kiểm namespace x-ref, loại thiết bị và `items[].item_id` theo restaurant_id. Chặn lựa chọn tùy ý giữa nhiều gói storage monthly khi user chưa phân biệt. ID user cung cấp trực tiếp được cho phép, nhưng chưa chứng minh tồn tại cho tới khi tool xử lý.
- **Khôi phục:** nếu planner lặp tool đang thiếu ID dù feedback đã có lookup, code chuyển lượt call đó sang read được gợi ý và ghi `recovery_override`. Với trường hợp đối tượng mơ hồ không còn lookup hoặc volume thiếu mức đích, call lặp được chuyển thành ask hợp lệ. Code không tự tạo write mới hoặc tự đánh dấu task hoàn tất.
- **Kết thúc:** planner nhận `execution_status` có index/kind/succeeded, observations cũng có trạng thái rõ; lỗi done trả `evidence_options`. Vẫn kiểm mọi goal và write đã thực thi; không tự sửa index dựa trên phỏng đoán ý nghĩa goal.
- **Value:** nguồn trả hóa đơn được chuẩn hóa bằng hint đã so số dư với đúng hóa đơn; có trace `normalized_source`. Thêm hint ngày mai cho forecast và nội dung cảnh báo mới nhất của đúng location khi user yêu cầu chép nguyên. Bỏ brightness optional nếu user chỉ bật đèn, không yêu cầu mức sáng.
- **Cache:** read chỉ được gọi lại sau write thành công cùng app; tạo note không vô hiệu hóa cache cảnh báo. Đây là policy theo ToolEnv hiện tại; không tự coi write app khác đã làm thay đổi app đang đọc.
- **Đóng gói:** Kaggle kiểm có `evidence_rules.py`; fingerprint workflow tính thêm module này. Khi upload phải upload toàn bộ baseline_v2.

Kiểm tra: **97 test qua**, bao gồm các regression namespace, loại thiết bị, nested menu, lookup, cache, ngày mai và issue→ask. Replay **121 gold calls của 50 mẫu** qua checker ID mới: **0 call chuẩn bị chặn**. Đây là kiểm tra checker với label ở test, không dùng label khi inference và không phải đánh giá năng lực Qwen.

Batch 50 trước sửa có proxy **29,67%**, 11/50 điểm 1, hash đúng 21/50; agent trung bình **66,59s**, p95 **131,41s**, chưa tính ASR. Hash phiên bản batch đó là `feb8ddf499aae6b3d63fe2117eceeb70fdb9510d8b2a3f1eca09528401e02444`. Phân tích từng mẫu: `../lifemate_baseline_v2/workflow_gold/error_analysis_50.md`. **Chưa chạy lại model trên Kaggle cho phiên bản mới.**

Giới hạn còn lại: tính đủ goal, giải điều kiện và suy thời gian/nội dung chưa có bộ chứng minh ngữ nghĩa tổng quát. Candidate đúng collection không luôn chứng minh đúng người; kiểm ambiguity storage hiện chỉ phủ gói monthly có dung lượng GB/TB. Prompt vẫn có thể bị model làm sai. Hint/guard không được coi là bằng chứng mọi mẫu đã được sửa; cần chạy lại cùng 50 IDs, seed, model và tool budget.

## Kết quả thực nghiệm làm mốc (trước các chốt mới)

Batch 30 mẫu trước đó (đã bị batch 10 mẫu ghi đè trong `lifemate_baseline_v2/workflow_gold`) có proxy trung bình **10,28%**, hash đúng **9/30**, proxy dương **5/30**, không có mẫu proxy hoàn hảo. Có **17/30** mẫu thiếu tool, **23/30** có tool thừa; agent mất trung bình **55,1 giây/mẫu**, p95 **115,5 giây** khi dùng transcript chuẩn, chưa tính ASR. Không còn artifact của batch 30 ở đường dẫn này và các file cũ không ghi mã phiên bản prompt/code; **không dùng chúng để kết luận hiệu quả của thay đổi mới**. Cần chạy lại cùng sample IDs, seed, giới hạn và phần cứng, rồi so `score_proxy`, hash, lỗi, thời gian và trace.

Batch 10 mẫu ngày 2026-10-29 sau khi thêm `task_plan` có **0/10 tool call** và **10/10 `validation_no_progress`**: Qwen luôn xuất `call` thiếu `goals`. Batch kế tiếp đạt **proxy trung bình 10%, 1/10 mẫu hoàn hảo, 2/10 hash đúng**, agent trung bình **68,2 giây/mẫu**. Các artifact này đã bị batch mới ghi đè.

Batch 30 mẫu trước đó (đã bị batch 50 ghi đè) (mốc **trước các sửa đổi trong mục Workflow đang chạy ở trên**) có **proxy trung bình 22,78%, 5/30 hoàn hảo, 12/30 hash đúng**, 21/30 thiếu tool; agent trung bình **47,6 giây**, p95 **96,1 giây**, chưa tính ASR. Có 12 `issue_no_progress`, 10 `validation_no_progress`, 7 `done`, 1 `ask_missing_info`; trace cho thấy selector đánh giá sai read phụ trợ, plan thiếu goal, Qwen bịa ID/email, và lặp read/write đã xong. Batch 30 dùng `workflow_sha256=c2ef07384cb868351c0a5892b9131638559fba0aa3c7a0fa0d1bec74e865504d`. **Chưa có batch Qwen mới cho code vừa sửa**; test scripted không thay thế kết quả thực nghiệm. Khi chạy lại, so cùng 30 ID/seed/model/phần cứng và hash mới.

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
