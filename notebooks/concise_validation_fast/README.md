# Full validation without token enforcement

Kaggle: https://www.kaggle.com/code/anartt/lifemate-concise-validation-fast

Select **T4 ×2**, Internet on, then Save & Run All yourself. QuickSave only updates the source.

Uses the fixed245 validation IDs (1300/245 train/valid split), Qwen3-4B-Instruct-2507 and completed local adapters Planner v4 best500 / Caller v5 best371. English training prompts and TypeScript tool contracts stay unchanged. Supplied Zipformer ASR processes current/history audio; no transcript oracle, finetuning, public-test run or submission.

Two independent vLLM workers process123/122 disjoint episodes with batch4 per GPU. FP16, context8192, LoRA rank16, CUDA Graphs enabled. Prefix caching is disabled on T4/SM75 due to the observed pinned-runtime kernel failure and enabled on SM80+. Chunked prefill stays disabled. The first graph capture adds startup time.

Both roles use `guided_decoding=None`; no enforcer filters tokens. JSON/schema validation and tool argument validation remain. One initial generation plus at most one revision per node per decision; Planner shares its revision between repair and optional completion review. Exhausted invalid output ends with `format_error`. The old synthetic probes that relied on forced JSON output are skipped, explicitly recorded in `preflight.json`.

Each episode allows **7 decisions and7 executed tool calls**. Prevented repeated reads still consume decisions. The Summarizer route terminates without a model generation. This limits resource consumption; it does not fix semantic reasoning or guarantee recovery from a repeated plan.

Pinned vLLM requires LMFE as a transitive package dependency, so its lock entry remains for installation consistency. Runtime does not attach it to the sampler.

Output: `/kaggle/working/concise_validation/concise_two_gpu_unguided_limits7_valid245/validation/`.

- `report.json`: local approximate contest score, tool F1 and final-state accuracy, recomputed over all245 predictions.
- `submission.jsonl`, `traces.jsonl`: combined predictions and raw generations/errors/tool results.
- `gpu0/`, `gpu1/`: worker logs, reports and batch caches.
- ASR transcripts and report.

Gold labels are used only by the scorer after prediction. Checkpoint, model and prompt provenance are checked before loading. Source uses a frozen core commit plus embedded runtime; rebuild with `python make_notebook.py` after editing `runtime/`.

Preserve the whole run directory for resume and set `RESUME_DIR` to its mode folder containing `config.json` and `validation/`. Changed settings require a new run name. An interrupted active batch is not silently replayed. A ten-hour budget pauses only at batch boundaries; no per-sample timeout.

CPU checks: `python checks.py`. Historical guided-run results remain in the external local archive; the current unguided7/7 full run has not been executed by this cleanup.
