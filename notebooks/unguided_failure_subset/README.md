# Four failing validation samples without LMFE

Kaggle: https://www.kaggle.com/code/anartt/lifemate-unguided-failure-subset

Open the notebook, select **T4 ×2** and enable Internet, then **Save Version → Save & Run All** yourself. One T4 runs the LLM with batch4; the other GPU is unused because this diagnostic has only four workflows. Publishing uses QuickSave and starts no session.

Selected IDs from the unchanged `split.txt`: `train_0017`, `train_0066`, `train_0088`, `train_0001`. These were failures in fast validation v5 and are deliberately selected diagnostics, not a representative validation score.

Runs supplied Zipformer ASR on current and history audio, then Qwen3-4B-Instruct-2507 + the existing Planner/Caller local adapters. No finetuning. English prompts, TypeScript schemas, greedy decoding, token limits (Planner320, Caller512), context8192, graph/cache compatibility stay as in the prior fast validation runtime. Rollouts are now limited to **7 decisions /7 executed tool calls**; prevented duplicate reads still consume a decision. Summarizer route ends without model generation.

**Both nodes use `guided_decoding=None`.** LMFE may remain installed as a pinned vLLM dependency, but is not attached to sampling. JSON parsing, schema checks, tool argument validation and one repair per node per decision remain active. The old synthetic preflight relied on grammar-constrained output and is explicitly skipped. Invalid output after the allowed repair ends with `format_error` without executing that invalid call.

Labels and prior predictions are never included in model messages. Gold labels are read by the scorer after all four predictions; `baseline_subset.json` is consumed only by final reporting. Prompt and checkpoint provenance are checked before loading. The frozen core commit plus embedded runtime is used rather than whatever branch happens to be latest.

Output: `/kaggle/working/unguided_failure_subset/unguided_failure_subset_v2_limits7/validation/`.

- `comparison.md` / `comparison.json`: per-ID old/new score, tool F1, state match, termination, repetition, thought length and actual tool calls.
- `report.json`: aggregate approximate contest metrics and JSON/schema violation counts.
- `traces.jsonl`: raw Planner/Caller responses, errors, repairs and tool observations.
- `submission.jsonl`, ASR transcripts/reports and completed batch cache.

Historical v5 batch composition and two-GPU concurrency differ; timing is not a controlled speed comparison. Repetition counts use whitespace six-grams occurring at least three times within one raw generation, including malformed JSON. Raw traces remain the evidence for diagnosis.

CPU checks: `uv run --no-project --python 3.12 --with jsonschema python checks.py`. Rebuild: `python make_notebook.py`. GPU inference is left for the user; CPU checks do not establish whether these four cases improve.
