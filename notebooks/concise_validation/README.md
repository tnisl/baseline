# Concise Planner/Caller validation

Inference-only evaluation on the unchanged 245 validation IDs. Uses Planner adapter from completed training v4 (restored best step500) and Caller from v5 (best step371), exact English prompts and TypeScript tool documentation from training. No trained Summarizer: its terminal route finishes directly. No public-test execution or submission.

The supplied ASR transcribes current and history user audio; gold text and labels do not enter generation. Batch inference4 with vLLM0.9.2, FP16 and two LoRA adapters on GPU0. A T4x2 machine can be selected, but this inference backend uses only the first GPU. No per-sample time penalty; batch-boundary 10-hour deadline and saved batch outputs support continuation.

Output: `/kaggle/working/concise_validation/concise_planner_v4_caller_v5_valid245/validation/`. `report.json` contains local approximate contest score100, mean tool-call F1, state accuracy and per-sample metrics. `submission.jsonl` holds actual executed calls/state hashes; `traces.jsonl` holds trajectories, termination reasons, repairs and generated decisions. Loss is not substituted for these metrics. Model quality is unmeasured until inference finishes.

Private input `anartt/lifemate-concise-planner-caller-adapters` version2 includes both complete restored adapters, tokenizer files and lineage. File hashes are verified. The two training data manifests have identical parsed JSON but different raw-byte hashes from serialization order; `data_contract.json` records both raw hashes and a common canonical content hash. Runtime checks each role against its recorded raw manifest hash rather than falsely requiring byte-identical files. The validation split and prompt provenance remain fixed.

CPU check: `python checks.py`. Regenerate: `python make_notebook.py`. This validates four concurrent workflows, termination without a Summarizer call, score on exact reference matches, and notebook syntax; GPU/package compatibility is established by actual runtime only. Do not report approximate local score as an official leaderboard result.
