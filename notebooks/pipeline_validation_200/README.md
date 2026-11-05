# Pipeline validation on split_200.txt with old adapters

Kaggle: https://www.kaggle.com/code/lctrnnguynkhi/lifemate-pipeline-validation-split200

Select T4 x2, Internet on; Save & Run All yourself. No GPU job is launched by publication.

Uses exactly the 200 supplied IDs in `runtime/split_200.txt`, original order preserved. **168 IDs overlap the old adapters’ training set; 32 overlap their validation set.** See `split_audit.json` and report metadata. The aggregate score must not be presented as held-out validation.

Planner v4 CE-best step500 and Caller v5 CE-best step371, independently trained local adapters, frozen old English prompts/TypeScript contracts. No new evidence-v3 adapter or prompt. Qwen/Qwen3-4B-Instruct-2507 pinned revision, two independent vLLM0.9.2 GPU workers,100 episodes each, batch4/GPU. No enforcer; graphs on, prefix caching off on T4. Maximum7 Planner decisions and7 executed calls; at most one revision per node per decision. Summarizer route terminates without generating a summary.

Full rollout: supplied Zipformer ASR for current/history audio, Planner -> Caller -> official tool environment -> appended observations -> Planner, then score tool F1/final state. No reference trajectory is provided to models; labels are used only for scoring after prediction. No training, public-test inference or submission. ASR runs on CPU before LLM inference, matching the old workflow.

Private inputs: `cymeu4l0t/zaloai-dataset`, `lctrnnguynkhi/lifemate-concise-planner-caller-adapters`, `lctrnnguynkhi/lifemate-concise-validation-asr`. Checkpoint/prompt/model and file hashes verified before use.

Output: `/kaggle/working/pipeline_validation_200/old_adapters_unguided_limits7_split200/` with `split_audit.json`, `config.json` and `validation/`. In validation: `report.json` (approximate contest score, tool F1, state match), `submission.jsonl`, `traces.jsonl`, ASR transcripts/report and per-GPU logs/batch outputs. Preserve the whole run root and set `RESUME_DIR` to continue. Ten-hour session budget checks batch boundaries; no per-sample timeout. Do not reuse a245-ID run directory.

Build: `python make_notebook.py`. CPU checks: `uv run --no-project --python 3.12 --with jsonschema python checks.py`. GPU inference has not been executed for this notebook.
