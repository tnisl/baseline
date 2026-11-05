# Planner / Caller: evidence, intent and throughput

Kaggle: https://www.kaggle.com/code/lctrnnguynkhi/lifemate-planner-caller-evidence

Select **T4 ×2**, Internet on, then Save & Run All yourself. Saving the notebook does not launch training. No initial smoke training. Both GPUs train each selected role using DDP; roles train sequentially and start independently from **Qwen/Qwen3-4B-Instruct-2507** revision `cdbee75f17c01a7cc42f958dc650907174af0554`. No Global, Summarizer or ASR training/inference.

```python
STAGES = ('planner', 'caller')  # Or ('planner',) / ('caller',)
EPOCHS = {'planner': 1, 'caller': 1}
RUN_NODE_EVAL = True
```

Use a fresh `RUN_NAME` for a different experiment. These prompts/data differ from prior adapters; do not resume an old v2 checkpoint or evaluate these adapters with the old prompt.

## Supervision

English prompts, TypeScript tool contracts; outputs remain JSON. Planner chooses `{thought,next,tool}`. Caller emits `{name,arguments}` for the selected tool. This keeps the existing role contract; it does not add a new Caller `need_replan` status or claim the original paper's Caller can freely change the selected API.

Planner targets preserve decisive conditions, observed entity details, derived time/amount values and remaining eligible bills when the evidence compiler can establish them. Values may appear in plans when justified. Domain policies distinguish reschedule/create, off/speed, payment/autopay, balance lookups and entity matching. Caller is taught latest corrections, intent-required optional fields and exact user facts. Half its training plans use a generic non-oracle thought; ALL validation Caller plans use that generic thought, so evaluation cannot simply copy current target values from a reference rationale.

Labels are **Codex-authored executable evidence rules**, plus100 manually reviewed training-only no-action targets. They are not independently generated/reviewed LLM rationales for every episode. The compiler uses the annotated current action as supervision and visible past evidence, never the current/future tool response. Derived time values are supervised annotations, not an independent arithmetic verifier. Read all examples/audits before claiming semantic completeness or equivalence to the paper.

Train-only augmentations, at most one of each category per episode:

- Planner: recover from a blocked unchanged read **or** reject a premature completion proposal and continue the pending annotated objective.
- Caller: review a proposed call with one annotated optional field omitted; reconstruct the complete reference call from the request/evidence. This teaches completeness but does not prove every optional field is always required.

The fixed split remains **1300 train /245 valid IDs**, with no overlap. A training episode with an unobserved opaque ID, tool-reported error or detectable bill-filter contradiction is quarantined and reported, not silently repaired from hidden state. The raw split count remains1300; eligible training episodes may be fewer. Validation references are not filtered for poor performance. Its11 no-action episodes have no explicit contest terminal-route label and are omitted from node CE/accuracy, reported separately. No validation synthetic augmentation, optimizer updates or per-ID prompt rules.

## Loss, speed and memory

Assistant-only causal cross-entropy, normalized by actual assistant-token count, with correct global token weighting across DDP ranks. Prompt/padding tokens remain masked. The objective is unchanged; loss is not a substitute for business metrics.

- FP16 on T4, NF4 base, LoRA rank16/alpha32/dropout0 on q/k/v/o/gate/up/down.
- Microbatch train/eval4 per GPU; accumulation1, effective batch8. Backbone activation checkpointing remains enabled for long batches.
- Unsloth compile ON, separate per-rank compile caches. First compilation costs time; speedup is not guaranteed without GPU measurements.
- Projection chunk512; projection checkpoint **OFF** by default to avoid its backward recomputation. This uses more VRAM. If long batches OOM, start a new run with `PROJECTION_CHECKPOINT=True`; never silently truncate evidence or lower batch size.
- Tokenization cached once during preparation; validation sorted by length. Initial CE and duplicate full CE after hash-verified best restoration skipped by default.
- Logs separate training token/s (excluding padding), assistant-token/s, train seconds/step, padding fraction, peak allocated VRAM, eval duration and checkpoint duration.

Training adds recovery examples and longer targets, so record/step count is larger than the prior notebook. Better throughput does **not** guarantee shorter total wall time. Generation evaluation also adds work.

Cosine LR1e-5,3% warmup; CE and generation eval every100 steps and at end, early-stop patience3/min_delta.001. CE-best and early stopping still use CE; periodic generation metrics do not change checkpoint selection. CE-best remains available even if post-stage node generation prefers another checkpoint.

Each periodic evaluation greedily generates on **all** validation records for the current role (Planner807 / Caller573), sharded across both resident Unsloth models, batch4/GPU, max256 new tokens for Planner /192 for Caller, no enforcer. Prompts contain messages only; target answers are used exclusively for scoring. It prints schema validity, Planner route/tool accuracy and initial/intermediate/terminal agreement, or Caller full-action match, argument-field F1 and per-field counts. Reports identify this as the resident Unsloth/Transformers backend; numerics may differ from the later vLLM run. No second model is loaded while DDP holds GPUs. Training hidden-state/loss flags, checkpointing and RNG states are restored afterward. This adds potentially substantial evaluation time.

## Generation metrics and checkpoint selection

After each role finishes, training processes exit and release GPU memory. A separate pinned **vLLM0.9.2 environment** evaluates CE-best and last checkpoints. Two independent GPU workers, batch4 each, greedy generation, no enforcer,256 Planner /192 Caller output tokens,8192 context. T4 prefix caching disabled; CUDA Graphs enabled; validated T4 LoRA compatibility patch retained. The LMFE package remains a transitive pinned dependency of vLLM but is not attached to generation.

Both candidates receive the same messages/token IDs in each paired probe. If adapter files are identical, evaluate once. Workers flush output and parent prints progress. This post-stage vLLM comparison is additional to the resident-model generation evaluations during training.

Planner reports route/tool agreement, initial/intermediate/terminal agreement, schema validity and calls proposed after reference completion. Caller reports full-action match, argument-field F1 and per-field accuracy, including IDs/times. These use exact/fuzzy annotation comparisons, not a universal semantic checker; valid alternative tool orders can mismatch labels.

Recommend one of **CE-best and last** by node match accuracy, then schema validity, with CE-best as tie-break. This is not a search over every saved checkpoint and is not full-workflow contest score. Full ASR/Planner/Caller rollout must be evaluated separately before deployment.

## Checkpoints and outputs

Root: `/kaggle/working/planner_caller_evidence/<RUN_NAME>/`.

- `prepared/`: new rationales/targets, cached tokens, manifest, per-episode `data_audit.json` and scoring references.
- `training/adapters/<role>/`: CE-best adapter, tokenizer, lineage and CE history.
- `training/checkpoints/<role>/`: last two complete resume snapshots, best snapshot and atomic `latest.json`. Includes optimizer/scheduler/scaler/per-rank RNG and batch cursor.
- `training/checkpoints/<role>/eval_history.json`: CE and generation metrics written at every eval, also preserved in resume snapshots and final `training_history.json`.
- `training/node_eval/<role>/step-XXXXXXXX/`: periodic generation `report.json` and raw outputs in `gpu0.jsonl` / `gpu1.jsonl`.
- `node_eval/<role>/`: candidate configuration, per-GPU logs/raw generations and `report.json`.
- `selected_adapters/<role>/`: self-contained recommended inference adapter, tokenizer and selection provenance. Exported `final_adapter` hash is explicitly a safetensors-file hash; original CE-best model-state hash is retained separately. Do not confuse these hash conventions.
- `selected_adapters/checkpoint_manifest.json`, `data_contract.json`: file hashes, roles and prompt/data provenance.
- `node_checkpoint_recommendations.json`: recommendation per completed role.

Save every25 attempted steps and after evaluation/early-stop/cooperative deadline. At10 hours request a cooperative checkpoint. A hard kill can lose updates since the last complete checkpoint. Preserve the WHOLE root, attach it as a private Kaggle input and set `RESUME_DIR`. Keep settings/source identical; completed roles and node evaluation reports are reused only after provenance checks. The two-role run may need more than one session; select one role if desired.

## CPU verification

```bash
uv run --no-project --python 3.12 --with numpy --with jsonschema --with safetensors --with torch==2.9.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu python checks.py
uv run --no-project --python 3.12 --with numpy --with jsonschema --with safetensors --with torch==2.9.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu python check_ddp_resume.py
python make_notebook.py
```

Checks cover evidence/future isolation, training-only no-action annotation, CE value/gradient parity with/without projection checkpointing, sorted weighted eval, role metrics and actual2-rank CPU checkpoint/resume parity. Local full preparation uses the supplied dataset/tokenizer without GPUs. These checks do not establish T4 VRAM compatibility, package startup on a new Kaggle image or quality gains.

Generation limits apply to both periodic and post-stage evaluation. Reports include `max_new_tokens` and `length_limited_count`; capped outputs are still scored and never silently counted as complete. Training context8192 and projection/loss chunk512 are unchanged. Use a fresh run root after this source change; old cached evaluation reports cannot be reused.
