# Planner and Caller with concise English plans

Run `kaggle_planner_caller_compiled.ipynb` on Kaggle with **T4 ×2** and Internet enabled. It embeds a complete runtime overlay over core commit `4894d331692293565635454a8c686b7c18314a04`; it does not depend on the latest branch contents. Dataset: `cymeu4l0t/zaloai-dataset`.

The notebook starts full training directly. **No automatic smoke training, deliberate pause/resume or synthetic GPU preflight.** Dependency consistency, data/schema checks, initial validation and periodic validation remain. Initial and scheduled validation now print start messages; scheduled validation prints loss, duration, best step and patience afterward.

Planner decides tool order, conditions, pending objectives and termination. Its English thought gives a short objective and parameter-source hints. Caller receives the current request, history, prior observations, plan and detailed selected tool contract, then resolves values and emits `{name, arguments}`. Tool documentation remains TypeScript; output actions remain JSON. No trained Summarizer is required to learn Planner's terminal route, but this training notebook does not modify any inference runtime to skip Summarizer.

Example (illustrative, not a held-out example):

```json
{"thought":"Perform the next pending alarm / delete operation. Resolve alarm_id from the alarm.list result.","next":"caller","tool":"alarm.delete"}
```

The target Caller action can contain `alarm_id`, but the thought does not copy its value. Label construction uses annotated actions to select the tool and attribute fields to visible sources, never future tool observations or final state. For unconstrained free text, the source hint refers to the request/context; history and observed values are matched when structured values are available. Planner inference may retain short user-supplied selectors to disambiguate objectives; labels currently use generic field-source rules and do not semantically resolve every repeated-tool objective. These are deterministic templates authored in code, not independently reviewed per-sample rationales. Conditional labels remind the model to respect conditions, rather than fabricating a calculated predicate. Review difficult conditional, repeated-tool and ambiguous-reference cases before asserting paper-equivalent reasoning quality. Legacy zero-action heuristic and terminal routes are retained; terminal labels do not claim tool success merely from the presence of a trajectory.

Default: independent local training from Qwen3-4B-Instruct-2507, **Planner then Caller**, 1 epoch each; no Global or Summarizer training. `LOCAL_STAGES=('caller',)` or `('planner',)` selects one role. Optional Global modes remain available for matching new labels/prompts. The episode split is unchanged: **1300 train / 245 valid**, episode-disjoint. Existing adapters/checkpoints from older prompts cannot be resumed under this new run. Their weights are not silently reused.

Unsloth compilation is **enabled** (`UNSLOTH_COMPILE_DISABLE=0`), before importing Unsloth. Each DDP rank uses separate TorchInductor/Triton caches. First-time compilation can delay startup; actual compile/backend compatibility and throughput must be checked on Kaggle. This setting enables Unsloth's supported compilation paths; it does not guarantee every operation is compiled. FP16 on T4, QLoRA rank16/alpha32, micro-batch train/valid **4 per GPU**, effective training batch **8**, cosine LR schedule, 3% warmup, early-stop patience3, eval every100 steps and at the end. Chunked assistant-only loss and length bucketing remain.

Checkpoint every25 attempted steps, after eval and cooperative pause/deadline; retain optimizer, scheduler, scaler, per-rank RNG and batch cursor. At 10 hours request a cooperative save. Preserve the **whole** mode directory printed at the end, mount as a private input, and set `RESUME_DIR` to resume. A hard cancellation can lose steps after the last complete checkpoint. New run name: `planner_caller_concise_compiled_v1`.

Validation loss uses gold plans and is not end-to-end accuracy. Removing complete argument values makes the task less trivial, but field-source plans are still supervised. Evaluate Planner-generated plans followed by Caller separately. No GPU run, ASR training/inference, workflow validation or submission is automatically launched by notebook creation.

CPU checks: `python check_contracts.py`. Rebuild embedded runtime with `python make_notebook.py` after editing `runtime/`. Never edit only the generated notebook payload.
