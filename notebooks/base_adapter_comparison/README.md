# Base vs fine-tuned Planner/Caller

Kaggle: https://www.kaggle.com/code/anartt/lifemate-base-adapter-comparison

Select **T4 ×2**, Internet on, then Save & Run All yourself. One GPU loads one vLLM engine; base requests disable both LoRA adapters, adapter requests select the trained role. This keeps the same hardware, tokenizer, FP16 weights, chat template, prompts, token limits and engine settings. The second GPU is unused. No training or enforcer; JSON/schema checks remain.

Default `SAMPLE_SET='diagnostic4'` uses `train_0017`, `train_0066`, `train_0088`, `train_0001`, all from split.txt. For a broader test, select `SAMPLE_SET='valid245'` and use a new `RUN_NAME`. Four selected failures cannot establish whether finetuning generally helps or hurts. Full245 takes substantially longer; no completion-time guarantee is made.

## Comparisons

1. **Fixed-prefix node probes:** Planner sees the request, context, ASR history and reference *past* calls/observations. Target next tool/route stays outside its messages. Caller receives the same reference-selected tool and fixed generic plan in both conditions; target argument values and current tool result are excluded. Each probe gets one generation, no repair or completion review, batch4. Save prompt/schema/token-ID hashes and assert equality across each pair.
2. **Independent workflows:** base/base versus trained Planner/Caller, each starting a fresh official ToolEnv for each episode. Inputs contain no reference plan, target arguments or future observations. Same ASR transcripts, maximum7 decisions /7 executed calls, one repair per node per decision; Summarizer route terminates without generation. Requests can diverge once their earlier actions differ, which is expected for this end-to-end comparison.

Node probes measure agreement with annotated trajectories, not official contest score. Alternative valid tool order can count as a Planner mismatch. Contest annotations do not distinguish terminal summarizer/give_up for zero-call episodes, so their terminal route is excluded from accuracy. Caller metrics are conditional on the correct selected tool with a generic plan, not the exact training rationale or a model-generated plan. Previous prefix thoughts use that same generic text in both modes; this is controlled diagnostic conditioning and differs from free rollout. No gold argument values are copied into the current plan.

The workflow score uses the existing local approximation: final-state hash gate plus tool-call F1. Different tool environments are used for base, adapter and scorer replay. Gold data is used for fixed-prefix conditioning and scoring, never as answers in free workflow inputs. Do not mix node metrics with deployment workflow score.

## Outputs

Root: `/kaggle/working/base_adapter_comparison/<RUN_NAME>/validation/`.

- `comparison.md`, `comparison.json`: overall base/adapter node accuracy, win/loss counts, workflow score/F1/state match.
- `node_outputs.jsonl`: paired raw responses, schema validity, expected/parsed action, per-field argument matches (e.g. time), prompt/token hashes and selected adapter role.
- `node_requests.json`, `node_targets.json`: inspectable diagnostic inputs and separate targets; target dictionaries never reach the model.
- `node_report.json`: node summaries, written before workflow evaluation starts.
- `base/`, `adapter/`: submissions, raw traces and workflow metrics.
- Shared ASR transcripts/report; run/checkpoint provenance configuration in the parent folder.

Results are flushed after each generation batch or workflow batch. Preserve partial output if the session ends. This notebook does not resume partial comparisons: choose a new run name for reruns. Paired node order alternates base-first /adapter-first by batch; timings still include warmup and are not controlled throughput measurements.

A consistent base-correct/adapter-wrong pattern under the same node inputs supports investigating finetuning regression. Both-wrong cases point to capability/input/prompt limitations too; one run cannot isolate all causes. This compares FP16 base with FP16+LoRA inference, not the training NF4 computation path.

CPU checks: `uv run --no-project --python 3.12 --with jsonschema python checks.py`. Rebuild with `python make_notebook.py` after changing `runtime/`. CPU fixtures verify isolation, switching, scoring and independent environments; they are not GPU/model results.
