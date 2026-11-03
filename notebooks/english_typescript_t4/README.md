# English instructions + TypeScript contracts on two T4 GPUs

Open `kaggle_en_ts_t4.ipynb` on Kaggle, select T4 ×2 and enable Internet. The new notebook defaults to `PROFILE = 'smoke'`. It has not been run on GPU by the coding agent.

## Paper distinction

The supplied [paper, Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/pdf/2402.01030), proposes CodeAct with **executable Python actions** (section 2.1). It does not propose TypeScript tool schemas. This experiment uses TypeScript **interface documentation** as requested, retains schema-validated JSON actions, and continues the repository's existing alpha-UMi global-to-local adaptation. It does not implement CodeAct's interpreter or arbitrary executable code actions.

## Prompts and contracts

The four role instructions and completion review are English. TypeScript contracts are generated from the installed public catalog for each sample, with typed required/optional fields, literal enums, unions/intersections for `x-one-of`, and comments preserving ranges, identifier sources and formats. Original Vietnamese catalog descriptions and user data are not machine-translated. Existing Codex rationale labels and Vietnamese answers are retained; no new teacher model or held-out demonstrations are used.

Example:

```typescript
interface InstalledTools {
  /** write; device_id.x-ref="smart_home.devices" */
  "smart_home.light_set"(args: {
    "device_id": string;
    "power"?: "on" | "off";
    "brightness"?: number;
  }): unknown;
}
```

JSON Schema is still used internally to validate tool arguments and constrain decoding. It is not the tool representation shown in the prompt. Inputs never include labels, reference state, or sample-ID rules. `split.txt` remains the user's exact 100-ID held-out split.

## T4 performance changes

- English instructions and compact TypeScript contracts reduce unnecessary prompt tokens; preparation reports actual lengths for every role and refuses to drop overlength records.
- Assistant-only head projection remains memory bounded. Default loss chunk increases from **32 to 256** tokens, reducing small vocabulary matmul/checkpoint calls; set `LOSS_CHUNK = 32` if memory requires the conservative path. At vocabulary 151,936, a 256-token FP32 logits buffer is about **148 MiB**, versus multi-GiB full-prompt logits. This excludes other tensors and is not a full VRAM bound.
- Tokenization is cached once per stage/partition and shared between DDP ranks. Cache keys include the prepared record checksum; model/tokenizer revision is pinned in the notebook.
- Per-step Python object agreement is replaced with a tensor overflow reduction; checkpoint/hash checks remain synchronized.
- Full validation loss runs every **100** steps, instead of 50, and at stage end. All validation records still contribute to token-weighted loss. Patience remains 3, so early stopping now reacts after three non-improving 100-step evaluations; set `EVAL_STEPS = 50` for the previous cadence.
- Training logs report measured seconds/step including intervening evaluation, validation event durations, and total scheduled-evaluation time. **No GPU throughput improvement is claimed before measurement.** Backbone, rank-16 QLoRA, batch 4/GPU, accumulation 1, cosine warmup 3%, and full epoch limits (1/1/1/2) are retained. Training and vLLM remain isolated package stacks.

## Smoke and full execution

`PROFILE = 'smoke'`, then Run All:

1. Verify input/rationale/split hashes, install locked runtimes, test two-GPU NCCL/AMP and package imports.
2. Prepare all instructions using the fixed train/valid membership. No validation example contributes to optimizer updates.
3. Each of global, planner, caller and summarizer executes **two optimizer attempts** on its longest 16 training records; evaluate on its longest 8 held-out records. At least one finite weight update per stage is required. This tests full-length contexts at the actual batch size, not truncated toy prompts.
4. Save/restore best checkpoints, verify both-rank hashes and independent local initialization from the global smoke adapter.
5. Run supplied ASR and vLLM multi-adapter inference for four held-out workflows, including synthetic guided-decoding preflight. Save `smoke_pass.json` only after assertions pass. Task scores can be low; completion of the smoke is an integration check, not a quality threshold.

For full training, set `PROFILE = 'full'` and Run All. The notebook always verifies/completes the smoke before full training. If its matching adapter artifacts already exist, completed smoke stages are skipped, then smoke inference is checked again. Use the same `RUN_NAME` to reuse completed stages; a failed partial stage needs a new `RUN_NAME`. There is no mid-stage optimizer resume. Change `RUN_NAME` when changing runtime/configuration.

Full mode trains all four stages on all 1,445 training requests, with no max-step pilot cap, then evaluates all 100 held-out episodes using supplied ASR and vLLM inference batch 4. ASR is never fine-tuned. Outputs are under `/kaggle/working/paper_en_ts_t4/<RUN_NAME>/{prepared,smoke,full}`. A successful smoke cannot guarantee every full-data batch fits or that the full run finishes within Kaggle's session limit.

Attach `cymeu4l0t/zaloai-dataset` (accessible shared input), plus private `anartt/lifemate-codex-workflow-rationales` and `anartt/lifemate-supplied-zipformer-asr`. The notebook is saved to the currently authenticated account `anartt`.

## Build and check

`runtime/` contains the editable experiment overlay and dependency locks. The notebook checks out the same pinned stable core used by the previous notebook, writes the embedded overlay, and checks its hash. Earlier notebooks and unrelated working changes are preserved.

```bash
python notebooks/english_typescript_t4/make_notebook.py
python notebooks/english_typescript_t4/checks.py
```

CPU checks cover English instructions, held-out canary exclusion, TypeScript contract generation, notebook compilation, completion-loss/gradient equivalence on a tiny Qwen3, two-rank optimizer/evaluation/checkpoint behavior and the smoke step cap. Actual package compatibility, VRAM and speed on T4 are tested when the user runs the notebook.
