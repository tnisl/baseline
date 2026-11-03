# Version 6: validate, then predict public test

Open `kaggle_v6_validation_test.ipynb`, select a T4 GPU, enable Internet and Run All. This notebook is inference only; it does not train missing local adapters. It uses the completed version-6 Global and Planner adapters, preserved in private input `cymeu4l0t/lifemate-v6-completed-adapters`.

Use **dataset version 2 or later**. The initial dataset upload skipped the nested adapter folders and contained only metadata. Version 2 includes `global/adapter_model.safetensors` and `planner/adapter_model.safetensors`, plus their configs and tokenizers. If an existing editor still shows the old input, update the dataset version under Input, or remove and re-add this private dataset. The notebook checks for all manifest files before loading weights.

Default `MODE='multi_fallback'` implements the requested configuration:

| Role | Actual adapter |
| --- | --- |
| Planner | Completed Planner adapter, restored best checkpoint at step 550 |
| Caller | Completed Global adapter, restored best checkpoint at step 550 |
| Summarizer | Same Global adapter |

No fake Caller/Summarizer checkpoint folders are created. Output reports record this mapping. Using Global under local-role prompts is a fallback configuration, not the fully fine-tuned three-local-adapter solution. Optional `MODE='single'` uses the Global single-agent workflow.

The original Vietnamese system prompts, JSON tool contracts, controller, output limits and guided decoding are preserved. The exact static prompt SHA256 (`0caf7a12a791a2414a6f7419447c32b1c1c07ffe41376f6a1782ad1e90631f7f`) is checked against both original lineages. No English/TypeScript or new anti-repeat policy is used.

## Execution

1. Verify every attached checkpoint file hash and the model/prompt revisions.
2. Install isolated, locked vLLM 0.9.2 and supplied Zipformer2 CPU ASR runtimes.
3. Transcribe current and user-history audio for the 100 held-out `split.txt` IDs. No provided reference transcript is used by the agent.
4. Generate validation predictions with up to four independent workflows per batch. Only after generation, replay gold actions in separate scoring environments and report local metrics.
5. Transcribe public-test audio, predict **every** `public_test.jsonl` ID with the public-test catalog and its own `zaic_tools`, and write `public_test/submission.jsonl`. Public test is not scored locally because gold is unavailable. Submission to Kaggle/competition is manual.

Inference uses GPU 0, FP16 base weights and the two real LoRA adapters; T4 ×2 can be selected, but the second card is not used by this original inference backend. There is no per-sample time penalty or timeout. This notebook has CPU checks; its new split-aware wrapper has not yet been executed on GPU.

## Long runs and continuation

The default session budget is 10 hours from the notebook's first cell. The budget is checked **before starting a new inference batch**, so active episodes are allowed to finish and save their actual calls. If the budget is exhausted, the process exits with code 75 and `progress.json`; validation must finish before the public-test phase starts. This avoids a hard deadline for individual samples, but an exceptionally long batch or infrastructure failure can still interrupt it.

Every finished batch is stored atomically under `<split>/batches/`. To continue in another Kaggle session, preserve the previous output as a private input (or attach prior notebook output), set `RESUME_DIR` to the prior run folder containing `config.json`, keep `RUN_NAME` and MODE unchanged, and Run All. The notebook copies the run folder into writable storage and checks configuration. Finished batches are reused without new ToolEnv calls; ASR cache is reused.

If the session was unexpectedly killed in an active batch, `active_batch.json` prevents automatic re-execution of unrecorded actions. This notebook does not claim to restore lost in-memory tool state. Inspect/preserve that incomplete batch rather than silently restarting it.

Outputs: `/kaggle/working/v6_validation_test/v6_planner_global_fallback_v1/{validation,public_test}`. Each split has `submission.jsonl`, `traces.jsonl`, `report.json`, ASR caches and completed batch files. A file produced before a paused run may be partial; submit public-test results only after `report.json` confirms every expected ID.

Required private inputs: `cymeu4l0t/zaloai-dataset`, `cymeu4l0t/lifemate-supplied-zipformer-asr`, `cymeu4l0t/lifemate-v6-completed-adapters`. No rationale dataset or optimizer packages are needed.

Regenerate with `python notebooks/v6_validation_test/make_notebook.py`. Run CPU fixture checks with `python notebooks/v6_validation_test/checks.py` from this checkout. Checks cover exact prompt/lineage hashes, public-test input sanitization, complete ID coverage, submission export and completed-batch reuse; they do not establish model quality.
