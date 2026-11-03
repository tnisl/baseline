# Qwen3 GLPFT: 2-GPU fine-tuning notebook

Open [the Kaggle notebook](https://www.kaggle.com/code/cymeu4l0t/lifemate-qwen3-glpft-finetune-2gpu), choose **GPU T4 ×2**, enable Internet and Run All. The notebook is uploaded with **Quick Save**, without launching a session.

| Setting | Value |
| --- | --- |
| Model | Qwen/Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Fine-tuning | Unsloth NF4 QLoRA, rank 16, alpha 32, explicit PyTorch DDP |
| GPUs | Exactly 2; one model replica/process per GPU |
| Training micro-batch | **4 per GPU**, global and all local stages |
| Validation-loss batch | **4 per GPU**; every held-out record evaluated once |
| Gradient accumulation | 1; effective training batch **8** |
| Epoch limits | Global 1, planner 1, caller 1, summarizer 2 |
| LR | Global 5e-5; local 1e-5; cosine schedule; 3% warmup |
| Selection | Loss evaluation every 50 attempts and at the final step; early stopping patience 3, min_delta .001; restore lowest-loss trained adapter |
| Workflow inference | vLLM 0.9.2, batch **4**, four independent environments, one GPU |
| ASR | Supplied non-streaming Zipformer2 ONNX, CPU inference; no ASR training |

`kaggle_finetune_2gpu.ipynb` embeds all runtime changes, dependency locks, and the exact `split.txt`. The Git checkout inside the notebook pins a stable core commit; the embedded overlay supplies the new training loop and current prompt policy. Existing validation-only notebooks are preserved.

The notebook needs three private datasets under the authenticated Kaggle account `cymeu4l0t`: `zaloai-dataset`, `lifemate-codex-workflow-rationales`, and `lifemate-supplied-zipformer-asr`. These inputs are attached in `kernel-metadata.json`; content hashes are checked before use.

## Data and stage lineage

The split is 1,445 training requests and exactly the user's 100 held-out IDs. Gold tool actions are replayed against ToolEnv; every generated training target must satisfy its inference schema, and overlength examples cause an error instead of being dropped. Assistant completion tokens alone contribute to the loss.

| Stage | Train examples | Valid examples | Maximum optimizer attempts |
| --- | ---: | ---: | ---: |
| Global | 4,759 | 327 | 595 |
| Planner | 4,759 | 327 | 595 |
| Caller | 3,314 | 227 | 415 |
| Summarizer | 1,334 | 100 | 334 over 2 epochs |

The last global batch is padded by one training example to give both ranks a full batch. Other stages also record any final-batch padding in lineage. Validation is sharded without padding; losses are weighted by the number of supervised tokens across both ranks.

Each local stage starts from a fresh copy of the best global adapter, with a hash assertion proving the initialization. Checkpoint writes are rank-zero-only, with explicit barriers and identical restore/early-stop decisions across ranks. A loss-only wrapper exposes the loss tensor to DDP static-graph handling, including reentrant gradient checkpointing. No Hugging Face Trainer loop or FLOP-counter collectives are used.

Completed stages can be skipped after lineage checks when rerunning the training cell. There is **no mid-stage optimizer resume**; choose a new `RUN_NAME` after an interrupted stage, or preserve and explicitly reuse completed adapters in a separate run.

## Labels and evaluation

The original Codex rationale artifact is unchanged (SHA256 `d14768df60e155521a57998a5c6d0c4c1cc3b6372b74cce7a955098fd3fdf497`). It mostly uses Codex-written tool-explanation templates with episode overrides; it is not 3,541 independently handwritten rationales. The artifact's old `partition` field is ignored. No Qwen teacher is used.

Summarizer targets are concise, deterministic facts from the final gold tool observation, capped at 180 answer tokens. This replaces the old full JSON observation dump that could exceed inference output limits. These are synthetic summaries, not human-written responses. The original zero-call → `give_up` convention is retained, including some no-op/cancellation requests; that taxonomy remains an acknowledged mismatch with the policy.

Training uses the provided clean transcripts. Post-training validation uses ASR for current audio and user history, with no oracle fallback. Held-out labels serve only validation loss and scoring after predictions; they do not enter optimizer updates or inference prompts. The prompt policy was developed using earlier validation diagnostics, so this is a development split, not a pristine test set.

## Verification

`verification.json` records CPU checks: two-rank Gloo token-weighted gradient equivalence, uneven validation, synchronized checkpoint restoration and early stopping, fixed split, all gold action replays, target schema/token budgets, and synthetic four-workflow controller/grammar tests. Reentrant checkpointing with DDP static graph is also tested with a small CPU model.

**The new Qwen/Unsloth DDP runtime has not been executed on GPU.** CPU tests cannot establish VRAM usage or CUDA compatibility. The notebook runs a two-GPU imports/NCCL/AMP backward preflight before loading Qwen. Fine-tuning and full inference are left for the user to run.

DDP behavior follows [PyTorch's DDP design](https://docs.pytorch.org/docs/2.8/notes/ddp.html) and [Unsloth's DDP guidance](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp). Training and vLLM inference use separate Python/CUDA environments to preserve the previously tested package combinations.
