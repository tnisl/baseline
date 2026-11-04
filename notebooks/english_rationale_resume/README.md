# English evidence rationales, 1300/245 split, resumable training

Open `kaggle_english_rationale_resume.ipynb`, select **T4 x2**, enable Internet and Run All. Default `PROFILE='full'` runs an integration smoke first, then full training for the selected stages. Saving the notebook does not start a GPU job. ASR and test inference are outside this notebook.

## Choose training stages

| Configuration | Stages | Initialization |
| --- | --- | --- |
| `TRAIN_MODE='local'`, `LOCAL_INIT='base'` (default) | Planner, Caller, Summarizer | Each independently from original Qwen; no Global is trained or required |
| `TRAIN_MODE='global'` | Global only | Original Qwen |
| `TRAIN_MODE='global_to_local'` | Global, then selected local stages | Each local independently from the new Global |
| `TRAIN_MODE='local'`, `LOCAL_INIT='global'` | Selected local stages only | Set `GLOBAL_ADAPTER` to a completed matching Global adapter directory |

`LOCAL_STAGES=('planner','caller','summarizer')` can select a subset, for example `('caller',)`. This does not alter the data split. `GLOBAL_ADAPTER` must be None unless using local mode with global initialization. Smoke/full checkpoints are separate in every mode. With an external Global, both smoke and full local training use that supplied parent, whose lineage must match the new 1300/245 labels and prompt; a smoke Global is rejected for full local training.

Each mode writes under `.../<RUN_NAME>/<MODE_TAG>/` and has a different config/resume signature. Local-from-base is a one-stage specialization variant, not GLPFT. Old notebook version-2 run directories have a different source signature: resume them with their original notebook version. A completed matching Global from such a run may still be reused through `GLOBAL_ADAPTER`.

## Data and rationale

The deterministic episode split has **1300 train and 245 validation**. It retains the original 100 validation IDs and adds 145 selected proportionally by first-tool app, capped action count, and conditional-request indicator. All role examples from an episode stay in its partition. Files are `runtime/train_ids.txt`, `runtime/split.txt`, and `runtime/split_manifest.json`. These are IDs, not private annotations. Semantic paraphrase groups have not been manually audited.

**Start Global from the original model when training it.** The old 1445/100-split Global saw some of the newly held-out 145 episodes and cannot serve as an uncontaminated parent for this experiment. Local-from-base avoids that parent; other local modes require a Global trained on these new labels/split. In all cases local branches initialize independently, never from another local role. Incompatible run directories/checkpoints are rejected by source/data/config signatures.

The annotation code is `runtime/umilife/english_labels.py`. English rationale targets include the selected operation's concrete values, ID provenance from prior observations/history, observed numerical thresholds where the Vietnamese request has a numeric literal, and computed offsets from observed calendar times. It does not access the current/future tool result. Original names and text fields remain quoted in their original language. Missing-information labels name the missing fields. Cancellation/acknowledgement zero-call examples use a terminal summarizer target; other zero-call examples retain give_up. This classification is a heuristic that needs review for ambiguous requests.

These are **deterministic evidence-compiled annotations**, authored as rules by Codex, not an LLM teacher and not a claim that every example has been individually reviewed. They improve numerical/provenance specificity but do not provide complete semantic parsing of all Vietnamese conditions or validate that every original gold trajectory is optimal. The notebook exports the complete role datasets under `prepared/` for review. No validation-specific examples/rules are added to system prompts.

System prompts, rationales and response labels are English; user requests/catalog descriptions/quoted values preserve the original data. Tool contracts retain compact TypeScript documentation and JSON actions. A thought is limited to 1024 characters. Contexts exceeding 8192 tokens are rejected, never silently truncated.

## Training

- Qwen/Qwen3-4B-Instruct-2507 pinned revision; Unsloth QLoRA r16/alpha32.
- Two DDP GPUs, microbatch **4/GPU**, validation-loss batch **4/GPU**, effective train batch 8.
- Global 1 epoch, Planner 1, Caller 1, Summarizer 2; LR 5e-5 Global / 1e-5 local; cosine, warmup 3%, early-stopping patience3/min_delta .001, eval every100 attempts and final step.
- Loss chunks256, cached tokenization shared across ranks, length bucketing. Validation loss remains assistant-only token CE, not decision-weighted and not a guarantee of free-generation quality. Evaluate Planner generation separately before concluding improvement.
- Mandatory smoke uses longest16 train/8 validation examples per role, two optimizer attempts, checkpoint every step. It deliberately exits after step1 and restarts once from that saved state. The full profile begins only after all smoke stages finish. This is a GPU integration check for the user to run, not a claim it was already run on Kaggle.

## Interrupted sessions and resume

The notebook saves every25 optimizer attempts, after evaluation, and at a cooperative 10-hour deadline. A SIGTERM handler requests a save at the next safe step boundary. A hard kill cannot execute a final save: work after the last complete checkpoint can be lost.

Each immutable `checkpoints/<role>/step-XXXXXXXX/` contains the adapter, AdamW optimizer, scheduler, AMP scaler, Python/NumPy/Torch/per-GPU RNG, epoch/batch cursor, early-stop/history state, and a snapshot of the associated best adapter. `latest.json` is replaced only after a whole checkpoint directory is committed. The most recent two step checkpoints are retained per role; best/final adapters are separate.

To resume across Kaggle sessions:

1. Preserve/download the **whole mode run directory**, by default `/kaggle/working/english_evidence_resume/english_evidence_1300_v2/local_base/`. Kaggle working storage is not independently durable and the notebook cannot guarantee retention if a session is discarded without outputs.
2. Upload/attach that directory as a private dataset or prior notebook output.
3. Set `RESUME_DIR` to its mounted path containing `run_config.json`, `prepared/`, `smoke/`, and `full/`. Keep TRAIN_MODE, LOCAL_INIT, LOCAL_STAGES, RUN_NAME, source, split, GPU count and training settings unchanged.
4. Run All. Completed stages are skipped; an interrupted stage resumes its optimizer/batch cursor. Full Global is never initialized from the smoke Global.

Resuming with different batch size, data, loss chunk or schedule is rejected. Checkpoints are trusted local artifacts (`torch.load(..., weights_only=False)` for optimizer/RNG state); do not attach untrusted checkpoint files.

## Validation performed during development

CPU/Gloo two-rank regression tests continuous training against pause/resume at step1 with dropout: final weights, update count, validation metrics/best step and scheduler agree exactly. Full-data preparation checks every target against its JSON schema and every tokenized context against8192. GPU/Unsloth integration must be tested by the notebook's smoke; no GPU training is launched automatically by this implementation task.
