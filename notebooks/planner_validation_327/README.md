# Planner free-generation evaluation on 327 fixed validation inputs

This notebook evaluates the completed version-6 Planner adapter on the exact 327 saved `planner.validation.jsonl` inputs used for the reported validation loss. They cover all 100 episodes from the user's `split.txt`: 227 caller decisions and 100 terminal decisions. No training, ASR, Caller/Summarizer generation, public-test prediction or tool execution takes place.

Select a T4 GPU, enable Internet, and Run All. T4 x2 is also accepted, but inference uses GPU 0 only. The notebook is saved for manual execution; it has not been GPU-tested as part of notebook creation.

## What is held fixed

- Original Vietnamese prompt policy and JSON tool presentation, with prompt SHA256 `0caf7a12a791a2414a6f7419447c32b1c1c07ffe41376f6a1782ad1e90631f7f`.
- Original clean current/history transcripts and previous gold thoughts/actions/observations, byte-for-byte preserving the saved messages. Gold observations are supplied as fixed context; no tools are replayed during evaluation.
- Completed Planner adapter and original Qwen/Qwen3-4B-Instruct-2507 revision.
- vLLM 0.9.2 FP16, LMFormatEnforcer, temperature 0, seed 42, max context 8192, max output 320 tokens. The same numerically checked T4 LoRA kernel workaround is retained.
- Each input receives one generation with the Planner adapter. No current gold thought, next route, selected tool or completion review is supplied. Tool enums contain all installed tools, not the correct tool only.
- Batch size 4, last batch size 3. There are 82 validation batches plus four synthetic preflight requests.

The stable core is pinned to commit `4894d331692293565635454a8c686b7c18314a04`; the notebook embeds its complete runtime overlay, including the new evaluator. The reused inference backend defines other roles for compatibility, but this evaluator submits only Planner requests. These inputs intentionally use gold prefixes for an offline diagnostic; they are not end-to-end competition inference inputs.

## Inputs and label separation

Attach private inputs `cymeu4l0t/lifemate-v6-completed-adapters` (dataset version >=2) and `cymeu4l0t/lifemate-v6-planner-validation-inputs`. The latter contains only flat files to avoid skipped nested-directory uploads:

- `planner_inputs.jsonl`: id, step, exact saved messages, and schema derived from installed public tools.
- `planner_labels.jsonl`: separate scorer-only targets, parsed after all 327 predictions have been saved.
- `evaluation_manifest.json`, `training_manifest.json`, `split.txt`: checksums, original training lineage and provenance.

`prepare_inputs.py` checks the original validation file hash, previous-only gold observations, supplied clean request, all installed tools, 327 unique step keys, and the 100 validation IDs. Recombining separated messages and labels reproduces the original validation JSONL hash exactly. No sample-specific prompt edits are made and benchmark-derived labels/inputs are not committed to GitHub.

## Metrics

`report.json` contains route accuracy across 327 inputs; joint route/tool accuracy; tool accuracy across the 227 caller inputs, counting wrong routes and malformed JSON as errors; terminal route accuracy across 100 inputs; first-step accuracy across 100 episodes; route confusion; schema failures; and episode-level agreement across all independently supplied reference states. These are strict reference agreement metrics, not official competition scores, semantic rationale accuracy or end-to-end workflow success. Alternative valid tool orders may be marked incorrect. The majority-route baseline is reported because the route classes are imbalanced.

The old rollout's first-step tool match was measured with ASR inputs. This notebook uses clean transcripts, so differences cannot automatically be attributed to the gold-prefix intervention alone. A comparison between training and inference backends is not included in this run.

Outputs are under `/kaggle/working/planner_validation_327/planner_v6_gold_prefix_327_v1/`: `predictions.jsonl`, `details.csv`, `details.jsonl`, `report.json`, `preflight.json`, `config.json`, and `batches/`.

Finished batches are saved atomically. To resume in another session, preserve the run directory as a private input and set `RESUME_DIR` to it. Keep the run settings unchanged. The 10-hour budget pauses only before new batches; there is no per-input timeout penalty. Interrupted unsaved generations may safely be regenerated because no tool actions occur.

Build with `python notebooks/planner_validation_327/make_notebook.py`. Run CPU checks with `python notebooks/planner_validation_327/checks.py`. CPU checks do not establish GPU/runtime compatibility or model quality.
