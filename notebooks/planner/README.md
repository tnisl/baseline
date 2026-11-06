# Planner — Unsloth training, Transformers validation

Notebook: [LifeMate Two Node Planner Qwen35](https://www.kaggle.com/code/cymeu4l0t/lifemate-two-node-planner-qwen35).

Kaggle version15 includes configurable GPU batching and quiet model loading. Its source cells were verified against the repository notebook after QuickSave; no training was launched.

Select **GPU T4 ×2**, enable Internet, and attach the private competition dataset. The standalone notebook embeds source and the fixed split of 200 validation IDs; it does not embed competition data. It is saved for manual execution, not launched automatically.

The Kaggle connector did not persist the requested dataset attachment. Before running, use **Add Input** to attach your private competition dataset (currently cymeu4l0t/zaloai-dataset). The notebook detects either its extracted train folder or a ZIP. Publishing status is recorded in published.json; Kaggle's save status does not establish that training has run.

Set PROFILE="smoke" for four optimizer updates on 16 training and eight validation episodes. Set PROFILE="full" for the configured EPOCHS and every reference-prefix decision from the 200 validation episodes. Use a new RUN_NAME each time. Full starts from the pretrained model, never a smoke adapter.

The notebook now defaults to CHECK_KERNELS=True and RUN_NAME="planner_quiet_load_smoke_01". This inspects Gated DeltaNet bindings after Unsloth has loaded/patched the model and profiles optimizer step2 on rank0, including the configured accumulation microbatches and backward. It does not install or force optional kernels. Set CHECK_KERNELS=False for ordinary training without profiling.

Kernel diagnostics are saved before validation:

- diagnostics/rankN.kernels.before.json and rankN.kernels.after.json: per-layer forward/conv/chunk/recurrent callable origins, wrapper chains, package versions and module fast-path flags. A missing optional package can coexist with bundled Unsloth kernels. The module flag may predate patching. Recurrent/update functions usually concern cached decoding, not training prefill.
- diagnostics/rank0.step2.profile.json: actual CUDA kernel names, call counts and times, CPU operators, candidate GDN/conv names and optional Unsloth counter deltas. Name matches help inspection; absence of a match is inconclusive. cuda_events_recorded=false means the profiler did not capture CUDA evidence, so it cannot establish the GPU path.
- diagnostics/rank0.step2.trace.json: Chrome trace for closer inspection. The kernel_profile_complete log prints whether CUDA events were captured and the largest kernels immediately after step2. Reports already written remain available if later validation is interrupted; show_kernel_diagnostics() prints them from the notebook.

Step2 timings include profiler overhead on rank0 and can affect other ranks. Distributed workers wait after export so that export time is not charged to the next optimizer step. Do not use step2 for throughput comparisons. Other steps can still include compilation/warmup; this four-step smoke is a kernel probe, not a controlled speed benchmark. CPU tests verify inventory/report/trace behavior, not T4 fast-path execution.

Configure NUM_GPU, MICRO_BATCH and GRAD_ACC in the first code cell. Effective batch = NUM_GPU × MICRO_BATCH × GRAD_ACC; defaults remain 2 × 2 × 2 = 8. For the same batch8, examples are 1 × 2 × 4 or 2 × 4 × 1. A larger microbatch requires more VRAM. Values must be positive integers and NUM_GPU must not exceed visible CUDA GPUs; Kaggle T4×2 allows 1 or 2. EVAL_BATCH remains a separate per-GPU validation setting. Smoke and full use the same selected training configuration. CLI equivalents: --num-gpu, --micro-batch, --grad-acc.

Thinking is disabled through the chat template. Only completion JSON is supervised. No input truncation, packing, synthetic recovery examples, revise, output repair or token-level enforcer.

Qwen3.5/T4 training preserves Unsloth's forced-FP32 computation path when requested by the library, while loading the unquantized base with FP16 storage. It does not force FP16 autocast onto Gated DeltaNet. Precision and memory are recorded for each rank. The exact combined DDP workload still requires a GPU smoke run; CPU tests are not a GPU compatibility or performance claim.

There is no validation during training. The final adapter is exported, training processes exit, and one fresh ordinary Transformers/PEFT process per selected GPU generates decisions on disjoint validation prefixes. Paired base evaluation disables the adapter on the identical frozen backbone. Metrics are computed from raw JSON output and include every invalid answer. They measure reference-prefix planning, not complete agent success or the official contest score.

The input schema, messages, completion encoding, training process and generation metrics are separated under src/twonode/. Caller can be added as its own independently supervised task later; it is not implemented here.

Rebuild after changing runtime source:

```bash
python notebooks/planner/make_notebook.py
```

The requirements pin the Qwen3.5/T4 dependency family. packages.txt captures the actual resolved environment. config.json records the model revision, and source_manifest.json records the embedded source hashes.

Keep notebook/output private. artifacts.zip contains the final adapter, prepared examples, manifests and reports; intermediate checkpoints and compile caches remain outside that archive. No training, GPU smoke or validation-quality run is performed by publishing the notebook.

Startup diagnostics record PID/rank, stage duration, process RSS, available system RAM and allocated/reserved GPU memory before/after import, model loading, LoRA attachment and Trainer initialization. If a stage takes longer than 120 seconds, faulthandler repeatedly records all Python thread stacks in diagnostics/rankN.STAGE.stacks.log. The timer is diagnostic, not a termination deadline. These files remain under the output directory even when startup fails; no optimizer/model changes are implied by this diagnostic update.

The v14 load-model stack dumps repeatedly stopped in Unsloth's stream wrapper while tqdm refreshed the loading bar. Hub/Transformers progress bars are now explicitly disabled in tokenization, training and validation, with TQDM_DISABLE/HF_HUB_DISABLE_PROGRESS_BARS set before child imports. Trainer uses disable_tqdm=True, retaining ordinary loss logs, startup markers and stack diagnostics. CPU tests verify suppression; a fresh T4 run must confirm resolution. EPOCHS=1 preserves the latest user-edited Kaggle setting.
