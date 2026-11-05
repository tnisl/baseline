# Diagnose slow/looping evaluation on six real prefixes

Kaggle: https://www.kaggle.com/code/lctrnnguynkhi/lifemate-evaluation-failure-probe

Run yourself on **T4 x2**, Internet on. Publication is QuickSave only. No finetuning, optimizer updates, ASR, tool execution, retries or enforcer.

Four historical looping states: `train_1070` step1, `train_0550` step1, `train_1124` step1, `train_0486` step0. Controls: `train_1405` step1, `train_0442` step0. Uses the exact saved validation-prefix messages from session365792172; target/reference fields are only used for scoring/CE probes, never passed to generation. Six original historical step100 outputs remain in the private input dataset.

**Step100 weights were lost on cancellation.** The latest saved adapter is **step75** and is copied unchanged with hashes. This notebook cannot reproduce step100 exactly. It does not use the older completed Planner v4 adapter or its old prompts.

Sixteen cases,96 outputs:

| Engine | Model | Batch sizes | GPU |
|---|---|---|---|
| Unsloth2025.10.9 / Transformers | Base with zero LoRA contribution; step75 | 1,4 | 0 |
| Transformers4.56.2 + PEFT0.17.1 NF4 | Base; step75 | 1,4 | 0 |
| Transformers4.56.2 + PEFT0.17.1 FP16 | Base; step75 | 1,4 | 0 |
| vLLM0.9.2 | Base; step75 | 1,4 | 1 |

Fresh subprocess for each engine/model case, sequential execution. Plain Transformers cases run first, with no Unsloth import or monkey patches, using SDPA and the same pinned Torch/Transformers/PEFT/bitsandbytes package versions as the Unsloth environment. HF cases share installed packages to avoid tripling disk usage; process state is isolated. Each compares batch1/4 with the same six messages, tokenizer IDs, greedy decoding, max512 tokens, no repetition penalty or schema enforcer. Unsloth compile ON, resident-style inference context with hidden-state/logits flags switched exactly as periodic eval. A supervised CE forward runs before generation and after each generation batch to verify restoration; no backward or optimizer. T4 vLLM compatibility patch, graphs ON, prefix caching OFF. Base/adapter weights are not shared between processes.

Compare **Unsloth NF4 with plain Transformers NF4**, **Transformers FP16 with vLLM FP16**, and **Transformers NF4 with FP16**. NF4 uses double quantization and FP16 compute. Actual quantization config, LoRA tensor dtypes and attention implementation are recorded for HF cases; remaining loading/kernel differences can still affect outputs. Plain FP16/NF4 base has no adapter; Unsloth base has verified zero LoRA contribution. Explicit greedy settings, max512 and EOS IDs151645/151643 apply to every case; effective HF arguments are saved separately from model defaults. This is a standalone evaluation probe, not a recreated DDP training session with optimizer/gradients/memory pressure. Startup, warmup, compilation and processing order affect latency comparisons.

Private input `lctrnnguynkhi/lifemate-evaluation-failure-inputs` contains authentic step75 adapter/config, six unchanged prepared records, catalog, historical step100 outputs and hashes. No full competition dataset required.

Outputs at `/kaggle/working/evaluation_failure_probe_precision/`:

- `report.md`, `report.json`: per-case schema/match/repetition/length counts, generation time and paired comparisons.
- `<engine>_<model>.jsonl`: raw text, exact generated token IDs/count, EOS/length/stop reason, prompt hash and batch duration.
- `<engine>_<model>_settings.json`: effective generation settings and precision.
- `<engine>_<model>.log`: startup, generation progress, pre/post CE restoration checks and underlying exceptions.
- `input_manifest.json`, `source.json`: source/data/checkpoint provenance. CPU source verification does not establish GPU compatibility.

HF token counts include EOS when present; vLLM token count follows returned token IDs and may omit its stop token. Compare finish/stop reasons as well as counts. HF per-row counts strip padded positions at first configured EOS. Batch duration is shared by rows in that batch; report sums it only once. Prompt IDs must match in all sixteen cases.

Interpretation:

- Batch4-only changes suggest batching/numerics/cache/padding effects, not proof of a specific bug.
- Adapter-only repetition on both engines suggests checkpoint behavior; it does not establish label noise.
- Unsloth-only repetition at matched NF4 points toward its loading/kernel/mode-transition path. Check saved dtype/config details before attributing a specific bug.
- Plain Transformers NF4-only repetition versus FP16 suggests sensitivity to quantization.
- Differences between plain Transformers FP16 and vLLM FP16 point toward inference implementation/numerics.
- No repetition at step75 does not clear the lost step100 checkpoint or the original training-time state.

Build `python make_notebook.py`; CPU checks `uv run --no-project --python 3.12 --with jsonschema python checks.py`. Failure-selected cases are diagnostics, not representative validation metrics. No GPU run launched.

CPU API integration: `uv run --no-project --python 3.12 --with torch==2.8.0+cpu --with transformers==4.56.2 --with peft==0.17.1 --extra-index-url https://download.pytorch.org/whl/cpu python check_plain_hf.py`. This verifies Qwen3/PEFT generation APIs, not GPU NF4/FP16 behavior.
