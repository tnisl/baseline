# Diagnose slow/looping evaluation on six real prefixes

Kaggle: https://www.kaggle.com/code/lctrnnguynkhi/lifemate-evaluation-failure-probe

Run yourself on **T4 x2**, Internet on. Publication is QuickSave only. No finetuning, optimizer updates, ASR, tool execution, retries or enforcer.

Four historical looping states: `train_1070` step1, `train_0550` step1, `train_1124` step1, `train_0486` step0. Controls: `train_1405` step1, `train_0442` step0. Uses the exact saved validation-prefix messages from session365792172; target/reference fields are only used for scoring/CE probes, never passed to generation. Six original historical step100 outputs remain in the private input dataset.

**Step100 weights were lost on cancellation.** The latest saved adapter is **step75** and is copied unchanged with hashes. This notebook cannot reproduce step100 exactly. It does not use the older completed Planner v4 adapter or its old prompts.

Eight cases,48 outputs:

| Engine | Model | Batch sizes | GPU |
|---|---|---|---|
| Unsloth2025.10.9 / Transformers | Base with zero LoRA contribution; step75 | 1,4 | 0 |
| vLLM0.9.2 | Base; step75 | 1,4 | 1 |

Fresh subprocess for each engine/model case, sequential execution. Each compares batch1/4 with the same six messages, tokenizer IDs, greedy decoding, max512 tokens, no repetition penalty or schema enforcer. Unsloth compile ON, resident-style inference context with hidden-state/logits flags switched exactly as periodic eval. A supervised CE forward runs before generation and after each generation batch to verify restoration; no backward or optimizer. T4 vLLM compatibility patch, graphs ON, prefix caching OFF. Base/adapter weights are not shared between processes.

**Unsloth uses NF4 and vLLM uses FP16**, matching the two existing runtimes. A cross-engine difference alone cannot separate backend from quantization. This is a standalone evaluation probe, not a recreated DDP training session with optimizer/gradients/memory pressure. Startup, warmup, compilation and processing order affect latency comparisons.

Private input `lctrnnguynkhi/lifemate-evaluation-failure-inputs` contains authentic step75 adapter/config, six unchanged prepared records, catalog, historical step100 outputs and hashes. No full competition dataset required.

Outputs at `/kaggle/working/evaluation_failure_probe/`:

- `report.md`, `report.json`: per-case schema/match/repetition/length counts, generation time and paired comparisons.
- `<engine>_<model>.jsonl`: raw text, exact generated token IDs/count, EOS/length/stop reason, prompt hash and batch duration.
- `<engine>_<model>_settings.json`: effective generation settings and precision.
- `<engine>_<model>.log`: startup, generation progress, pre/post CE restoration checks and underlying exceptions.
- `input_manifest.json`, `source.json`: source/data/checkpoint provenance. CPU source verification does not establish GPU compatibility.

HF token counts include EOS when present; vLLM token count follows returned token IDs and may omit its stop token. Compare finish/stop reasons as well as counts. HF per-row counts strip padded positions at first configured EOS. Batch duration is shared by rows in that batch; report sums it only once. Prompt IDs must match in all eight cases.

Interpretation:

- Batch4-only changes suggest batching/numerics/cache/padding effects, not proof of a specific bug.
- Adapter-only repetition on both engines suggests checkpoint behavior; it does not establish label noise.
- Unsloth-only repetition suggests the backend/precision/mode-transition combination needs isolation.
- No repetition at step75 does not clear the lost step100 checkpoint or the original training-time state.

Build `python make_notebook.py`; CPU checks `uv run --no-project --python 3.12 --with jsonschema python checks.py`. Failure-selected cases are diagnostics, not representative validation metrics. No GPU run launched.
