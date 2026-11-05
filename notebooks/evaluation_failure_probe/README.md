# Experiment1: base precision x LoRA dtype

Kaggle: https://www.kaggle.com/code/lctrnnguynkhi/lifemate-evaluation-failure-probe

Four cases, one authentic Planner checkpoint at step75:

| Case | Base weights | LoRA parameters |
|---|---|---|
| nf4_lora_fp32 | NF4, double quantization, FP16 compute | FP32 |
| nf4_lora_fp16 | NF4, double quantization, FP16 compute | FP16 |
| fp16_lora_fp32 | FP16 | FP32 |
| fp16_lora_fp16 | FP16 | FP16 |

All cases use plain Transformers4.56.2 + PEFT0.17.1, Torch2.8.0, bitsandbytes0.47.0 and **SDPA**. The process asserts no Unsloth/Unsloth Zoo import. The pinned training package lock is reused for version parity; installed packages do not imply imported patches. Four fresh subprocesses run sequentially on the same GPU0. Select T4 x2 on Kaggle; this controlled test uses only its first GPU. Publication is QuickSave, no automatic run.

For each case, run the same six original validation prefixes at **batch1 and4**:48 outputs in total. Four historical looping states: train_1070 step1, train_0550 step1, train_1124 step1, train_0486 step0. Controls: train_1405 step1, train_0442 step0. Same pinned Qwen/Qwen3-4B-Instruct-2507 revision, original English prompt/TypeScript schema, greedy, max512 output tokens, EOS151645/151643 and pad151643. The512 cap stays for comparison with previous diagnostic runs; it is independent of the finetuning notebook's256/192 caps. No training, optimizer, ASR, tool execution, repairs or enforcer.

The adapter is explicitly cast after PEFT loading, then **reloaded from the original checkpoint into that dtype**. This avoids testing FP32 values already rounded by an initial FP16 load. Every adapter tensor is compared against original source weights cast to the requested dtype; dtype is verified again after every batch. Only LoRA parameters are cast, never the quantized base model. Any mismatch fails visibly without fallback.

Private input remains `lctrnnguynkhi/lifemate-evaluation-failure-inputs`; manifests and file hashes are verified. The source contains exact unchanged messages from session365792172. Target labels only enter scoring, never generation. Historical step100 outputs remain available for reference; its weights were not saved, so this is not an exact reproduction of step100.

Outputs: `/kaggle/working/evaluation_failure_probe_dtype2x2/`.

- `report.md`, `report.json`: repetition/schema/match/length metrics and comparisons holding either base precision or LoRA dtype fixed.
- `<case>_step75.jsonl`: raw output, exact generated token IDs, stop reason, prompt hash and batch time.
- `<case>_step75_settings.json`: actual quantization config, requested/actual LoRA dtype, source-weight verification and attention implementation. Summary rejects an incorrect dtype or attention backend.
- `<case>_step75.log`: full startup/generation/error output.
- `input_manifest.json`, `source.json`: provenance; separate output root avoids mixing previous experiment results.

Interpretation: if both NF4 cases loop and both FP16 cases do not, base quantization/loading is implicated more strongly. If changing LoRA dtype changes behavior while holding base precision fixed, adapter arithmetic/rounding matters. These observations do not prove a particular library bug, and dtype interacts with the compute kernels. The six selected prefixes are not a validation score. Startup/warmup and generated lengths affect timing comparisons. Failed runs retain raw outputs even if final summary is unavailable.

Build: `python make_notebook.py`.

CPU checks:

```bash
uv run --no-project --python 3.12 --with jsonschema python checks.py
uv run --no-project --python 3.12 --with jsonschema --with torch==2.8.0+cpu --with transformers==4.56.2 --with peft==0.17.1 --extra-index-url https://download.pytorch.org/whl/cpu python check_plain_hf.py
```

The real tiny Qwen3/PEFT CPU test checks explicit FP16/FP32 adapter casts, source-weight restoration, unchanged base tensors and left-padded generation. It does not establish GPU NF4 behavior. No GPU run launched.
