# Planner backend / guided decoding comparison

Controlled diagnostic for the long/repeating Planner thoughts seen in full validation. Notebook `kaggle_planner_backend_comparison.ipynb`, T4x2, Internet on. No finetuning or ASR; no validation IDs are used as probe inputs and no tool workflow is deployed. Prompts are unchanged from the completed Planner adapter; TypeScript tool schemas remain unchanged.

Twelve train-only probes are selected deterministically: eight initial states, two gold-prefix states and two terminal states. Prior observations are replayed only on training episodes. Request messages/schema are stored separately from reference next/tool labels. Reference labels are read only by the output assessor, never inserted into generation. Probe match is diagnostic on seen training states, not held-out accuracy or contest score.

Seven conditions, each on the same12 inputs:

| Backend | Adapter | Grammar | CUDA Graph |
|---|---|---|---|
| Transformers | base only | off | off |
| Transformers | trained Planner | off | off |
| Transformers | trained Planner | LMFE | off |
| vLLM | base only | off | off |
| vLLM | trained Planner | off | off |
| vLLM | trained Planner | LMFE | off |
| vLLM | trained Planner | LMFE | on |

Both run the same pinned model revision, saved Planner tokenizer, FP16 base and LoRA factors, greedy decoding with repetition penalty1, seed42, batch1 and max320 new tokens. The actual rendered prompt, input token IDs and schema hashes must agree across all conditions or report generation fails. HF uses SDPA; vLLM uses XFormers and the existing T4 LoRA compatibility patch. These kernels need not be bitwise equivalent. The experiment does not recreate NF4 training precision or prove batch4 behavior. Prefix caching is disabled throughout to isolate the other factors and avoid the known SM75 compiler failure. The Graph condition uses a fresh subprocess after eager engines exit.

Transformers runs GPU0 and vLLM GPU1 concurrently, then the vLLM Graph control runs GPU1. Saved outputs are flushed after every probe. On a worker error the parent prints its log and stops the other worker. Use a fresh RUN_NAME on rerun; incomplete result files are not silently mixed or overwritten. No automatic full validation or other notebooks are started by this notebook.

Outputs under `/kaggle/working/planner_backend_comparison/planner_backend_diagnostic_v1/`:

- `report.json`, `summary.csv`: JSON validity, thought length, repeated6-gram rates, near1024-character counts, tool/reference match and timing per condition.
- `transformers.jsonl`, `vllm.jsonl`, `vllm_graph.jsonl`: raw outputs, parsed JSON, finish reason, hashes and per-probe measurements.
- `*_environment.json`: framework versions, model revision and hardware. HF also checks nonzero adapter effect on next-token logits compared with base.
- `requests.json`, `references.json`, `probe_manifest.json`: reproducible probe inputs and scorer-only targets.
- `transformers.log`, `vllm.log`: worker logs. Graph control also prints to Kaggle Logs.

Repetition is measured on raw text even when JSON parsing fails; otherwise unguided prose could appear falsely less repetitive. Thought length averages in the report are conditional on valid structured outputs. Timings include first-request compilation/warmup and host grammar overhead; this is not a warmed throughput benchmark. Output token counts use backend-specific EOS conventions and must not be treated as exact cross-backend tokenizer disagreements.

Interpretation: repetition in both adapted backends but not base suggests an adapter/training contribution; only guided modes suggests a grammar interaction; only Graph suggests a graph/LoRA path issue. These are diagnostic signals, not causal proof from twelve inputs. Numerical precision/kernel differences remain. Investigate per-input outputs before changing labels, prompts or retraining.

Rebuild: `python make_notebook.py`. CPU checks: `uv run --no-project --python 3.12 --with jsonschema python checks.py`. No model weights or private request text are committed.
