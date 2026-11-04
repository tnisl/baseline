# Faster validation: two GPUs, CUDA Graphs, prefix cache, one revision

Notebook: `kaggle_concise_validation_fast.ipynb`. Select **T4 ×2**, Internet on. Saving does not start a GPU job. The previous evaluation notebook/session is independent and remains unchanged.

Two separate vLLM workers each load the same base + Planner/Caller adapters on one T4. The fixed 245 validation IDs are partitioned by position into disjoint sets of 123 and122; each worker runs batch4 (up to8 concurrent workflows across the machine). No tensor parallelism or GPU-to-GPU model synchronization is needed. Batch grouping changes, so floating-point/scheduling differences can affect generated outputs despite temperature0.

`enforce_eager=False` enables CUDA Graph capture; `enable_prefix_caching=True` enables automatic prefix caching. Two LoRA slots, maximum four sequences per worker, FP16, context8192, GPU memory utilization0.85 to reserve more graph overhead. Chunked prefill remains disabled. Separate Triton/vLLM caches per worker. T4 LoRA compatibility patch writes atomically. First graph capture adds startup cost; GPU speedup and memory compatibility have not been benchmarked locally. Unsupported graph/cache settings fail visibly, with no silent fallback to eager. Prefix caching retains vLLM's adapter-aware cache isolation.

Each Planner decision has at most two generation attempts: initial + **one revision shared between error repair and optional completion review**. Each Caller step also has initial + one correction. Exhausted contract attempts terminate the sample with format_error; there is no outer retry loop that restarts the budget. Legitimate multi-step Planner→Caller→tool iteration remains, bounded by16 decisions /10 executed tool calls. Repeated-read prevention still sends feedback to the next decision. `revision_counts`, `repairs` and decision indices in traces make limits auditable. Runtime policy changes can change F1/score, so the fast run uses a fresh run name and does not reuse old-policy batch predictions.

Same Planner v4 best500, Caller v5 best371; same English prompts, TypeScript contracts and split1300/245. ASR runs once on825 current/history audio files before splitting LLM inference; no transcript oracle. No Summarizer model, finetuning, public-test inference or automatic submission.

Output root: `/kaggle/working/concise_validation/concise_two_gpu_graph_cache_retry1_valid245/validation/`. Worker artifacts are in `gpu0/`, `gpu1/`, including detailed `worker.log`, per-generation token counts/timing, batches, submissions, traces and reports. Parent prints worker progress and restores original split order when merging. The combined `report.json` recomputes local approximate tool F1, state accuracy and hash-gated score over all245 episodes; partial/duplicate/overlapping predictions are rejected. No averaging of worker scores with unequal sample counts.

Preserve the entire run directory for resume; set `RESUME_DIR` to the mode folder containing `config.json` and `validation/`. Only completed batches under the identical new config can be reused. A killed active batch is flagged rather than silently replayed. Parent terminates its own remaining worker if another worker fails. Ten-hour batch-boundary deadline remains; no per-sample timeout.

CPU verification: `python checks.py` tests four workflows, one-revision error recovery/exhaustion, shared Planner completion-review budget, no Summarizer calls, exact scoring, merged output ordering and duplicate rejection. `python make_notebook.py` rebuilds the embedded runtime.

Primary vLLM documentation: https://docs.vllm.ai/en/v0.9.2/ and https://docs.vllm.ai/en/v0.9.1/design/automatic_prefix_caching.html .
