"""Kaggle-ready LifeMate baseline experiments.

Run from any cwd: python baseline_v7/run.py --mode all --data-root train+public_test
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import time

from .pipeline import Agent
from .common import (
    agent_summary, asr_case, asr_summary, dumps, evaluate_agent_case,
    find_root, import_toolenv, load_catalog, load_train, write_json,
)
from .reporting import plot_agent, plot_asr, write_agent_report, write_asr_report
from .reproducibility import prepare_process, runtime_versions, seed_everything


def workflow_sha256():
    """Fingerprint prompt assembly, pipeline and shared behavior for comparisons."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = sorted(p for p in here.rglob('*') if p.is_file() and p.suffix in ('.py', '.txt'))
    for path in paths:
        name = path.relative_to(here).as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_csv(path, cases):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cases:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in cases for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in cases:
            writer.writerow({key: dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def asr_experiment(rows, root, output_dir, model_id, batch_size, seed):
    from .models import PhoWhisper
    output_dir = Path(output_dir) / "asr"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = PhoWhisper(model_id)
    predictions = []
    path = output_dir / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, prediction in enumerate(predictor.transcribe(rows, root, batch_size), 1):
            prediction["seed"] = seed
            predictions.append(prediction)
            handle.write(dumps(prediction) + "\n")
            handle.flush()
            print(f"\rASR {index}/{len(rows)}", end="", flush=True)
    print()
    del predictor
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    by_id = {item["id"]: item for item in predictions}
    cases = [asr_case(row, by_id[row["id"]]) for row in rows]
    summary = asr_summary(cases)
    summary["model_id"] = model_id
    summary["seed"] = seed
    summary["latency_note"] = "Per-sample ASR latency is the full batch wall time; every item in the batch waits for the batch to finish."
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "cases.csv", cases)
    write_csv(output_dir / "errors.csv", sorted(
        (x for x in cases if x["word_errors"]), key=lambda x: (x["wer"] or 0, x["word_errors"]), reverse=True))
    write_asr_report(output_dir / "report.md", summary, cases)
    plot_asr(output_dir / "charts.png", cases, summary)
    print(f"ASR outputs: {output_dir}")
    return by_id


def load_or_run_asr(rows, root, output_dir, model_id, batch_size, seed):
    path = Path(output_dir) / "asr" / "predictions.jsonl"
    if path.is_file():
        predictions = read_jsonl(path)
        by_id = {item["id"]: item for item in predictions}
        if (all(row["id"] in by_id for row in rows)
                and all(by_id[row["id"]].get("model_id") == model_id
                        and by_id[row["id"]].get("seed") == seed
                        and by_id[row["id"]].get("requested_batch_size") == batch_size
                        and by_id[row["id"]].get("latency_basis") == "full_batch_wall_time"
                        for row in rows)):
            print(f"Reusing ASR predictions: {path}")
            return by_id
    return asr_experiment(rows, root, output_dir, model_id, batch_size, seed)



def agent_experiment(rows, transcripts, root, output_dir, mode, agent, catalog, ToolEnv, asr_predictions=None, workers=1):
    run_dir = Path(output_dir) / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    # Never leave an older successful summary next to traces from an aborted run.
    for name in ("summary.json", "report.md", "charts.png", "cases.csv", "errors.csv",
                 "predictions.jsonl", "traces.jsonl"):
        (run_dir / name).unlink(missing_ok=True)
    status = run_dir / "run_status.json"
    write_json(status, {"status": "running", "requested_samples": len(rows)})
    try:
        cases = _agent_experiment(rows, transcripts, root, output_dir, mode, agent,
                                 catalog, ToolEnv, asr_predictions, workers)
    except Exception as exc:
        write_json(status, {"status": "aborted", "error_type": type(exc).__name__,
                            "error": str(exc), "note": "Partial traces only; no valid batch summary."})
        raise
    write_json(status, {"status": "complete", "samples": len(cases)})
    return cases


def _agent_experiment(rows, transcripts, root, output_dir, mode, agent, catalog, ToolEnv, asr_predictions=None, workers=1):
    run_dir = Path(output_dir) / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    traces_path = run_dir / "traces.jsonl"
    cases = []
    experiment_start = time.perf_counter()
    from .batching import iter_outcomes
    with predictions_path.open("w", encoding="utf-8") as predictions_file, \
         traces_path.open("w", encoding="utf-8") as traces_file:
        outcomes = iter_outcomes(agent, rows, transcripts, workers)
        for index, (row, outcome) in enumerate(zip(rows, outcomes), 1):
            prediction = {key: outcome[key] for key in ("id", "tool_calls", "state_hash")}
            predictions_file.write(dumps(prediction) + "\n")
            predictions_file.flush()
            traces_file.write(dumps({
                "id": row["id"], "transcript": outcome["transcript"],
                "stop_reason": outcome["stop_reason"],
                "planner_calls": outcome["planner_calls"], "caller_calls": outcome["caller_calls"],
                "summarizer_calls": outcome["summarizer_calls"], "final_answer": outcome["final_answer"],
                "trace": outcome["trace"]
            }) + "\n")
            traces_file.flush()
            case = evaluate_agent_case(row, outcome, root, catalog, ToolEnv)
            if asr_predictions is not None:
                asr_item = asr_case(row, asr_predictions[row["id"]])
                case.update({
                    "asr_wer": asr_item["wer"], "asr_cer": asr_item["cer"],
                    "asr_latency_s": asr_item["asr_latency_s"],
                    "pipeline_latency_s": asr_item["asr_latency_s"] + case["agent_latency_s"],
                })
            cases.append(case)
            print(f"\r{mode} {index}/{len(rows)} score={case['score_proxy']:.2f} "
                  f"hash={int(case['state_hash_match'])} calls={case['tool_count']}", end="", flush=True)
    print()
    summary = agent_summary(cases)
    summary["experiment_wall_s"] = time.perf_counter() - experiment_start
    summary["samples_per_second"] = len(cases) / max(summary["experiment_wall_s"], 1e-9)
    summary["inference_workers"] = workers
    summary["latency_note"] = "Concurrent latency includes batch queue time; use workers=1 for isolated latency."
    summary["mode"] = mode
    summary["architecture"] = "planner_caller_summarizer"
    summary["termination_owner"] = "summarizer"
    summary["environment_error_policy"] = "review_by_summarizer"
    summary["feedback_source"] = "env.validate/env.call"
    summary["model_id"] = agent.llm.model_id
    summary["llm_backend"] = getattr(agent.llm, "backend", "transformers")
    summary["llm_dtype"] = getattr(agent.llm, "dtype", None)
    summary["llm_temperature"] = agent.llm.temperature
    summary["seed"] = agent.llm.seed
    summary["workflow_sha256"] = workflow_sha256()
    summary["metric_note"] = "score_proxy approximates contest call matching; hash match is exact via zaic_tools."
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "cases.csv", cases)
    write_csv(run_dir / "errors.csv", [x for x in cases if x["score_proxy"] < 1])
    write_agent_report(run_dir / "report.md", summary, cases)
    plot_agent(run_dir / "charts.png", cases, summary)
    print(f"{mode} outputs: {run_dir}")
    return cases


def compare_runs(output_dir, workflow_cases, baseline_cases, asr_predictions, rows):
    workflow = {row["id"]: row for row in workflow_cases}
    baseline = {row["id"]: row for row in baseline_cases}
    comparisons = []
    for row in rows:
        sample_id = row["id"]
        a, b = workflow[sample_id], baseline[sample_id]
        gold_ok, asr_ok = a["score_proxy"] == 1.0, b["score_proxy"] == 1.0
        category = (
            "both_correct" if gold_ok and asr_ok else
            "asr_sensitive" if gold_ok else
            "asr_helped" if asr_ok else
            "shared_failure"
        )
        comparisons.append({
            "id": sample_id, "category": category,
            "reference": row["audio_text"], "asr_transcript": asr_predictions[sample_id]["text"],
            "workflow_score_proxy": a["score_proxy"], "baseline_score_proxy": b["score_proxy"],
            "workflow_hash_match": a["state_hash_match"],
            "baseline_hash_match": b["state_hash_match"],
            "workflow_error_flags": a["error_flags"],
            "baseline_error_flags": b["error_flags"],
            "asr_wer": b.get("asr_wer"), "asr_cer": b.get("asr_cer"),
            "baseline_pipeline_latency_s": b.get("pipeline_latency_s"),
            "gold_tool_calls": a["gold_tool_calls"],
            "workflow_tool_calls": a["tool_calls"], "baseline_tool_calls": b["tool_calls"],
        })
    from collections import Counter
    category_counts = dict(Counter(row["category"] for row in comparisons))
    write_csv(Path(output_dir) / "comparison.csv", comparisons)
    write_json(Path(output_dir) / "comparison_summary.json", {
        "samples": len(comparisons), "categories": category_counts,
        "note": "asr_sensitive means gold-transcript agent was correct, ASR-transcript agent was not. It does not by itself prove the ASR error caused the failure."
    })
    print(f"Comparison: {category_counts}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("asr", "workflow", "baseline", "all"), default="all")
    parser.add_argument("--data-root", default=None, help="Folder containing train/train.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=50, help="Number of train samples; 0 means all")
    parser.add_argument("--sample-id", default=None,
                        help="Run exactly one train sample by ID, ignoring --limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asr-model", default="vinai/PhoWhisper-medium")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--llm-backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--llm-temperature", type=float, default=0.7,
                        help="Qwen sampling temperature; 0 uses greedy decoding")
    parser.add_argument("--vllm-max-model-len", type=int, default=12288)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-attention-backend",
                        choices=("TRITON_ATTN", "FLASH_ATTN", "FLASHINFER"),
                        default="TRITON_ATTN")
    parser.add_argument("--vllm-dtype", choices=("float16", "bfloat16"),
                        default="float16", help="Use float16 on Tesla T4")
    parser.add_argument("--vllm-use-cuda-graphs", action="store_true",
                        help="Faster on some GPUs, but reserves extra VRAM")
    parser.add_argument("--inference-workers", type=int, default=1,
                        help="Independent samples in flight; >1 requires vLLM")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=None,
                        help="Defaults to inference-workers")
    parser.add_argument("--batch-wait-ms", type=float, default=10)
    parser.add_argument("--no-structured-outputs", action="store_true",
                        help="Disable per-node constrained JSON generation for A/B")
    parser.add_argument("--asr-batch-size", type=int, default=1)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=24,
                        help="Maximum rounds; each has one planner, at most one caller and one summarizer")
    parser.add_argument("--save-node-inputs", action="store_true",
                        help="Save exact rendered node inputs in traces for later dataset construction")
    parser.add_argument("--max-no-progress", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=11500)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_process(args.seed, vllm=args.llm_backend == "vllm" and args.mode != "asr")
    seed_everything(args.seed)
    if args.limit < 0 or args.asr_batch_size < 1 or args.max_no_progress < 1:
        raise ValueError("--limit must be >= 0, --asr-batch-size and --max-no-progress must be >= 1")
    if min(args.max_tool_calls, args.max_decisions, args.max_prompt_tokens, args.max_new_tokens) < 1:
        raise ValueError("Tool, decision and token limits must be positive")
    if not 0 <= args.llm_temperature <= 2:
        raise ValueError("--llm-temperature must be in [0, 2]")
    if args.max_prompt_tokens + args.max_new_tokens > 16000:
        raise ValueError("Prompt plus generation limit must not exceed 16000 tokens")
    if args.llm_backend == "vllm":
        if args.max_prompt_tokens + args.max_new_tokens > args.vllm_max_model_len:
            raise ValueError("Prompt plus generation limit exceeds --vllm-max-model-len")
        if not 0 < args.vllm_gpu_memory_utilization <= 1:
            raise ValueError("--vllm-gpu-memory-utilization must be in (0, 1]")
        if args.vllm_tensor_parallel_size < 1:
            raise ValueError("--vllm-tensor-parallel-size must be >= 1")
    if args.inference_workers < 1 or args.batch_wait_ms < 0:
        raise ValueError("workers must be >=1 and batch wait >=0")
    if args.inference_workers > 1 and args.llm_backend != "vllm":
        raise ValueError("Concurrent inference requires vLLM")
    args.vllm_max_num_seqs = args.vllm_max_num_seqs or args.inference_workers
    if args.vllm_max_num_seqs < args.inference_workers:
        raise ValueError("max-num-seqs must be >= inference-workers")
    root = find_root(args.data_root)
    if args.sample_id:
        rows = [row for row in load_train(root, 0, args.seed)
                if row["id"] == args.sample_id]
        if not rows:
            raise ValueError(f"Train sample not found: {args.sample_id}")
    else:
        rows = load_train(root, args.limit, args.seed)
    output_dir = Path(args.output_dir or (
        "/kaggle/working/lifemate_baseline_v7" if Path("/kaggle/working").is_dir()
        else Path(__file__).resolve().parents[2] / "outputs"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["structured_output_backend"] = "guidance" if args.llm_backend == "vllm" and not args.no_structured_outputs else None
    config.update({"architecture": "planner_caller_summarizer",
                   "feedback_source": "env.validate/env.call", "model_authored_feedback": False,
                   "planner_guidance": "instruction_max_240_chars", "summarizer": True,
                   "termination_owner": "summarizer", "environment_error_policy": "review_by_summarizer",
                   "resolved_data_root": str(root), "sample_ids": [row["id"] for row in rows],
                   "sample_count": len(rows), "reproducibility": runtime_versions(),
                   "workflow_sha256": workflow_sha256()})
    write_json(output_dir / "run_config.json", config)
    print(f"Data root: {root} | samples: {len(rows)} | output: {output_dir}")
    print("Architecture: planner_caller_summarizer | summarizer owns completion", flush=True)

    asr_predictions = None
    if args.mode in ("asr", "all"):
        asr_predictions = asr_experiment(rows, root, output_dir, args.asr_model,
                                         args.asr_batch_size, args.seed)
    elif args.mode == "baseline":
        asr_predictions = load_or_run_asr(rows, root, output_dir, args.asr_model,
                                          args.asr_batch_size, args.seed)
    if args.mode == "asr":
        return

    catalog = load_catalog(root)
    ToolEnv = import_toolenv(root)
    if args.llm_backend == "vllm" and not args.no_structured_outputs:
        from .contracts import preflight_schemas
        count = preflight_schemas(catalog)
        print(f"Structured output preflight: {count} schema variants compiled with guidance", flush=True)
    if args.llm_backend == "vllm":
        from .models import QwenVLLM
        llm = QwenVLLM(
            args.llm_model,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            attention_backend=args.vllm_attention_backend,
            dtype=args.vllm_dtype,
            enforce_eager=not args.vllm_use_cuda_graphs,
            seed=args.seed,
            temperature=args.llm_temperature,
            max_num_seqs=args.vllm_max_num_seqs,
            structured_outputs=not args.no_structured_outputs,
        )
    else:
        from .models import Qwen
        llm = Qwen(args.llm_model, temperature=args.llm_temperature,
                   seed=args.seed)
    if args.inference_workers > 1:
        from .batching import BatchedLLM
        llm = BatchedLLM(llm, args.inference_workers, args.batch_wait_ms)
    agent = Agent(
        llm, catalog, ToolEnv, root,
        max_tool_calls=args.max_tool_calls, max_decisions=args.max_decisions,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens, max_no_progress=args.max_no_progress,
        save_node_inputs=args.save_node_inputs,
    )
    try:
        workflow_cases = baseline_cases = None
        if args.mode in ("workflow", "all"):
            workflow_cases = agent_experiment(
                rows, {row["id"]: row["audio_text"] for row in rows},
                root, output_dir, "workflow_gold", agent, catalog, ToolEnv, workers=args.inference_workers)
        if args.mode in ("baseline", "all"):
            baseline_cases = agent_experiment(
                rows, {row["id"]: asr_predictions[row["id"]]["text"] for row in rows},
                root, output_dir, "baseline_asr", agent, catalog, ToolEnv, asr_predictions, workers=args.inference_workers)
        if workflow_cases is not None and baseline_cases is not None:
            compare_runs(output_dir, workflow_cases, baseline_cases, asr_predictions, rows)
    finally:
        if args.inference_workers > 1:
            llm.close()
    print(f"Finished. Results: {output_dir}")


if __name__ == "__main__":
    main()
