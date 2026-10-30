"""Small Markdown reports and charts; import matplotlib only when plotting."""
from __future__ import annotations

from collections import Counter
from pathlib import Path


def pct(value):
    return "n/a" if value is None else f"{value:.2%}"


def num(value):
    return "n/a" if value is None else f"{value:.1f}"


def safe_text(value, length=100):
    return str(value).replace("|", "/").replace("\n", " ")[:length]


def write_asr_report(path, summary, cases):
    lines = [
        "# PhoWhisper ASR only",
        "",
        f"Model: {summary['model_id']}",
        f"Workflow SHA256: {summary.get('workflow_sha256', 'n/a')}",
        f"Samples: {summary['samples']}",
        f"Corpus WER: {pct(summary['corpus_wer'])}",
        f"Corpus CER: {pct(summary['corpus_cer'])}",
        f"Exact normalized transcripts: {summary['exact_normalized_transcripts']}",
        f"Word edits: S={summary['word_S']}, D={summary['word_D']}, I={summary['word_I']}",
        f"Audio duration mean/p95: {num(summary['duration_s']['mean'])}/{num(summary['duration_s']['p95'])} s",
        f"ASR latency mean/p95: {num(summary['latency_s']['mean'])}/{num(summary['latency_s']['p95'])} s",
        f"RTF mean/p95: {num(summary['rtf']['mean'])}/{num(summary['rtf']['p95'])}",
        "",
        "Per-sample ASR latency is the full batch wall time; every item waits for the batch to finish.",
        "",
        "## Highest WER cases",
        "",
        "| ID | WER | Reference | Prediction |",
        "|---|---:|---|---|",
    ]
    worst = sorted((x for x in cases if x["wer"] is not None),
                   key=lambda x: (x["wer"], x["word_errors"]), reverse=True)[:20]
    for row in worst:
        lines.append(f"| {row['id']} | {pct(row['wer'])} | {safe_text(row['reference'])} | {safe_text(row['transcript'])} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_agent_report(path, summary, cases):
    prompt = summary["prompt_tokens_per_node_call"]
    loops = summary["tool_iterations_per_sample"]
    decisions = summary["llm_calls_per_sample"]
    lines = [
        f"# {summary['mode']}",
        "",
        f"Model: {summary['model_id']}",
        f"LLM temperature: {summary.get('llm_temperature', 'n/a')}",
        f"Samples: {summary['samples']}",
        f"Mean score proxy: {pct(summary['mean_score_proxy'])}",
        f"Exact state-hash accuracy: {pct(summary['state_hash_accuracy'])}",
        f"Perfect proxy cases: {summary['perfect_proxy']}",
        f"Prompt context tokens mean/p95/max: {num(prompt['mean'])}/{num(prompt['p95'])}/{num(prompt['max'])}",
        f"Tool calls mean/p95/max: {num(loops['mean'])}/{num(loops['p95'])}/{num(loops['max'])}",
        f"LLM requests mean/p95/max: {num(decisions['mean'])}/{num(decisions['p95'])}/{num(decisions['max'])}",
        f"Planner calls mean/p95: {num(summary['planner_calls_per_sample']['mean'])}/{num(summary['planner_calls_per_sample']['p95'])}",
        f"Caller requests mean/p95: {num(summary['caller_calls_per_sample']['mean'])}/{num(summary['caller_calls_per_sample']['p95'])}",
        f"Planning rounds mean/p95: {num(summary['decisions_per_sample']['mean'])}/{num(summary['decisions_per_sample']['p95'])}",
        f"Tool failures mean/p95: {num(summary['tool_failures_per_sample']['mean'])}/{num(summary['tool_failures_per_sample']['p95'])}",
        f"Agent latency mean/p95: {num(summary['agent_latency_s']['mean'])}/{num(summary['agent_latency_s']['p95'])} s",
        f"Pipeline latency mean/p95: {num(summary['pipeline_latency_s']['mean'])}/{num(summary['pipeline_latency_s']['p95'])} s",
        f"Pipeline cases under 30 seconds: {summary['pipeline_under_30s'] if summary['pipeline_under_30s'] is not None else 'n/a'}",
        f"Context-trimmed samples: {summary['context_trimmed_samples']}",
        "",
        "Score proxy uses the local state hash and an approximate call matcher. It is not the official leaderboard score.",
        "",
        "## Stop reasons",
        "",
        "| Reason | Cases |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(summary["stop_reasons"].items()))
    lines.extend(["", "## Error flags", "", "| Flag | Cases |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in sorted(summary["error_flags"].items()))
    lines.extend(["", "## Lowest-scoring cases", "", "| ID | Score | Hash | Flags | Request | Transcript |",
                  "|---|---:|---|---|---|---|"])
    for row in sorted(cases, key=lambda x: (x["score_proxy"], not x["state_hash_match"]))[:20]:
        lines.append(f"| {row['id']} | {pct(row['score_proxy'])} | {row['state_hash_match']} | "
                     f"{safe_text(', '.join(row['error_flags']), 45)} | {safe_text(row['reference'], 80)} | "
                     f"{safe_text(row['transcript'], 80)} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_asr(path, cases, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    wer = [min(x["wer"], 2.0) for x in cases if x["wer"] is not None]
    axes[0].hist(wer, bins=20, color="#58a5d8")
    axes[0].set(xlabel="Utterance WER (clipped at 2.0)", ylabel="Samples",
                title=f"Corpus WER = {pct(summary['corpus_wer'])}")
    duration = [x["duration_s"] for x in cases]
    latency = [x["asr_latency_s"] for x in cases]
    axes[1].scatter(duration, latency, alpha=.5, s=14, color="#eb9365")
    axes[1].set(xlabel="Audio duration (s)", ylabel="ASR latency (s)", title="Duration vs inference")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_agent(path, cases, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    contexts = [n for x in cases for n in x["prompt_tokens_each"]]
    axes[0].hist(contexts, bins=20, color="#58a5d8", alpha=.7, label="Selector calls")
    axes[0].legend()
    axes[0].set(xlabel="Prompt tokens / LLM call", ylabel="Calls", title="Context length")
    counts = Counter(x["tool_iterations"] for x in cases)
    axes[1].bar(list(sorted(counts)), [counts[k] for k in sorted(counts)], color="#e8bd63")
    axes[1].set(xlabel="Tool calls / sample", ylabel="Samples", title="Tool iterations")
    flags = Counter(flag for x in cases for flag in x["error_flags"])
    common = flags.most_common(7)
    axes[2].barh([key for key, _ in reversed(common)], [n for _, n in reversed(common)], color="#e47f7f")
    axes[2].set(xlabel="Cases", title="Most common errors")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
