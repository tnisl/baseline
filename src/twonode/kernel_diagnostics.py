"""Inspect patched Gated DeltaNet bindings and summarize an actual training trace."""
from collections import Counter
import importlib.metadata
import os
from pathlib import Path
import sys

from .io import write_json


BINDINGS = ("causal_conv1d_fn", "causal_conv1d_update",
            "chunk_gated_delta_rule", "recurrent_gated_delta_rule")


def callable_info(function):
    if function is None:
        return {"available": False, "implementation_hint": "unavailable"}
    chain, seen = [], set()
    while function is not None and id(function) not in seen:
        seen.add(id(function))
        name = getattr(function, "__qualname__", type(function).__qualname__)
        module = getattr(function, "__module__", type(function).__module__)
        code = getattr(function, "__code__", None)
        chain.append({"name": name, "module": module,
                      "file": code.co_filename if code else None})
        function = getattr(function, "__wrapped__", None) or getattr(function, "_torchdynamo_orig_callable", None)
    identity = " ".join(f"{item['module']}.{item['name']}" for item in chain).lower()
    if any(item["name"].split(".")[-1].startswith("torch_") for item in chain):
        hint = "torch_fallback"
    elif "fla" in identity or "triton" in identity:
        hint = "fla_or_triton_binding"
    elif "causal_conv1d" in identity:
        hint = "causal_conv1d_binding"
    else:
        hint = "unknown"
    return {"available": True, "implementation_hint": hint, "wrapper_chain": chain}


def kernel_inventory(model):
    layers = []
    for name, module in model.named_modules():
        if "gateddeltanet" not in type(module).__name__.lower() and not hasattr(module, "chunk_gated_delta_rule"):
            continue
        bindings = {key: callable_info(getattr(module, key)) for key in BINDINGS if hasattr(module, key)}
        if "causal_conv1d_fn" in bindings and not bindings["causal_conv1d_fn"]["available"]:
            bindings["causal_conv1d_fn"]["implementation_hint"] = "torch_conv1d_fallback_hint"
        layers.append({"layer": name, "class": f"{type(module).__module__}.{type(module).__name__}",
                       "forward": callable_info(module.forward), "bindings": bindings})
    versions = {}
    for package in ("torch", "transformers", "unsloth", "unsloth_zoo", "triton",
                    "flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # This flag can predate Unsloth patches; retain it as evidence, never as the verdict.
    module_flags = {name: bool(module.is_fast_path_available)
                    for name, module in list(sys.modules.items()) if module is not None
                    and "qwen3_5" in name and hasattr(module, "is_fast_path_available")}
    hints = Counter(f"{key}:{value['implementation_hint']}" for layer in layers
                    for key, value in layer["bindings"].items())
    return {"gated_delta_layers": len(layers), "binding_counts": dict(hints), "layers": layers,
            "package_versions": versions, "module_fast_path_flags": module_flags,
            "environment": {key: os.environ.get(key) for key in (
                "UNSLOTH_FORCE_FLOAT32", "UNSLOTH_COMPILE_DISABLE",
                "UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D", "UNSLOTH_ENABLE_CCE")},
            "interpretation": "Bindings and package presence do not prove execution or hardware support. "
                              "Training prefill uses chunk/conv forward; recurrent/update bindings concern cached decode."}


def vendor_counters():
    """Read optional already-loaded Unsloth counters without importing/changing kernels."""
    result = {}
    for name, module in list(sys.modules.items()):
        if module is None or "unsloth" not in name:
            continue
        counters = vars(module).get("GDN_CAUSAL_CONV1D_STATS")
        if isinstance(counters, dict):
            result[name] = {key: value for key, value in counters.items() if isinstance(value, int)}
    return result


def summarize_events(events):
    kernels, operators = {}, {}
    for event in events:
        cuda = str(event.device_type).split(".")[-1] == "CUDA"
        target = kernels if cuda else operators
        item = target.setdefault(event.name, {"name": event.name, "calls": 0, "self_time_us": 0.0})
        item["calls"] += 1
        item["self_time_us"] += float(event.device_time_total if cuda else event.self_cpu_time_total)
    kernels = sorted(kernels.values(), key=lambda item: item["self_time_us"], reverse=True)
    operators = sorted(operators.values(), key=lambda item: item["self_time_us"], reverse=True)
    patterns = {
        "gated_delta_or_fla_candidates": ("gated_delta", "delta_rule", "chunk_fwd", "chunk_bwd", "fused_recurrent"),
        "causal_conv_candidates": ("causal_conv",),
        "other_conv_candidates": ("conv1d", "convolution", "cudnn"),
        "matrix_multiply_candidates": ("gemm", "matmul", "sgemm", "hgemm"),
    }
    candidates = {group: [item for item in kernels if any(word in item["name"].lower() for word in words)]
                  for group, words in patterns.items()}
    return {"cuda_events_recorded": bool(kernels), "cuda_kernel_calls": sum(k["calls"] for k in kernels),
            "cuda_kernel_time_us": sum(k["self_time_us"] for k in kernels),
            "cuda_kernels": kernels, "top_cpu_operators": operators[:80], "kernel_name_candidates": candidates,
            "interpretation": "CUDA names prove those kernels appeared in this step; name groups are search hints, "
                              "not a universal fast-path classifier. Empty matches are inconclusive. "
                              "Kernel-time sums can overlap across streams and are not wall time."}


def save_profile(profiler, output, rank, step, counters_before):
    directory = Path(output) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / f"rank{rank}.step{step}.trace.json"
    profiler.export_chrome_trace(str(trace))
    summary = summarize_events(profiler.events())
    after = vendor_counters()
    summary.update({"rank": rank, "optimizer_step": step, "trace": trace.name,
                    "vendor_counters_after": after,
                    "vendor_counter_deltas": {module: {key: value - counters_before.get(module, {}).get(key, 0)
                                               for key, value in values.items()} for module, values in after.items()},
                    "timing_note": "Profiled step includes instrumentation overhead; exclude it from throughput comparisons."})
    write_json(directory / f"rank{rank}.step{step}.profile.json", summary)
    return summary
