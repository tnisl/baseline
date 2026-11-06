"""Prepare → tokenize → Unsloth DDP → clean Transformers validation subprocesses."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .config import RunConfig
from .data import prepare
from .evaluate import combine
from .io import write_json


def launch(config, train_python, eval_python):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible is not None else [str(i) for i in range(config.world_size)]
    if len(devices) < config.world_size or any(not device.strip() for device in devices[:config.world_size]):
        raise ValueError("NUM_GPU exceeds CUDA_VISIBLE_DEVICES")
    devices = devices[:config.world_size]
    environment = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONHASHSEED=str(config.seed),
                       TQDM_DISABLE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
                       CUDA_VISIBLE_DEVICES=",".join(devices), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    config_path = str(config.path / "config.json")
    subprocess.run([eval_python, "-u", "-m", "twonode.tokenization", "--config", config_path],
                   env=environment, check=True)
    # Train exits completely before starting any validation model process.
    subprocess.run([train_python, "-u", "-m", "torch.distributed.run", "--standalone",
                    f"--nproc_per_node={config.world_size}", "--module", "twonode.train", "--config", config_path],
                   env=environment, check=True)
    processes = []
    try:
        for rank in range(config.world_size):
            worker_env = dict(environment, CUDA_VISIBLE_DEVICES=devices[rank])
            # torchrun's distributed variables must not leak into plain inference.
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                worker_env.pop(key, None)
            processes.append(subprocess.Popen([eval_python, "-u", "-m", "twonode.evaluate",
                "--config", config_path, "--worker-rank", str(rank)], env=worker_env))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Transformers validation workers failed: {codes}; see their logs")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    return combine(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--eval-batch", type=int, default=2)
    parser.add_argument("--num-gpu", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-acc", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--train-python", default=sys.executable)
    parser.add_argument("--eval-python", default=sys.executable)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--check-kernels", action="store_true", help="Inspect GDN bindings and profile optimizer step 2")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new empty output directory; never overwrite or silently resume a run")
    config = RunConfig(output=str(args.output.resolve()), profile=args.profile,
                       revision=args.revision, max_length=args.max_length, eval_batch=args.eval_batch,
                       epochs=args.epochs, learning_rate=args.learning_rate, evaluate_base=not args.skip_base,
                       check_kernels=args.check_kernels, world_size=args.num_gpu,
                       micro_batch=args.micro_batch, gradient_accumulation=args.grad_acc)
    config.path.mkdir(parents=True, exist_ok=True)
    manifest = prepare(args.data_root, args.split_file, config.prepared, config.profile == "smoke")
    print(json.dumps({"data_counts": manifest["counts"], "quarantine": manifest["quarantine"]}), flush=True)
    # Resolve once so tokenizer, Unsloth and ordinary Transformers share exact weights.
    resolved = subprocess.check_output([args.eval_python, "-c",
        "from huggingface_hub import HfApi; import sys; print(HfApi().model_info(sys.argv[1], revision=sys.argv[2]).sha)",
        config.model, config.revision], text=True).strip()
    if len(resolved) != 40 or any(ch not in "0123456789abcdef" for ch in resolved):
        raise ValueError("Model revision did not resolve to an immutable SHA")
    config.revision = resolved
    config.save()
    source = Path(__file__).resolve().parents[2]
    if (source / "source_manifest.json").exists():
        shutil.copy2(source / "source_manifest.json", config.path / "source_manifest.json")
    with (config.path / "packages.txt").open("w") as stream:
        subprocess.run([args.train_python, "-m", "pip", "freeze"], stdout=stream, check=True)
    reports = launch(config, args.train_python, args.eval_python)
    write_json(config.path / "run_complete.json", {"profile": config.profile,
               "model_revision": resolved, "validation_examples": reports["adapter"]["samples"],
               "validation_episodes": reports["adapter"]["episodes"],
               "adapter_accuracy": reports["adapter"]["accuracy"]})
    print(json.dumps({name: {k: v for k, v in report.items() if k not in ("by_tool", "by_group", "by_app")}
                      for name, report in reports.items()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
