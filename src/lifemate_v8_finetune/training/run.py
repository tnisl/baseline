"""One reproducible, logged entry point for Kaggle and local GPU training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import os
import zipfile

from ..common import write_json
from .data import ROLES, stable_hash

ROOT = Path(__file__).resolve().parents[3]
PROFILES = {
    "smoke": {"data_limit": 20, "max_steps": 2, "gradient_accumulation": 1, "logging_steps": 1},
    "train": {"data_limit": 0, "max_steps": -1, "gradient_accumulation": 16, "logging_steps": 5},
}


def run_logged(command, log_path):
    """Preserve the original traceback while streaming progress to Kaggle."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    print("Console log:", log_path, flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=environment,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"Job exited with code {code}. Full traceback: {log_path}")


def verify_adapter(path, role):
    path = Path(path)
    for name in ("adapter_config.json", "adapter_model.safetensors", "role_metadata.json", "training_metrics.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    metadata = json.loads((path / "role_metadata.json").read_text())
    metrics = json.loads((path / "training_metrics.json").read_text())
    if metadata.get("role") != role or metadata.get("global_finetune") is not False:
        raise ValueError(f"Invalid role-local adapter metadata: {path}")
    verification = metrics.get("adapter_verification", {})
    if not verification.get("finite") or not verification.get("updated"):
        raise ValueError(f"Adapter has no verified finite weight update: {path}")
    return metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    p.add_argument("--data-root")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    p.add_argument("--holdout-ids", default=str(ROOT / "configs/holdout_ids_seed42.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=.05)
    p.add_argument("--negative-weight", type=float, default=.1)
    p.add_argument("--ranking-beta", type=float, default=1.)
    p.add_argument("--ranking-margin", type=float, default=.2)
    p.add_argument("--negatives-per-example", type=int, default=2)
    p.add_argument("--counterfactuals-per-case", type=int, default=2)
    p.add_argument("--validation-fraction", type=float, default=.1)
    p.add_argument("--save-steps", type=int, default=50)
    for name in ("data-limit", "max-steps", "gradient-accumulation", "logging-steps"):
        p.add_argument("--" + name, type=int, default=None)
    p.add_argument("--full-precision-base", action="store_true")
    p.add_argument("--cuda-debug", action="store_true")
    p.add_argument("--overwrite-adapters", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if len(set(args.roles)) != len(args.roles):
        raise ValueError("Each role may occur only once")
    settings = vars(args).copy()
    for name, default in PROFILES[args.profile].items():
        if settings[name] is None:
            settings[name] = default
    output = Path(args.output_dir).expanduser().resolve()
    data = output / "data"
    training = output / "local_lora"
    logs = output / "logs"
    for role in args.roles:
        if (training / "adapters" / role / "adapter_config.json").exists() and not args.overwrite_adapters:
            raise FileExistsError(f"Adapter {role} already exists. Choose a new output directory or --overwrite-adapters.")
    status = {"status": "preparing", "profile": args.profile, "roles": args.roles, "completed_roles": [],
              "started_at": datetime.now(timezone.utc).isoformat(), "global_finetune": False,
              "config": settings}
    try:
        status["source_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        status["source_commit"] = None
    write_json(output / "run_status.json", status)
    write_json(output / "run_config.json", status)
    try:
        prepare = [sys.executable, ROOT / "prepare_data.py", "--output-dir", data,
                   "--seed", args.seed, "--limit", settings["data_limit"],
                   "--validation-fraction", args.validation_fraction,
                   "--negatives-per-example", args.negatives_per_example,
                   "--counterfactuals-per-case", args.counterfactuals_per_case]
        if args.data_root:
            prepare += ["--data-root", args.data_root]
        if args.holdout_ids:
            prepare += ["--holdout-ids", args.holdout_ids]
        run_logged(prepare, logs / "prepare_data.console.log")
        manifest = json.loads((data / "manifest.json").read_text())
        status["dataset_manifest_sha256"] = stable_hash(manifest)
        metrics = {}
        for role in args.roles:
            status.update(status="training", current_role=role)
            write_json(output / "run_status.json", status)
            command = [sys.executable, ROOT / "train_role.py", "--role", role,
                       "--data-dir", data, "--output-dir", training, "--model", args.model]
            for name in ("seed", "gpu_index", "max_seq_length", "epochs", "learning_rate", "batch_size",
                         "lora_r", "lora_alpha", "lora_dropout", "negative_weight", "ranking_beta",
                         "ranking_margin", "save_steps", "max_steps", "gradient_accumulation", "logging_steps"):
                command += ["--" + name.replace("_", "-"), settings[name]]
            if args.full_precision_base:
                command.append("--full-precision-base")
            if args.cuda_debug:
                command.append("--cuda-debug")
            if args.overwrite_adapters:
                command.append("--overwrite-adapter")
            run_logged(command, training / "training_runs" / f"{role}.console.log")
            metrics[role] = verify_adapter(training / "adapters" / role, role)
            status["completed_roles"].append(role)
            write_json(output / "run_status.json", status)
        archive = output / "lora_adapters.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for role in args.roles:
                for path in sorted((training / "adapters" / role).rglob("*")):
                    if path.is_file():
                        z.write(path, Path("adapters") / role / path.relative_to(training / "adapters" / role))
        status.update(status="complete", current_role=None,
                      finished_at=datetime.now(timezone.utc).isoformat(), adapter_archive=str(archive))
        write_json(output / "metrics.json", metrics)
        write_json(output / "run_status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        status.update(status="failed", error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "run_status.json", status)
        raise


if __name__ == "__main__":
    main()
