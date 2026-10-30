"""Set process and library seeds before loading ASR or LLM models."""
from __future__ import annotations

import importlib.metadata
import os
import random
import sys


def prepare_process(seed: int, *, vllm: bool) -> None:
    """Restart once when needed so Python's hash seed applies from startup."""
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("--seed must be between 0 and 4294967295")
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(seed)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if vllm:
        environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if any(os.environ.get(key) != value for key, value in environment.items()):
        # Preserve `python -m lifemate_v7` as well as direct script entry points.
        original = getattr(sys, 'orig_argv', None)
        argv = [sys.executable, *original[1:]] if original else [sys.executable, *sys.argv]
        os.execvpe(sys.executable, argv, environment)


def seed_everything(seed: int) -> None:
    """Seed host RNGs and request deterministic PyTorch kernels where available."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def runtime_versions() -> dict:
    """Record enough of the software environment to compare runs."""
    versions = {}
    for package in ("torch", "transformers", "vllm", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "vllm_v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
        "torch_deterministic_algorithms": "warn_only",
    }
