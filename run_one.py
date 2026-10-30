"""Run the baseline_v4 workflow for one train ID without replacing batch results.

Example:
    python baseline_v4/run_one.py train_1703 --data-root train+public_test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run import main


def single_sample_args(argv):
    parser = argparse.ArgumentParser(
        description="Run the baseline_v4 workflow for one train sample. Other run.py flags pass through."
    )
    parser.add_argument("sample_id", help="Train ID, for example train_1703")
    known, extra = parser.parse_known_args(argv)
    sample_id = known.sample_id
    if not sample_id.startswith("train_"):
        parser.error("Cần ID train, ví dụ train_1703.")
    args = ["--mode", "workflow", "--sample-id", sample_id]
    if not any(arg == "--output-dir" or arg.startswith("--output-dir=")
               for arg in extra):
        base = (Path("/kaggle/working/lifemate_baseline_v4")
                if Path("/kaggle/working").is_dir()
                else Path.cwd() / "lifemate_baseline_v4")
        args += ["--output-dir", str(base / "single" / sample_id)]
    return args + extra


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *single_sample_args(sys.argv[1:])]
    main()
