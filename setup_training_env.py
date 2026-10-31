"""Create an isolated text-only training environment with a pinned CUDA stack."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    target = Path(args.venv_dir).expanduser().resolve()
    # with_pip=False also works on notebook images without the ensurepip package.
    venv.EnvBuilder(with_pip=False, symlinks=True).create(target)
    python = target / "bin/python"
    print("Isolated training Python:", python, flush=True)
    for requirements in ("pytorch-cu128.txt", "finetune.txt"):
        subprocess.run([
            sys.executable, "-m", "pip", "--python", str(python), "install",
            "--disable-pip-version-check", "-r", str(root / "requirements" / requirements),
        ], check=True)
    subprocess.run([str(python), "-c",
                    "import torch,transformers,peft,bitsandbytes; "
                    "print('Pinned training stack:',torch.__version__,torch.version.cuda,"
                    "transformers.__version__,peft.__version__,bitsandbytes.__version__)"], check=True)


if __name__ == "__main__":
    main()
