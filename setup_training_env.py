"""Use an isolated CUDA 12.8 stack so Kaggle image updates do not change kernels."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-dir",required=True)
    args=parser.parse_args()
    target=Path(args.venv_dir).resolve()
    venv.EnvBuilder(with_pip=False,symlinks=True).create(target)
    python=target/"bin/python"
    root=Path(__file__).resolve().parent
    for requirements in ("requirements-pytorch-cu128.txt","requirements.txt"):
        subprocess.run([sys.executable,"-m","pip","--python",str(python),"install",
                        "--no-cache-dir","--disable-pip-version-check","-r",str(root/requirements)],check=True)
    subprocess.run([str(python),"-c",
        "import torch,transformers,peft;print('Runtime:',torch.__version__,torch.version.cuda,"
        "transformers.__version__,peft.__version__);"
        "assert torch.cuda.is_available();print('GPU:',torch.cuda.get_device_name(0))"],check=True)


if __name__=="__main__":
    main()
