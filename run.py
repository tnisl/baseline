"""Entry point: python baseline_v7/run.py --mode workflow ..."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from lifemate_v7.runner import main

if __name__ == '__main__':
    main()
