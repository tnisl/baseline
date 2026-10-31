"""Run one train sample without replacing batch results."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from lifemate_v8.runner import main


def single_sample_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample_id', help='Train ID')
    known, extra = parser.parse_known_args(argv)
    if not known.sample_id.startswith('train_'):
        parser.error('Expected a train ID')
    args = ['--mode', 'workflow', '--sample-id', known.sample_id]
    if not any(x == '--output-dir' or x.startswith('--output-dir=') for x in extra):
        base = (Path('/kaggle/working/lifemate_baseline_v8') if Path('/kaggle/working').is_dir()
                else Path(__file__).resolve().parent / 'outputs')
        args += ['--output-dir', str(base / 'single' / known.sample_id)]
    return args + extra


if __name__ == '__main__':
    sys.argv = [sys.argv[0], *single_sample_args(sys.argv[1:])]
    main()
