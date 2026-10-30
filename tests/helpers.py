from copy import deepcopy
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent / 'train+public_test'
sys.path.insert(0, str(BASE / 'src'))


def sample(sample_id='train_0236'):
    with (ROOT / 'train/train.jsonl').open() as handle:
        return next(row for line in handle if (row := json.loads(line))['id'] == sample_id)


def plan(instruction='Lấy dữ kiện cần thiết cho bước hiện tại.'):
    return {'next': 'caller', 'instruction': instruction}


def call(name, **arguments):
    return {'name': name, 'arguments': arguments}


class ScriptedLLM:
    model_id, backend, dtype, temperature, seed = 'scripted', 'test', 'none', 0, 42
    structured_outputs = True

    def __init__(self, outputs):
        self.outputs, self.inputs = iter(outputs), []

    def count(self, messages):
        return 100

    def complete(self, messages, max_new_tokens=512, output_schema=None):
        self.inputs.append((deepcopy(messages), deepcopy(output_schema)))
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return {'text': output if isinstance(output, str) else json.dumps(output, ensure_ascii=False),
                'prompt_tokens': self.count(messages), 'generation_tokens': 5, 'latency_s': .01}
