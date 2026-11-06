"""Serializable run settings, shared by isolated training/evaluation processes."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .io import write_json


@dataclass
class RunConfig:
    output: str
    model: str = "Qwen/Qwen3.5-4B"
    revision: str = "main"
    profile: str = "smoke"
    seed: int = 42
    max_length: int = 8192
    max_new_tokens: int = 64
    micro_batch: int = 2
    gradient_accumulation: int = 2
    world_size: int = 2
    epochs: float = 2.0
    learning_rate: float = 5e-5
    rank: int = 16
    alpha: int = 32
    smoke_steps: int = 4
    eval_batch: int = 2
    evaluate_base: bool = True
    check_kernels: bool = False

    def __post_init__(self):
        if self.profile not in ("smoke", "full"):
            raise ValueError("profile must be smoke or full")
        if (self.micro_batch, self.gradient_accumulation, self.world_size) != (2, 2, 2):
            raise ValueError("This baseline requires microbatch 2 × accumulation 2 × two GPUs = 8")
        if not 0 < self.max_length <= 16384 or self.max_new_tokens >= self.max_length:
            raise ValueError("Invalid context/output budget")
        if self.smoke_steps <= 0 or self.epochs <= 0 or self.eval_batch <= 0:
            raise ValueError("Training steps, epochs and eval batch must be positive")
        if self.check_kernels and self.profile == "smoke" and self.smoke_steps < 2:
            raise ValueError("Kernel profiling requires at least two optimizer steps")

    @property
    def path(self):
        return Path(self.output)

    @property
    def prepared(self):
        return self.path / "prepared"

    def save(self):
        write_json(self.path / "config.json", asdict(self))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
