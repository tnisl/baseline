"""Check deterministic setup without loading model weights or a GPU."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproducibility import prepare_process, seed_everything  # noqa: E402


class ReproducibilityTests(unittest.TestCase):
    def test_prepare_process_restarts_with_seeded_environment(self):
        with patch.dict(os.environ, {"PYTHONHASHSEED": "7"}):
            with patch("reproducibility.os.execvpe") as restart:
                prepare_process(42, vllm=True)
        restart.assert_called_once()
        environment = restart.call_args.args[2]
        self.assertEqual(environment["PYTHONHASHSEED"], "42")
        self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(environment["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")

    def test_prepare_process_does_not_restart_when_environment_matches(self):
        values = {"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                  "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
        with patch.dict(os.environ, values):
            with patch("reproducibility.os.execvpe") as restart:
                prepare_process(42, vllm=True)
        restart.assert_not_called()

    def test_seed_everything_sets_each_rng_and_deterministic_flags(self):
        events = []
        np = SimpleNamespace(random=SimpleNamespace(seed=lambda value: events.append(("numpy", value))))
        cudnn = SimpleNamespace(benchmark=True, deterministic=False)
        torch = SimpleNamespace(
            manual_seed=lambda value: events.append(("torch", value)),
            cuda=SimpleNamespace(manual_seed_all=lambda value: events.append(("cuda", value))),
            backends=SimpleNamespace(cudnn=cudnn),
            use_deterministic_algorithms=lambda *args, **kwargs: events.append(("algorithms", args, kwargs)),
        )
        with patch.dict(sys.modules, {"numpy": np, "torch": torch}):
            seed_everything(42)
            first = random.random()
            seed_everything(42)
            self.assertEqual(random.random(), first)
        self.assertIn(("numpy", 42), events)
        self.assertIn(("torch", 42), events)
        self.assertIn(("cuda", 42), events)
        self.assertIn(("algorithms", (True,), {"warn_only": True}), events)
        self.assertFalse(cudnn.benchmark)
        self.assertTrue(cudnn.deterministic)

    def test_hash_seed_range(self):
        for seed in (-1, 2**32):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    prepare_process(seed, vllm=False)


if __name__ == "__main__":
    unittest.main()
