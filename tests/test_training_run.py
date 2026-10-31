"""Runner contracts without model downloads or GPU allocation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from helpers import BASE
from lifemate_v8_finetune.training.run import ROOT, PROFILES, parser, run_logged, verify_adapter


class TrainingRunTests(unittest.TestCase):
    def test_profiles_and_entrypoint_resolve_without_torch(self):
        self.assertEqual(ROOT, BASE)
        self.assertTrue((ROOT / "finetune.py").is_file())
        self.assertEqual(PROFILES["smoke"]["max_steps"], 2)
        self.assertEqual(PROFILES["smoke"]["data_limit"], 20)
        self.assertEqual(PROFILES["train"]["data_limit"], 0)
        args = parser().parse_args(["--output-dir", "/tmp/test"])
        self.assertEqual(args.roles, ["planner", "caller", "summarizer"])
        self.assertEqual(args.profile, "smoke")

    def test_child_traceback_is_preserved_and_failure_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "job.log"
            with self.assertRaises(RuntimeError):
                run_logged([sys.executable, "-c", "print('marker',flush=True); raise ValueError('expected_failure')"], path)
            log = path.read_text()
            self.assertIn("marker", log)
            self.assertIn("Traceback", log)
            self.assertIn("expected_failure", log)

    def test_only_updated_finite_adapter_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "adapter_config.json").write_text("{}")
            (p / "adapter_model.safetensors").write_bytes(b"presence-only")
            (p / "role_metadata.json").write_text(json.dumps({"role": "caller", "global_finetune": False}))
            for finite, updated in ((True, False), (False, True)):
                (p / "training_metrics.json").write_text(json.dumps({"adapter_verification": {"finite": finite, "updated": updated}}))
                with self.assertRaises(ValueError):
                    verify_adapter(p, "caller")
            (p / "training_metrics.json").write_text(json.dumps({"adapter_verification": {"finite": True, "updated": True}}))
            self.assertTrue(verify_adapter(p, "caller")["adapter_verification"]["updated"])
            with self.assertRaises(ValueError):
                verify_adapter(p, "planner")
