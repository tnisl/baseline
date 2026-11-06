import json
import io
from pathlib import Path
import tempfile
import time
import unittest

from twonode.diagnostics import disable_progress_bars, startup_stage


class DiagnosticsTests(unittest.TestCase):
    def test_loader_progress_does_not_write_to_stream(self):
        from transformers.utils import logging as transformers_logging
        from huggingface_hub.utils import are_progress_bars_disabled
        transformers_logging.enable_progress_bar()
        disable_progress_bars()
        stream = io.StringIO()
        with transformers_logging.tqdm(total=3, desc="Loading weights", file=stream) as bar:
            for i in range(3):
                bar.update(1)
                bar.set_postfix({"Materializing param": f"layer{i}.weight"})
                bar.refresh()
        self.assertEqual(stream.getvalue(), "")
        self.assertTrue(are_progress_bars_disabled())

    def test_silent_phase_produces_traceback_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            with startup_stage(directory, 0, "load_test", timeout=.02):
                time.sleep(.06)
            root = Path(directory) / "diagnostics"
            self.assertIn("test_silent_phase", (root / "rank0.load_test.stacks.log").read_text())
            records = [json.loads(s) for s in (root / "rank0.stages.jsonl").read_text().splitlines()]
            self.assertEqual([r["event"] for r in records], ["start", "complete"])
            self.assertIn("VmRSS", records[0]["memory"])

    def test_phase_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with startup_stage(directory, 1, "failed_test"):
                    raise RuntimeError("failure")
            path = Path(directory) / "diagnostics/rank1.stages.jsonl"
            self.assertEqual(json.loads(path.read_text().splitlines()[-1])["event"], "failed")
