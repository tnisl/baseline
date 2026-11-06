import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from twonode.config import RunConfig
from twonode.run import launch


class LaunchTests(unittest.TestCase):
    def test_configurable_batch_and_save_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            for gpu, micro, accumulation in ((1, 2, 4), (2, 4, 1), (2, 1, 3)):
                config = RunConfig(output=directory, world_size=gpu, micro_batch=micro,
                                   gradient_accumulation=accumulation)
                config.save()
                restored = RunConfig.load(config.path / "config.json")
                self.assertEqual(restored.effective_batch, gpu * micro * accumulation)
                self.assertEqual(restored, config)
            for field in ("world_size", "micro_batch", "gradient_accumulation"):
                for invalid in (0, -1, 1.5, True):
                    with self.assertRaisesRegex(ValueError, "positive integers"):
                        RunConfig(output=directory, **{field: invalid})

    def test_one_and_two_gpu_launches_preserve_device_ids_and_eval_shards(self):
        for gpu in (1, 2):
            with self.subTest(num_gpu=gpu), tempfile.TemporaryDirectory() as directory:
                config = RunConfig(output=directory, world_size=gpu)
                worker = Mock()
                worker.wait.return_value = 0
                worker.poll.return_value = 0
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,7", "LOCAL_RANK": "9"}), \
                     patch("twonode.run.subprocess.run") as run, \
                     patch("twonode.run.subprocess.Popen", return_value=worker) as popen, \
                     patch("twonode.run.combine", return_value={"ok": True}):
                    self.assertEqual(launch(config, "train-python", "eval-python"), {"ok": True})
                training = run.call_args_list[1]
                self.assertIn(f"--nproc_per_node={gpu}", training.args[0])
                self.assertEqual(training.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "4" if gpu == 1 else "4,7")
                self.assertEqual(popen.call_count, gpu)
                for rank, call in enumerate(popen.call_args_list):
                    self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], ("4", "7")[rank])
                    self.assertNotIn("LOCAL_RANK", call.kwargs["env"])
                    self.assertEqual(call.args[0][-1], str(rank))

    def test_requesting_more_than_visible_devices_fails_before_launch(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
             patch("twonode.run.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                launch(RunConfig(output="unused", world_size=2), "train-python", "eval-python")
            run.assert_not_called()
