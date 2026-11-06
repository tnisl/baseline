import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from twonode.config import RunConfig
from twonode.kernel_diagnostics import callable_info, kernel_inventory, save_profile, summarize_events


def torch_chunk_gated_delta_rule():
    pass


class GatedDeltaNetFixture:
    causal_conv1d_fn = None
    chunk_gated_delta_rule = staticmethod(torch_chunk_gated_delta_rule)

    def forward(self):
        pass

    def named_modules(self):
        yield "linear_attn", self


class KernelDiagnosticsTests(unittest.TestCase):
    def test_inventory_distinguishes_missing_conv_and_torch_delta_fallback(self):
        report = kernel_inventory(GatedDeltaNetFixture())
        self.assertEqual(report["gated_delta_layers"], 1)
        bindings = report["layers"][0]["bindings"]
        self.assertEqual(bindings["causal_conv1d_fn"]["implementation_hint"], "torch_conv1d_fallback_hint")
        self.assertEqual(bindings["chunk_gated_delta_rule"]["implementation_hint"], "torch_fallback")
        self.assertNotIn("recurrent_gated_delta_rule", bindings)

        def wrapper():
            pass
        wrapper.__wrapped__ = torch_chunk_gated_delta_rule
        wrapper.__wrapped__._torchdynamo_orig_callable = wrapper
        try:
            self.assertEqual(len(callable_info(wrapper)["wrapper_chain"]), 2)
        finally:
            del torch_chunk_gated_delta_rule._torchdynamo_orig_callable

    def test_cpu_names_do_not_count_as_executed_cuda_kernels(self):
        events = [SimpleNamespace(name="chunk_gated_delta_rule", device_type="DeviceType.CPU",
                                  self_cpu_time_total=12),
                  SimpleNamespace(name="_causal_conv1d_fwd_kernel", device_type="DeviceType.CUDA",
                                  device_time_total=7),
                  SimpleNamespace(name="_causal_conv1d_fwd_kernel", device_type="DeviceType.CUDA",
                                  device_time_total=3)]
        report = summarize_events(events)
        self.assertEqual(report["cuda_kernel_calls"], 2)
        self.assertEqual(report["cuda_kernel_time_us"], 10)
        self.assertEqual(report["kernel_name_candidates"]["gated_delta_or_fla_candidates"], [])
        self.assertEqual(len(report["kernel_name_candidates"]["causal_conv_candidates"]), 1)
        self.assertFalse(summarize_events(events[:1])["cuda_events_recorded"])

    def test_real_cpu_profile_exports_trace_without_claiming_cuda(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                matrix = torch.ones(8, 8, requires_grad=True)
                (matrix @ matrix).sum().backward()
            report = save_profile(profiler, directory, 0, 2, {})
            self.assertFalse(report["cuda_events_recorded"])
            self.assertTrue(any("mm" in item["name"] for item in report["top_cpu_operators"]))
            root = Path(directory) / "diagnostics"
            self.assertTrue(json.loads((root / report["trace"]).read_text())["traceEvents"])
            self.assertEqual(json.loads((root / "rank0.step2.profile.json").read_text())["optimizer_step"], 2)

    def test_kernel_check_configuration_survives_save_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(output=directory, check_kernels=True)
            config.save()
            self.assertTrue(RunConfig.load(Path(directory) / "config.json").check_kernels)
            with self.assertRaisesRegex(ValueError, "at least two"):
                RunConfig(output=directory, check_kernels=True, smoke_steps=1)
