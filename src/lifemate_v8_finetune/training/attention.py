"""Keep exact SDPA attention memory efficient on pre-Ampere CUDA devices."""
from __future__ import annotations


def configure_training_attention(torch):
    """HF 4.57's native-GQA choice otherwise forces T4 to the quadratic math path.

    Use HF's existing repeat_kv path and PyTorch's fused efficient kernel on T4.
    The mathematical attention, masks and trainable parameters are unchanged.
    This runs once in the isolated training process, before any model forward.
    Ampere and newer GPUs retain the standard native-GQA dispatch.
    """
    if not torch.cuda.is_available():
        return {"policy": "standard_sdpa", "reason": "not_cuda"}
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:
        return {"policy": "standard_sdpa", "compute_capability": list(capability)}

    from transformers.integrations import sdpa_attention

    if not callable(getattr(sdpa_attention, "use_gqa_in_sdpa", None)):
        raise RuntimeError("Expected the pinned Transformers SDPA selector; check requirements/finetune.txt")

    # This selects the existing repeat_kv implementation in sdpa_attention_forward.
    # No model forward or mask implementation is replaced.
    def explicit_kv_selector(*args, **kwargs):
        return False

    sdpa_attention.use_gqa_in_sdpa = explicit_kv_selector
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    # Fail clearly when a shape is unsupported, rather than allocating O(sequence^2).
    torch.backends.cuda.enable_math_sdp(False)
    return {"policy": "explicit_kv_fused_efficient_sdpa", "compute_capability": list(capability),
            "native_gqa": False, "quadratic_math_fallback": False}
