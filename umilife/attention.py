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


def attention_preflight(torch):
    """Check the actual CUDA forward/backward kernel before downloading Qwen weights."""
    import torch.nn.functional as functional
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tensors = [torch.randn(1, 4, 64, 128, device="cuda", dtype=dtype, requires_grad=True)
               for _ in range(3)]
    output = functional.scaled_dot_product_attention(*tensors, is_causal=True)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(output).all()) and all(bool(torch.isfinite(x.grad).all()) for x in tensors)
    if not finite:
        raise RuntimeError("CUDA attention preflight produced nonfinite output/gradient")
    return {"forward_backward": "passed", "dtype": str(dtype), "shape": [1, 4, 64, 128]}


def chunked_sdpa_attention_forward(module, query, key, value, attention_mask,
                                   dropout=0.0, scaling=None, is_causal=None,
                                   chunk_size=256, **kwargs):
    """Exact full-key attention in query tiles; checkpoint each tile during training.

    A tile still attends to every permitted key. This is neither local nor sparse
    attention. Checkpointing prevents the full quadratic score graph being retained.
    Used only if the CUDA build cannot run its efficient SDPA kernel on the device.
    """
    import torch
    from torch.nn import functional
    from torch.utils.checkpoint import checkpoint
    from transformers.integrations.sdpa_attention import repeat_kv

    if chunk_size <= 0 or query.shape[1] % key.shape[1]:
        raise ValueError("Invalid attention tile size or query/key head count")
    repeat = query.shape[1] // key.shape[1]
    key, value = repeat_kv(key, repeat), repeat_kv(value, repeat)
    q_length, k_length = query.shape[2], key.shape[2]
    if is_causal is None:
        is_causal = q_length > 1 and attention_mask is None and getattr(module, "is_causal", True)
    outputs = []
    key_positions = torch.arange(k_length, device=query.device) if is_causal else None

    def attend(q, k, v, bias):
        return functional.scaled_dot_product_attention(q, k, v, attn_mask=bias,
            dropout_p=dropout, scale=scaling, is_causal=False)

    for start in range(0, q_length, chunk_size):
        end = min(start + chunk_size, q_length)
        bias = attention_mask
        if bias is not None:
            bias = bias[..., :k_length]
            if bias.shape[-2] != 1:
                bias = bias[..., start:end, :]
        if is_causal:
            allowed = key_positions[None, :] <= torch.arange(start, end, device=query.device)[:, None]
            if bias is None:
                bias = allowed
            elif bias.dtype == torch.bool:
                bias = bias & allowed
            else:
                bias = bias.masked_fill(~allowed, float("-inf"))
        q = query[:, :, start:end, :]
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q, key, value)):
            output = checkpoint(attend, q, key, value, bias, use_reentrant=False)
        else:
            output = attend(q, key, value, bias)
        outputs.append(output)
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None


def enable_chunked_sdpa_fallback(torch):
    """Enable a bounded exact implementation when no fused T4 kernel is usable."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["sdpa"] = chunked_sdpa_attention_forward
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    return {"policy": "checkpointed_query_tiled_sdpa", "query_tile_size": 256,
            "full_key_attention": True, "input_truncation": False}
