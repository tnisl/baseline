"""Pinned vLLM 0.9.2 / Triton3.3 compatibility policy, independent of Torch."""
def engine_options(capability):
    # SM75 prefix-prefill fails compilation on the observed FP16 path. Keep graphs.
    prefix=tuple(capability)>=(8,0)
    return {'cuda_graphs':True,'prefix_caching':prefix,
            'prefix_caching_policy':'sm80_plus',
            'prefix_caching_reason':('enabled_on_sm80_plus' if prefix else
                'disabled_on_pre_sm80_due_to_prefix_prefill_triton_conversion_failure'),
            'compute_capability':list(capability)}
