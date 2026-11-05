"""Pinned vLLM0.9.2 T4 LoRA kernel compatibility, shared with validated inference."""
import os
from pathlib import Path
def prepare_t4_lora_kernel():
    """Work around Triton 3.3 fp16 conversion failure on SM75; preserve FP32 accumulation."""
    import torch,importlib.util,importlib.metadata,hashlib
    path=Path(importlib.util.find_spec("vllm").origin).parent/"lora/ops/triton_ops/kernel_utils.py"
    original="        accumulator += tl.dot(\n            tiled_a,\n            tiled_b,\n        )"
    patched="        # LifeMate SM75 compatibility: explicit exact fp16->fp32 before dot.\n        accumulator += tl.dot(tiled_a.to(tl.float32), tiled_b.to(tl.float32), input_precision=\"ieee\")"
    source=path.read_text()
    capability=torch.cuda.get_device_capability(0)
    if capability[0]>=8:
        if patched in source:path.write_text(source.replace(patched,original))
        return {"applied":False,"capability":list(capability)}
    assert importlib.metadata.version("vllm")=="0.9.2"
    assert importlib.metadata.version("triton")=="3.3.0"
    assert original in source or patched in source,"Unexpected LoRA kernel source"
    if original in source:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False,suffix='.py') as tmp:
            tmp.write(source.replace(original,patched));temporary=tmp.name
        os.replace(temporary,path)
    # Exercise the exact patched helper on the GPU against an FP32 PyTorch reference.
    global triton,tl,mm_k
    import triton
    import triton.language as tl
    from vllm.lora.ops.triton_ops.kernel_utils import mm_k
    @triton.jit
    def check_kernel(A,B,C,K:tl.constexpr):
        rows=tl.arange(0,32)
        cols=tl.arange(0,16)
        kk=tl.arange(0,32)
        a=A+rows[:,None]*K+kk[None,:]
        b=B+kk[:,None]*16+cols[None,:]
        acc=mm_k(a,b,1,16,kk,K,32,16,32,True,1,False,tl.float16)
        tl.store(C+rows[:,None]*16+cols[None,:],acc)
    errors=[]
    torch.manual_seed(42)
    for k in (128,2560):
        a=torch.randn(32,k,device="cuda",dtype=torch.float16)
        b=torch.randn(k,16,device="cuda",dtype=torch.float16)
        c=torch.empty(32,16,device="cuda",dtype=torch.float32)
        check_kernel[(1,)](a,b,c,k,num_warps=4)
        ref=a.float()@b.float()
        torch.testing.assert_close(c,ref,rtol=5e-4,atol=2e-3)
        errors.append(float((c-ref).abs().max()))
    return {"applied":True,"capability":list(capability),"operation":"LoRA dot with FP32 accumulation",
            "reference_test":"pass","maximum_errors":errors,"kernel_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
