"""Memory-bounded causal CE on assistant tokens; Unsloth returns hidden states."""
import os
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def completion_loss(model,batch,chunk_size=None):
    chunk_size=chunk_size or int(os.environ.get("LIFEMATE_LOSS_CHUNK","512"))
    assert chunk_size>0
    # UNSLOTH_RETURN_HIDDEN_STATES=1 bypasses its fused vocabulary-sized loss allocation.
    output=model(**batch,use_cache=False,return_dict=True)
    hidden=output.logits
    head=model.get_output_embeddings()
    assert hidden.shape[-1]==head.weight.shape[1],'Expected Unsloth hidden-state output, not vocabulary logits'
    labels=batch['labels'][:,1:]
    mask=labels!=-100
    selected=hidden[:,:-1][mask];targets=labels[mask]
    assert targets.numel()>0
    def chunk_nll(h,t):
        logits=F.linear(h.to(head.weight.dtype),head.weight,head.bias)
        return F.cross_entropy(logits.float(),t,reduction='sum')
    total=hidden.new_zeros((),dtype=torch.float32)
    for start in range(0,len(targets),chunk_size):
        h=selected[start:start+chunk_size];t=targets[start:start+chunk_size]
        if os.environ.get("LOSS_PROJECTION_CHECKPOINT","0")=="1" and torch.is_grad_enabled() and h.requires_grad:
            loss=checkpoint(chunk_nll,h,t,use_reentrant=False)
        else:loss=chunk_nll(h,t)
        total=total+loss
    return total/targets.numel()
