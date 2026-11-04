"""Executed by the user before Qwen loading: imports, NCCL and AMP backward on both GPUs."""
import os,json
import torch
local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
from unsloth import FastLanguageModel
import torch.distributed as dist
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel
from umilife.ddp_engine import agreement

def main():
    assert torch.cuda.device_count()==2 and int(os.environ['WORLD_SIZE'])==2
    dist.init_process_group('nccl',device_id=torch.device('cuda',local),timeout=timedelta(minutes=5))
    dtype=torch.bfloat16 if torch.cuda.get_device_capability(local)[0]>=8 else torch.float16
    model=torch.nn.Linear(16,8).to(torch.device('cuda',local));wrapped=DistributedDataParallel(model,device_ids=[local],static_graph=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16,init_scale=1024)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=dtype):loss=wrapped(torch.randn(4,16,device=torch.device('cuda',local))).float().square().mean()
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    x=torch.tensor([1.],device=torch.device('cuda',local));dist.all_reduce(x);assert x.item()==2
    agreement('NCCL_AMP_BATCH4_PASS')
    if dist.get_rank()==0:print(json.dumps({'status':'pass','world_size':2,'micro_batch':4,'backend':'nccl','amp':str(dtype)}),flush=True)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
