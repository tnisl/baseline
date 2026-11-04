"""Run with torchrun --standalone --nproc_per_node=2: exact CPU optimizer resume regression."""
import os,sys,json,random
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).parent/'runtime/umilife'))
from ddp_engine import fit

def collate(rows):return {k:torch.tensor([r[k] for r in rows]) for k in ('input_ids','labels')}
class Toy(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(3))
 def forward(self,input_ids,labels,**kw):
  weight=torch.nn.functional.dropout(self.weight,p=.2,training=self.training)
  logits=weight.expand(len(input_ids)*2,3)
  return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits,labels[:,1:].reshape(-1)))
def main():
 dist.init_process_group('gloo');rank=dist.get_rank();device=torch.device('cpu');root=Path(sys.argv[1]);root.mkdir(exist_ok=True)
 rows=[{'input_ids':[1,1,1],'labels':[-100,i%3,(i+1)%3]} for i in range(32)]
 def save(m,p):torch.save(m.state_dict(),p/'model.pt')
 def load(m,p):m.load_state_dict(torch.load(p/'model.pt',weights_only=True))
 os.environ.update(TRAIN_RESUME_SIGNATURE='cpu-test',SAVE_STEPS='1')
 kwargs=dict(epochs=1,lr=.01,save_model=save,load_model=load,eval_steps=2,patience=20)
 torch.manual_seed(123+rank);random.seed(123+rank)
 full=Toy();baseline=fit(full,rows,rows[:5],collate,device,root/'baseline',**kwargs)
 torch.manual_seed(123+rank);random.seed(123+rank)
 os.environ['SMOKE_PAUSE_STEP']='1'
 try:fit(Toy(),rows,rows[:5],collate,device,root/'resume',**kwargs);raise AssertionError('Expected pause')
 except SystemExit as e:assert e.code==75
 resumed=Toy();result=fit(resumed,rows,rows[:5],collate,device,root/'resume',**kwargs)
 torch.testing.assert_close(resumed.weight,full.weight,rtol=0,atol=0)
 assert result['optimizer_updates']==baseline['optimizer_updates']==4
 for a,b in zip(result['history'],baseline['history']):
  assert {k:v for k,v in a.items() if k!='evaluation_seconds'}=={k:v for k,v in b.items() if k!='evaluation_seconds'}
 state=torch.load(root/'resume/step-00000004/training_state.pt',weights_only=False)
 assert state['scheduler']['last_epoch']==4 and len(state['rng'])==2
 assert result['early_stopping']['best_step']==baseline['early_stopping']['best_step']
 if rank==0:print('EXACT_CPU_RESUME_PASS',result['final_trainable_sha256'])
 dist.destroy_process_group()
if __name__=='__main__':main()
