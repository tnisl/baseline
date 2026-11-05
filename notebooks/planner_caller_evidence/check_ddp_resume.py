"""Two CPU/Gloo ranks: uninterrupted versus cooperative pause + exact resume."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'runtime'))
import umilife
umilife.__path__.append(str(H.parents[1]/'umilife'))

def worker(folder):
 import torch
 import torch.distributed as dist
 from types import SimpleNamespace
 from umilife.ddp_engine import fit
 from umilife.train import Collator
 torch.set_num_threads(1);torch.manual_seed(42);dist.init_process_group('gloo')
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.25]));self.config=SimpleNamespace(use_cache=False)
  def generate(self,*args,**kwargs):raise AssertionError('Generation must not run during fit')
  def forward(self,input_ids,labels,**kwargs):
   mask=labels!=-100;return SimpleNamespace(loss=((self.weight*input_ids.float()-labels.float())[mask]**2).mean())
 class Data:
  def __init__(self,n):self.items=[{'input_ids':[1]*(2+i%4),'attention_mask':[1]*(2+i%4),'labels':[-100]+[2]*(1+i%4)} for i in range(n)]
  def __len__(self):return len(self.items)
  def __getitem__(self,i):return self.items[i]
 model=Model()
 def save(m,p):torch.save(m.state_dict(),Path(p)/'weights.pt')
 def load(m,p):m.load_state_dict(torch.load(Path(p)/'weights.pt',weights_only=True))
 try:
  result=fit(model,Data(32),Data(9),Collator(0),torch.device('cpu'),Path(folder),1,.05,save,load,eval_steps=2,train_batch_size=2,eval_batch_size=2,patience=10)
  if dist.get_rank()==0:(Path(folder)/'result.json').write_text(json.dumps(result))
 finally:dist.destroy_process_group()

if __name__=='__main__':
 if '--worker' in sys.argv:worker(sys.argv[-1]);raise SystemExit
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);env=os.environ.copy();env.update(OMP_NUM_THREADS='1',SAVE_STEPS='2',TRAIN_RESUME_SIGNATURE='cpu-resume-v3',INITIAL_EVAL='0',VERIFY_RESTORED_EVAL='0')
  command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(Path(__file__).resolve()),'--worker']
  for mode in ('full','pause','resume'):
   folder=root/('full' if mode=='full' else 'resumed');e=env.copy()
   if mode=='pause':e['SMOKE_PAUSE_STEP']='2'
   result=subprocess.run(command+[str(folder)],env=e,capture_output=True,text=True)
   if mode=='pause':
    assert result.returncode!=0 and json.loads((folder/'latest.json').read_text())['step']==2
   else:assert result.returncode==0,result.stdout[-3000:]+result.stderr[-3000:]
  a=json.loads((root/'full/result.json').read_text());b=json.loads((root/'resumed/result.json').read_text())
  assert a['global_step']==b['global_step']==8
  assert a['final_trainable_sha256']==b['final_trainable_sha256']
  assert [(x['step'],x['eval_loss']) for x in a['history']]==[(x['step'],x['eval_loss']) for x in b['history']]
  assert all(x['evaluation_kind']=='teacher_forced_ce' and 'node_eval' not in x for x in a['history'])
  history=json.loads((root/'resumed/eval_history.json').read_text())
  assert [(x['step'],x['eval_loss']) for x in history]==[(x['step'],x['eval_loss']) for x in b['history']]
  assert a['before']['eval_loss'] is None and b['before']['eval_loss'] is None
  print('PASS: CPU2-rank DDP, checkpoint at step2, optimizer/scheduler/RNG/cursor resume, final hash/CE match, teacher-forced CE only, no duplicate initial/final CE')
