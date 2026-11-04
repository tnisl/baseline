import json,hashlib,ast,textwrap
from pathlib import Path
H=Path(__file__).resolve().parent
payload={str(p.relative_to(H/'runtime')):p.read_text() for p in sorted((H/'runtime').rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
sha=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();cells=[]
def cell(kind,s):
 c={'cell_type':kind,'metadata':{},'source':textwrap.dedent(s).strip()+'\n','id':f'en-resume-{len(cells):02}'}
 if kind=='code':ast.parse(c['source']);c.update(outputs=[],execution_count=None)
 cells.append(c)
cell('markdown','''
# Planner + Caller — concise English plans — compiled Unsloth
Select T4 x2, Internet on, Run All. No job is launched by saving this notebook.
Train Planner and Caller only, independently from original Qwen. No Summarizer or Global training by default. English prompts and concise field-source rationales; tool contracts are TypeScript, actions remain JSON. The split remains 1300 train / 245 validation episodes.
This notebook starts full finetuning directly: no automatic smoke training, forced pause/resume, or synthetic GPU preflight. Package consistency and input checks remain. Unsloth compilation is enabled; its first compilation can take time and GPU speedup must be measured.
Planner decides the next tool, constraints and termination; Caller resolves argument values from the request, history and past observations. Labels name parameter sources without serializing the target arguments. Deterministic labels require semantic review, especially complex conditions and repeated same-tool objectives.
Planner and Caller: 1 epoch each. Train/validation microbatch 4 per GPU, effective train batch 8. Unsloth QLoRA rank16, cosine/warmup, patience3; loss chunks256, cached tokenization.
Checkpoint every25 updates and after evaluation: adapter, optimizer, LR scheduler, scaler, per-rank RNG and batch cursor. At 10 hours request a cooperative save/pause. Hard kill can lose work since the last complete checkpoint. Kaggle working storage is not permanent: preserve outputs and attach them as a private input for a later session. RESUME_DIR must point to the previous whole run directory including prepared/, full/ and checkpoints/best.
This notebook only prepares and finetunes; no ASR, workflow/public-test inference or automatic submission. Validation loss remains teacher-forced and is not a replacement for Planner free-generation evaluation. No validation-specific prompt exceptions are used.
''')
cell('code','''
from pathlib import Path
import os,sys,json,hashlib,subprocess,zipfile,shutil,time
PROFILE='full' # Full training only; no automatic smoke test
TRAIN_MODE='local' # 'local', 'global', or 'global_to_local'
LOCAL_INIT='base' # local mode: 'base' or 'global'
GLOBAL_ADAPTER=None # completed Global adapter directory; only for local + LOCAL_INIT='global'
LOCAL_STAGES=('planner','caller') # optionally select a subset
RUN_NAME='planner_caller_concise_compiled_v1'
assert TRAIN_MODE in ('local','global','global_to_local') and LOCAL_INIT in ('base','global')
if TRAIN_MODE=='local':assert (GLOBAL_ADAPTER is not None)==(LOCAL_INIT=='global')
else:assert GLOBAL_ADAPTER is None,'GLOBAL_ADAPTER only applies to local mode'
MODE_TAG=TRAIN_MODE+'_'+LOCAL_INIT if TRAIN_MODE=='local' else TRAIN_MODE
RESUME_DIR=None # Mounted previous whole run directory, not just a single adapter
WORK=Path('/kaggle/working/planner_caller_compiled')/RUN_NAME/MODE_TAG
WORK.mkdir(parents=True,exist_ok=True)
if RESUME_DIR is not None:shutil.copytree(Path(RESUME_DIR),WORK,dirs_exist_ok=True)
SOURCE=Path('/kaggle/working/planner_caller_source')
MODEL='Qwen/Qwen3-4B-Instruct-2507';REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
EPOCHS={'global':1,'planner':1,'caller':1,'summarizer':2}
DEADLINE=time.time()+10*3600
assert PROFILE=='full'
assert set(LOCAL_STAGES) <= {'planner','caller'} and LOCAL_STAGES
if not SOURCE.exists():subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/tnisl/baseline.git',str(SOURCE)],check=True)
subprocess.run(['git','-C',str(SOURCE),'fetch','origin','4894d331692293565635454a8c686b7c18314a04'],check=True)
subprocess.run(['git','-C',str(SOURCE),'checkout','--detach','FETCH_HEAD'],check=True)
''')
cell('code',f'''
PAYLOAD={payload!r}
SOURCE_SHA={sha!r}
assert hashlib.sha256(json.dumps(PAYLOAD,sort_keys=True).encode()).hexdigest()==SOURCE_SHA
for name,content in PAYLOAD.items():
 p=SOURCE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
config={{'source_sha':SOURCE_SHA,'model':MODEL,'revision':REVISION,'epochs':EPOCHS,'micro_batch':4,'world_size':2,'train_mode':TRAIN_MODE,'local_init':LOCAL_INIT if TRAIN_MODE=='local' else None,'compile_enabled':True,'local_stages':list(LOCAL_STAGES),'global_parent_sha':hashlib.sha256((Path(GLOBAL_ADAPTER)/'adapter_model.safetensors').read_bytes()).hexdigest() if GLOBAL_ADAPTER else None}}
if (WORK/'run_config.json').exists():assert json.loads((WORK/'run_config.json').read_text())==config,'Incompatible resume directory'
else:(WORK/'run_config.json').write_text(json.dumps(config,indent=2))
''')
cell('code','''
sys.path.insert(0,str(SOURCE))
from umilife.paths import find_data_root
DATA_ROOT,archive=find_data_root('/kaggle/input')
if DATA_ROOT is None:
 assert archive is not None,'Attach the competition dataset'
 dest=Path('/tmp/english_evidence_data');dest.mkdir(exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  assert all((dest/m.filename).resolve().is_relative_to(dest.resolve()) for m in z.infolist())
  z.extractall(dest)
 DATA_ROOT,_=find_data_root(dest)
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.9.10'],check=True)
UV=[sys.executable,'-m','uv'];os.environ['UV_NATIVE_TLS']='true'
venv=Path('/tmp/english_evidence_unsloth');PYTHON=str(venv/'bin/python')
if not Path(PYTHON).exists():subprocess.run([*UV,'venv','--python','3.12','--seed',str(venv)],check=True)
subprocess.run([*UV,'pip','sync','--index-strategy','unsafe-best-match','--python',PYTHON,str(SOURCE/'requirements_train_2gpu.lock')],check=True)
subprocess.run([*UV,'pip','check','--python',PYTHON],check=True)
ENV=os.environ.copy()
for key in ('PYTHONPATH','PYTHONHOME'):ENV.pop(key,None)
ENV.update(CUDA_VISIBLE_DEVICES='0,1',PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
 UNSLOTH_COMPILE_DISABLE='0',LIFEMATE_LOSS_CHUNK='256',SAVE_STEPS='25',TRAIN_DEADLINE=str(DEADLINE),HF_HOME='/tmp/lifemate_hf_cache')
purelib=Path(subprocess.check_output([PYTHON,'-I','-c',"import sysconfig;print(sysconfig.get_path('purelib'))"],text=True).strip())
ENV['LD_LIBRARY_PATH']=':'.join(str(p) for p in (purelib/'nvidia').glob('*/lib'))+':'+ENV.get('LD_LIBRARY_PATH','')
def execute(module,args,ddp=False,env=None,check=True):
 command=[PYTHON,'-u','-m']+(['torch.distributed.run','--standalone','--nproc_per_node=2','--module',module] if ddp else [module])
 return subprocess.run(command+list(map(str,args)),cwd=SOURCE,env=env or ENV,check=check)
# No synthetic GPU preflight or model smoke. Verify real training startup in the log.
for key in ('SMOKE_PAUSE_STEP',):ENV.pop(key,None)
DATA=WORK/'prepared'
if not (DATA/'manifest.json').exists():
 execute('umilife.prepare_2gpu',['--data-root',DATA_ROOT,'--output',DATA,'--split-file',SOURCE/'split.txt','--model',MODEL,'--revision',REVISION])
m=json.loads((DATA/'manifest.json').read_text());assert m['train_episodes']==1300 and m['validation_episodes']==245
assert m['split_sha256']==hashlib.sha256((SOURCE/'split.txt').read_bytes()).hexdigest()
print('Data:',m['counts']);print('Annotations:',m['rationale_provenance'])
''')
cell('code','''
from umilife.training_modes import stages_for,parent_for,verify_parent
SELECTED_STAGES=stages_for(TRAIN_MODE,LOCAL_STAGES)
print('Selected stages:',SELECTED_STAGES,'Local initialization:',LOCAL_INIT if TRAIN_MODE=='local' else TRAIN_MODE)
def train_profile(profile):
 output=WORK/profile
 for stage in SELECTED_STAGES:
  epochs=EPOCHS[stage]
  parent=parent_for(stage,TRAIN_MODE,LOCAL_INIT,GLOBAL_ADAPTER,output)
  init_from='global' if parent is not None else 'base'
  if parent is not None:
   verify_parent(parent,MODEL,REVISION,hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),m['prompts_sha256'],False)
  done=output/'adapters'/stage/'lineage.json'
  if done.exists():
   record=json.loads(done.read_text())
   assert record['dataset_manifest_sha256']==hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
   assert record['smoke']==(False) and record['initialization']==init_from
   assert record['global_adapter_sha256']==(hashlib.sha256((parent/'adapter_model.safetensors').read_bytes()).hexdigest() if parent else None)
   print('Already complete:',profile,stage);continue
  args=['--stage',stage,'--data-dir',DATA,'--output-dir',output,'--model',MODEL,'--revision',REVISION,
   '--init-from',init_from,'--epochs',epochs,'--train-batch-size',4,'--eval-batch-size',4,'--eval-steps',100]
  stage_env=ENV.copy()
  if parent is not None:args+=['--global-adapter',parent]
  result=execute('umilife.train_2gpu',args,ddp=True,env=stage_env,check=False)
  assert result.returncode==0,'Training paused or failed. Preserve WORK and inspect logs/latest checkpoint before resuming.'
 print('Completed:',profile)
train_profile('full')
print('Preserve this directory:',WORK)
''')
for k,v in payload.items():
 if k.endswith('.py'):ast.parse(v,filename=k)
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'}}
(H/'kaggle_planner_caller_compiled.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
meta={'id':'anartt/lifemate-planner-caller-compiled','title':'LifeMate Planner Caller Compiled','code_file':'kaggle_planner_caller_compiled.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['cymeu4l0t/zaloai-dataset'],'competition_sources':[],'kernel_sources':[]}
(H/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('BUILT',sha)
