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
# Planner + Caller — evidence, intent and throughput
T4 x2, Internet on. Two independent local QLoRA adapters from the original Qwen3-4B, one epoch each. Choose STAGES to train either role alone. No Global, Summarizer, ASR or automatic workflow inference. No initial smoke run.
English prompts and TypeScript contracts. Evidence-grounded supervision includes conditions, observed entities, times and pending goals; richer plans may contain justified values. Training-only repair examples teach argument completeness and reuse of existing reads.100 training no-action targets were manually reviewed; all other rationales are a transparent Codex-authored evidence compiler, not independently generated per-example teacher reasoning. Split IDs remain1300/245; suspect training episodes are quarantined with a report, never silently repaired from hidden state. Validation no-action route targets are omitted because contest annotations do not identify the route.
Unsloth compile enabled, FP16/4-bit base, rank16 alpha32, microbatch train/eval4 per GPU, effective batch8, cosine LR/warmup, patience3. Assistant-token CE is unchanged mathematically: projection chunk512, projection checkpoint OFF for throughput, backbone checkpoint ON for long T4 batches. This costs more VRAM and is not yet GPU-benchmarked. No evidence is truncated. Tokenization is cached once; validation is length-sorted. Initial CE and repeated CE after hash-verified best restoration are skipped by default. Every100 steps and at training end, evaluate CE AND greedy generation on ALL reference-prefix validation records using the resident Unsloth models, batch4/GPU, max256 output tokens for Planner and192 for Caller, no enforcer. Print/save schema validity, Planner route/tool accuracy and terminal errors, or Caller action match and argument F1. These generation passes add wall time. Early stopping and CE-best selection still use CE. Logs separate train throughput, padding, CE/generation eval, checkpoint I/O and peak VRAM.
Save every25 attempted steps, after eval and cooperative deadline. Preserve the WHOLE WORK directory and attach it to resume. Hard cancellation can lose updates since the last checkpoint. Old adapters/runs are incompatible with this new prompt/data version.
After each completed role, free training GPUs and run vLLM generation evaluation for CE-best and last checkpoints on the same reference-prefix inputs, batch4/GPU, no enforcer. Report route/tool agreement, terminal errors, action match and per-field/argument metrics. Recommend one of those two candidates; do not call it a full-pipeline contest score. Review exported rationales and error cases before promoting adapters.
''')
cell('code','''
from pathlib import Path
import os,sys,json,hashlib,subprocess,zipfile,shutil,time
STAGES=('planner',)  # ('planner',) or ('caller',) trains only that node
EPOCHS={'planner':1,'caller':1}
RUN_NAME='planner_caller_evidence_v3'
RESUME_DIR=None  # Previous entire WORK directory mounted as a private dataset
RUN_NODE_EVAL=True
LOSS_CHUNK=512
PROJECTION_CHECKPOINT=False  # More VRAM, less backward recomputation
EVAL_STEPS=100
SAVE_STEPS=25
WORK=Path('/kaggle/working/planner_caller_evidence')/RUN_NAME
WORK.mkdir(parents=True,exist_ok=True)
if RESUME_DIR is not None:shutil.copytree(Path(RESUME_DIR),WORK,dirs_exist_ok=True)
SOURCE=Path('/kaggle/working/planner_caller_evidence_source')
MODEL='Qwen/Qwen3-4B-Instruct-2507';REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
DEADLINE=time.time()+10*3600
assert STAGES and len(set(STAGES))==len(STAGES) and set(STAGES)<={'planner','caller'}
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
config={{'source_sha':SOURCE_SHA,'model':MODEL,'revision':REVISION,'epochs':EPOCHS,'micro_batch':4,'world_size':2,'stages':list(STAGES),'compile_enabled':True,'loss_chunk':LOSS_CHUNK,'projection_checkpoint':PROJECTION_CHECKPOINT,'eval_steps':EVAL_STEPS,'save_steps':SAVE_STEPS,'initialization':'independent_base','output_token_limits':{{'planner':256,'caller':192}}}}
if (WORK/'run_config.json').exists():assert json.loads((WORK/'run_config.json').read_text())==config,'Incompatible resume directory'
else:(WORK/'run_config.json').write_text(json.dumps(config,indent=2))
''')
cell('code','''
sys.path.insert(0,str(SOURCE))
from umilife.paths import find_data_root
DATA_ROOT,archive=find_data_root('/kaggle/input')
if DATA_ROOT is None:
 assert archive is not None,'Attach the competition dataset'
 dest=Path('/tmp/evidence_v3_data');dest.mkdir(exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  assert all((dest/m.filename).resolve().is_relative_to(dest.resolve()) for m in z.infolist())
  z.extractall(dest)
 DATA_ROOT,_=find_data_root(dest)
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.9.10'],check=True)
UV=[sys.executable,'-m','uv'];os.environ['UV_NATIVE_TLS']='true'
venv=Path('/tmp/evidence_v3_unsloth');PYTHON=str(venv/'bin/python')
if not Path(PYTHON).exists():subprocess.run([*UV,'venv','--python','3.12','--seed',str(venv)],check=True)
subprocess.run([*UV,'pip','sync','--index-strategy','unsafe-best-match','--python',PYTHON,str(SOURCE/'requirements_train_2gpu.lock')],check=True)
subprocess.run([*UV,'pip','check','--python',PYTHON],check=True)
ENV=os.environ.copy()
for key in ('PYTHONPATH','PYTHONHOME'):ENV.pop(key,None)
ENV.update(CUDA_VISIBLE_DEVICES='0,1',PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
 UNSLOTH_COMPILE_DISABLE='0',LIFEMATE_LOSS_CHUNK=str(LOSS_CHUNK),LOSS_PROJECTION_CHECKPOINT=str(int(PROJECTION_CHECKPOINT)),INITIAL_EVAL='0',VERIFY_RESTORED_EVAL='0',SAVE_STEPS=str(SAVE_STEPS),TRAIN_DEADLINE=str(DEADLINE),HF_HOME='/tmp/lifemate_hf_cache')
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
print('Data:',m['counts']);print('Annotations:',m['rationale_provenance']);print('Quarantined train episodes:',m['train_quarantined_episodes']);print('Audit:',DATA/'data_audit.json')
''')
cell('code','''
# Build a separate vLLM0.9.2 environment only if generation evaluation is requested.
if RUN_NODE_EVAL:
 infer_venv=Path('/tmp/evidence_v3_vllm');INFER_PYTHON=str(infer_venv/'bin/python')
 if not Path(INFER_PYTHON).exists():subprocess.run([*UV,'venv','--python','3.12','--seed',str(infer_venv)],check=True)
 subprocess.run([*UV,'pip','sync','--index-strategy','unsafe-best-match','--python',INFER_PYTHON,str(SOURCE/'requirements_infer_2gpu.lock')],check=True)
 subprocess.run([*UV,'pip','check','--python',INFER_PYTHON],check=True)
 INFER_ENV={k:v for k,v in ENV.items() if not k.startswith('UNSLOTH_')}
 INFER_ENV.update(VLLM_USE_V1='0',VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USAGE_STATS_ENABLED='0',DO_NOT_TRACK='1')
 infer_purelib=Path(subprocess.check_output([INFER_PYTHON,'-I','-c',"import sysconfig;print(sysconfig.get_path('purelib'))"],text=True).strip())
 INFER_ENV['LD_LIBRARY_PATH']=':'.join(str(p) for p in (infer_purelib/'nvidia').glob('*/lib'))+':'+os.environ.get('LD_LIBRARY_PATH','')
''')
cell('code','''
OUTPUT=WORK/'training'
recommendations={}
for stage in STAGES:
 target=OUTPUT/'adapters'/stage
 done=target/'lineage.json'
 if done.exists():
  record=json.loads(done.read_text())
  assert record['dataset_manifest_sha256']==hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
  assert record['stage']==stage and record['initialization']=='base' and not record['smoke']
  print('Already complete:',stage)
 else:
  print('TRAINING:',stage,'2 GPUs, microbatch4, independent base initialization',flush=True)
  args=['--stage',stage,'--data-dir',DATA,'--data-root',DATA_ROOT,'--output-dir',OUTPUT,'--model',MODEL,'--revision',REVISION,
        '--init-from','base','--epochs',EPOCHS[stage],'--train-batch-size',4,'--eval-batch-size',4,'--eval-steps',EVAL_STEPS]
  result=execute('umilife.train_2gpu',args,ddp=True,check=False)
  assert result.returncode==0,'Paused or failed: preserve WORK, inspect logs/latest checkpoint, then resume the same run.'
 if RUN_NODE_EVAL:
  print('NODE GENERATION EVAL:',stage,'CE-best vs last; not a full workflow score',flush=True)
  report_dir=WORK/'node_eval'/stage
  args=['--stage',stage,'--prepared',DATA,'--adapter',target,'--checkpoints',OUTPUT/'checkpoints'/stage,
        '--data-root',DATA_ROOT,'--model',MODEL,'--revision',REVISION,'--output',report_dir]
  subprocess.run([INFER_PYTHON,'-u','-m','umilife.node_eval',*map(str,args)],cwd=SOURCE,env=INFER_ENV,check=True)
  recommendations[stage]=json.loads((report_dir/'report.json').read_text())
  (WORK/'node_checkpoint_recommendations.json').write_text(json.dumps(recommendations,ensure_ascii=False,indent=2))
selected=WORK/'selected_adapters'
if RUN_NODE_EVAL:
 roles=sorted(p.name for p in selected.iterdir() if p.is_dir() and (p/'lineage.json').exists())
 raw_sha=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
 contract={'canonical_manifest_sha256':hashlib.sha256(json.dumps(m,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'raw_manifest_sha256':{r:raw_sha for r in roles}}
 (selected/'data_contract.json').write_text(json.dumps(contract,indent=2))
 files={str(p.relative_to(selected)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected.rglob('*')) if p.is_file() and p.name!='checkpoint_manifest.json'}
 (selected/'checkpoint_manifest.json').write_text(json.dumps({'roles':roles,'model':MODEL,'revision':REVISION,'prompt_sha256':m['prompts_sha256'],'files':files},indent=2))
 print('Exported selected adapters:',selected)
print('Preserve entire directory:',WORK)
''')
for k,v in payload.items():
 if k.endswith('.py'):ast.parse(v,filename=k)
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'}}
(H/'kaggle_planner_caller_evidence.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
meta={'id':'lctrnnguynkhi/lifemate-planner-caller-evidence','title':'LifeMate Planner Caller Evidence','code_file':'kaggle_planner_caller_evidence.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['cymeu4l0t/zaloai-dataset'],'competition_sources':[],'kernel_sources':[]}
(H/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('BUILT',sha)
