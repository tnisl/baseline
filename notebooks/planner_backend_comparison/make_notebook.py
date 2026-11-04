import ast,json,hashlib,textwrap
from pathlib import Path
HERE=Path(__file__).resolve().parent
payload={str(p.relative_to(HERE/'runtime')):p.read_text() for p in sorted((HERE/'runtime').rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
sha=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip()+'\n'})
def code(s):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':textwrap.dedent(s).strip()+'\n'})
md('# Planner repetition: Transformers vs vLLM, guided vs unguided\nT4x2, Internet on. Twelve fixed training-only probes:8 initial,2 gold-prefix,2 terminal. No ASR, validation tuning, tool rollout, training or submission. Both backends use FP16, same tokenizer/prompt/schema/max320 new tokens, greedy batch1. Four core adapted conditions plus two base-model controls and one vLLM CUDA-Graph control. Transformers runs GPU0, vLLM GPU1; graph control runs afterward in a fresh subprocess. Raw outputs and prompt/token IDs fingerprints are saved. Hardware/attention kernels differ (SDPA vs XFormers); this is not a bitwise numerical-equivalence test. It does not reproduce the training NF4 precision. First-request warmup is included in timings.')
code('''
from pathlib import Path
import os,sys,json,hashlib,subprocess,zipfile,time,shutil
MODEL='Qwen/Qwen3-4B-Instruct-2507'
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
MODE='multi_local'  # trained local Planner and Caller, terminal route without Summarizer
assert MODE=='multi_local'
RUN_NAME='planner_backend_diagnostic_v1'
SESSION_STARTED=time.monotonic()
SESSION_HOURS=10  # Stop before new batches after this budget; no per-sample timeout.
RESUME_DIR=None  # Optional mounted prior run directory containing config.json and split subfolders.
WORK=Path('/kaggle/working/planner_backend_comparison')/RUN_NAME;WORK.mkdir(parents=True,exist_ok=True)
if RESUME_DIR is not None:shutil.copytree(Path(RESUME_DIR),WORK,dirs_exist_ok=True)
SOURCE=Path('/kaggle/working/planner_comparison_source')
CORE_REV='4894d331692293565635454a8c686b7c18314a04'
if not SOURCE.exists():subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/tnisl/baseline.git',str(SOURCE)],check=True)
subprocess.run(['git','-C',str(SOURCE),'fetch','origin',CORE_REV],check=True)
subprocess.run(['git','-C',str(SOURCE),'checkout','--detach','FETCH_HEAD'],check=True)
''')
code(f'''
PAYLOAD={payload!r}
SOURCE_SHA={sha!r}
assert hashlib.sha256(json.dumps(PAYLOAD,sort_keys=True).encode()).hexdigest()==SOURCE_SHA
for name,content in PAYLOAD.items():
    path=SOURCE/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
assert hashlib.sha256((SOURCE/'split.txt').read_bytes()).hexdigest()=='ce713fbd99bf0bad81420935477f545af1b97fa96e5b606c6ea9377e465b7427'
''')
code('''
sys.path.insert(0,str(SOURCE))
from umilife.paths import find_data_root
DATA_ROOT,archive=find_data_root('/kaggle/input')
if DATA_ROOT is None:
    destination=Path('/tmp/v6_competition_data');destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert all((destination/m.filename).resolve().is_relative_to(destination.resolve()) for m in z.infolist())
        z.extractall(destination)
    DATA_ROOT,_=find_data_root(destination)
assert hashlib.sha256((DATA_ROOT/'train/train.jsonl').read_bytes()).hexdigest()=='24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8'
def input_dir(slug,marker):
    candidates=[p.parent for p in Path('/kaggle/input').rglob(marker) if p.parent.name==slug]
    return next((p for p in candidates if (p/marker).is_file()),None)
ADAPTERS=input_dir('lifemate-concise-planner-caller-adapters','checkpoint_manifest.json')

assert ADAPTERS is not None,'Attach the completed Planner/Caller checkpoint dataset'
manifest=json.loads((ADAPTERS/'checkpoint_manifest.json').read_text())
assert manifest['roles']==['planner','caller']
missing=[name for name in manifest['files'] if not (ADAPTERS/name).is_file()]
assert not missing, f'Incomplete checkpoint input: {missing}. Update lifemate-concise-planner-caller-adapters to dataset version 2 or later.'
print('Checkpoint input:',ADAPTERS,'Files:',len(manifest['files']))
for name,expected in manifest['files'].items():assert hashlib.sha256((ADAPTERS/name).read_bytes()).hexdigest()==expected,name
from umilife.prompt_policy import PROMPTS,digest,POLICY_VERSION
for role in ('planner','caller'):
    meta=json.loads((ADAPTERS/role/'lineage.json').read_text())
    assert meta['prompt_sha256']==digest(PROMPTS) and meta['model']==MODEL and meta['revision']==REVISION
config={'mode':MODE,'model':MODEL,'revision':REVISION,'source_sha':SOURCE_SHA,
        'checkpoint_manifest_sha':hashlib.sha256((ADAPTERS/'checkpoint_manifest.json').read_bytes()).hexdigest(),
        'prompt_sha':digest(PROMPTS),'policy':POLICY_VERSION,'batch_size':1,'max_tokens':320,'probes':12,'precision':'float16','seed':42,'asr':False,'train_only':True}
if (WORK/'config.json').exists():assert json.loads((WORK/'config.json').read_text())==config,'Different run settings; use a fresh RUN_NAME'
else:(WORK/'config.json').write_text(json.dumps(config,indent=2))
print('Checkpoint roles:',manifest['roles'],'Mode:',MODE)
''')
code('''
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.9.10'],check=True)
UV=[sys.executable,'-m','uv'];os.environ['UV_NATIVE_TLS']='true'
def runtime(name,lock,gpus):
    path=Path('/tmp')/name;python=str(path/'bin/python')
    if not Path(python).exists():subprocess.run([*UV,'venv','--python','3.12','--python-preference','only-managed','--seed',str(path)],check=True)
    subprocess.run([*UV,'pip','sync','--index-strategy','unsafe-best-match','--python',python,str(SOURCE/lock)],check=True)
    subprocess.run([*UV,'pip','check','--python',python],check=True)
    purelib=Path(subprocess.check_output([python,'-I','-c',"import sysconfig;print(sysconfig.get_path('purelib'))"],text=True).strip())
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=gpus,PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',
        VLLM_USE_V1='0',VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_USAGE_STATS_ENABLED='0',DO_NOT_TRACK='1',HF_HOME='/tmp/lifemate_hf_cache')
    env['LD_LIBRARY_PATH']=':'.join([str(p) for p in sorted((purelib/'nvidia').glob('*/lib'))]+[os.environ.get('LD_LIBRARY_PATH','')])
    return python,env
INFER_PYTHON,INFER_ENV=runtime('lifemate_vllm092_env','requirements_infer_2gpu.lock','0,1')

subprocess.run([INFER_PYTHON,'-c',"import torch,vllm;assert torch.cuda.device_count()==2;assert vllm.__version__=='0.9.2'"],env=INFER_ENV,check=True)
''')
code("subprocess.run([INFER_PYTHON,'-u','-m','umilife.prepare_comparison','--data-root',str(DATA_ROOT),'--adapters',str(ADAPTERS),'--work',str(WORK)],cwd=SOURCE,env=INFER_ENV,check=True)\nworkers=[];handles=[]\ntry:\n for backend,gpu in [('transformers','0'),('vllm','1')]:\n  env=INFER_ENV.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,TRITON_CACHE_DIR='/tmp/compare_triton_'+gpu,VLLM_CACHE_ROOT='/tmp/compare_vllm_'+gpu)\n  log=(WORK/(backend+'.log')).open('w',buffering=1);handles.append(log)\n  command=[INFER_PYTHON,'-u','-m','umilife.compare_worker','--backend',backend,'--work',str(WORK),'--adapters',str(ADAPTERS),'--model',MODEL,'--revision',REVISION,'--max-tokens','320']\n  workers.append((backend,subprocess.Popen(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)))\n counts={}\n while any(p.poll() is None for _,p in workers):\n  for backend,p in workers:\n   if p.poll() not in (None,0):raise RuntimeError(backend+' failed: '+(WORK/(backend+'.log')).read_text()[-10000:])\n   path=WORK/(backend+'.jsonl');n=len(path.read_text().splitlines()) if path.exists() else 0\n   if counts.get(backend)!=n:counts[backend]=n;print(backend,n,'/36',flush=True)\n  time.sleep(2)\n for backend,p in workers:\n  if p.returncode:raise RuntimeError(backend+' failed: '+(WORK/(backend+'.log')).read_text()[-10000:])\nfinally:\n for _,p in workers:\n  if p.poll() is None:\n   p.terminate()\n   try:p.wait(timeout=10)\n   except subprocess.TimeoutExpired:p.kill();p.wait()\n for f in handles:f.close()\n# Fresh process for CUDA Graph control, so the eager engine cannot retain GPU memory.\nenv=INFER_ENV.copy();env.update(CUDA_VISIBLE_DEVICES='1',TRITON_CACHE_DIR='/tmp/compare_triton_1',VLLM_CACHE_ROOT='/tmp/compare_vllm_1')\nsubprocess.run([INFER_PYTHON,'-u','-m','umilife.compare_worker','--backend','vllm','--graph','--work',str(WORK),'--adapters',str(ADAPTERS),'--model',MODEL,'--revision',REVISION,'--max-tokens','320'],cwd=SOURCE,env=env,check=True)\nsubprocess.run([INFER_PYTHON,'-u','-m','umilife.summarize_comparison','--work',str(WORK)],cwd=SOURCE,env=INFER_ENV,check=True)\nprint('Results:',WORK/'report.json',WORK/'summary.csv')\n")
for i,c in enumerate(cells):
 c['id']=f'v6-infer-{i:02}'
 if c['cell_type']=='code':ast.parse(c['source'])
for name,s in payload.items():
 if name.endswith('.py'):ast.parse(s)
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'}}
(HERE/'kaggle_planner_backend_comparison.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
meta={'id':'anartt/lifemate-planner-backend-comparison','title':'LifeMate Planner Backend Comparison','code_file':'kaggle_planner_backend_comparison.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
'dataset_sources':['cymeu4l0t/zaloai-dataset','anartt/lifemate-concise-planner-caller-adapters'],'competition_sources':[],'kernel_sources':[]}
(HERE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Built',sha)
