import ast,json,hashlib,textwrap
from pathlib import Path
HERE=Path(__file__).resolve().parent
payload={str(p.relative_to(HERE/'runtime')):p.read_text() for p in sorted((HERE/'runtime').rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
sha=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip()+'\n'})
def code(s):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':textwrap.dedent(s).strip()+'\n'})
md('# Base vs fine-tuned adapters\nPaired unguided comparison with the same tokenizer, English prompts, TypeScript contracts, FP16 engine, ASR transcripts and token limits. Default: the four previously failing examples; set SAMPLE_SET to valid245 for all held-out episodes. Fixed-prefix probes isolate Planner tool choice and Caller arguments. Independent base/adapter rollouts measure end-to-end behavior with 7 decisions /7 executed calls. Reference data in fixed-prefix probes is diagnostic conditioning, not deployment input. No finetuning, enforcer or automatic submission. Save/Run yourself.')
code('''
from pathlib import Path
import os,sys,json,hashlib,subprocess,zipfile
MODEL='Qwen/Qwen3-4B-Instruct-2507'
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
RUN_NAME='base_adapter_diagnostic4_v1'
SAMPLE_SET='diagnostic4'  # 'diagnostic4' or 'valid245'; change RUN_NAME for a new selection
assert SAMPLE_SET in ('diagnostic4','valid245')
WORK=Path('/kaggle/working/base_adapter_comparison')/RUN_NAME;WORK.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/kaggle/working/base_adapter_source')
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
ASR_MODEL=input_dir('lifemate-concise-validation-asr','model_manifest.json')
assert ADAPTERS is not None and ASR_MODEL is not None,'Attach completed v6 adapters and supplied ASR datasets'
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
ids=(SOURCE/('diagnostic_ids.txt' if SAMPLE_SET=='diagnostic4' else 'split.txt')).read_text().splitlines()
assert len(ids)==(4 if SAMPLE_SET=='diagnostic4' else 245) and len(set(ids))==len(ids)
config={'sample_set':SAMPLE_SET,'mode':'base_vs_adapter','model':MODEL,'revision':REVISION,'source_sha':SOURCE_SHA,
        'checkpoint_manifest_sha':hashlib.sha256((ADAPTERS/'checkpoint_manifest.json').read_bytes()).hexdigest(),
        'prompt_sha':digest(PROMPTS),'policy':POLICY_VERSION,'batch_size_per_gpu':4,'gpu_workers':1,'guided_decoding':False,'max_decisions':7,'max_tool_calls':7,'ids':ids,'cuda_graphs':True,'prefix_caching_policy':'sm80_plus','revisions_per_role':1,'asr':True}
if (WORK/'config.json').exists():assert json.loads((WORK/'config.json').read_text())==config,'Different run settings; use a fresh RUN_NAME'
else:(WORK/'config.json').write_text(json.dumps(config,indent=2))
print('Checkpoint roles:',manifest['roles'],'Comparison: base vs adapter')
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
INFER_PYTHON,INFER_ENV=runtime('lifemate_vllm092_env','requirements_infer_2gpu.lock','0')
ASR_PYTHON,ASR_ENV=runtime('lifemate_supplied_asr_env','requirements_asr_2gpu.lock','')
subprocess.run([INFER_PYTHON,'-c',"import torch,vllm;assert torch.cuda.device_count()>=1;assert vllm.__version__=='0.9.2'"],env=INFER_ENV,check=True)
''')
code('''
output=WORK/'validation';output.mkdir(exist_ok=True)
assert set(ids)<=set((SOURCE/'split.txt').read_text().splitlines())
(output/'split.txt').write_text('\\n'.join(ids)+'\\n')
print('IDs:',ids,'Compare base vs adapter; enforcer OFF; ASR ON; batch4; no training')
subprocess.run([ASR_PYTHON,'-u','-m','umilife.transcribe_supplied','--data-root',str(DATA_ROOT),
    '--work',str(output),'--model-dir',str(ASR_MODEL),'--split','train'],cwd=SOURCE,env=ASR_ENV,check=True)
subprocess.run([INFER_PYTHON,'-u','-m','umilife.compare_finetune','--data-root',str(DATA_ROOT),
    '--work',str(output),'--model',MODEL,'--revision',REVISION,'--adapters',str(ADAPTERS)],cwd=SOURCE,env=INFER_ENV,check=True)
report=json.loads((output/'comparison.json').read_text())
assert report['samples']==len(ids) and report['guided_decoding'] is False
print((output/'comparison.md').read_text())
print('Results:',output)
''')
for i,c in enumerate(cells):
 c['id']=f'v6-infer-{i:02}'
 if c['cell_type']=='code':ast.parse(c['source'])
for name,s in payload.items():
 if name.endswith('.py'):ast.parse(s)
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'}}
(HERE/'kaggle_base_adapter_comparison.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
meta={'id':'anartt/lifemate-base-adapter-comparison','title':'LifeMate Base Adapter Comparison','code_file':'kaggle_base_adapter_comparison.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
'dataset_sources':['cymeu4l0t/zaloai-dataset','anartt/lifemate-concise-planner-caller-adapters','anartt/lifemate-concise-validation-asr'],'competition_sources':[],'kernel_sources':[]}
(HERE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Built',sha)
