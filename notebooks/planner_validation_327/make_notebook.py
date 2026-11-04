import ast
import hashlib
import json
import textwrap
from pathlib import Path

HERE=Path(__file__).resolve().parent
payload={str(p.relative_to(HERE/'runtime')):p.read_text() for p in sorted((HERE/'runtime').rglob('*'))
         if p.is_file() and '__pycache__' not in str(p)}
source_sha=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
cells=[]
def cell(kind, source):
    result={'cell_type':kind,'metadata':{},'source':textwrap.dedent(source).strip()+'\n'}
    if kind=='code': result.update(execution_count=None,outputs=[]);ast.parse(result['source'])
    cells.append(result)

cell('markdown','''
# Planner: free generation on all 327 validation inputs

Select a T4 GPU, enable Internet, and Run All. This notebook performs no training and executes no tools.

It uses the **exact saved Planner validation messages from training version 6**: 327 step inputs from the 100 `split.txt` episodes, with clean reference transcripts and previous gold trajectory. Each current target is stored separately and is read by the scorer only after every prediction is saved. This is an offline diagnostic, not an end-to-end/ASR benchmark or public-test prediction run.

The original Vietnamese prompt, JSON schemas, vLLM 0.9.2 FP16 backend and completed Planner adapter are preserved. One free generation per input, temperature 0, maximum 320 output tokens, batch size 4 (last batch 3). No completion review or repair retries. Four additional synthetic preflight requests test the engine/grammar independently of validation data.

Report separate route accuracy, caller tool accuracy on 227 caller steps, terminal accuracy on 100 terminal steps, first-step accuracy, schema failures and per-example CSV. Exact reference-tool matching does not adjudicate alternative valid tool orders or rationale semantics. The all-steps episode metric is not rollout task success.

Inputs are private: `cymeu4l0t/lifemate-v6-planner-validation-inputs` and `cymeu4l0t/lifemate-v6-completed-adapters` **version 2 or later**. Finished batches can be reused if their output directory is preserved. Interrupted unsaved generations can safely be regenerated because no tools execute.
''')
cell('code','''
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
MODEL='Qwen/Qwen3-4B-Instruct-2507'
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
RUN_NAME='planner_v6_gold_prefix_327_v1'
SESSION_STARTED=time.monotonic()
SESSION_HOURS=10
RESUME_DIR=None  # Optional previous run directory containing config.json and batches/.
WORK=Path('/kaggle/working/planner_validation_327')/RUN_NAME
WORK.mkdir(parents=True,exist_ok=True)
if RESUME_DIR is not None:shutil.copytree(Path(RESUME_DIR),WORK,dirs_exist_ok=True)
SOURCE=Path('/kaggle/working/planner_validation_327_source')
CORE_REV='4894d331692293565635454a8c686b7c18314a04'
if not SOURCE.exists():
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/tnisl/baseline.git',str(SOURCE)],check=True)
subprocess.run(['git','-C',str(SOURCE),'fetch','origin',CORE_REV],check=True)
subprocess.run(['git','-C',str(SOURCE),'checkout','--detach','FETCH_HEAD'],check=True)
''')
cell('code',f'''
PAYLOAD={payload!r}
SOURCE_SHA={source_sha!r}
assert hashlib.sha256(json.dumps(PAYLOAD,sort_keys=True).encode()).hexdigest()==SOURCE_SHA
for name,content in PAYLOAD.items():
    path=SOURCE/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
''')
cell('code','''
def input_dir(slug,marker):
    candidates=[Path('/kaggle/input')/slug,Path('/kaggle/input/datasets/cymeu4l0t')/slug]
    result=next((p for p in candidates if (p/marker).is_file()),None)
    assert result is not None,f'Attach private input cymeu4l0t/{slug}'
    return result
INPUTS=input_dir('lifemate-v6-planner-validation-inputs','evaluation_manifest.json')
ADAPTERS=input_dir('lifemate-v6-completed-adapters','checkpoint_manifest.json')
for folder,manifest_name in [(INPUTS,'evaluation_manifest.json'),(ADAPTERS,'checkpoint_manifest.json')]:
    manifest=json.loads((folder/manifest_name).read_text())
    missing=[n for n in manifest['files'] if not (folder/n).is_file()]
    assert not missing,f'Incomplete input {folder}: {missing}; adapters require dataset version >=2'
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
provenance=json.loads((INPUTS/'evaluation_manifest.json').read_text())
assert provenance['samples']==327 and provenance['episodes']==100
lineage=json.loads((ADAPTERS/'planner/lineage.json').read_text())
assert lineage['dataset_manifest_sha256']==provenance['training_manifest_sha256']
assert lineage['model']==MODEL and lineage['revision']==REVISION
print('Inputs:',INPUTS,'Planner adapter:',ADAPTERS/'planner')
print('Gold-prefix examples:',provenance['samples'],'Episodes:',provenance['episodes'])
''')
cell('code','''
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.9.10'],check=True)
UV=[sys.executable,'-m','uv'];os.environ['UV_NATIVE_TLS']='true'
environment=Path('/tmp/lifemate_planner_vllm092_env')
if not (environment/'bin/python').exists():
    subprocess.run([*UV,'venv','--python','3.12','--python-preference','only-managed','--seed',str(environment)],check=True)
PYTHON=str(environment/'bin/python')
subprocess.run([*UV,'pip','sync','--index-strategy','unsafe-best-match','--python',PYTHON,str(SOURCE/'requirements_infer_2gpu.lock')],check=True)
subprocess.run([*UV,'pip','check','--python',PYTHON],check=True)
purelib=Path(subprocess.check_output([PYTHON,'-I','-c',"import sysconfig;print(sysconfig.get_path('purelib'))"],text=True).strip())
ENV=os.environ.copy()
for key in ('PYTHONPATH','PYTHONHOME'):ENV.pop(key,None)
ENV.update(CUDA_VISIBLE_DEVICES='0',PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',
    VLLM_USE_V1='0',VLLM_ATTENTION_BACKEND='XFORMERS',VLLM_WORKER_MULTIPROC_METHOD='spawn',
    VLLM_USAGE_STATS_ENABLED='0',DO_NOT_TRACK='1',HF_HOME='/tmp/lifemate_hf_cache')
ENV['LD_LIBRARY_PATH']=':'.join([str(p) for p in sorted((purelib/'nvidia').glob('*/lib'))]+[os.environ.get('LD_LIBRARY_PATH','')])
subprocess.run([PYTHON,'-c',"import torch,vllm;assert torch.cuda.is_available();assert vllm.__version__=='0.9.2';print(torch.cuda.get_device_name(0))"],env=ENV,check=True)
''')
cell('code','''
subprocess.run([PYTHON,'-u','-m','umilife.eval_planner',
    '--inputs',str(INPUTS),'--adapters',str(ADAPTERS),'--work',str(WORK),
    '--model',MODEL,'--revision',REVISION,'--source-sha',SOURCE_SHA,
    '--deadline-monotonic',str(SESSION_STARTED+SESSION_HOURS*3600)],cwd=SOURCE,env=ENV,check=True)
''')
cell('code','''
report=json.loads((WORK/'report.json').read_text())
assert report['status']=='complete' and report['metrics']['samples']==327
metrics=report['metrics']
for key in ('route_accuracy','route_and_tool_accuracy','caller_tool_accuracy',
            'terminal_route_accuracy','first_route_and_tool_accuracy'):
    print(f'{key}: {metrics[key]:.2%}')
print('Schema failures:',metrics['schema_invalid'])
print('Full report:',WORK/'report.json')
print('Per-example results:',WORK/'details.csv')
print(json.dumps(metrics['route_confusion'],ensure_ascii=False,indent=2))
''')
for i,c in enumerate(cells):c['id']=f'planner327-{i:02}'
for name,content in payload.items():
    if name.endswith('.py'):ast.parse(content)
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'accelerator':'GPU'}}
(HERE/'kaggle_planner_validation_327.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
metadata={'id':'cymeu4l0t/lifemate-planner-validation-327','title':'LifeMate Planner validation 327',
    'code_file':'kaggle_planner_validation_327.ipynb','language':'python','kernel_type':'notebook',
    'is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
    'dataset_sources':['cymeu4l0t/lifemate-v6-planner-validation-inputs','cymeu4l0t/lifemate-v6-completed-adapters'],
    'competition_sources':[],'kernel_sources':[]}
(HERE/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('BUILT_PLANNER_327',source_sha)
