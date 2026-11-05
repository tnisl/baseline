import ast,json,hashlib,textwrap
from pathlib import Path
H=Path(__file__).resolve().parent
payload={str(p.relative_to(H/'runtime')):p.read_text() for p in sorted((H/'runtime').rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
sha=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();cells=[]
def cell(kind,source):
 source=textwrap.dedent(source).strip()+'\n';c={'cell_type':kind,'metadata':{},'source':source,'id':f'probe-{len(cells):02}'}
 if kind=='code':ast.parse(source);c.update(execution_count=None,outputs=[])
 cells.append(c)
cell('markdown','''
# Evaluation failure probe: six prefixes
Four actual looping states plus two controls from the interrupted evidence-v3 Planner evaluation. Exact saved messages; no ASR, tool execution, training, repair or enforcer. Step100 weights were not saved: use authentic step75 and retain historical step100 outputs for reference. This does NOT reproduce step100 exactly.
Compare base (zero-initialized LoRA) and step75 on Unsloth NF4, plain Transformers NF4, plain Transformers FP16, and vLLM FP16, batch1 and4, greedy max512 tokens.96 outputs total. Independent fresh subprocesses per model/backend avoid adapter cache contamination. Compare matched-precision pairs; loading, adapter dtypes and attention kernels may still differ. Startup/warmup costs are not controlled benchmarks.
Save source only; run yourself on T4 x2 with Internet. Outputs include exact token IDs, EOS/length stop, repetition indicators, prompt hashes, batch times and post-generation CE/flag-restoration checks. Reference answers are scored only after generation and never fed as generation input.
''')
cell('code','''
from pathlib import Path
import os,sys,json,hashlib,subprocess,time
SOURCE=Path('/kaggle/working/eval_probe_source')
WORK=Path('/kaggle/working/evaluation_failure_probe_precision');WORK.mkdir(parents=True,exist_ok=True)
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
inputs=list(Path('/kaggle/input').rglob('probe_manifest.json'));assert len(inputs)==1,'Attach the private evaluation failure inputs'
INPUT=inputs[0].parent;manifest=json.loads(inputs[0].read_text())
assert manifest['source_session_id']==365792172 and manifest['adapter_step']==75 and len(manifest['samples'])==6
for name,wanted in manifest['files'].items():assert hashlib.sha256((INPUT/name).read_bytes()).hexdigest()==wanted,name
(WORK/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
(WORK/'source.json').write_text(json.dumps({{'runtime_sha256':SOURCE_SHA,'guided_decoding':False}},indent=2))
print('Selected samples:',manifest['samples'])
''')
cell('code','''
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.9.10'],check=True)
UV=[sys.executable,'-m','uv'];os.environ['UV_NATIVE_TLS']='true'
def runtime(backend):
 path=Path('/tmp')/('eval_failure_vllm' if backend=='vllm' else 'eval_failure_hf');python=str(path/'bin/python')
 if not Path(python).exists():subprocess.run([*UV,'venv','--python','3.12','--seed',str(path)],check=True)
 lock='requirements_train_2gpu.lock' if backend!='vllm' else 'requirements_infer_2gpu.lock'
 subprocess.run([*UV,'pip','sync','--index-strategy','unsafe-best-match','--python',python,str(SOURCE/lock)],check=True)
 subprocess.run([*UV,'pip','check','--python',python],check=True)
 purelib=Path(subprocess.check_output([python,'-I','-c',"import sysconfig;print(sysconfig.get_path('purelib'))"],text=True).strip())
 env=os.environ.copy()
 for key in list(env):
  if key in ('PYTHONPATH','PYTHONHOME') or key.startswith('UNSLOTH_'):env.pop(key,None)
 env.update(PYTHONNOUSERSITE='1',TOKENIZERS_PARALLELISM='false',HF_HOME='/tmp/lifemate_hf_cache',OMP_NUM_THREADS='2',
     CUDA_VISIBLE_DEVICES='1' if backend=='vllm' else '0',VLLM_USE_V1='0',VLLM_ATTENTION_BACKEND='XFORMERS',
     VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USAGE_STATS_ENABLED='0',DO_NOT_TRACK='1')
 if backend=='unsloth':env.update(UNSLOTH_COMPILE_DISABLE='0',LIFEMATE_LOSS_CHUNK='512',LOSS_PROJECTION_CHECKPOINT='0')
 env['LD_LIBRARY_PATH']=':'.join(str(p) for p in (purelib/'nvidia').glob('*/lib'))+':'+os.environ.get('LD_LIBRARY_PATH','')
 subprocess.run([python,'-c','import torch;assert torch.cuda.device_count()==1'],env=env,check=True)
 return python,env
RUNTIMES={b:runtime(b) for b in ('transformers_nf4','transformers_fp16','unsloth','vllm')}
''')
cell('code','''
# Fresh model processes for each variant, sequential execution, releasing both GPUs between cases.
# Per-case progress is streamed directly to notebook logs and retained in WORK.
for backend in ('transformers_nf4','transformers_fp16','unsloth','vllm'):
 python,env=RUNTIMES[backend]
 for variant in ('base','step75'):
  log=WORK/f'{backend}_{variant}.log'
  command=[python,'-u','-m','umilife.eval_probe','--backend',backend,'--variant',variant,'--input',str(INPUT),'--output',str(WORK)]
  with log.open('w') as stream:
   process=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
   try:
    for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
    assert process.wait()==0,f'{backend}/{variant} failed; inspect {log}'
   finally:
    if process.poll() is None:process.terminate();process.wait(timeout=20)
''')
cell('code','''
python,env=RUNTIMES['vllm']
subprocess.run([python,'-u','-m','umilife.probe_summary','--work',str(WORK),'--input',str(INPUT)],cwd=SOURCE,env=env,check=True)
print('Read report.md, report.json, *_settings.json, *.jsonl and *.log in',WORK)
''')
for name,s in payload.items():
 if name.endswith('.py'):ast.parse(s,filename=name)
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'}}
(H/'kaggle_evaluation_failure_probe.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
meta={'id':'lctrnnguynkhi/lifemate-evaluation-failure-probe','title':'LifeMate Evaluation Failure Probe','code_file':'kaggle_evaluation_failure_probe.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['lctrnnguynkhi/lifemate-evaluation-failure-inputs'],'competition_sources':[],'kernel_sources':[]}
(H/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('BUILT',sha)
