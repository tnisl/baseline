"""CPU-only checks for old-policy recovery and public-test inference/export."""
import ast,importlib.util,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
sp=importlib.util.spec_from_file_location('umilife.prompt_policy',HERE/'runtime/umilife/prompt_policy.py');policy=importlib.util.module_from_spec(sp);sp.loader.exec_module(policy);sys.modules['umilife.prompt_policy']=policy
sp=importlib.util.spec_from_file_location('recovered_infer',HERE/'runtime/umilife/infer_2gpu.py');infer=importlib.util.module_from_spec(sp);sp.loader.exec_module(infer)
expected='0caf7a12a791a2414a6f7419447c32b1c1c07ffe41376f6a1782ad1e90631f7f'
assert policy.digest(policy.PROMPTS)==expected
assert 'typescript' not in policy.POLICY_VERSION
for name in ('global','planner'):
 path=ROOT/'outputs/kaggle/finetune_2gpu_v6_timeout/paper_finetune_2gpu/full_2gpu_chunked_loss_v2/adapters'/name/'lineage.json'
 assert json.loads(path.read_text())['prompt_sha256']==expected
class Backend:
    schema_violations=0
    batch_sizes=[]
    def __init__(self,*a):pass
    def preflight(self):return {'synthetic':'pass'}
    def validate(self,r):assert 'GOLD_CANARY' not in json.dumps(r)
    def generate(self,requests):
        self.batch_sizes.append(len(requests));out=[]
        for r in requests:
            self.validate(r)
            value={'thought':'Nothing pending','next':'summarizer'} if r['role']=='planner' else {'answer':'Done'}
            out.append({'raw':json.dumps(value),'schema_valid':True})
        return out
class Env:
    def __init__(self,*a,**kw):self.calls=[]
    def get_tool_calls(self):return self.calls
    def get_state_hash(self):return 'synthetic'
with tempfile.TemporaryDirectory() as d:
 root=Path(d);work=root/'work';work.mkdir();part=root/'public_test';part.mkdir()
 rows=[{'id':f'public_{i}','audio':f'{i}.wav','context':{'installed_apps':{},'current_time':'2026-11-01T00:00:00+07:00'},'history':[],'label':'GOLD_CANARY','audio_text':'GOLD_CANARY'} for i in range(4)]
 (part/'public_test.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
 (work/'ids.txt').write_text('\n'.join(r['id'] for r in rows)+'\n')
 (work/'asr_transcripts.json').write_text(json.dumps({r['audio']:'synthetic ASR input' for r in rows}))
 (work/'asr_report.json').write_text(json.dumps({'oracle_fallback':False,'validation_samples':4}))
 args=['infer','--data-root',str(root),'--work',str(work),'--model','fixture','--revision','fixture','--adapters','fixture','--ids-file',str(work/'ids.txt'),'--split','public_test','--mode','multi_fallback']
 with patch.object(infer,'Backend',Backend),patch.object(infer,'load_catalog',lambda *a:{}),patch.object(infer,'import_toolenv',lambda *a:Env),patch.object(sys,'argv',args):
  infer.main();infer.main()  # Reuse completed batches without re-executing tools.
 report=json.loads((work/'report.json').read_text());assert report['samples']==4 and 'metrics' not in report
 assert len((work/'submission.jsonl').read_text().splitlines())==4
 assert not (work/'active_batch.json').exists()
 assert len(list((work/'batches').glob('*.json')))==1
 # A fresh job pauses before creating any episode when its batch-boundary budget expires.
 fresh=root/'paused';fresh.mkdir()
 for name in ('ids.txt','asr_transcripts.json','asr_report.json'):(fresh/name).write_bytes((work/name).read_bytes())
 paused_args=[str(fresh) if x==str(work) else str(fresh/'ids.txt') if x==str(work/'ids.txt') else x for x in args]+['--deadline-monotonic','1']
 with patch.object(infer,'Backend',Backend),patch.object(infer,'load_catalog',lambda *a:{}),patch.object(infer,'import_toolenv',lambda *a:Env),patch.object(sys,'argv',paused_args):
  try:infer.main();raise AssertionError('Expected batch-boundary pause')
  except SystemExit as e:assert e.code==75
 assert json.loads((fresh/'progress.json').read_text())['completed']==0
 assert not (fresh/'active_batch.json').exists()
n=json.loads((HERE/'kaggle_v6_validation_test.ipynb').read_text())
for c in n['cells']:
 if c['cell_type']=='code':ast.parse(c['source'])
print('OLD_POLICY_HASH_PUBLIC_EXPORT_AND_BATCH_REUSE_PASS')
