import sys,json,ast,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'runtime'))
import umilife
umilife.__path__.append(str(H.parents[1]/'umilife'))
from umilife.infer_2gpu import rollout,run_batch
from umilife.prompt_policy import PROMPTS
from umilife.evaluate import evaluate
class Env:
 def __init__(self):self.calls=[]
 def validate(self,n,a):return {'ok':True}
 def call(self,n,a):self.calls.append({'name':n,'arguments':a});return {'ok':True}
 def get_tool_calls(self):return self.calls
 def get_state_hash(self):return 'same'
class Backend:
 def validate(self,r):assert r['role'] in ('planner','caller')
 def generate(self,rs):
  out=[]
  for r in rs:
   if r['role']=='caller':x={'name':'demo.write','arguments':{'value':'requested'}}
   else:
    done=bool(json.loads(r['messages'][1]['content'])['trajectory'])
    x={'thought':'Finish.' if done else 'Perform the requested operation.','next':'summarizer' if done else 'caller'}
    if not done:x['tool']='demo.write'
   out.append({'raw':json.dumps(x),'schema_valid':True})
  return out
specs={'demo.write':{'kind':'write','description':'Write a value','parameters':{'type':'object','properties':{'value':{'type':'string'}},'required':['value']}}}
rows=[{'id':f'x{i}','context':{},'history':[]} for i in range(4)]
r=run_batch([rollout(row,'Perform the requested write.',specs,Env(),'multi') for row in rows],Backend())
assert all(x[1]['reason']=='complete' and len(x[0]['tool_calls'])==1 for x in r)
assert all(t.get('role')!='summarizer' for x in r for t in x[1]['trace'])
report=evaluate([x[0] for x in r],[x[0] for x in r],specs)
assert report['tool_f1']==report['state_accuracy']==1 and report['score_100']==100
nb=json.loads((H/'kaggle_concise_validation_fast.ipynb').read_text())
for c in nb['cells']:
 if c['cell_type']=='code':ast.parse(c['source'])
assert not any("public_test=run_split" in c['source'] for c in nb['cells'])
print('PASS: four workflows, terminal route without Summarizer, scoring, notebook syntax, validation only')

# Broken model output must consume the same bounded retry budget, never restart it.
from collections import Counter
class Broken(Backend):
 def __init__(self,role,one_failure=False):self.role=role;self.counts=Counter();self.one_failure=one_failure
 def generate(self,rs):
  responses=[]
  for r in rs:
   self.counts[r['role']]+=1
   fail=r['role']==self.role and (not self.one_failure or self.counts[r['role']]==1)
   responses.append({'raw':'bad','schema_valid':False,'schema_error':'fixture failure'} if fail else super().generate([r])[0])
  return responses
for role in ('planner','caller'):
 backend=Broken(role);env=Env();result=run_batch([rollout(rows[0],'write',specs,env,'multi')],backend)[0]
 assert result[1]['reason']=='format_error'
 assert backend.counts[role]==2 and env.calls==[],(role,backend.counts)
 backend=Broken(role,one_failure=True);env=Env();result=run_batch([rollout(rows[0],'write',specs,env,'multi')],backend)[0]
 assert result[1]['reason']=='complete' and len(env.calls)==1
 calls=Counter((t['decision'],t['role']) for t in result[1]['trace'] if 'raw' in t)
 assert max(calls.values())<=2,calls
# A first invalid Planner response consumes its revision, so no third completion review.
class TerminalAfterFailure(Broken):
 def generate(self,rs):
  out=[]
  for r in rs:
   self.counts[r['role']]+=1
   out.append({'raw':'bad','schema_valid':False} if self.counts[r['role']]==1 else
              {'raw':json.dumps({'thought':'No operation needed.','next':'summarizer'}),'schema_valid':True})
  return out
b=TerminalAfterFailure('planner');result=run_batch([rollout(rows[0],'no action',specs,Env(),'multi')],b)[0]
assert b.counts['planner']==2 and result[1]['reason']=='complete'

# Scoring merged workers must cover IDs exactly once and restore requested order.
import tempfile
import umilife.parallel_validation as parallel
with tempfile.TemporaryDirectory() as temp:
 w=Path(temp);(w/'train').mkdir()
 gold=[{'id':r['id'],'label':{'tool_calls':[{'name':'demo.write','arguments':{'value':'requested'}}]}} for r in rows[:2]]
 (w/'train/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in gold))
 (w/'asr_report.json').write_text('{}')
 for rank in range(2):
  part=w/f'gpu{rank}';part.mkdir()
  (part/'submission.jsonl').write_text(json.dumps(r[rank][0])+'\n')
  (part/'traces.jsonl').write_text(json.dumps(r[rank][1])+'\n')
  (part/'report.json').write_text('{}')
 parallel.load_catalog=lambda *a:specs
 parallel.import_toolenv=lambda *a:lambda *args,**kwargs:Env()
 report=parallel.merge(w,w,[rows[1]['id'],rows[0]['id']])
 assert report['metrics']['score_100']==100 and report['samples']==2
 assert [x['id'] for x in parallel.read_jsonl(w/'submission.jsonl')]==[rows[1]['id'],rows[0]['id']]
 (w/'gpu1/submission.jsonl').write_text((w/'gpu0/submission.jsonl').read_text())
 try:parallel.merge(w,w,[rows[1]['id'],rows[0]['id']]);raise RuntimeError('Duplicate IDs were accepted')
 except AssertionError:pass
print('PASS: one-revision budgets, recovered errors, shared Planner review budget, two-worker merge and duplicate rejection')
