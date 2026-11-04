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
nb=json.loads((H/'kaggle_concise_validation.ipynb').read_text())
for c in nb['cells']:
 if c['cell_type']=='code':ast.parse(c['source'])
assert not any("public_test=run_split" in c['source'] for c in nb['cells'])
print('PASS: four workflows, terminal route without Summarizer, scoring, notebook syntax, validation only')
