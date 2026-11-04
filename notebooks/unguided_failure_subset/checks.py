"""CPU checks for the changed decoding path, bounded repairs and report comparison."""
import ast,hashlib,json,sys,tempfile,types
from pathlib import Path
from collections import Counter
from unittest.mock import patch
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'runtime'))
import umilife
umilife.__path__.append(str(H.parents[1]/'umilife'))
from umilife.infer_2gpu import Backend,rollout,run_batch
from umilife.summarize_subset import generation_stats,summarize
from umilife.prompt_policy import PROMPTS,digest
assert digest(PROMPTS)=='6a339bb5157de0f61682f487c7911074df7fa6f305ae1c2763e94376f4b934af'
for name in ('prompt_policy.py','typescript_schema.py','runtime_options.py','transcribe_supplied.py'):
    assert (H/'runtime/umilife'/name).read_bytes()==(H.parent/'concise_validation_fast/runtime/umilife'/name).read_bytes()
ids=(H/'runtime/diagnostic_ids.txt').read_text().splitlines()
assert len(ids)==4 and len(set(ids))==4 and set(ids)<=set((H/'runtime/split.txt').read_text().splitlines())
nb=json.loads((H/'kaggle_unguided_failure_subset.ipynb').read_text())
for cell in nb['cells']:
    if cell['cell_type']=='code':ast.parse(cell['source'])
payload_cell=ast.parse(nb['cells'][2]['source'])
payload=ast.literal_eval(payload_cell.body[0].value)
assert payload=={str(p.relative_to(H/'runtime')):p.read_text() for p in (H/'runtime').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
# Execute the real sampler without importing GPU libraries. No grammar, regex or processor.
b=Backend.__new__(Backend);b.max_tokens={'planner':320,'caller':512}
with patch.dict(sys.modules,{'vllm':types.SimpleNamespace(SamplingParams=lambda **kw:kw)}):
    for role in ('planner','caller'):
        settings=b.make_sampling({'role':role})
        assert settings=={'temperature':0,'seed':42,'max_tokens':b.max_tokens[role],'guided_decoding':None}
# Exercise the real post-generation validator with valid and broken model outputs.
b.requests={'planner':None,'caller':None};b.validate=lambda r:None;b.batch_sizes=[];b.roles_seen=Counter();b.schema_violations=0
b.tokenizer=types.SimpleNamespace(apply_chat_template=lambda *a,**kw:'prompt')
outputs=[json.dumps({'thought':'Do the operation.','next':'caller','tool':'demo.write'}),'{bad',json.dumps({'next':'invented'})]
b.llm=types.SimpleNamespace(generate=lambda *a,**kw:[types.SimpleNamespace(outputs=[types.SimpleNamespace(text=x,finish_reason='stop',stop_reason=None,token_ids=[1])]) for x in outputs])
schema={'type':'object','properties':{'thought':{'type':'string'},'next':{'enum':['caller']},'tool':{'enum':['demo.write']}},'required':['thought','next','tool'],'additionalProperties':False}
request={'role':'planner','schema':schema,'messages':[]}
with patch.dict(sys.modules,{'vllm':types.SimpleNamespace(SamplingParams=lambda **kw:kw)}):
    responses=b.generate([request]*3)
assert [r['schema_valid'] for r in responses]==[True,False,False] and b.schema_violations==2
assert all(r['guided_decoding'] is False for r in responses)
class Env:
    def __init__(self):self.calls=[]
    def validate(self,*args):return {'ok':True}
    def call(self,name,arguments):self.calls.append({'name':name,'arguments':arguments});return {'ok':True}
    def get_tool_calls(self):return self.calls
    def get_state_hash(self):return 'same'
class Broken:
    def __init__(self,role):self.role=role;self.counts=Counter()
    def validate(self,r):pass
    def generate(self,requests):
        out=[]
        for r in requests:
            role=r['role'];self.counts[role]+=1
            if role==self.role:out.append({'raw':'bad','schema_valid':False,'schema_error':'fixture'})
            else:out.append({'raw':json.dumps({'thought':'Perform the operation.','next':'caller','tool':'demo.write'}),'schema_valid':True})
        return out
specs={'demo.write':{'kind':'write','description':'Write value','parameters':{'type':'object','properties':{'value':{'type':'string'}},'required':['value']}}}
for role in ('planner','caller'):
    backend=Broken(role);env=Env()
    result=run_batch([rollout({'id':'x','context':{},'history':[]},'Write requested value.',specs,env,'multi')],backend)[0]
    assert result[1]['reason']=='format_error' and backend.counts[role]==2 and not env.calls
# Historical report is read only after inference; compare all selected IDs exactly once.
baseline=json.loads((H/'runtime/baseline_subset.json').read_text())
with tempfile.TemporaryDirectory() as temp:
    w=Path(temp)
    (w/'report.json').write_text(json.dumps({'metrics':{'per_sample':[x['metrics'] for x in baseline['samples'].values()]}}))
    (w/'traces.jsonl').write_text(''.join(json.dumps({'id':sid,'reason':'complete','trace':[{'role':'planner','raw':json.dumps({'thought':'Perform the operation.'}),'schema_valid':True}]})+'\n' for sid in ids))
    (w/'submission.jsonl').write_text(''.join(json.dumps({'id':sid,'tool_calls':[]})+'\n' for sid in ids))
    comparison=summarize(w,H/'runtime/baseline_subset.json')
    assert len(comparison['samples'])==4 and (w/'comparison.md').exists()
print('PASS: unchanged prompts/contracts, four held-out IDs, embedded runtime, unguided both nodes, JSON/schema rejection, one repair, no invalid tool execution, historical comparison')
