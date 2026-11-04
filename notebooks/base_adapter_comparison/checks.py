"""CPU verification of paired inputs, reference isolation, scoring and adapter toggle."""
import ast,json,sys,tempfile
from pathlib import Path
from copy import deepcopy
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'runtime'))
import umilife
umilife.__path__.append(str(H.parents[1]/'umilife'))
from umilife.compare_finetune import safe_row,probes_for,assess,paired_nodes,summarize_nodes,workflows,ComparedBackend
from umilife.prompt_policy import digest,PROMPTS
specs={'demo.write':{'kind':'write','description':'Save value','parameters':{'type':'object','properties':{'value':{'type':'string'}},'required':['value']}}}
raw={'id':'x','audio':'x.wav','audio_text':'ORACLE_SECRET','history':[],
     'context':{'current_time':'2026-11-04T10:10:00+07:00','installed_apps':{'demo':{'tools':['demo.write'],'state':'HIDDEN_STATE'}}},
     'label':{'tool_calls':[{'name':'demo.write','arguments':{'value':'TARGET_SECRET'}}],
              'tool_results':[{'ok':True,'future':'FUTURE_SECRET'}],'final_state':'FINAL_SECRET'}}
transcripts={'x.wav':'Save the requested value.'};row=safe_row(raw,transcripts)
probes=list(probes_for(raw,row,transcripts['x.wav'],specs));assert len(probes)==3
for p in probes[:2]:
    messages=json.dumps(p['request']['messages'])
    for secret in ('ORACLE_SECRET','HIDDEN_STATE','TARGET_SECRET','FUTURE_SECRET','FINAL_SECRET'):assert secret not in messages,secret
changed=deepcopy(raw);changed['label']['tool_calls'][0]['arguments']['value']='OTHER_TARGET';changed['label']['tool_results'][0]['future']='OTHER_FUTURE'
other=list(probes_for(changed,row,transcripts['x.wav'],specs))
assert probes[0]['request']==other[0]['request'] and probes[1]['request']==other[1]['request']
assert 'FUTURE_SECRET' in json.dumps(probes[2]['request']['messages']) # Now a past observation.
empty=deepcopy(raw);empty['label']={'tool_calls':[],'tool_results':[]}
p=list(probes_for(empty,row,transcripts['x.wav'],specs))[0]
assert assess(p,{'raw':'{"thought":"Done","next":"summarizer"}','schema_valid':True},specs)['scorable'] is False
b=ComparedBackend.__new__(ComparedBackend);b.adapter_requests={'planner':object(),'caller':object()}
b.select('base');assert all(v is None for v in b.requests.values())
b.select('adapter');assert b.requests==b.adapter_requests
class Fake:
    def select(self,mode):self.mode=mode
    def validate(self,r):pass
    def generate(self,requests):
        out=[]
        for r in requests:
            content=json.loads(r['messages'][1]['content'])
            if r['role']=='planner':
                value={'thought':'Save requested value.','next':'summarizer' if content['trajectory'] else 'caller'}
                if value['next']=='caller':value['tool']='demo.write'
            else:value={'name':'demo.write','arguments':{'value':'TARGET_SECRET' if self.mode=='adapter' else 'wrong'}}
            out.append({'raw':json.dumps(value),'schema_valid':True,'mode':self.mode,'guided_decoding':False,
                'prompt_sha256':digest(r['messages']),'schema_sha256':digest(r['schema']),
                'input_ids_sha256':digest(r['messages'])})
        return out
class Env:
    def __init__(self):self.calls=[]
    def validate(self,*args):return {'ok':True}
    def call(self,name,arguments):self.calls.append({'name':name,'arguments':arguments});return {'ok':True}
    def get_tool_calls(self):return self.calls
    def get_state_hash(self):return json.dumps(self.calls,sort_keys=True)
envs=[]
def factory(sid):
    e=Env();envs.append(e);return e
with tempfile.TemporaryDirectory() as temp:
    w=Path(temp)
    rows=paired_nodes(probes,Fake(),specs,w);report=summarize_nodes(rows)
    assert report['caller']['paired_outcomes']['adapter_win']==1
    assert report['caller']['conditions']['base']['accuracy']==0
    assert report['planner']['conditions']['adapter']['accuracy']==1
    assert len(rows)==6
    result=workflows([raw],[row],transcripts,specs,factory,Fake(),w)
    assert result['base']['metrics']['score_100']==0 and result['adapter']['metrics']['score_100']==100
    assert len(envs)==3 and len({id(e) for e in envs})==3 and all(len(e.calls)==1 for e in envs)
# Embedded source must match actual runtime and prompts must match completed training.
assert digest(PROMPTS)=='6a339bb5157de0f61682f487c7911074df7fa6f305ae1c2763e94376f4b934af'
nb=json.loads((H/'kaggle_base_adapter_comparison.ipynb').read_text())
for c in nb['cells']:
    if c['cell_type']=='code':ast.parse(c['source'])
payload=ast.literal_eval(ast.parse(nb['cells'][2]['source']).body[0].value)
assert payload=={str(p.relative_to(H/'runtime')):p.read_text() for p in (H/'runtime').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
print('PASS: target/future/oracle isolation, past observations, no-action unscored, base disables adapters, paired fingerprints, win/loss scoring, independent workflow environments, embedded syntax/prompt parity')
