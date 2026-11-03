"""CPU checks for the new instruction/schema experiment (no GPU job)."""
import ast,hashlib,importlib,json,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
pkg=types.ModuleType('english_ts_runtime');pkg.__path__=[str(HERE/'runtime/umilife')];sys.modules[pkg.__name__]=pkg
policy=importlib.import_module('english_ts_runtime.prompt_policy')
ts=importlib.import_module('english_ts_runtime.typescript_schema')
assert all(text.isascii() for text in [*policy.PROMPTS.values(),policy.COMPLETION_REVIEW])
specs={'demo.lookup':{'kind':'read','description':'Find item','parameters':{'type':'object','properties':{
    'id':{'type':'string','x-ref':'demo.items'},'name':{'type':'string'},
    'limit':{'type':'integer','minimum':1},'mode':{'type':'string','enum':['a','b']}},
    'required':['mode'],'x-one-of':[['id'],['name']]}}}
rendered=ts.typescript_catalog(specs)
assert 'interface InstalledTools' in rendered and '"limit"?: number' in rendered and '"mode": "a" | "b"' in rendered
assert '& (' in rendered and 'x-ref=' in rendered and 'minimum=1' in rendered
row={'id':'GOLD_CANARY','label':'GOLD_CANARY','audio_text':'GOLD_CANARY','context':{'current_time':'2026-11-01T12:00:00+07:00'},'history':[]}
for role in policy.PROMPTS:
 messages=policy.build_messages(role,row,'Find the provided item',specs,[],{'tool':'demo.lookup'})
 assert 'GOLD_CANARY' not in str(messages)
 if role!='summarizer':assert 'interface InstalledTools' in messages[-1]['content']
 assert messages[0]['content'].isascii()
notebook=json.loads((HERE/'kaggle_en_ts_t4.ipynb').read_text())
for cell in notebook['cells']:
 if cell['cell_type']=='code':ast.parse(cell['source'])
assert 'train_profile(\'smoke\')' in notebook['cells'][7]['source']
assert "gate['runtime_sha']==RUNTIME_SHA" in notebook['cells'][10]['source']
node=ast.parse(notebook['cells'][2]['source']).body[0];payload=ast.literal_eval(node.value)
assert payload['umilife/prompt_policy.py']==(HERE/'runtime/umilife/prompt_policy.py').read_text()
root=HERE.parents[2]
catalog=json.loads((root/'train+public_test/train/train_tools_catalog.json').read_text())
public={t['name']:t for app in catalog['apps'] for t in app['tools']}
Path('/tmp/en_ts_tests/catalog.ts').write_text(ts.typescript_catalog(public))
print(json.dumps({'status':'pass','public_tools':len(public),'english_instructions':True,'gold_canary_excluded':True,'notebook_code_compiles':True,'smoke_gate':True}))
