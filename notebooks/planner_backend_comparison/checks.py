import sys,json,tempfile,ast
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'runtime'))
from umilife.comparison_common import assess,training_probes
from umilife.summarize_comparison import summarize
schema={'type':'object','properties':{'thought':{'type':'string'},'next':{'enum':['caller']},'tool':{'enum':['demo.read']}},'required':['thought','next','tool']}
expected={'next':'caller','tool':'demo.read'}
a=assess(json.dumps({'thought':'one two three four five six '*6,**expected}),schema,expected)
assert a['schema_valid'] and a['repeats_6gram_3plus'] and a['raw_repeats_6gram_3plus']
b=assess('one two three four five six '*6,schema,expected)
assert not b['schema_valid'] and b['raw_repeats_6gram_3plus']
rows=[{'id':str(i),'audio_text':'Nếu needed' if i%2 else 'normal','history':[{}],'label':{'tool_calls':[{}] if i%3 else []}} for i in range(30)]
chosen=training_probes(rows,{'1','2'},8);assert len(chosen)==8 and not {'1','2'}&{r['id'] for r in chosen}
with tempfile.TemporaryDirectory() as temp:
 w=Path(temp);(w/'requests.json').write_text(json.dumps([{'probe':'p'}]))
 for backend in ('transformers','vllm'):
  rs=[{'probe':'p','backend':backend,'mode':mode,'cuda_graphs':False,'max_new_tokens':320,'prompt_sha256':'same','input_ids_sha256':'same','schema_sha256':'same','seconds':1,**a} for mode in ('base_unguided','adapter_unguided','adapter_guided')]
  (w/(backend+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rs))
 rs=[{**rs[-1],'cuda_graphs':True}];(w/'vllm_graph.jsonl').write_text(json.dumps(rs[0])+'\n')
 r=summarize(w);assert len(r['conditions'])==7
 rs[0]['input_ids_sha256']='different';(w/'vllm_graph.jsonl').write_text(json.dumps(rs[0])+'\n')
 try:summarize(w);raise RuntimeError('Mismatched tokenizer accepted')
 except AssertionError:pass
for f in (H/'runtime').rglob('*.py'):ast.parse(f.read_text())
nb=json.loads((H/'kaggle_planner_backend_comparison.ipynb').read_text())
for c in nb['cells']:
 if c['cell_type']=='code':ast.parse(c['source'])
print('PASS: repetition on valid/invalid JSON, train-only selection, seven conditions, token-ID parity enforcement, syntax')
