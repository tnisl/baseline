import ast,copy,json,sys,tempfile,unittest
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'runtime'))
import umilife
umilife.__path__.append(str(H.parents[1]/'umilife'))
from umilife.eval_probe import trim_generated,repetition,groups,fingerprint
from umilife.probe_summary import summarize_probe
class Checks(unittest.TestCase):
 def test_stops_and_loops(self):
  self.assertEqual(trim_generated([5,9,0,0],{9}),([5,9],'eos'))
  self.assertEqual(trim_generated([5]*512,{9}),([5]*512,'length'))
  self.assertGreaterEqual(repetition('Preserve the requested fuel type and condition. '*8),4)
 def test_prompt_only_and_batch_parity(self):
  class Tok:
   def apply_chat_template(self,messages,**kw):return [ord(c) for c in messages[0]['content']]
  rows=[{'messages':[{'role':'user','content':s}],'target':'SECRET'} for s in ['longer','short']]
  one=groups(rows,Tok(),1);many=groups(rows,Tok(),4)
  self.assertEqual([fingerprint(p[1]) for g in one for p in g],[fingerprint(p[1]) for g in many for p in g])
  changed=copy.deepcopy(rows);changed[0]['target']='OTHER_SECRET'
  self.assertEqual(groups(changed,Tok(),4)[0][1][1],many[0][1][1])
 def test_paired_summary_and_mismatch(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'input';source.mkdir();samples=[{'id':f'x{i}','step':0} for i in range(6)]
   (source/'probe_manifest.json').write_text(json.dumps({'samples':samples}))
   for b in ('unsloth','vllm'):
    for v in ('base','step75'):
     rows=[{**s,'backend':b,'variant':v,'batch_size':n,'batch_index':i//n,'batch_seconds':2.,'input_ids_sha256':s['id'],
      'generated_tokens':10,'finish_reason':'eos','max_repeated_6gram':0,'raw':'{}','schema_valid':True,'correct':True} for n in (1,4) for i,s in enumerate(samples)]
     (root/f'{b}_{v}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
   result=summarize_probe(root,source);self.assertEqual(len(result['cases']),8)
   self.assertEqual(result['cases']['unsloth/base/batch4']['generation_seconds'],4.)
   self.assertTrue(all(r['prompt_token_ids_identical'] for r in result['paired']))
   path=root/'vllm_step75.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()];rows[0]['input_ids_sha256']='wrong';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
   with self.assertRaisesRegex(AssertionError,'parity'):summarize_probe(root,source)
 def test_source_and_payload(self):
  nb=json.loads((H/'kaggle_evaluation_failure_probe.ipynb').read_text())
  for c in nb['cells']:
   if c['cell_type']=='code':ast.parse(c['source'])
  payload=ast.literal_eval(ast.parse(nb['cells'][2]['source']).body[0].value)
  self.assertEqual(payload,{str(p.relative_to(H/'runtime')):p.read_text() for p in (H/'runtime').rglob('*') if p.is_file() and '__pycache__' not in str(p)})
  self.assertEqual(payload['umilife/prompt_policy.py'],(H.parent/'planner_caller_evidence/runtime/umilife/prompt_policy.py').read_text())
  selection=json.loads((H/'selection.json').read_text());self.assertEqual(selection['adapter_step'],75);self.assertFalse(selection['exact_step100_reproduction'])
if __name__=='__main__':unittest.main()
