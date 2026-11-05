"""CPU checks for evidence labels, causal loss/gradients, eval batching and metrics."""
import ast,contextlib,copy,json,os,sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'runtime'))
import umilife
umilife.__path__.append(str(H.parents[1]/'umilife'))
import torch
from umilife.evidence_labels import rationale,issues
from umilife.completion_loss import completion_loss
from umilife.ddp_engine import weighted_eval
from umilife.train import Collator
from umilife.prompt_policy import build_messages
from umilife.node_eval import assess,summarize,export_adapter,OUTPUT_TOKEN_LIMITS
from umilife.training_eval import generation_context,make_evaluator
class Checks(unittest.TestCase):
 def test_predicates_and_no_future_result(self):
  row={'audio_text':'Hoá đơn tới hạn trong 4 ngày, riêng cái nào quá 400 nghìn thì để đó.','context':{'current_time':'2026-10-01T10:00:00+07:00'},'history':[]}
  prior=[{'action':{'name':'bill_payment.list_bills','arguments':{}},'observation':{'ok':True,'bills':[
   {'id':'b1','type':'water','amount':200000,'due_date':'2026-10-03','paid':False},
   {'id':'b2','type':'internet','amount':300000,'due_date':'2026-10-04','paid':False},
   {'id':'b3','type':'electricity','amount':800000,'due_date':'2026-10-02','paid':False}]}}]
  action={'name':'bill_payment.pay','arguments':{'bill_id':'b1','source':'bank'}}
  text=rationale(row,action,prior)
  self.assertIn('200,000',text);self.assertIn('internet',text);self.assertIn('spending limit',text)
  row['label']={'future_observation':'SECRET'};self.assertEqual(text,rationale(row,action,prior))
  wrong={'name':'bill_payment.pay','arguments':{'bill_id':'b3','source':'bank'}}
  self.assertIn('bill_violates_request_filter',issues(row,prior,wrong,{'ok':True}))
 def test_train_only_zero_annotations(self):
  data=json.loads((H/'runtime/zero_action_train_labels.json').read_text())['labels']
  train=set((H/'runtime/train_ids.txt').read_text().splitlines());valid=set((H/'runtime/split.txt').read_text().splitlines())
  self.assertEqual(len(data),100);self.assertTrue(set(data)<=train);self.assertFalse(set(data)&valid)
  self.assertEqual(data['train_0327']['next'],'summarizer')
 def test_loss_and_gradients_equivalent(self):
  class Toy(torch.nn.Module):
   def __init__(self):super().__init__();self.emb=torch.nn.Embedding(17,7);self.head=torch.nn.Linear(7,17,bias=False)
   def get_output_embeddings(self):return self.head
   def forward(self,input_ids,**kwargs):return type('Output',(),{'logits':self.emb(input_ids)})()
  torch.manual_seed(3);model=Toy();reference=copy.deepcopy(model)
  batch={'input_ids':torch.tensor([[1,2,3,4,5],[2,3,4,5,0]]),'labels':torch.tensor([[-100,-100,3,4,5],[-100,3,4,5,-100]])}
  logits=reference.head(reference.emb(batch['input_ids']))[:,:-1]
  expected=torch.nn.functional.cross_entropy(logits.float().reshape(-1,17),batch['labels'][:,1:].reshape(-1),ignore_index=-100)
  expected.backward()
  for flag in ('0','1'):
   model.zero_grad(set_to_none=True)
   with patch.dict(os.environ,{'LOSS_PROJECTION_CHECKPOINT':flag}):loss=completion_loss(model,batch,chunk_size=2);loss.backward()
   torch.testing.assert_close(loss,expected)
   for actual,wanted in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(actual.grad,wanted.grad)
 def test_validation_sorted_and_weighted(self):
  class Items:
   def __init__(self):self.items=[{'input_ids':[1]*n,'attention_mask':[1]*n,'labels':[-100]+[2]*(n-1)} for n in (12,3,9,2)]
   def __len__(self):return len(self.items)
   def __getitem__(self,i):return self.items[i]
  seen=[];collate=Collator(0)
  def collect(rows):seen.append(max(len(r['input_ids']) for r in rows));return collate(rows)
  def loss(model,batch):return torch.tensor(2.)
  with patch('umilife.ddp_engine.dist.get_rank',return_value=0),patch('umilife.ddp_engine.dist.get_world_size',return_value=1),patch('umilife.ddp_engine.dist.all_reduce',side_effect=lambda *a,**k:None):
   result=weighted_eval(torch.nn.Linear(1,1),Items(),collect,torch.device('cpu'),contextlib.nullcontext,2,loss)
  self.assertEqual(result,2.);self.assertEqual(seen,[3,12])
 def test_node_metrics_and_contracts(self):
  specs={'demo.write':{'kind':'write','description':'Save a value','parameters':{'type':'object','properties':{'value':{'type':'string'},'time':{'type':'string'}},'required':['value']}}}
  row={'context':{},'history':[]};plan={'thought':'Save the requested value at the requested time.','next':'caller','tool':'demo.write'}
  record={'messages':build_messages('caller',row,'request',specs,[],plan),'target':json.dumps({'name':'demo.write','arguments':{'value':'requested','time':'19:00'}})}
  good=assess('caller',record,record['target'],specs)
  missing=assess('caller',record,'{"name":"demo.write","arguments":{"value":"requested"}}',specs)
  self.assertTrue(good['correct']);self.assertTrue(missing['schema_valid']);self.assertFalse(missing['correct']);self.assertFalse(missing['field_matches']['time'])
  self.assertEqual(summarize([good,missing],'caller')['match_accuracy'],.5)
  record={'messages':build_messages('planner',row,'request',specs,[]),'target':json.dumps(plan)}
  self.assertTrue(assess('planner',record,json.dumps(plan),specs)['correct'])
  self.assertFalse(assess('planner',record,'{"thought":"x","next":"caller","tool":"other.write"}',{**specs,'other.write':specs['demo.write']})['schema_valid'])
  evaluated={**assess('planner',record,json.dumps(plan),specs),'step':0}
  self.assertEqual(summarize([evaluated],'planner')['tool_accuracy'],1.)
 def test_training_generation_and_state_restore(self):
  from types import SimpleNamespace
  class Tokenizer:
   pad_token_id=0;padding_side='right'
   def apply_chat_template(self,messages,**kwargs):
    assert all(m['role']!='assistant' for m in messages)
    return [1]*len(messages[0]['content'])
   def batch_decode(self,tokens,**kwargs):return ['{"name":"demo.write","arguments":{"value":"requested"}}']*len(tokens)
  class Model(torch.nn.Module):
   def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.config=SimpleNamespace(use_cache=False);self.generation_config=SimpleNamespace(eos_token_id=[1]);self.restored=False
   def gradient_checkpointing_enable(self,**kwargs):self.restored=True
   def generate(self,input_ids,attention_mask,**kwargs):
    assert os.environ['UNSLOTH_RETURN_HIDDEN_STATES']=='0' and os.environ['UNSLOTH_RETURN_LOGITS']=='1'
    assert not self.training and not torch.is_grad_enabled() and self.config.use_cache
    assert kwargs['synced_gpus'] is False and kwargs['logits_to_keep']==1 and not kwargs['do_sample']
    assert kwargs['max_new_tokens']==192
    assert torch.equal(attention_mask.sum(1),torch.tensor([2,3]))
    assert torch.equal(input_ids[0],torch.tensor([0,1,1]))
    torch.rand(3);return torch.cat([input_ids,torch.ones((len(input_ids),1),dtype=torch.long)],1)
  model=Model();tokenizer=Tokenizer();specs={'demo.write':{'parameters':{'type':'object','properties':{'value':{'type':'string'}},'required':['value']}}}
  records=[{'id':f'valid_{i}','step':0,'messages':[{'role':'system','content':'x'*n},{'role':'user','content':'{}'}],
            'target':'{"name":"demo.write","arguments":{"value":"requested"}}'} for i,n in enumerate((3,2))]
  def gather(out,rows):out[0]=rows
  with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'UNSLOTH_RETURN_HIDDEN_STATES':'1','UNSLOTH_RETURN_LOGITS':'0'}),patch('umilife.training_eval.read_jsonl',return_value=records),patch('umilife.training_eval.load_catalog',return_value=specs),patch('umilife.training_eval.dist.get_rank',return_value=0),patch('umilife.training_eval.dist.get_world_size',return_value=1),patch('umilife.training_eval.dist.all_gather_object',side_effect=gather):
   before=torch.get_rng_state().clone()
   evaluator=make_evaluator('caller','unused','unused',tokenizer,torch.device('cpu'),tmp,lambda m:m.eval(),lambda m:m.train())
   result=evaluator(model,100)
   self.assertEqual(result['metrics']['argument_f1'],1.);self.assertEqual(result['metrics']['samples'],2)
   self.assertEqual(OUTPUT_TOKEN_LIMITS,{'planner':256,'caller':192});self.assertEqual(result['max_new_tokens'],192)
   self.assertEqual(result['metrics']['length_limited_count'],0)
   self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(model.training and model.restored)
   self.assertFalse(model.config.use_cache);self.assertEqual(tokenizer.padding_side,'right')
   self.assertEqual(os.environ['UNSLOTH_RETURN_HIDDEN_STATES'],'1');self.assertEqual(os.environ['UNSLOTH_RETURN_LOGITS'],'0')
   self.assertTrue((Path(tmp)/'step-00000100/report.json').exists())
   with self.assertRaisesRegex(RuntimeError,'injected'):
    with generation_context(model,tokenizer,torch.device('cpu'),lambda m:m.eval(),lambda m:m.train()):raise RuntimeError('injected')
   self.assertTrue(model.training);self.assertFalse(model.config.use_cache);self.assertEqual(os.environ['UNSLOTH_RETURN_HIDDEN_STATES'],'1')
 def test_selected_export_keeps_training_artifact(self):
  from safetensors.torch import save_file,load_file
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);ce=root/'ce';ce.mkdir();last=root/'last';last.mkdir();selected=root/'selected'/'caller'
   save_file({'lora':torch.zeros(2,2)},ce/'adapter_model.safetensors')
   save_file({'lora':torch.ones(2,2)},last/'adapter_model.safetensors')
   (ce/'tokenizer_config.json').write_text('{}')
   export_adapter(ce,last/'adapter_model.safetensors',selected,{'final_adapter':{'sha256':'prior','finite':True}},{'recommended_checkpoint':'last'})
   self.assertTrue(torch.equal(load_file(str(ce/'adapter_model.safetensors'))['lora'],torch.zeros(2,2)))
   self.assertTrue(torch.equal(load_file(str(selected/'adapter_model.safetensors'))['lora'],torch.ones(2,2)))
   meta=json.loads((selected/'lineage.json').read_text())
   self.assertEqual(meta['final_adapter']['hash_kind'],'safetensors_file');self.assertEqual(meta['final_adapter']['parameters'],4)
   self.assertTrue((selected/'tokenizer_config.json').exists())
 def test_embedded_notebook(self):
  nb=json.loads((H/'kaggle_planner_caller_evidence.ipynb').read_text())
  for c in nb['cells']:
   if c['cell_type']=='code':ast.parse(c['source'])
  payload=ast.literal_eval(ast.parse(nb['cells'][2]['source']).body[0].value)
  self.assertEqual(payload,{str(p.relative_to(H/'runtime')):p.read_text() for p in (H/'runtime').rglob('*') if p.is_file() and '__pycache__' not in str(p)})
if __name__=='__main__':unittest.main()
