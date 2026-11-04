import argparse,json,os,time,contextlib,hashlib
from pathlib import Path
from umilife.comparison_common import assess,token_hash,fingerprint

def main():
 p=argparse.ArgumentParser()
 for name in ('backend','work','adapters','model','revision'):p.add_argument('--'+name,required=True)
 p.add_argument('--graph',action='store_true');p.add_argument('--max-tokens',type=int,default=320)
 a=p.parse_args();work=Path(a.work);requests=json.loads((work/'requests.json').read_text());refs={r['probe']:r for r in json.loads((work/'references.json').read_text())}
 import torch,transformers
 torch.manual_seed(42);torch.cuda.set_device(0)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(Path(a.adapters)/'planner'),local_files_only=True)
 prompts=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True) for r in requests]
 tokens=[tok.encode(s,add_special_tokens=False) for s in prompts]
 assert all(len(ids)+a.max_tokens<=8192 for ids in tokens),'No evidence may be truncated'
 gpu=os.environ.get('CUDA_VISIBLE_DEVICES');path=work/(a.backend+('_graph' if a.graph else '')+'.jsonl')
 def save(index,mode,raw,output_tokens,elapsed,finish):
  result={'probe':requests[index]['probe'],'mode':mode,'backend':a.backend,'cuda_graphs':a.graph,
    'physical_gpu':gpu,'precision':'float16','seed':42,'batch_size':1,'max_new_tokens':a.max_tokens,
    'prompt_sha256':hashlib.sha256(prompts[index].encode()).hexdigest(),'input_ids_sha256':token_hash(tokens[index]),
    'schema_sha256':fingerprint(requests[index]['schema']),'input_tokens':len(tokens[index]),
    'output_tokens':output_tokens,'seconds':elapsed,'finish_reason':finish,'raw':raw,
    **assess(raw,requests[index]['schema'],refs[requests[index]['probe']])}
  with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k in ('probe','mode','seconds','thought_words','schema_valid','repeats_6gram_3plus')}),flush=True)
 if path.exists():raise RuntimeError('Fresh output required; use a new run directory')
 versions={'torch':torch.__version__,'transformers':transformers.__version__,'model':a.model,'revision':a.revision,'gpu_name':torch.cuda.get_device_name(0),'tokenizer_path':str(Path(a.adapters)/'planner')}
 if a.backend=='transformers':
  from transformers import AutoModelForCausalLM,GenerationConfig
  from peft import PeftModel
  from lmformatenforcer import JsonSchemaParser
  from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
  model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.float16,attn_implementation='sdpa',device_map={'':0},trust_remote_code=False)
  model=PeftModel.from_pretrained(model,str(Path(a.adapters)/'planner'),is_trainable=False,autocast_adapter_dtype=False).to(dtype=torch.float16).eval()
  first={'input_ids':torch.tensor([tokens[0]],device='cuda'),'attention_mask':torch.ones((1,len(tokens[0])),dtype=torch.long,device='cuda')}
  with torch.inference_mode():
   base=None
   with model.disable_adapter():base=model(**first,use_cache=False).logits[:,-1].float().cpu()
   adapted=model(**first,use_cache=False).logits[:,-1].float().cpu()
  delta=float((base-adapted).abs().max());assert delta>0,'Adapter has no measured effect'
  versions.update(attention='sdpa',adapter_dtype='float16',adapter_logit_delta_max=delta)
  for mode in ('base_unguided','adapter_unguided','adapter_guided'):
   for i,r in enumerate(requests):
    guided=mode.endswith('_guided');kwargs={}
    if guided:kwargs['prefix_allowed_tokens_fn']=build_transformers_prefix_allowed_tokens_fn(tok,JsonSchemaParser(r['schema']))
    input_ids=torch.tensor([tokens[i]],device='cuda');mask=torch.ones_like(input_ids)
    cfg=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=a.max_tokens,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id or tok.eos_token_id,bos_token_id=tok.bos_token_id,repetition_penalty=1.0)
    torch.cuda.synchronize();start=time.monotonic()
    with torch.inference_mode(),(model.disable_adapter() if mode.startswith('base') else contextlib.nullcontext()):
     output=model.generate(input_ids=input_ids,attention_mask=mask,generation_config=cfg,**kwargs)
    torch.cuda.synchronize();elapsed=time.monotonic()-start;new=output[0,len(tokens[i]):].tolist()
    save(i,mode,tok.decode(new,skip_special_tokens=True),len(new),elapsed,'eos' if new and new[-1]==tok.eos_token_id else 'length')
 else:
  from umilife.infer_2gpu import Backend
  from vllm import SamplingParams
  from vllm.sampling_params import GuidedDecodingParams
  class ControlledBackend(Backend):
   def make_sampling(self,request):
    return SamplingParams(temperature=0,top_p=1,top_k=-1,repetition_penalty=1,max_tokens=a.max_tokens,seed=42,
       guided_decoding=GuidedDecodingParams(json=request['schema'],backend='lm-format-enforcer') if request['guided'] else None)
  os.environ['COMPARE_CUDA_GRAPHS']='1' if a.graph else '0'
  backend=ControlledBackend(a.model,a.revision,a.adapters)
  # Verify the exact tokenizer IDs used by the actual engine, not just external AutoTokenizer.
  for r,prompt,ids in zip(requests,prompts,tokens):
   assert backend.tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)==prompt
   assert backend.tokenizer.encode(prompt,add_special_tokens=False)==ids
  adapter=backend.requests['planner'];versions.update(attention='xformers',vllm=__import__('vllm').__version__,prefix_caching=backend.runtime_options['prefix_caching'])
  modes=('adapter_guided',) if a.graph else ('base_unguided','adapter_unguided','adapter_guided')
  for mode in modes:
   backend.requests['planner']=None if mode.startswith('base') else adapter
   for i,r in enumerate(requests):
    request={'role':'planner','messages':r['messages'],'schema':r['schema'],'guided':mode.endswith('_guided')}
    start=time.monotonic();response=backend.generate([request])[0];elapsed=time.monotonic()-start
    save(i,mode,response['raw'],len(backend.tokenizer.encode(response['raw'],add_special_tokens=False)),elapsed,response['finish_reason'])
 (work/(a.backend+('_graph' if a.graph else '')+'_environment.json')).write_text(json.dumps(versions,indent=2))
if __name__=='__main__':main()
