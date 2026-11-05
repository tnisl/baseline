"""Small controlled generation probes; no optimizer, no ASR or tool execution."""
import argparse,json,os,time,hashlib,re
from pathlib import Path
from collections import Counter
from umilife.core import read_jsonl,write_json
from umilife.node_eval import assess
MODEL='Qwen/Qwen3-4B-Instruct-2507'
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
CASES={'nf4_lora_fp32':('nf4','float32'),'nf4_lora_fp16':('nf4','float16'),
       'fp16_lora_fp32':('fp16','float32'),'fp16_lora_fp16':('fp16','float16')}
BACKENDS=tuple(CASES)
EOS_IDS=[151645,151643]
HF_GENERATION={'max_new_tokens':512,'do_sample':False,'num_beams':1,'use_cache':True,'logits_to_keep':1,
               'pad_token_id':151643,'eos_token_id':EOS_IDS,'synced_gpus':False,'return_dict_in_generate':False}

def model_settings(model):
    quant=getattr(model.config,'quantization_config',None)
    # Unsloth can expose a raw config dict with torch.dtype values instead of a BNB config object.
    quant=json.loads(json.dumps(quant.to_dict() if hasattr(quant,'to_dict') else quant,default=str))
    return {'quantization_config':quant,
            'lora_parameter_dtypes':dict(Counter(str(p.dtype) for n,p in model.named_parameters() if 'lora_' in n)),
            'attention_implementation':getattr(model.config,'_attn_implementation',None)}

def fingerprint(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
def repetition(raw):
    words=re.findall(r'\w+',raw.lower());counts=Counter(tuple(words[i:i+6]) for i in range(len(words)-5))
    return max(counts.values(),default=0)
def trim_generated(ids,eos_ids):
    for i,token in enumerate(ids):
        if token in eos_ids:return ids[:i+1],'eos'
    return ids,'length'
def groups(records,tokenizer,batch_size):
    pairs=[(r,tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)) for r in records]
    pairs.sort(key=lambda x:len(x[1]))
    assert all(len(ids)+512<=8192 for _,ids in pairs)
    return [pairs[i:i+batch_size] for i in range(0,len(pairs),batch_size)]
def save_rows(stream,group,tokenizer,outputs,catalog,backend,variant,batch_size,seconds,batch_index):
    for (record,prompt_ids),(ids,finish,stop) in zip(group,outputs,strict=True):
        raw=tokenizer.decode(ids,skip_special_tokens=True)
        row={'id':record['id'],'step':record['step'],'backend':backend,'variant':variant,'batch_size':batch_size,
             'batch_index':batch_index,'batch_seconds':seconds,'input_tokens':len(prompt_ids),'input_ids_sha256':fingerprint(prompt_ids),
             'generated_tokens':len(ids),'output_token_ids':ids,'finish_reason':finish,'stop_reason':stop,
             'raw':raw,'max_repeated_6gram':repetition(raw),**assess('planner',record,raw,catalog)}
        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
        print(json.dumps({k:row[k] for k in ('id','step','backend','variant','batch_size','batch_index','batch_seconds','generated_tokens','finish_reason','schema_valid','correct','max_repeated_6gram')}),flush=True)

def verify_lora_dtype(model,dtype):
    params=[p for n,p in model.named_parameters() if 'lora_' in n]
    assert params and all(p.dtype==dtype for p in params),'LoRA dtype changed or no adapter was loaded'

def cast_lora_dtype(model,dtype):
    # Cast only adapter parameters, never call model.to(dtype) on a quantized base.
    for name,param in model.named_parameters():
        if 'lora_' in name:
            param.data=param.data.to(dtype=dtype);param.requires_grad_(False)
    verify_lora_dtype(model,dtype)

def restore_source_lora(model,weights,dtype):
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    import torch
    result=set_peft_model_state_dict(model,weights)
    assert not result.unexpected_keys and not [k for k in result.missing_keys if 'lora_' in k]
    actual=get_peft_model_state_dict(model)
    assert set(actual)==set(weights),'Adapter keys differ from the saved source'
    # Reload after casting: FP16 base loading may already have rounded adapter values.
    for name,value in actual.items():
        assert torch.equal(value.detach().cpu(),weights[name].to(dtype=dtype)),name
    verify_lora_dtype(model,dtype)

def plain_hf(a,records,catalog,stream):
    import sys,torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    from peft import PeftModel
    from peft.utils.save_and_load import load_peft_weights
    assert not any(n=='unsloth' or n.startswith(('unsloth.','unsloth_zoo')) for n in sys.modules),'Plain Transformers must not import Unsloth patches'
    set_seed(42)
    base_precision,lora_precision=CASES[a.backend]
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16) if base_precision=='nf4' else None
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float16,
        device_map={'':0},quantization_config=config,attn_implementation='sdpa',trust_remote_code=False)
    model=PeftModel.from_pretrained(model,str(Path(a.input)/'adapter_step75'),is_trainable=False,autocast_adapter_dtype=False)
    before=model_settings(model)['lora_parameter_dtypes']
    requested=getattr(torch,lora_precision)
    cast_lora_dtype(model,requested)
    restore_source_lora(model,load_peft_weights(str(Path(a.input)/'adapter_step75'),device='cpu'),requested)
    model.eval();tokenizer.padding_side='left';model.config.use_cache=True
    assert not any(n=='unsloth' or n.startswith(('unsloth.','unsloth_zoo')) for n in sys.modules)
    write_json(Path(a.output)/f'{a.backend}_{a.variant}_settings.json',{'effective_generation':HF_GENERATION,
        **model_settings(model),'base_precision':base_precision,'requested_lora_dtype':str(requested),
        'lora_dtypes_before_explicit_cast':before,'lora_source_weights_verified':True,'unsloth_imported':False,'adapter_step':75})
    for batch_size in (1,4):
        for batch_index,group in enumerate(groups(records,tokenizer,batch_size)):
            width=max(len(ids) for _,ids in group)
            batch={'input_ids':torch.tensor([[151643]*(width-len(ids))+ids for _,ids in group],device='cuda'),
                   'attention_mask':torch.tensor([[0]*(width-len(ids))+[1]*len(ids) for _,ids in group],device='cuda')}
            torch.cuda.synchronize();started=time.monotonic()
            with torch.inference_mode():out=model.generate(**batch,**HF_GENERATION)
            torch.cuda.synchronize();seconds=time.monotonic()-started
            verify_lora_dtype(model,requested)
            outputs=[(*trim_generated(ids,set(EOS_IDS)),None) for ids in out[:,width:].cpu().tolist()]
            save_rows(stream,group,tokenizer,outputs,catalog,a.backend,a.variant,batch_size,seconds,batch_index)

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=BACKENDS,required=True);p.add_argument('--variant',choices=('step75',),required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    records=read_jsonl(Path(a.input)/'records.jsonl');catalog=json.loads((Path(a.input)/'catalog.json').read_text())
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    with (output/f'{a.backend}_{a.variant}.jsonl').open('w') as stream:
        plain_hf(a,records,catalog,stream)
if __name__=='__main__':main()
