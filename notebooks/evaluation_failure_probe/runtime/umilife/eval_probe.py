"""Small controlled generation probes; no optimizer, no ASR or tool execution."""
import argparse,json,os,time,hashlib,re
from pathlib import Path
from collections import Counter
from umilife.core import read_jsonl,write_json
from umilife.node_eval import assess
MODEL='Qwen/Qwen3-4B-Instruct-2507'
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
BACKENDS=('unsloth','transformers_nf4','transformers_fp16','vllm')
EOS_IDS=[151645,151643]
HF_GENERATION={'max_new_tokens':512,'do_sample':False,'num_beams':1,'use_cache':True,'logits_to_keep':1,
               'pad_token_id':151643,'eos_token_id':EOS_IDS,'synced_gpus':False,'return_dict_in_generate':False}

def model_settings(model):
    quant=getattr(model.config,'quantization_config',None)
    return {'quantization_config':quant.to_dict() if hasattr(quant,'to_dict') else quant,
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

def hf(a,records,catalog,stream):
    os.environ.update(UNSLOTH_RETURN_HIDDEN_STATES='1',UNSLOTH_RETURN_LOGITS='0')
    import torch
    from unsloth import FastLanguageModel
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from umilife.training_eval import generation_context
    from umilife.completion_loss import completion_loss
    from umilife.train import encode,Collator
    from umilife.models import TARGET_MODULES,seed_all
    seed_all(42)
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=MODEL,revision=REVISION,use_exact_model_name=True,
        max_seq_length=8192,dtype=torch.float16,load_in_4bit=True,fast_inference=False,device_map={'':0},trust_remote_code=False)
    model=FastLanguageModel.get_peft_model(model,r=16,lora_alpha=32,lora_dropout=0,target_modules=TARGET_MODULES,
        bias='none',use_gradient_checkpointing=True,random_state=42)
    if a.variant=='step75':
        result=set_peft_model_state_dict(model,load_peft_weights(str(Path(a.input)/'adapter_step75'),device='cpu'))
        assert not result.unexpected_keys
        assert not [k for k in result.missing_keys if 'lora_' in k]
    else:assert all(torch.count_nonzero(p)==0 for n,p in model.named_parameters() if 'lora_B' in n),'Base requires zero LoRA contribution'
    FastLanguageModel.for_training(model);model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    probe={k:v.to('cuda') for k,v in Collator(tokenizer.pad_token_id)([encode(tokenizer,records[0])]).items()}
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):initial_loss=completion_loss(model,probe)
    assert torch.isfinite(initial_loss);model.train()
    print(json.dumps({'event':'pre_generation_ce_probe','loss':float(initial_loss),'optimizer_updates':0}),flush=True)
    generation=model.generation_config.to_dict();eos=generation.get('eos_token_id')
    eos_ids=set(EOS_IDS)
    write_json(Path(a.output)/f'{a.backend}_{a.variant}_settings.json',{'quantization':'NF4','generation_config_defaults':generation,'effective_generation':HF_GENERATION,**model_settings(model),'compile':os.environ.get('UNSLOTH_COMPILE_DISABLE'),'adapter_step':75 if a.variant=='step75' else None})
    for batch_size in (1,4):
        for batch_index,group in enumerate(groups(records,tokenizer,batch_size)):
            width=max(len(ids) for _,ids in group)
            batch={'input_ids':torch.tensor([[tokenizer.pad_token_id]*(width-len(ids))+ids for _,ids in group],device='cuda'),
                   'attention_mask':torch.tensor([[0]*(width-len(ids))+[1]*len(ids) for _,ids in group],device='cuda')}
            torch.cuda.synchronize();started=time.monotonic()
            with generation_context(model,tokenizer,torch.device('cuda',0),FastLanguageModel.for_inference,FastLanguageModel.for_training):
                out=model.generate(**batch,**HF_GENERATION)
                torch.cuda.synchronize();seconds=time.monotonic()-started
                outputs=[(*trim_generated(ids,eos_ids),None) for ids in out[:,width:].cpu().tolist()]
            save_rows(stream,group,tokenizer,outputs,catalog,a.backend,a.variant,batch_size,seconds,batch_index)
            # Exercise the original hidden-state CE path after switching back; no backward/update.
            probe={k:v.to('cuda') for k,v in Collator(tokenizer.pad_token_id)([encode(tokenizer,group[0][0])]).items()}
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):loss=completion_loss(model,probe)
            assert torch.isfinite(loss) and model.training and not model.config.use_cache
            assert os.environ['UNSLOTH_RETURN_HIDDEN_STATES']=='1' and os.environ['UNSLOTH_RETURN_LOGITS']=='0'
            print(json.dumps({'event':'restored_ce_probe','batch_size':batch_size,'batch_index':batch_index,'loss':float(loss),'restored':True}),flush=True)

def plain_hf(a,records,catalog,stream):
    import sys,torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    from peft import PeftModel
    assert not any(n=='unsloth' or n.startswith(('unsloth.','unsloth_zoo')) for n in sys.modules),'Plain Transformers must not import Unsloth patches'
    set_seed(42)
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16) if a.backend=='transformers_nf4' else None
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float16,
        device_map={'':0},quantization_config=config,attn_implementation='sdpa',trust_remote_code=False)
    if a.variant=='step75':model=PeftModel.from_pretrained(model,str(Path(a.input)/'adapter_step75'),is_trainable=False,autocast_adapter_dtype=False)
    model.eval();tokenizer.padding_side='left';model.config.use_cache=True
    assert not any(n=='unsloth' or n.startswith(('unsloth.','unsloth_zoo')) for n in sys.modules)
    write_json(Path(a.output)/f'{a.backend}_{a.variant}_settings.json',{'effective_generation':HF_GENERATION,
        **model_settings(model),'unsloth_imported':False,'adapter_step':75 if a.variant=='step75' else None})
    for batch_size in (1,4):
        for batch_index,group in enumerate(groups(records,tokenizer,batch_size)):
            width=max(len(ids) for _,ids in group)
            batch={'input_ids':torch.tensor([[151643]*(width-len(ids))+ids for _,ids in group],device='cuda'),
                   'attention_mask':torch.tensor([[0]*(width-len(ids))+[1]*len(ids) for _,ids in group],device='cuda')}
            torch.cuda.synchronize();started=time.monotonic()
            with torch.inference_mode():out=model.generate(**batch,**HF_GENERATION)
            torch.cuda.synchronize();seconds=time.monotonic()-started
            outputs=[(*trim_generated(ids,set(EOS_IDS)),None) for ids in out[:,width:].cpu().tolist()]
            save_rows(stream,group,tokenizer,outputs,catalog,a.backend,a.variant,batch_size,seconds,batch_index)

def vllm_probe(a,records,catalog,stream):
    import torch
    from umilife.t4_compat import prepare_t4_lora_kernel
    print(json.dumps(prepare_t4_lora_kernel()),flush=True)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from umilife.runtime_options import engine_options
    options=engine_options(torch.cuda.get_device_capability(0))
    engine=LLM(model=MODEL,revision=REVISION,dtype='float16',tensor_parallel_size=1,enable_lora=True,max_loras=1,
        max_lora_rank=16,max_model_len=8192,max_num_seqs=4,max_num_batched_tokens=8192,gpu_memory_utilization=.85,
        enable_prefix_caching=options['prefix_caching'],enforce_eager=not options['cuda_graphs'],enable_chunked_prefill=False,
        seed=42,generation_config='vllm',disable_log_stats=True)
    tokenizer=engine.get_tokenizer();sampling=SamplingParams(temperature=0,seed=42,max_tokens=512,stop_token_ids=EOS_IDS,guided_decoding=None)
    lora=LoRARequest('step75',1,str(Path(a.input)/'adapter_step75')) if a.variant=='step75' else None
    write_json(Path(a.output)/f'{a.backend}_{a.variant}_settings.json',{'quantization':'FP16 base','sampling':str(sampling),'runtime':options,'eos_token_id':tokenizer.eos_token_id})
    for batch_size in (1,4):
        for batch_index,group in enumerate(groups(records,tokenizer,batch_size)):
            started=time.monotonic()
            out=engine.generate([{'prompt_token_ids':ids} for _,ids in group],sampling,lora_request=lora,use_tqdm=False)
            seconds=time.monotonic()-started
            outputs=[(list(r.outputs[0].token_ids),r.outputs[0].finish_reason,r.outputs[0].stop_reason) for r in out]
            save_rows(stream,group,tokenizer,outputs,catalog,a.backend,a.variant,batch_size,seconds,batch_index)

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=BACKENDS,required=True);p.add_argument('--variant',choices=('base','step75'),required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    records=read_jsonl(Path(a.input)/'records.jsonl');catalog=json.loads((Path(a.input)/'catalog.json').read_text())
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    with (output/f'{a.backend}_{a.variant}.jsonl').open('w') as stream:
        (hf if a.backend=='unsloth' else vllm_probe if a.backend=='vllm' else plain_hf)(a,records,catalog,stream)
if __name__=='__main__':main()
