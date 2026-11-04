"""Notebook runtime: two-rank Unsloth QLoRA, explicit DDP loop and best adapter."""
import os
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE','1')
os.environ.setdefault('UNSLOTH_RETURN_LOGITS','0')
os.environ['UNSLOTH_RETURN_HIDDEN_STATES']='1'
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

def main():
    import argparse,json,hashlib,importlib.metadata
    from pathlib import Path
    import torch
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    from unsloth import FastLanguageModel
    import torch.distributed as dist
    from datetime import timedelta
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from umilife.core import digest,write_json,dumps
    from umilife.models import TARGET_MODULES,seed_all
    from umilife.train import Dataset,Collator,adapter_state
    from umilife.prompt_policy import PROMPTS,POLICY_AUDIT
    from umilife.ddp_engine import fit,agreement
    from umilife.training_modes import validate_initialization,verify_parent
    from umilife.completion_loss import completion_loss
    p=argparse.ArgumentParser()
    for k in ('stage','data-dir','output-dir','model','revision'):p.add_argument('--'+k,required=True)
    p.add_argument('--train-batch-size',type=int,default=4);p.add_argument('--eval-batch-size',type=int,default=4)
    p.add_argument('--smoke',action='store_true');p.add_argument('--eval-steps',type=int,default=100)
    p.add_argument('--init-from',choices=('base','global'),required=True)
    p.add_argument('--global-adapter');p.add_argument('--epochs',type=int,required=True)
    a=p.parse_args()
    assert a.train_batch_size>0 and a.eval_batch_size>0
    assert int(os.environ['WORLD_SIZE'])==2 and torch.cuda.device_count()==2
    assert a.stage in ('global','planner','caller','summarizer')
    assert a.epochs==1 if a.stage=='global' else a.epochs>0
    validate_initialization(a.stage,a.init_from,a.global_adapter)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20),device_id=torch.device('cuda',local))
    seed_all(42)
    data=Path(a.data_dir);out=Path(a.output_dir);target=out/'adapters'/a.stage
    assert not (target/'lineage.json').exists(),'Stage already complete'
    if target.exists():
        import shutil;shutil.rmtree(target)
    manifest=json.loads((data/'manifest.json').read_text())
    prompt_sha=POLICY_AUDIT['static_prompts_sha256']
    assert manifest['train_episodes']==1300 and manifest['validation_episodes']==245
    assert manifest['prompts_sha256']==prompt_sha and manifest['rationale_authors']==['Codex']
    for part in ('train','validation'):
        f=data/f'{a.stage}.{part}.jsonl';assert digest(f)==manifest['files_sha256'][f.name]
    dtype=torch.bfloat16 if torch.cuda.get_device_capability(local)[0]>=8 else torch.float16
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=a.model,revision=a.revision,
        use_exact_model_name=True,max_seq_length=8192,dtype=dtype,load_in_4bit=True,
        fast_inference=False,device_map={'':local},trust_remote_code=False)
    model=FastLanguageModel.get_peft_model(model,r=16,lora_alpha=32,lora_dropout=0,
        target_modules=TARGET_MODULES,bias='none',use_gradient_checkpointing=True,random_state=42)
    parent=None
    def load_adapter(model,path):
        result=set_peft_model_state_dict(model,load_peft_weights(str(path),device='cpu'))
        assert not result.unexpected_keys,result.unexpected_keys
    if a.global_adapter:
        parent=Path(a.global_adapter)
        lineage=verify_parent(parent,a.model,a.revision,digest(data/'manifest.json'),prompt_sha,a.smoke)
        load_adapter(model,parent)
    initial=adapter_state(model)
    if parent:assert initial['sha256']==lineage['final_adapter']['sha256'],'Global initialization mismatch'
    assert all('lora_' in n for n,v in model.named_parameters() if v.requires_grad)
    FastLanguageModel.for_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    def cached_dataset(part):
        path=data/f'{a.stage}.{part}.jsonl'
        cache=data/f'{a.stage}.{part}.{digest(path)[:16]}.tokens.pt'
        if dist.get_rank()==0 and not cache.exists():
            encoded=Dataset(path,tokenizer,8192)
            temp=cache.with_suffix('.tmp');torch.save({'items':encoded.items,'stats':encoded.stats},temp);temp.replace(cache)
        dist.barrier()
        payload=torch.load(cache,map_location='cpu',weights_only=True)
        encoded=Dataset.__new__(Dataset);encoded.items=payload['items'];encoded.stats=payload['stats']
        return encoded
    train=cached_dataset('train');valid=cached_dataset('validation')
    assert train.stats['overlength']==valid.stats['overlength']==0,'No training examples may be silently dropped'
    if a.smoke:
        train.items=sorted(train.items,key=lambda x:len(x['input_ids']),reverse=True)[:16]
        valid.items=sorted(valid.items,key=lambda x:len(x['input_ids']),reverse=True)[:8]
        assert len(train.items)==16 and len(valid.items)==8
    os.environ['TRAIN_RESUME_SIGNATURE']=hashlib.sha256(json.dumps({'manifest':digest(data/'manifest.json'),'stage':a.stage,'epochs':a.epochs,'smoke':a.smoke,'batch':a.train_batch_size,'eval_batch':a.eval_batch_size,'eval_steps':a.eval_steps,'loss_chunk':os.environ.get('LIFEMATE_LOSS_CHUNK','256'),'model':a.model,'revision':a.revision,'init_from':a.init_from,'parent':digest(parent/'adapter_model.safetensors') if parent else None},sort_keys=True).encode()).hexdigest()
    def save_adapter(model,path):model.save_pretrained(path,safe_serialization=True)
    result=fit(model,train,valid,Collator(tokenizer.pad_token_id),torch.device('cuda',local),
        out/'checkpoints'/a.stage,a.epochs,5e-5 if a.stage=='global' else 1e-5,
        save_adapter,load_adapter,amp_dtype=dtype,train_batch_size=a.train_batch_size,
        eval_batch_size=a.eval_batch_size,loss_function=completion_loss,
        eval_steps=1 if a.smoke else a.eval_steps,max_steps=2 if a.smoke else None)
    final=adapter_state(model);assert final['finite'] and final['sha256']!=initial['sha256']
    agreement(final['sha256'])
    if dist.get_rank()==0:
        save_adapter(model,target);tokenizer.save_pretrained(target)
        history=result.pop('history');write_json(target/'training_history.json',history)
        record={**result,'stage':a.stage,'model':a.model,'revision':a.revision,
            'initialization':a.init_from,'method':('Unsloth QLoRA global' if a.stage=='global' else 'Unsloth QLoRA local-from-'+a.init_from),'loss_implementation':'assistant-only chunked CE, checkpointed projection','loss_chunk':int(os.environ.get('LIFEMATE_LOSS_CHUNK','256')),'smoke':a.smoke,'initial_adapter':initial,'final_adapter':final,
            'parent':str(parent) if parent else None,
            'global_adapter_sha256':digest(parent/'adapter_model.safetensors') if parent else None,
            'dataset_manifest_sha256':digest(data/'manifest.json'),'prompt_sha256':prompt_sha,
            'tokenization':{'train':train.stats,'validation':valid.stats},
            'actual_records':{'train':len(train),'validation':len(valid)},
            'peak_cuda_gb_rank0':torch.cuda.max_memory_allocated()/2**30,
            'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','unsloth','unsloth_zoo','peft','bitsandbytes')}}
        write_json(target/'lineage.json',record);print(dumps(record),flush=True)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
