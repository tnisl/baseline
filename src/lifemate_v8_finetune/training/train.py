"""One independent LoRA per role: masked SFT plus optional preference-ranking loss.

This is reference-free pairwise ranking, NOT DPO. The base model stays frozen.
Only selected completion logits are materialized to avoid full prompt x vocabulary logits.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import json
import math
import os
import sys
import time

from ..common import write_json
from .data import prompt_hashes, stable_hash
from .encoding import PairCollator, RoleDataset
from .attention import configure_training_attention, attention_preflight, enable_chunked_sdpa_fallback


def select_visible_gpu(index,environ):
    """Select one ordinal from the inherited visible list before importing torch."""
    if index<0:raise ValueError('gpu-index must be nonnegative')
    inherited=environ.get('CUDA_VISIBLE_DEVICES')
    if inherited is None:return str(index)
    devices=[x.strip() for x in inherited.split(',') if x.strip()]
    if not devices or '-1' in devices:
        raise ValueError('CUDA_VISIBLE_DEVICES disables GPUs; enable a GPU session')
    if index>=len(devices):raise ValueError('gpu-index is outside the inherited CUDA_VISIBLE_DEVICES list')
    return devices[index]


def runtime_report(torch):
    """Capture versions before model kernels run; do not inspect CUDA after a fault."""
    packages={}
    for name in ('torch','transformers','peft','accelerate','bitsandbytes'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    cuda=torch.cuda.is_available()
    return {'python':sys.version.split()[0],'packages':packages,
            'torch_cuda':torch.version.cuda,'cuda_available':cuda,
            'cuda_device_count':torch.cuda.device_count() if cuda else 0,
            'gpu':torch.cuda.get_device_name() if cuda else None,
            'compute_capability':list(torch.cuda.get_device_capability()) if cuda else None,
            'torch_cuda_arch_list':torch.cuda.get_arch_list() if cuda else [],
            'cuda_launch_blocking':os.environ.get('CUDA_LAUNCH_BLOCKING'),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}


def adapter_state(model):
    """Hash actual LoRA weights without retaining another model on GPU or CPU."""
    import torch
    digest=hashlib.sha256()
    finite=True
    parameters=0
    for name,parameter in sorted(model.named_parameters()):
        if 'lora_' not in name:continue
        value=parameter.detach().cpu().contiguous()
        finite=finite and bool(torch.isfinite(value).all())
        parameters+=value.numel()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    if not parameters:raise ValueError('No LoRA weights to verify')
    return {'sha256':digest.hexdigest(),'finite':finite,'parameters':parameters}


def sequence_scores(model,batch):
    import torch
    import torch.nn.functional as F
    labels=batch['labels']
    active=labels[:,1:]!=-100
    positions=active.any(dim=0).nonzero(as_tuple=False).flatten()
    if not positions.numel():raise ValueError('No target prediction positions')
    outputs=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                  use_cache=False,logits_to_keep=positions)
    logits=outputs.logits
    if logits.shape[1]!=len(positions):
        raise ValueError('Model does not honor logits_to_keep; use the pinned Qwen3 stack')
    targets=labels[:,positions+1]
    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1),
                         ignore_index=-100,reduction='none').reshape_as(targets)
    count=(targets!=-100).sum(dim=1)
    if (count==0).any():raise ValueError('Example has no supervised tokens')
    return -loss.sum(dim=1)/count


def local_loss(model,inputs,*,negative_weight=.2,ranking_beta=1.,ranking_margin=.2):
    import torch.nn.functional as F
    chosen=sequence_scores(model,inputs['chosen'])
    sft=-chosen.mean()
    paired=inputs['paired']
    pair=sft.new_zeros(())
    accuracy=sft.new_zeros(())
    if negative_weight and paired.any():
        rejected_batch={k:v[paired] for k,v in inputs['rejected'].items()}
        rejected=sequence_scores(model,rejected_batch)
        gap=chosen[paired]-rejected
        pair=F.softplus(ranking_margin-ranking_beta*gap).mean()
        accuracy=(gap>0).float().mean()
    return sft+negative_weight*pair,{'sft_loss':sft.detach(),'ranking_loss':pair.detach(),
                                    'preference_accuracy':accuracy.detach()}


def trainer_class():
    from transformers import Trainer
    class LocalTrainer(Trainer):
        def __init__(self,*args,negative_weight=.2,ranking_beta=1.,ranking_margin=.2,
                     role_label='role',trace_batches=False,**kwargs):
            super().__init__(*args,**kwargs)
            self.negative_weight=negative_weight
            self.ranking_beta,self.ranking_margin=ranking_beta,ranking_margin
            # Loss is explicitly averaged per example; Trainer handles accumulation scaling.
            self.model_accepts_loss_kwargs=False
            self.latest_local_metrics={}
            self.eval_totals={}
            self.role_label,self.trace_batches=role_label,trace_batches
            self.train_microbatches=0

        def training_step(self,model,inputs,num_items_in_batch=None):
            self.train_microbatches+=1
            n=self.train_microbatches
            visible=n<=3 or (self.trace_batches and n%10==0)
            if visible:
                print(f'[{self.role_label}] microbatch={n} optimizer_step={self.state.global_step}'
                      f' forward/backward start chosen_shape={tuple(inputs["chosen"]["input_ids"].shape)}',flush=True)
            loss=super().training_step(model,inputs,num_items_in_batch)
            if visible:print(f'[{self.role_label}] microbatch={n} forward/backward returned',flush=True)
            return loss

        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            loss,stats=local_loss(model,inputs,negative_weight=self.negative_weight,
                ranking_beta=self.ranking_beta,ranking_margin=self.ranking_margin)
            self.latest_local_metrics={k:float(v) for k,v in stats.items()}
            if not model.training:
                n=inputs['chosen']['input_ids'].shape[0]
                pairs=int(inputs['paired'].sum()) if self.negative_weight else 0
                totals=self.eval_totals
                totals['examples']=totals.get('examples',0)+n
                totals['pairs']=totals.get('pairs',0)+pairs
                totals['sft']=totals.get('sft',0.)+float(stats['sft_loss'])*n
                totals['rank']=totals.get('rank',0.)+float(stats['ranking_loss'])*pairs
                totals['correct']=totals.get('correct',0.)+float(stats['preference_accuracy'])*pairs
            return (loss,{'loss':loss}) if return_outputs else loss

        def evaluate(self,*args,**kwargs):
            self.eval_totals={}
            metrics=super().evaluate(*args,**kwargs)
            t=self.eval_totals
            prefix=kwargs.get('metric_key_prefix','eval')
            metrics.update({prefix+'_sft_loss':t.get('sft',0.)/max(1,t.get('examples',0)),
                            prefix+'_ranking_loss':t.get('rank',0.)/max(1,t.get('pairs',0)),
                            prefix+'_preference_accuracy':t.get('correct',0.)/t['pairs'] if t.get('pairs') else None,
                            prefix+'_pairs':t.get('pairs',0)})
            return metrics

        def prediction_step(self,model,inputs,prediction_loss_only,ignore_keys=None):
            import torch
            # Inputs are nested chosen/rejected batches, not model.forward kwargs.
            inputs=self._prepare_inputs(inputs)
            with torch.no_grad(),self.compute_loss_context_manager():
                loss=self.compute_loss(model,inputs)
            return loss.detach(),None,None
    return LocalTrainer


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=('planner','caller','summarizer'),required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--model',default='Qwen/Qwen3-4B-Instruct-2507')
    p.add_argument('--max-seq-length',type=int,default=8192)
    p.add_argument('--epochs',type=float,default=1);p.add_argument('--max-steps',type=int,default=-1)
    p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--gradient-accumulation',type=int,default=16)
    p.add_argument('--lora-r',type=int,default=16);p.add_argument('--lora-alpha',type=int,default=32)
    p.add_argument('--lora-dropout',type=float,default=.05)
    p.add_argument('--full-precision-base',action='store_true',help='Disable 4-bit base; still train LoRA only')
    p.add_argument('--negative-weight',type=float,default=.2)
    p.add_argument('--ranking-beta',type=float,default=1.);p.add_argument('--ranking-margin',type=float,default=.2)
    p.add_argument('--save-steps',type=int,default=100);p.add_argument('--logging-steps',type=int,default=10)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume-from-checkpoint')
    p.add_argument('--gpu-index',type=int,default=0,
                   help='Use one GPU, selected by ordinal within inherited CUDA_VISIBLE_DEVICES')
    p.add_argument('--overwrite-adapter',action='store_true')
    p.add_argument('--cuda-debug',action='store_true',
                   help='Set CUDA_LAUNCH_BLOCKING=1 before importing torch; diagnostic only, slower')
    return p.parse_args()


def main():
    args=parse_args()
    inherited_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES']=select_visible_gpu(args.gpu_index,os.environ)
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Only one GPU process per role is supported; do not launch with torchrun/DDP')
    if args.cuda_debug:os.environ['CUDA_LAUNCH_BLOCKING']='1'
    if min(args.max_seq_length,args.batch_size,args.gradient_accumulation,args.lora_r,args.lora_alpha,
           args.save_steps,args.logging_steps)<=0 or args.epochs<=0 or args.learning_rate<=0:
        raise ValueError('Training sizes/rates must be positive')
    if args.max_steps==0 or args.max_steps< -1 or args.negative_weight<0 or args.ranking_beta<=0:
        raise ValueError('Invalid max_steps or ranking parameters')
    if (not 0<=args.lora_dropout<1 or args.ranking_margin<0
            or not all(math.isfinite(x) for x in (args.epochs,args.learning_rate,args.lora_dropout,
                args.negative_weight,args.ranking_beta,args.ranking_margin))):
        raise ValueError('Training rates/margins must be finite, and dropout must be in [0, 1)')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,TrainingArguments,set_seed
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    if not torch.cuda.is_available():raise RuntimeError('Enable a Kaggle GPU before training Qwen3-4B')
    if torch.cuda.device_count()!=1:raise RuntimeError('Training requires exactly one visible GPU per process')
    attention_policy=configure_training_attention(torch)
    runtime=runtime_report(torch)
    runtime.update(role=args.role,load_in_4bit=not args.full_precision_base,
                   requested_gpu_index=args.gpu_index,inherited_cuda_visible_devices=inherited_visible_devices,
                   attention_implementation='sdpa',cuda_debug=args.cuda_debug)
    runtime['attention_policy']=attention_policy
    write_json(Path(args.output_dir)/'training_runs'/f'{args.role}.runtime.json',runtime)
    print(json.dumps({'training_runtime':runtime},ensure_ascii=False,indent=2),flush=True)
    print(f'[{args.role}] Checking CUDA attention forward/backward before model load...',flush=True)
    try:
        runtime['attention_preflight']=attention_preflight(torch)
    except RuntimeError as exc:
        # Only backend-selection failures are recoverable. Never continue after a
        # CUDA illegal address/device assertion or silently swallow an OOM.
        unsupported=any(marker in str(exc).lower() for marker in
                        ('invalid backend','no available kernel','no viable backend','no viable kernel'))
        if attention_policy['policy']!='explicit_kv_fused_efficient_sdpa' or not unsupported:raise
        print(f'[{args.role}] Fused SDPA unavailable; using exact checkpointed query tiles.',flush=True)
        runtime['attention_policy']=enable_chunked_sdpa_fallback(torch)
        runtime['attention_policy']['fused_preflight_error']=str(exc)
        runtime['attention_preflight']=attention_preflight(torch)
    write_json(Path(args.output_dir)/'training_runs'/f'{args.role}.runtime.json',runtime)
    print(f'[{args.role}] CUDA attention preflight passed.',flush=True)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size()!=1:
        raise ValueError('This notebook trains one role per single-GPU process; distributed training is not configured')
    set_seed(args.seed)
    data=Path(args.data_dir);out=Path(args.output_dir)
    adapter=out/'adapters'/args.role
    if (adapter/'adapter_config.json').exists() and not args.overwrite_adapter and not args.resume_from_checkpoint:
        raise FileExistsError(f'{adapter} exists; choose resume or --overwrite-adapter')
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['prompt_sha256']!=prompt_hashes():raise ValueError('Dataset prompt hashes differ from inference prompts; rebuild data')
    print(f'[{args.role}] Checking train/validation data checksums...',flush=True)
    for split in ('train','validation'):
        name=f'{args.role}.{split}.jsonl'
        actual=hashlib.sha256((data/name).read_bytes()).hexdigest()
        if manifest.get('files_sha256',{}).get(name)!=actual:
            raise ValueError('Dataset file checksum mismatch; rebuild data: '+name)
    print(f'[{args.role}] Loading tokenizer: {args.model}',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    print(f'[{args.role}] Tokenizer loaded. Preparing datasets on CPU before loading GPU model.',flush=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    negative_weight=0. if args.role=='planner' else args.negative_weight
    train=RoleDataset(data/f'{args.role}.train.jsonl',tokenizer,args.max_seq_length,
                      use_negatives=negative_weight>0,seed=args.seed,progress=True,
                      total=manifest['counts'][f'{args.role}.train']['examples'])
    valid=RoleDataset(data/f'{args.role}.validation.jsonl',tokenizer,args.max_seq_length,
                      use_negatives=negative_weight>0,seed=args.seed,progress=True,
                      total=manifest['counts'][f'{args.role}.validation']['examples'])
    stats={'train':train.stats,'validation':valid.stats}
    print(json.dumps({'role':args.role,'tokenization':stats},ensure_ascii=False,indent=2),flush=True)
    bf16=torch.cuda.is_bf16_supported()
    dtype=torch.bfloat16 if bf16 else torch.float16
    model_kwargs={'torch_dtype':dtype,'device_map':{'':torch.cuda.current_device()},'attn_implementation':'sdpa'}
    if not args.full_precision_base:
        model_kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    # Every CLI role starts from args.model, never from another role's adapter.
    print(f'[{args.role}] Loading base model on GPU (4-bit={not args.full_precision_base})...',flush=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,**model_kwargs)
    print(f'[{args.role}] Base model loaded. Preparing LoRA...',flush=True)
    model.config.use_cache=False
    if not args.full_precision_base:
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={'use_reentrant':False})
    else:
        model.enable_input_require_grads()
    model=get_peft_model(model,LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        bias='none',task_type='CAUSAL_LM'))
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name in trainable):
        raise RuntimeError('Only LoRA weights may be trainable')
    model.print_trainable_parameters()
    config=vars(args)|{'global_finetune':False,'starting_checkpoint':args.model,
        'role':args.role,'negative_objective':'SFT + reference-free pairwise ranking (not DPO)',
        'effective_negative_weight':negative_weight,'dtype':str(dtype),'load_in_4bit':not args.full_precision_base,
        'gpu':torch.cuda.get_device_name(0),'prompt_sha256':prompt_hashes(),
        'dataset_manifest_sha256':stable_hash(manifest),'tokenization':stats,
        'runtime':runtime,
        'summarizer_loss_scope':'next only; answer text masked',
        'trainable_parameter_names':trainable}
    write_json(out/'training_runs'/f'{args.role}.config.json',config)
    training_args=TrainingArguments(output_dir=str(out/'checkpoints'/args.role),
        num_train_epochs=args.epochs,max_steps=args.max_steps,learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},warmup_ratio=.03,weight_decay=.01,
        lr_scheduler_type='cosine',optim='adamw_torch',fp16=not bf16,bf16=bf16,
        logging_steps=args.logging_steps,logging_first_step=True,eval_strategy='steps',eval_steps=args.save_steps,
        save_strategy='steps',save_steps=args.save_steps,save_total_limit=2,
        prediction_loss_only=True,remove_unused_columns=False,label_names=[],
        report_to='none',seed=args.seed,data_seed=args.seed,dataloader_num_workers=0)
    trainer=trainer_class()(model=model,args=training_args,train_dataset=train,eval_dataset=valid,
        data_collator=PairCollator(tokenizer.pad_token_id),processing_class=tokenizer,
        role_label=args.role,trace_batches=args.cuda_debug,
        negative_weight=negative_weight,ranking_beta=args.ranking_beta,ranking_margin=args.ranking_margin)
    started=time.perf_counter()
    initial_adapter=adapter_state(model)
    print(f'[{args.role}] Starting trainer.train()...',flush=True)
    result=trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    evaluation=trainer.evaluate()
    final_adapter=adapter_state(model)
    verification={'initial_sha256':initial_adapter['sha256'],'final_sha256':final_adapter['sha256'],
                  'updated':initial_adapter['sha256']!=final_adapter['sha256'],
                  'finite':final_adapter['finite'],'parameters':final_adapter['parameters']}
    if not verification['finite'] or not verification['updated']:
        raise RuntimeError('LoRA update verification failed: '+json.dumps(verification)+
                           '. Use at least two optimizer steps when warmup starts at learning rate zero.')
    trainer.save_model(str(adapter));tokenizer.save_pretrained(adapter)
    # Runtime checks this file before routing an adapter to a node.
    write_json(adapter/'role_metadata.json',{'role':args.role,'base_model':args.model,
        'prompt_sha256':prompt_hashes()[args.role],'global_finetune':False,
        'dataset_manifest_sha256':stable_hash(manifest),'lora_r':args.lora_r,
        'loss_scope':'next' if args.role=='summarizer' else 'completion'})
    metrics={'train':result.metrics,'validation':evaluation,'last_batch':trainer.latest_local_metrics,
             'adapter_verification':verification,
             'wall_s':time.perf_counter()-started,'peak_cuda_memory_gb':torch.cuda.max_memory_allocated()/2**30}
    write_json(adapter/'training_metrics.json',metrics)
    print(json.dumps({'adapter':str(adapter),'metrics':metrics},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
