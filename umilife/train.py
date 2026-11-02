"""QLoRA GLPFT: global ReAct SFT, then three independently continued copies."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from .core import read_jsonl,write_json,digest,dumps,SYSTEMS,ROLES
from .models import MODEL_ID,TARGET_MODULES

def encode(tokenizer, record):
    prefix=tokenizer.apply_chat_template(record["messages"],tokenize=True,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(record["messages"]+[{"role":"assistant","content":record["target"]}],
                                       tokenize=True,add_generation_prompt=False)
    if full[:len(prefix)] != prefix: raise ValueError("Chat template changed assistant prefix")
    if len(full)<=len(prefix): raise ValueError("No assistant tokens")
    return {"input_ids":full,"attention_mask":[1]*len(full),"labels":[-100]*len(prefix)+full[len(prefix):]}

class Dataset:
    def __init__(self,path,tokenizer,max_length=8192,limit=None):
        self.items=[];self.stats={"loaded":0,"used":0,"overlength":0,"max_tokens":0}
        for record in read_jsonl(path):
            self.stats["loaded"]+=1
            x=encode(tokenizer,record);self.stats["max_tokens"]=max(self.stats["max_tokens"],len(x["input_ids"]))
            if len(x["input_ids"])>max_length:self.stats["overlength"]+=1;continue
            self.items.append(x);self.stats["used"]+=1
            if limit and len(self.items)>=limit:break
        if not self.items:raise ValueError("No usable examples: "+str(path))
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]

class Collator:
    def __init__(self,pad):self.pad=pad
    def __call__(self,items):
        import torch
        n=max(len(x["input_ids"]) for x in items)
        return {k:torch.tensor([x[k]+[v]*(n-len(x[k])) for x in items],dtype=torch.long)
                for k,v in (("input_ids",self.pad),("attention_mask",0),("labels",-100))}

def adapter_state(model):
    import torch
    h=hashlib.sha256();n=0;finite=True
    for name,parameter in sorted(model.named_parameters()):
        if "lora_" not in name:continue
        value=parameter.detach().cpu().contiguous()
        h.update(name.encode());h.update(value.view(torch.uint8).numpy().tobytes())
        n+=value.numel();finite &= bool(torch.isfinite(value).all())
    if not n:raise ValueError("No trainable adapter")
    return {"sha256":h.hexdigest(),"parameters":n,"finite":finite}

def completion_loss(model,batch):
    # Qwen's logits_to_keep avoids allocating prompt_length * 151k vocabulary.
    import torch.nn.functional as F
    labels=batch["labels"];positions=(labels[:,1:]!=-100).any(dim=0).nonzero().flatten()
    if not positions.numel():raise ValueError("Empty completion mask")
    output=model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],
                 use_cache=False,logits_to_keep=positions)
    if output.logits.shape[1]!=len(positions):raise ValueError("logits_to_keep unsupported")
    targets=labels[:,positions+1]
    return F.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),
                           targets.reshape(-1),ignore_index=-100)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("global",*ROLES),required=True)
    p.add_argument("--data-dir",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--global-adapter");p.add_argument("--model",default=MODEL_ID);p.add_argument("--revision")
    p.add_argument("--epochs",type=float);p.add_argument("--learning-rate",type=float)
    p.add_argument("--max-steps",type=int,default=-1);p.add_argument("--gradient-accumulation",type=int,default=8)
    p.add_argument("--max-length",type=int,default=8192);p.add_argument("--lora-r",type=int,default=16)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--eval-limit",type=int,default=64)
    p.add_argument("--resume-from-checkpoint")
    a=p.parse_args()
    if a.stage!="global" and not a.global_adapter:raise ValueError("Local training MUST initialize from global adapter")
    if a.stage=="global" and a.global_adapter:raise ValueError("Global stage starts at base")
    if not 256<=a.max_length<=16384:raise ValueError("Invalid context length")
    # Must be configured before CUDA is imported.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES","0");os.environ["TOKENIZERS_PARALLELISM"]="false"
    from .models import load_base,seed_all
    seed_all(a.seed)
    import torch
    from transformers import Trainer,TrainingArguments
    from peft import LoraConfig,get_peft_model,PeftModel,prepare_model_for_kbit_training
    data=Path(a.data_dir);out=Path(a.output_dir);target=out/"adapters"/a.stage
    if target.exists():raise FileExistsError("Refusing to overwrite completed stage: "+str(target))
    manifest=json.loads((data/"manifest.json").read_text())
    prompt_hash=hashlib.sha256(dumps(SYSTEMS).encode()).hexdigest()
    if manifest["prompts_sha256"]!=prompt_hash:raise ValueError("Training/inference prompts differ")
    for part in ("train","validation"):
        file=data/f"{a.stage}.{part}.jsonl"
        if digest(file)!=manifest["files_sha256"][file.name]:raise ValueError("Dataset checksum mismatch")
    tokenizer,model=load_base(a.model,four_bit=True,training=True,revision=a.revision)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={"use_reentrant":False})
    parent=None
    if a.stage=="global":
        model=get_peft_model(model,LoraConfig(r=a.lora_r,lora_alpha=2*a.lora_r,lora_dropout=.05,
            target_modules=TARGET_MODULES,bias="none",task_type="CAUSAL_LM"))
    else:
        parent=Path(a.global_adapter)
        lineage=json.loads((parent/"lineage.json").read_text())
        if lineage["stage"]!="global":raise ValueError("Parent must be global")
        if lineage["dataset_manifest_sha256"]!=digest(data/"manifest.json"):raise ValueError("Instruction reuse violated")
        if lineage["model"]!=a.model or lineage["revision"]!=a.revision:raise ValueError("Base revision mismatch")
        model=PeftModel.from_pretrained(model,parent,is_trainable=True)
    if any("lora_" not in n for n,v in model.named_parameters() if v.requires_grad):
        raise ValueError("Only adapter parameters may be trainable")
    initial=adapter_state(model)
    if parent and initial["sha256"]!=lineage["final_adapter"]["sha256"]:
        raise ValueError("Local initialization differs from global weights")
    train=Dataset(data/f"{a.stage}.train.jsonl",tokenizer,a.max_length)
    valid=Dataset(data/f"{a.stage}.validation.jsonl",tokenizer,a.max_length,a.eval_limit)
    bf16=torch.cuda.is_bf16_supported()
    class CompletionTrainer(Trainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.model_accepts_loss_kwargs=False
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            loss=completion_loss(model,inputs)
            return (loss,{"loss":loss}) if return_outputs else loss
        def prediction_step(self,model,inputs,prediction_loss_only,ignore_keys=None):
            with torch.no_grad(),self.compute_loss_context_manager():
                loss=self.compute_loss(model,self._prepare_inputs(inputs))
            return loss.detach(),None,None
    epochs=a.epochs if a.epochs is not None else (2 if a.stage in ("global","summarizer") else 1)
    lr=a.learning_rate if a.learning_rate else (5e-5 if a.stage=="global" else 1e-5)
    args=TrainingArguments(output_dir=str(out/"checkpoints"/a.stage),num_train_epochs=epochs,
        max_steps=a.max_steps,learning_rate=lr,per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.gradient_accumulation,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},fp16=not bf16,bf16=bf16,
        warmup_ratio=.03,weight_decay=.01,lr_scheduler_type="cosine",optim="adamw_torch",
        eval_strategy="no",save_strategy="steps",save_steps=100,save_total_limit=1,
        logging_steps=5,logging_first_step=True,prediction_loss_only=True,
        remove_unused_columns=False,report_to="none",seed=a.seed,data_seed=a.seed,
        dataloader_num_workers=0,group_by_length=True)
    trainer=CompletionTrainer(model=model,args=args,train_dataset=train,eval_dataset=valid,
                               data_collator=Collator(tokenizer.pad_token_id),processing_class=tokenizer)
    before=trainer.evaluate(metric_key_prefix="before")
    started=time.monotonic();result=trainer.train(resume_from_checkpoint=a.resume_from_checkpoint)
    after=trainer.evaluate(metric_key_prefix="after")
    final=adapter_state(model)
    if not final["finite"] or final["sha256"]==initial["sha256"]:raise RuntimeError("Adapter did not update finitely")
    trainer.save_model(str(target));tokenizer.save_pretrained(target)
    record={"stage":a.stage,"model":a.model,"revision":a.revision,"method":"QLoRA GLPFT",
        "parent":str(parent) if parent else None,"global_adapter_sha256":digest(parent/"adapter_model.safetensors") if parent else None,
        "initial_adapter":initial,"final_adapter":final,"dataset_manifest_sha256":digest(data/"manifest.json"),
        "prompt_sha256":prompt_hash,"epochs":epochs,"max_steps":a.max_steps,"learning_rate":lr,
        "gradient_accumulation":a.gradient_accumulation,"tokenization":{"train":train.stats,"validation":valid.stats},
        "before":before,"after":after,"train":result.metrics,"wall_s":time.monotonic()-started,
        "peak_cuda_gb":torch.cuda.max_memory_allocated()/2**30,
        "attention_policy":model.attention_policy_report,
        "versions":{n:importlib.metadata.version(n) for n in ("torch","transformers","peft","bitsandbytes","accelerate")}}
    write_json(target/"lineage.json",record)
    print(dumps(record),flush=True)
if __name__=="__main__":main()
