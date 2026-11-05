"""Greedy reference-prefix node metrics on the two resident training models."""
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from umilife.core import read_jsonl,load_catalog,write_json
from umilife.node_eval import assess,summarize


def generation_context(model,tokenizer,device,for_inference,for_training):
    # Unsloth's training forward returns hidden states for our CE. Generation needs logits.
    from contextlib import contextmanager
    @contextmanager
    def context():
        keys=('UNSLOTH_RETURN_HIDDEN_STATES','UNSLOTH_RETURN_LOGITS')
        saved={k:os.environ.get(k) for k in keys}
        padding=tokenizer.padding_side;cache=model.config.use_cache
        python_rng=random.getstate();numpy_rng=np.random.get_state()
        devices=[device.index] if device.type=='cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                os.environ.update(UNSLOTH_RETURN_HIDDEN_STATES='0',UNSLOTH_RETURN_LOGITS='1')
                for_inference(model);tokenizer.padding_side='left';model.config.use_cache=True
                with torch.inference_mode():yield
        finally:
            for k,v in saved.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
            for_training(model)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.config.use_cache=cache;tokenizer.padding_side=padding
            random.setstate(python_rng);np.random.set_state(numpy_rng)
            if device.type=='cuda':torch.cuda.empty_cache()
    return context()


def make_evaluator(role,prepared,data_root,tokenizer,device,output,for_inference,for_training,batch_size=4):
    rank=dist.get_rank();world=dist.get_world_size()
    records=read_jsonl(Path(prepared)/f'{role}.validation.jsonl')
    catalog=load_catalog(data_root,'train')
    # Encode messages only: the reference answer never enters generation.
    local=[(r,tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True))
           for r in records[rank::world]]
    local.sort(key=lambda pair:len(pair[1]))
    assert all(len(ids)+512<=8192 for _,ids in local),'Context exceeded; no evidence truncation'
    def evaluate(model,step):
        rows=[];folder=Path(output)/f'step-{step:08d}';folder.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        with generation_context(model,tokenizer,device,for_inference,for_training):
            with (folder/f'gpu{rank}.jsonl').open('w') as stream:
                for start in range(0,len(local),batch_size):
                    group=local[start:start+batch_size];width=max(len(ids) for _,ids in group)
                    batch={'input_ids':torch.tensor([[tokenizer.pad_token_id]*(width-len(ids))+ids for _,ids in group],device=device),
                           'attention_mask':torch.tensor([[0]*(width-len(ids))+[1]*len(ids) for _,ids in group],device=device)}
                    generated=model.generate(**batch,max_new_tokens=512,do_sample=False,num_beams=1,
                        use_cache=True,logits_to_keep=1,pad_token_id=tokenizer.pad_token_id,
                        synced_gpus=False,return_dict_in_generate=False)
                    texts=tokenizer.batch_decode(generated[:,width:].cpu(),skip_special_tokens=True)
                    for (record,_),raw in zip(group,texts,strict=True):
                        row={'id':record['id'],'step':record['step'],'raw':raw,**assess(role,record,raw,catalog)}
                        rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    stream.flush()
                    if start==0 or (start//batch_size+1)%10==0 or start+batch_size>=len(local):
                        print(json.dumps({'event':'training_node_eval_progress','role':role,'train_step':step,'rank':rank,
                            'completed':min(start+batch_size,len(local)),'total':len(local)}),flush=True)
        gathered=[None]*world;dist.all_gather_object(gathered,rows)
        all_rows=[r for shard in gathered for r in shard]
        assert len(all_rows)==len(records)
        report={'role':role,'backend':'resident_unsloth_transformers','greedy':True,'max_new_tokens':512,
                'batch_per_gpu':batch_size,'metrics':summarize(all_rows,role),
                'note':'Reference-prefix node agreement, not full workflow score; backend differs from post-stage vLLM.'}
        if rank==0:
            write_json(folder/'report.json',{'train_step':step,**report})
            print(json.dumps({'event':'training_node_eval_end','train_step':step,'seconds':time.monotonic()-started,**report}),flush=True)
        return report
    return evaluate
