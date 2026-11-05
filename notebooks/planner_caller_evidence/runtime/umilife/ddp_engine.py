"""Explicit, symmetric DDP collectives; no Trainer monkey-patched control flow."""
import contextlib,hashlib,json,math,random,shutil,time,os,signal
from pathlib import Path
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def batches(lengths,epoch,rank,world,batch_size=4):
    # Shuffle then length-bucket across both GPUs to reduce padding/stragglers.
    indices=list(range(len(lengths)));random.Random(42+epoch).shuffle(indices)
    width=batch_size*world
    ordered=[]
    for start in range(0,len(indices),width*32):
        ordered.extend(sorted(indices[start:start+width*32],key=lambda i:lengths[i],reverse=True))
    padding=(-len(ordered))%width
    ordered.extend((ordered*math.ceil(padding/max(1,len(ordered))))[:padding])
    return [ordered[i+rank*batch_size:i+(rank+1)*batch_size] for i in range(0,len(ordered),width)]


def state_digest(model):
    h=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def agreement(value):
    values=[None]*dist.get_world_size();dist.all_gather_object(values,value)
    assert len(set(values))==1,values
    return value


def weighted_eval(model,items,collate,device,amp,batch_size=4,loss_function=None):
    # No DDP forward: rank workloads can differ by one validation example.
    model.eval();rank=dist.get_rank();world=dist.get_world_size()
    totals=torch.zeros(2,device=device,dtype=torch.float64)
    local=sorted(range(rank,len(items),world),key=lambda i:len(items[i]['input_ids']))
    with torch.inference_mode():
        for start in range(0,len(local),batch_size):
            x={k:v.to(device) for k,v in collate([items[i] for i in local[start:start+batch_size]]).items()}
            n=(x['labels'][:,1:]!=-100).sum()
            with amp():loss=loss_function(model,x) if loss_function else model(**x,use_cache=False,return_dict=True).loss
            totals[0]+=loss.double()*n;totals[1]+=n
    dist.all_reduce(totals)
    result=(totals[0]/totals[1]).item()
    assert math.isfinite(result) and totals[1]>0,'Nonfinite validation loss'
    model.train()
    return result


class LossModule(torch.nn.Module):
    # DDP static_graph must see the loss tensor directly even for custom ModelOutput objects.
    def __init__(self,model,loss_function=None):super().__init__();self.model=model;self.loss_function=loss_function
    def forward(self,**kwargs):
        if self.loss_function:return self.loss_function(self.model,{k:v for k,v in kwargs.items() if k not in ("use_cache","return_dict")})
        return self.model(**kwargs).loss


def fit(model,train,valid,collate,device,output,epochs,lr,save_model,load_model,
        eval_steps=50,patience=3,min_delta=.001,amp_dtype=None,train_batch_size=4,eval_batch_size=4,loss_function=None,max_steps=None,evaluation_callback=None):
    rank=dist.get_rank();world=dist.get_world_size();assert world==2
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    latest=output/'latest.json'
    assert not (output/'best').exists() or latest.exists(),'Incomplete legacy checkpoint cannot resume'
    params=[p for p in model.parameters() if p.requires_grad]
    wrapped=DistributedDataParallel(LossModule(model,loss_function),device_ids=[device.index] if device.type=='cuda' else None,
        broadcast_buffers=False,find_unused_parameters=False,static_graph=True)
    amp=(lambda:torch.autocast(device_type=device.type,dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and amp_dtype==torch.float16),init_scale=1024)
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
    lengths=[len(x['input_ids']) for x in train]
    steps_per_epoch=len(batches(lengths,0,rank,world,train_batch_size));total=steps_per_epoch*epochs
    if max_steps is not None:total=min(total,max_steps)
    warmup=max(1,math.ceil(.03*total))
    def factor(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    initial=agreement(state_digest(model))
    # Resume loads the saved baseline; throughput mode skips duplicate baseline evaluation.
    before=None
    if not latest.exists() and os.environ.get('INITIAL_EVAL','0')=='1':
        if rank==0:print(json.dumps({'event':'initial_validation_start','records':len(valid)}),flush=True)
        before=weighted_eval(model,valid,collate,device,amp,eval_batch_size,loss_function)
        if rank==0:print(json.dumps({'event':'initial_validation_end','eval_loss':before}),flush=True)
    elif rank==0:print(json.dumps({'event':'initial_validation_skipped','resume':latest.exists()}),flush=True)
    eval_seconds=0.;last_log=time.monotonic();last_log_step=0
    best=math.inf;significant=math.inf;bad=0;best_step=0;history=[]
    step=0;updates=0;skipped=0;stop=False;started=time.monotonic()
    last_epoch=0.;saved_hash=None
    resume_epoch=resume_batch=0
    train_seconds=checkpoint_seconds=0.;interval_seconds=0.;interval_input=interval_target=interval_padded=0
    interrupted=[False]
    signal.signal(signal.SIGTERM,lambda *_:interrupted.__setitem__(0,True))
    signature=os.environ.get('TRAIN_RESUME_SIGNATURE','')
    def persist(epoch,next_batch):
        nonlocal checkpoint_seconds
        checkpoint_started=time.monotonic()
        rng={'numpy':np.random.get_state(),'python':random.getstate(),'cpu':torch.get_rng_state(),
             'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
        rngs=[None]*world;dist.all_gather_object(rngs,rng)
        dist.barrier()
        if rank==0:
            folder=output/f'step-{step:08d}';temp=output/'saving.tmp'
            if temp.exists():shutil.rmtree(temp)
            temp.mkdir();save_model(model,temp)
            if (output/'best').exists():shutil.copytree(output/'best',temp/'best_snapshot')
            torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'scaler':scaler.state_dict(),'rng':rngs,'signature':signature,'world_size':world,
                'epoch':epoch,'next_batch':next_batch,'step':step,'updates':updates,'skipped':skipped,
                'best':best,'significant':significant,'bad':bad,'best_step':best_step,'history':history,
                'saved_hash':saved_hash,'initial':initial,'before':before,'last_epoch':last_epoch,
                'eval_seconds':eval_seconds,'total':total},temp/'training_state.pt')
            if folder.exists():shutil.rmtree(folder)
            temp.rename(folder)
            pointer=output/'latest.tmp';pointer.write_text(json.dumps({'folder':folder.name,'step':step}));pointer.replace(latest)
            for old in sorted(output.glob('step-*'))[:-2]:shutil.rmtree(old)
        dist.barrier()
        elapsed=time.monotonic()-checkpoint_started;checkpoint_seconds+=elapsed
        if rank==0:print(json.dumps({'event':'checkpoint_saved','step':step,'seconds':elapsed}),flush=True)
    if latest.exists():
        folder=output/json.loads(latest.read_text())['folder']
        state=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=False)
        assert state['signature']==signature and state['world_size']==world and state['total']==total,'Resume settings mismatch'
        if rank==0:
            if (output/'best').exists():shutil.rmtree(output/'best')
            if (folder/'best_snapshot').exists():shutil.copytree(folder/'best_snapshot',output/'best')
        dist.barrier()
        load_model(model,folder);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        step=state['step'];updates=state['updates'];skipped=state['skipped'];best=state['best'];significant=state['significant'];bad=state['bad'];best_step=state['best_step'];history=state['history'];saved_hash=state['saved_hash'];initial=state['initial'];before=state['before'];last_epoch=state['last_epoch'];eval_seconds=state['eval_seconds']
        resume_epoch=state['epoch'];resume_batch=state['next_batch'];stop=bad>=patience
        np.random.set_state(state['rng'][rank]['numpy']);random.setstate(state['rng'][rank]['python']);torch.set_rng_state(state['rng'][rank]['cpu'])
        if device.type=='cuda':torch.cuda.set_rng_state(state['rng'][rank]['cuda'],device)
        last_log_step=step
        print(json.dumps({'resumed_step':step,'stage_checkpoint':str(folder)}),flush=True)
    for epoch in range(epochs):
        if epoch<resume_epoch:continue

        local_batches=batches(lengths,epoch,rank,world,train_batch_size)
        for batch_index,indices in enumerate(local_batches):
            if epoch==resume_epoch and batch_index<resume_batch:continue
            if step>=total or bad>=patience:break
            step_started=time.monotonic()
            model.train();optimizer.zero_grad(set_to_none=True)
            x={k:v.to(device) for k,v in collate([train[i] for i in indices]).items()}
            n=(x['labels'][:,1:]!=-100).sum()
            global_n=n.clone();dist.all_reduce(global_n)
            with amp():
                raw_loss=wrapped(**x,use_cache=False,return_dict=True)
                # DDP averages rank gradients; recover true global token-mean gradient.
                loss=raw_loss*(world*n/global_n)
            finite=torch.isfinite(loss).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            assert finite.item()==1,'Nonfinite training loss on a rank'
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            overflow=scaler.get_scale()<old_scale
            flag=torch.tensor(int(overflow),device=device);dist.all_reduce(flag)
            assert flag.item() in (0,world),"AMP overflow decisions diverged"
            if overflow:skipped+=1
            else:scheduler.step();updates+=1
            step+=1;last_epoch=epoch+(batch_index+1)/len(local_batches)
            elapsed=time.monotonic()-step_started;train_seconds+=elapsed;interval_seconds+=elapsed
            interval_input+=int(x['attention_mask'].sum().item());interval_target+=int(n.item());interval_padded+=x['input_ids'].numel()
            if step%10==0 or step==1:
                count=torch.tensor([interval_input,interval_target,interval_padded],device=device,dtype=torch.float64);dist.all_reduce(count)
                duration=torch.tensor(interval_seconds,device=device,dtype=torch.float64);dist.all_reduce(duration,op=dist.ReduceOp.MAX)
                report={'step':step,'planned_steps':total,'epoch':last_epoch,'local_loss':raw_loss.item(),
                        'lr':optimizer.param_groups[0]['lr'],'grad_norm':float(norm),'optimizer_updates':updates,
                        'train_seconds_per_step':duration.item()/(step-last_log_step),
                        'input_tokens_per_second':count[0].item()/max(duration.item(),1e-9),
                        'assistant_tokens_per_second':count[1].item()/max(duration.item(),1e-9),
                        'padding_fraction':1-count[0].item()/max(count[2].item(),1),
                        'peak_allocated_gb':torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0,
                        'seconds_per_step_including_eval':(time.monotonic()-last_log)/(step-last_log_step)}
                last_log=time.monotonic();last_log_step=step
                interval_input=interval_target=interval_padded=0;interval_seconds=0.
                if rank==0:print(json.dumps(report),flush=True)
            at_end=step==total
            if step%eval_steps==0 or at_end:
                if rank==0:print(json.dumps({'event':'validation_start','step':step,'records':len(valid)}),flush=True)
                eval_started=time.monotonic()
                metric=weighted_eval(model,valid,collate,device,amp,eval_batch_size,loss_function)
                ce_seconds=time.monotonic()-eval_started
                node_report=None;generation_started=time.monotonic()
                if evaluation_callback is not None:
                    optimizer.zero_grad(set_to_none=True)
                    node_report=evaluation_callback(model,step)
                generation_seconds=time.monotonic()-generation_started if evaluation_callback is not None else 0.
                duration=time.monotonic()-eval_started;eval_seconds+=duration
                improved=metric<best
                if metric<significant-min_delta:significant=metric;bad=0
                else:bad+=1
                # Every rank computes the same scalar/state; only I/O is rank-zero-only.
                if improved:
                    best=metric;best_step=step
                    dist.barrier()
                    if rank==0:
                        temp=output/'best.tmp'
                        if temp.exists():shutil.rmtree(temp)
                        temp.mkdir();save_model(model,temp)
                        (temp/'checkpoint.json').write_text(json.dumps({'step':step,'eval_loss':metric,'epoch':last_epoch}))
                        if (output/'best').exists():shutil.rmtree(output/'best')
                        temp.rename(output/'best')
                    dist.barrier()
                    saved_hash=agreement(state_digest(model))
                event={'step':step,'epoch':last_epoch,'eval_loss':metric,'best_step':best_step,
                       'best_loss':best,'patience_used':bad,'improved':improved,'evaluation_seconds':duration,
                       'ce_evaluation_seconds':ce_seconds,'generation_evaluation_seconds':generation_seconds}
                if node_report is not None:event['node_eval']=node_report
                history.append(event)
                if rank==0:
                    (output/'eval_history.tmp').write_text(json.dumps(history,ensure_ascii=False,indent=2))
                    (output/'eval_history.tmp').replace(output/'eval_history.json')
                    print(json.dumps(event),flush=True)
                stop=bad>=patience
                agreement(json.dumps([step,best,best_step,bad,stop]))
            pause=torch.tensor(int(interrupted[0] or step==int(os.environ.get('SMOKE_PAUSE_STEP','-1')) or time.time()>=float(os.environ.get('TRAIN_DEADLINE','inf'))),device=device)
            dist.all_reduce(pause,op=dist.ReduceOp.MAX)
            if step%int(os.environ.get('SAVE_STEPS','25'))==0 or step%eval_steps==0 or at_end or stop or pause.item():
                persist(epoch,batch_index+1)
            if pause.item():raise SystemExit(75)
            if stop or at_end:break
        if stop or step>=total:break
    assert updates>0 and best_step>0
    dist.barrier();load_model(model,output/'best');dist.barrier()
    final=agreement(state_digest(model));assert final==saved_hash and final!=initial
    after=best
    if os.environ.get('VERIFY_RESTORED_EVAL','0')=='1':
        after=weighted_eval(model,valid,collate,device,amp,eval_batch_size,loss_function)
        assert abs(after-best)<max(.005,abs(best)*.005),'Best checkpoint restoration changed loss'
    result={'world_size':world,'micro_batch':train_batch_size,'eval_batch_per_device':eval_batch_size,'gradient_accumulation':1,
            'effective_batch':train_batch_size*world,'epochs':epochs,'actual_epoch':last_epoch,'global_step':step,
            'optimizer_updates':updates,'amp_skipped_updates':skipped,'planned_steps':total,
            'learning_rate':lr,'scheduler':'cosine','warmup_ratio':.03,
            'early_stopping':{'patience':patience,'min_delta':min_delta,'eval_steps':eval_steps,
                'restore_best':True,'best_step':best_step,'best_metric':best,'stopped_early':stop},
            'before':{'eval_loss':before},'after':{'eval_loss':after,'source':'rerun' if os.environ.get('VERIFY_RESTORED_EVAL','0')=='1' else 'saved_validation_of_hash_verified_best'},'history':history,
            'initial_trainable_sha256':initial,'final_trainable_sha256':final,
            'wall_s':time.monotonic()-started,'train_padding_examples_per_epoch':(-len(train))%(train_batch_size*world),
            'training_seconds_this_session':train_seconds,'checkpoint_seconds_this_session':checkpoint_seconds,'validation_repeats':0,'scheduled_evaluation_seconds':eval_seconds}
    dist.barrier();del wrapped
    return result
