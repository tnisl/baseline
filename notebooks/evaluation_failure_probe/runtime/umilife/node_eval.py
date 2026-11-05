"""Post-stage vLLM generation metrics for CE-best and last local checkpoints."""
import argparse,json,os,sys,subprocess,time,re,shutil
from pathlib import Path
from collections import Counter
from jsonschema import Draft202012Validator,ValidationError
from umilife.core import read_jsonl,write_json,parse_json,load_catalog,digest as file_digest
from umilife.prompt_policy import request_schema,digest
from umilife.evaluate import matches

def assess(role,record,raw,catalog):
    target=parse_json(record['target']);messages=json.loads(record['messages'][1]['content'])
    if role=='planner':
        names=re.findall(r'^\s*"([^"\n]+)"\(args:',messages['tools_ts'],re.M)
        specs={name:catalog[name] for name in names}
        schema=request_schema('planner',specs)
    else:schema=request_schema('caller',{target['name']:catalog[target['name']]},{'tool':target['name']})
    value=None;valid=False;error=None
    try:
        value=parse_json(raw);Draft202012Validator(schema).validate(value)
        if role=='planner' and value.get('next')=='caller' and (not value.get('tool') or value['tool'] not in catalog):raise ValueError('Missing/unknown selected tool')
        valid=True
    except (ValueError,ValidationError) as exc:error=str(exc)[:300]
    result={'schema_valid':valid,'schema_error':error,'parsed':value,'target':target}
    if role=='planner':
        route=bool(valid and value['next']==target['next']);tool=bool(valid and value.get('tool')==target.get('tool')) if target['next']=='caller' else None
        result.update(correct=bool(route and (target['next']!='caller' or tool)),route_match=route,tool_match=tool,terminal=target['next']!='caller')
    else:
        correct=bool(valid and matches(value,target,catalog));fields={}
        pred=value.get('arguments',{}) if valid else {}
        for k,v in target['arguments'].items():
            fields[k]=bool(valid and k in pred and matches({'name':value['name'],'arguments':{k:pred[k]}},{'name':target['name'],'arguments':{k:v}},catalog))
        count=sum(fields.values());denom=len(pred)+len(target['arguments'])
        result.update(correct=correct,field_matches=fields,argument_f1=2*count/denom if denom else float(valid and value['name']==target['name']))
    return result

def summarize(rows,role):
    n=len(rows);assert n
    report={'samples':n,'match_accuracy':sum(r['correct'] for r in rows)/n,'schema_valid_rate':sum(r['schema_valid'] for r in rows)/n}
    if role=='planner':
        report['route_accuracy']=sum(r['route_match'] for r in rows)/n
        calls=[r for r in rows if not r['terminal']]
        report['tool_accuracy']=sum(r['tool_match'] for r in calls)/len(calls) if calls else None
        report['tool_samples']=len(calls)
        for kind in ('initial','intermediate','terminal'):
            sub=[r for r in rows if ('terminal' if r['terminal'] else 'initial' if r['step']==0 else 'intermediate')==kind]
            report[kind]={'n':len(sub),'accuracy':sum(r['correct'] for r in sub)/len(sub) if sub else None}
        report['continued_after_reference_end']=sum(r['terminal'] and isinstance(r['parsed'],dict) and r['parsed'].get('next')=='caller' for r in rows)
    else:
        fields={}
        for row in rows:
            for k,v in row['field_matches'].items():fields.setdefault(k,{'n':0,'correct':0});fields[k]['n']+=1;fields[k]['correct']+=v
        for value in fields.values():value['accuracy']=value['correct']/value['n']
        report.update(argument_f1=sum(r['argument_f1'] for r in rows)/n,fields=fields)
    return report

def worker(a):
    from umilife.t4_compat import prepare_t4_lora_kernel
    print(json.dumps(prepare_t4_lora_kernel()),flush=True)
    import torch
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from umilife.runtime_options import engine_options
    work=Path(a.output);config=json.loads((work/'config.json').read_text());options=engine_options(torch.cuda.get_device_capability(0))
    engine=LLM(model=a.model,revision=a.revision,tokenizer=config['tokenizer'],dtype='float16',tensor_parallel_size=1,
        enable_lora=True,max_loras=2,max_cpu_loras=2,max_lora_rank=16,max_model_len=8192,max_num_seqs=4,max_num_batched_tokens=8192,
        gpu_memory_utilization=.85,enforce_eager=not options['cuda_graphs'],enable_prefix_caching=options['prefix_caching'],
        enable_chunked_prefill=False,seed=42,generation_config='vllm',disable_log_stats=True)
    tokenizer=engine.get_tokenizer();records=read_jsonl(Path(a.prepared)/f'{a.stage}.validation.jsonl')[a.rank::2]
    catalog=load_catalog(a.data_root,'train');sampling=SamplingParams(temperature=0,seed=42,max_tokens=512,guided_decoding=None)
    candidates={name:LoRARequest(name,i+1,path) for i,(name,path) in enumerate(config['candidates'].items())}
    target=work/f'gpu{a.rank}.jsonl'
    with target.open('w') as out:
        for start in range(0,len(records),4):
            group=records[start:start+4]
            prompts=[tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True) for r in group]
            token_ids=[tokenizer.encode(p,add_special_tokens=False) for p in prompts]
            assert all(len(ids)+512<=8192 for ids in token_ids),'Context exceeded; no evidence truncation'
            order=list(candidates)
            if start//4%2:order.reverse()
            for name in order:
                tick=time.monotonic();results=engine.generate(prompts,sampling,lora_request=candidates[name],use_tqdm=False)
                for record,ids,result in zip(group,token_ids,results,strict=True):
                    row={'id':record['id'],'step':record['step'],'candidate':name,'prompt_sha256':digest(record['messages']),
                        'input_ids_sha256':digest(ids),'raw':result.outputs[0].text,'finish_reason':result.outputs[0].finish_reason,
                        **assess(a.stage,record,result.outputs[0].text,catalog)}
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps({'stage':a.stage,'rank':a.rank,'candidate':name,'completed':min(start+4,len(records)),
                                  'total':len(records),'batch_seconds':time.monotonic()-tick}),flush=True)

def export_adapter(adapter,weights,export,lineage,report):
    # Export a self-contained inference adapter without overwriting the CE-best training artifact.
    temp=export.with_name(export.name+'.tmp')
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir(parents=True)
    for path in adapter.iterdir():
        if path.is_file() and path.name not in ('adapter_model.safetensors','lineage.json'):shutil.copy2(path,temp/path.name)
    shutil.copy2(weights,temp/weights.name)
    import torch
    from safetensors import safe_open
    parameters=0
    with safe_open(weights,framework='pt',device='cpu') as handle:
        for key in handle.keys():
            value=handle.get_tensor(key);assert torch.isfinite(value).all(),key;parameters+=value.numel()
    selected_lineage={**lineage,'selection':report,
        'ce_best_final_adapter':lineage['final_adapter'],
        'final_adapter':{'sha256':file_digest(weights),'hash_kind':'safetensors_file','finite':True,'parameters':parameters}}
    write_json(temp/'lineage.json',selected_lineage)
    if export.exists():shutil.rmtree(export)
    temp.rename(export)
    return export

def main():
    p=argparse.ArgumentParser()
    for name in ('stage','prepared','adapter','checkpoints','data-root','model','revision','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--rank',type=int)
    a=p.parse_args()
    if a.rank is not None:worker(a);return
    work=Path(a.output);work.mkdir(parents=True,exist_ok=True);adapter=Path(a.adapter);cp=Path(a.checkpoints)
    latest=cp/json.loads((cp/'latest.json').read_text())['folder']
    lineage=json.loads((adapter/'lineage.json').read_text())
    assert lineage['stage']==a.stage and lineage['model']==a.model and lineage['revision']==a.revision
    assert lineage['dataset_manifest_sha256']==file_digest(Path(a.prepared)/'manifest.json')
    candidates={'ce_best':str(adapter),'last':str(latest)}
    if file_digest(adapter/'adapter_model.safetensors')==file_digest(latest/'adapter_model.safetensors'):candidates.pop('last')
    config={'stage':a.stage,'model':a.model,'revision':a.revision,'tokenizer':str(adapter),'candidates':candidates,
        'weights_sha256':{k:file_digest(Path(v)/'adapter_model.safetensors') for k,v in candidates.items()},
        'manifest_sha256':file_digest(Path(a.prepared)/'manifest.json'),'prompt_sha256':json.loads((Path(a.prepared)/'manifest.json').read_text())['prompts_sha256']}
    if (work/'report.json').exists():
        assert json.loads((work/'config.json').read_text())==config,'Stale node eval output'
        print((work/'report.json').read_text());return
    write_json(work/'config.json',config)
    processes=[];logs=[];last_seen=[None,None]
    try:
        for rank in range(2):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),TRITON_CACHE_DIR=f'/tmp/node_eval_triton_{rank}')
            log=(work/f'gpu{rank}.log').open('w');logs.append(log)
            command=[sys.executable,'-u','-m','umilife.node_eval',*sys.argv[1:],'--rank',str(rank)]
            processes.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Node generation worker failed')
            for rank,log in enumerate(logs):
                log.flush()
                for line in reversed((work/f'gpu{rank}.log').read_text().splitlines()[-12:]):
                    try:event=json.loads(line)
                    except ValueError:continue
                    if isinstance(event,dict) and 'completed' in event:
                        if line!=last_seen[rank]:print(json.dumps({'node_eval_progress':event}),flush=True);last_seen[rank]=line
                        break
            time.sleep(5)
        assert all(p.returncode==0 for p in processes),'Node generation worker failed'
    except BaseException:
        for process in processes:
            if process.poll() is None:process.terminate()
        for rank in range(len(logs)):
            logs[rank].flush();print((work/f'gpu{rank}.log').read_text()[-6000:],flush=True)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs:log.close()
    rows=[r for rank in range(2) for r in read_jsonl(work/f'gpu{rank}.jsonl')]
    expected={(r['id'],r['step']) for r in read_jsonl(Path(a.prepared)/f'{a.stage}.validation.jsonl')}
    for name in candidates:
        selected=[r for r in rows if r['candidate']==name]
        assert len(selected)==len(expected) and {(r['id'],r['step']) for r in selected}==expected
    for key in expected:
        pair=[r for r in rows if (r['id'],r['step'])==key]
        assert len({(r['prompt_sha256'],r['input_ids_sha256']) for r in pair})==1
    metrics={name:summarize([r for r in rows if r['candidate']==name],a.stage) for name in candidates}
    winner=max(metrics,key=lambda k:(metrics[k]['match_accuracy'],metrics[k]['schema_valid_rate'],k=='ce_best'))
    report={'stage':a.stage,'metrics':metrics,'recommended_checkpoint':winner,'recommended_path':candidates[winner],
            'guided_decoding':False,'batch_per_gpu':4,'gpu_workers':2,
            'note':'Reference-prefix node agreement, not full workflow/contest score. CE-best and last only; no claim of globally best checkpoint.'}
    export=work.parent.parent/'selected_adapters'/a.stage
    export_adapter(adapter,Path(candidates[winner])/'adapter_model.safetensors',export,lineage,report)
    report['exported_adapter']=str(export)
    write_json(work/'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
