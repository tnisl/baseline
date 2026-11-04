"""Two independent GPU workers; disjoint episodes, batch4/GPU, one combined score."""
import argparse,json,os,shutil,subprocess,sys,time
from collections import Counter
from pathlib import Path
from umilife.core import read_jsonl,write_json,load_catalog,import_toolenv
from umilife.evaluate import evaluate

def merge(root,work,ids):
    predictions=[];traces=[];reports=[]
    for rank in range(2):
        part=work/f'gpu{rank}'
        predictions.extend(read_jsonl(part/'submission.jsonl'));traces.extend(read_jsonl(part/'traces.jsonl'))
        reports.append(json.loads((part/'report.json').read_text()))
    assert len(predictions)==len(ids)==len(set(x['id'] for x in predictions))
    assert set(x['id'] for x in predictions)==set(ids)
    assert len(traces)==len(ids)==len(set(x['id'] for x in traces))
    lookup={x['id']:x for x in predictions};trace_lookup={x['id']:x for x in traces}
    predictions=[lookup[i] for i in ids];traces=[trace_lookup[i] for i in ids]
    for name,rows in [('submission',predictions),('traces',traces)]:
        target=work/f'{name}.jsonl';temp=target.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));temp.replace(target)
    catalog=load_catalog(root,'train');ToolEnv=import_toolenv(root,'train');references=[]
    for row in read_jsonl(root/'train/train.jsonl'):
        if row['id'] not in lookup:continue
        env=ToolEnv('train',row['id'],root=root)
        for action in row['label']['tool_calls']:env.call(action['name'],action['arguments'])
        references.append({'id':row['id'],'tool_calls':row['label']['tool_calls'],'state_hash':env.get_state_hash()})
    report={'samples':len(ids),'metrics':evaluate(predictions,references,catalog,traces),
            'asr':json.loads((work/'asr_report.json').read_text()),'workers':reports,
            'termination_reasons':dict(Counter(t['reason'] for t in traces)),
            'runtime':{'independent_gpus':2,'batch_per_gpu':4,'total_concurrent_workflows':8,
                       'cuda_graphs':True,'prefix_caching':True,'max_revisions_per_role_per_decision':1}}
    write_json(work/'report.json',report);return report

def main():
    p=argparse.ArgumentParser()
    for name in ('data-root','work','model','revision','adapters','ids-file'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-monotonic',type=float);p.add_argument('--mode',default='multi_local');p.add_argument('--split',default='validation')
    a=p.parse_args();assert a.mode=='multi_local' and a.split=='validation'
    import torch
    assert torch.cuda.device_count()==2,'Select GPU T4 x2; no silent single-GPU fallback'
    work=Path(a.work);root=Path(a.data_root);ids=Path(a.ids_file).read_text().splitlines()
    assert len(ids)==245 and len(set(ids))==245
    processes=[];logs=[];counts=[None,None];ready=set()
    try:
        for rank in range(2):
            part=work/f'gpu{rank}';part.mkdir(exist_ok=True)
            (part/'split.txt').write_text('\n'.join(ids[rank::2])+'\n')
            for name in ('asr_transcripts.json','asr_report.json'):shutil.copyfile(work/name,part/name)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(rank),LIFEMATE_WORKER_GPU=str(rank),
                TRITON_CACHE_DIR=f'/tmp/lifemate_eval_triton_{rank}',VLLM_CACHE_ROOT=f'/tmp/lifemate_eval_vllm_{rank}',OMP_NUM_THREADS='2')
            command=[sys.executable,'-u','-m','umilife.infer_2gpu','--data-root',str(root),'--work',str(part),
                '--model',a.model,'--revision',a.revision,'--adapters',a.adapters,
                '--ids-file',str(part/'split.txt'),'--split','validation','--mode','multi_local']
            if a.deadline_monotonic:command+=['--deadline-monotonic',str(a.deadline_monotonic)]
            log=(part/'worker.log').open('a',buffering=1);logs.append(log)
            processes.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
            print(json.dumps({'event':'worker_start','gpu':rank,'samples':len(ids[rank::2]),'cuda_graphs':True,'prefix_caching':True}),flush=True)
        while any(p.poll() is None for p in processes):
            for rank,process in enumerate(processes):
                if process.poll() not in (None,0,75):
                    print((work/f'gpu{rank}'/'worker.log').read_text()[-10000:],flush=True)
                    raise RuntimeError(f'GPU worker {rank} failed: {process.returncode}; inspect worker.log')
                if rank not in ready and (work/f'gpu{rank}'/'preflight.json').exists():
                    ready.add(rank);print(json.dumps({'event':'worker_ready','gpu':rank}),flush=True)
                progress=work/f'gpu{rank}'/'progress.json'
                if progress.exists():
                    x=json.loads(progress.read_text());done=x['completed']
                    if done!=counts[rank]:
                        counts[rank]=done;print(json.dumps({'event':'progress','gpu':rank,**x}),flush=True)
            time.sleep(2)
        if any(p.returncode==75 for p in processes):raise SystemExit(75)
        for rank,p in enumerate(processes):
            assert p.returncode==0,f'GPU worker {rank} failed; inspect worker.log'
        report=merge(root,work,ids)
        print(json.dumps({k:v for k,v in report['metrics'].items() if k!='per_sample'},indent=2),flush=True)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()
if __name__=='__main__':main()
