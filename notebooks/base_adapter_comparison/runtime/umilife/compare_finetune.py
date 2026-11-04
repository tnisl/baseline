"""Paired unguided diagnosis: fixed-prefix node probes and independent rollouts."""
import argparse,json,time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from umilife.core import read_jsonl,write_json,parse_json,allowed_specs,load_catalog,import_toolenv
from umilife.data import normalized_row
from umilife.evaluate import matches,evaluate
from umilife.prompt_policy import build_messages,request_schema,digest
from umilife.infer_2gpu import Backend,rollout,run_batch

MODES=('base','adapter')
CALLER_THOUGHT='Carry out the requested operation using the request, context, and previous tool observations; respect all user constraints.'


def safe_row(raw,transcripts):
    context={k:deepcopy(raw['context'][k]) for k in ('current_time','timezone','location','user_profile','installed_apps') if k in raw['context']}
    # Only app/tool names, never any state embedded alongside installed tools.
    context['installed_apps']={k:{'tools':list(v.get('tools',[]))} for k,v in context['installed_apps'].items()}
    history=[]
    for turn in raw.get('history',[]):
        keys=('role','audio') if turn['role']=='user' else ('role','text','tool_calls','tool_results')
        history.append({k:deepcopy(turn[k]) for k in keys if k in turn})
    return normalized_row({'id':raw['id'],'context':context,'history':history},transcripts)


def probes_for(raw,row,text,specs):
    """Gold CURRENT arguments/results are targets only; inputs contain past steps."""
    actions=raw['label']['tool_calls'];observations=raw['label']['tool_results'];trajectory=[]
    assert len(actions)==len(observations)
    for step in range(len(actions)+1):
        expected={'next':'caller','tool':actions[step]['name']} if step<len(actions) else {'next':'summarizer'}
        # Contest has no explicit no-action route label; do not invent a give_up target.
        if not actions:expected={'next':None}
        request={'role':'planner','messages':build_messages('planner',row,text,specs,trajectory),
                 'schema':request_schema('planner',specs)}
        yield {'probe':f"{row['id']}:{step}:planner",'id':row['id'],'step':step,'request':request,'expected':expected}
        if step==len(actions):break
        action=actions[step]
        plan={'thought':CALLER_THOUGHT,'next':'caller','tool':action['name']}
        request={'role':'caller','messages':build_messages('caller',row,text,specs,trajectory,plan),
                 'schema':request_schema('caller',specs,plan)}
        yield {'probe':f"{row['id']}:{step}:caller",'id':row['id'],'step':step,'request':request,'expected':deepcopy(action)}
        trajectory.append({'step':step,'thought':CALLER_THOUGHT,'action':deepcopy(action),'observation':deepcopy(observations[step])})


def assess(probe,response,catalog):
    expected=probe['expected'];role=probe['request']['role'];value=None
    try:value=parse_json(response['raw'])
    except ValueError:pass
    valid=response.get('schema_valid') is True
    result={'schema_valid':valid,'correct':False,'expected':expected,'parsed':value}
    if role=='planner':
        result['scorable']=expected['next'] is not None
        result['route_match']=bool(valid and value.get('next')==expected['next']) if result['scorable'] else None
        result['tool_match']=bool(valid and value.get('tool')==expected.get('tool')) if expected.get('tool') else None
        result['correct']=bool(result['route_match'] and (expected['next']!='caller' or result['tool_match'])) if result['scorable'] else None
    else:
        result['scorable']=True
        result['correct']=bool(valid and matches(value,expected,catalog))
        fields={}
        for key,target in expected['arguments'].items():
            fields[key]=bool(valid and isinstance(value.get('arguments'),dict) and key in value['arguments'] and matches(
                {'name':value.get('name'),'arguments':{key:value['arguments'][key]}},
                {'name':expected['name'],'arguments':{key:target}},catalog))
        result['argument_matches']=fields
    words=(value.get('thought','') if isinstance(value,dict) else '').split()
    grams=Counter(tuple(words[i:i+6]) for i in range(len(words)-5))
    result.update(thought_words=len(words),thought_repeats_6gram_3plus=max(grams.values(),default=0)>=3)
    return result


class ComparedBackend(Backend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.adapter_requests=dict(self.requests);self.mode='adapter'
    def select(self,mode):
        assert mode in MODES
        self.mode=mode
        self.requests=dict(self.adapter_requests) if mode=='adapter' else {role:None for role in self.adapter_requests}
    def generate(self,requests):
        responses=super().generate(requests)
        for request,response in zip(requests,responses,strict=True):
            rendered=self.tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True)
            tokens=self.tokenizer.encode(rendered,add_special_tokens=False)
            response.update(mode=self.mode,input_ids_sha256=digest(tokens),
                adapter_role=request['role'] if self.requests[request['role']] is not None else None)
        return responses


def paired_nodes(probes,backend,catalog,work):
    rows=[]
    with (work/'node_outputs.jsonl').open('w') as f:
        for start in range(0,len(probes),4):
            group=probes[start:start+4];paired={}
            # Alternate order to avoid always putting one condition first.
            for mode in (MODES if start//4%2==0 else MODES[::-1]):
                backend.select(mode);tick=time.monotonic()
                responses=backend.generate([p['request'] for p in group])
                elapsed=time.monotonic()-tick
                for p,response in zip(group,responses,strict=True):
                    record={'probe':p['probe'],'id':p['id'],'step':p['step'],'role':p['request']['role'],
                        'batch_elapsed_s':elapsed,'batch_size':len(group),**response,**assess(p,response,catalog)}
                    rows.append(record);paired.setdefault(p['probe'],[]).append(record)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            for pair in paired.values():
                assert len(pair)==2 and {r['mode'] for r in pair}==set(MODES)
                for key in ('prompt_sha256','schema_sha256','input_ids_sha256'):assert pair[0][key]==pair[1][key],key
            print(json.dumps({'stage':'paired_nodes','completed':min(start+4,len(probes)),'total':len(probes)}),flush=True)
    return rows


def summarize_nodes(rows):
    result={}
    for role in ('planner','caller'):
        group=[r for r in rows if r['role']==role];by_probe={}
        for r in group:by_probe.setdefault(r['probe'],{})[r['mode']]=r
        summaries={}
        for mode in MODES:
            selected=[p[mode] for p in by_probe.values()];scored=[r for r in selected if r['scorable']]
            summaries[mode]={'probes':len(selected),'scored':len(scored),'correct':sum(r['correct'] for r in scored),
                'accuracy':sum(r['correct'] for r in scored)/len(scored) if scored else None,
                'schema_invalid':sum(not r['schema_valid'] for r in selected),
                'repeating_thoughts':sum(r['thought_repeats_6gram_3plus'] for r in selected)}
        changes=Counter()
        for pair in by_probe.values():
            if pair['base']['scorable']:
                b=pair['base']['correct'];a=pair['adapter']['correct']
                changes['adapter_win' if a and not b else 'adapter_loss' if b and not a else 'both_correct' if a else 'both_wrong']+=1
        result[role]={'conditions':summaries,'paired_outcomes':dict(changes)}
    return result


def workflows(raws,rows,transcripts,catalog,ToolEnv,backend,work):
    predictions={mode:[] for mode in MODES};traces={mode:[] for mode in MODES}
    for start in range(0,len(rows),4):
        for mode in (MODES if start//4%2==0 else MODES[::-1]):
            backend.select(mode);part=work/mode;part.mkdir(exist_ok=True)
            # Environments are passed through the bound factory, fresh per mode/sample.
            gens=[rollout(row,transcripts[raw['audio']],allowed_specs(row,catalog),ToolEnv(raw['id']),'multi')
                  for raw,row in zip(raws[start:start+4],rows[start:start+4],strict=True)]
            completed=run_batch(gens,backend)
            for pred,trace in completed:
                predictions[mode].append(pred);traces[mode].append(trace)
                for name,record in [('submission',pred),('traces',trace)]:
                    with (part/f'{name}.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(json.dumps({'stage':'workflow','mode':mode,'completed':len(predictions[mode]),'total':len(rows)}),flush=True)
    references=[]
    for raw in raws:
        env=ToolEnv(raw['id'])
        for action in raw['label']['tool_calls']:env.call(action['name'],action['arguments'])
        references.append({'id':raw['id'],'tool_calls':raw['label']['tool_calls'],'state_hash':env.get_state_hash()})
    reports={}
    for mode in MODES:
        reports[mode]={'metrics':evaluate(predictions[mode],references,catalog,traces[mode]),
            'termination_reasons':dict(Counter(t['reason'] for t in traces[mode])),
            'schema_invalid':sum(r.get('schema_valid') is False for t in traces[mode] for r in t['trace'])}
        write_json(work/mode/'report.json',reports[mode])
    return reports


def main():
    p=argparse.ArgumentParser()
    for name in ('data-root','work','model','revision','adapters'):p.add_argument('--'+name,required=True)
    a=p.parse_args();work=Path(a.work);root=Path(a.data_root)
    assert not (work/'node_outputs.jsonl').exists(),'Use a new RUN_NAME for another comparison; preserve existing outputs.'
    ids=(work/'split.txt').read_text().splitlines();assert ids and len(ids)==len(set(ids))
    lookup={r['id']:r for r in read_jsonl(root/'train/train.jsonl')};raws=[lookup[i] for i in ids]
    transcripts=json.loads((work/'asr_transcripts.json').read_text())
    rows=[safe_row(r,transcripts) for r in raws];catalog=load_catalog(root,'train');Env=import_toolenv(root,'train')
    probes=[p for raw,row in zip(raws,rows,strict=True) for p in probes_for(raw,row,transcripts[raw['audio']],allowed_specs(row,catalog))]
    # Requests/targets are separate artifacts. Only request dictionaries reach the model.
    write_json(work/'node_requests.json',[{'probe':p['probe'],**p['request']} for p in probes])
    write_json(work/'node_targets.json',[{'probe':p['probe'],'expected':p['expected']} for p in probes])
    backend=ComparedBackend(a.model,a.revision,a.adapters)
    node_rows=paired_nodes(probes,backend,catalog,work)
    node_report=summarize_nodes(node_rows);write_json(work/'node_report.json',node_report)
    workflow_report=workflows(raws,rows,transcripts,catalog,lambda sid:Env('train',sid,root=root),backend,work)
    report={'ids':ids,'samples':len(ids),'guided_decoding':False,'batch_size':4,'max_decisions':7,'max_tool_calls':7,
            'nodes':node_report,'workflows':workflow_report,'runtime':backend.runtime_options,
            'notes':['Selected failures are diagnostic, not representative validation.',
                'Planner metrics are annotation next-tool agreement; alternative valid orders may differ.',
                'Caller probes condition on the reference tool, with a fixed generic plan shared by both modes.',
                'Node inputs use reference past calls/observations only. No-action terminal route is not scored.',
                'Workflow inputs contain no reference plan, future observation or target arguments.',
                'Same FP16 engine/tokenizer; base disables both adapters. This does not isolate training quantization effects.',
                'Timing includes warmup/scheduling; do not interpret it as a controlled speed benchmark.']}
    write_json(work/'comparison.json',report)
    lines=['# Base vs adapter comparison','',f'Episodes: {len(ids)}. Unguided, batch4, 7 decisions /7 tool calls.','',
        '| Test | Base | Adapter |','|---|---:|---:|']
    for role in ('planner','caller'):
        c=node_report[role]['conditions']
        lines.append(f"| {role} fixed-prefix agreement | {c['base']['accuracy']} | {c['adapter']['accuracy']} |")
    for key in ('score_100','tool_f1','state_accuracy'):
        lines.append(f"| Workflow {key} | {workflow_report['base']['metrics'][key]:.4f} | {workflow_report['adapter']['metrics'][key]:.4f} |")
    lines+=['',*report['notes']];(work/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
