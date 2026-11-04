"""Fixed episode split, concise field-source plans, prior-evidence-only inputs."""
import argparse,hashlib,json
from copy import deepcopy
from collections import Counter
from pathlib import Path
from jsonschema import Draft202012Validator
from umilife.core import read_jsonl,write_json,dumps,digest,allowed_specs,load_catalog,import_toolenv
from umilife.data import annotation_tasks,normalized_row
from umilife.prompt_policy import PROMPTS,build_messages,request_schema,POLICY_AUDIT
from umilife.train import encode
from umilife.english_labels import rationale,terminal as grounded_terminal,VERSION

RATIONAL_SHA='d14768df60e155521a57998a5c6d0c4c1cc3b6372b74cce7a955098fd3fdf497'
SPLIT_SHA='631f8cfab61ddde833477f743cf1a65a9012526e0178e97046a36ee864e3d505'
TRAIN_SHA='24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8'

def main():
    p=argparse.ArgumentParser()
    for name in ('data-root','output','split-file','model','revision'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(a.data_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert digest(root/'train/train.jsonl')==TRAIN_SHA
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    ids=Path(a.split_file).read_text().splitlines();valid=set(ids);assert len(ids)==len(valid)==245
    rows=read_jsonl(root/'train/train.jsonl');assert len(rows)==1545
    assignments={r['id']:'validation' if r['id'] in valid else 'train' for r in rows}
    assert valid=={k for k,v in assignments.items() if v=='validation'}
    catalog=load_catalog(root,'train');ToolEnv=import_toolenv(root,'train')
    counts=Counter();references=[];token_stats={};schema_cache={}
    handles={(role,part):(out/f'{role}.{part}.jsonl').open('w') for role in ('global','planner','caller','summarizer') for part in ('train','validation')}
    def add(role,row,specs,trajectory,target,plan=None):
        part=assignments[row['id']]
        schema=request_schema(role,specs,plan);key=dumps(schema)
        if key not in schema_cache:schema_cache[key]=Draft202012Validator(schema)
        schema_cache[key].validate(target)
        rec={'id':row['id'],'role':role,'step':len(trajectory),
             'messages':build_messages(role,normalized_row(row),row['audio_text'],specs,trajectory,plan),
             'target':dumps(target)}
        encoded=encode(tokenizer,rec);n=len(encoded['input_ids']);assert n<=8192,('Overlength',row['id'],role,n)
        stat=token_stats.setdefault(f'{role}.{part}',{'records':0,'max_tokens':0,'max_target_tokens':0})
        stat['records']+=1;stat['max_tokens']=max(stat['max_tokens'],n)
        stat['max_target_tokens']=max(stat['max_target_tokens'],sum(x!=-100 for x in encoded['labels']))
        handles[role,part].write(dumps(rec)+'\n');counts[f'{role}.{part}']+=1
    try:
        for row in rows:
            specs=allowed_specs(row,catalog);env=ToolEnv('train',row['id'],root=root);trajectory=[]
            gold=row['label']['tool_calls'];results=row['label']['tool_results'];assert len(gold)==len(results)
            for step,(action,expected) in enumerate(zip(gold,results,strict=True)):
                assert env.validate(action['name'],action['arguments']).get('ok')
                observed=env.call(action['name'],action['arguments']);assert observed==expected,(row['id'],step)
                thought=rationale(row,action,trajectory)
                plan={'thought':thought,'next':'caller','tool':action['name']}
                add('planner',row,specs,trajectory,plan)
                add('caller',row,specs,trajectory,action,plan)
                add('global',row,specs,trajectory,{**plan,'action':action})
                trajectory.append({'thought':thought,'action':action,'observation':observed})
            references.append({'id':row['id'],'tool_calls':gold,'state_hash':env.get_state_hash()})
            plan,answer=grounded_terminal(row,trajectory)
            add('planner',row,specs,trajectory,plan)
            add('global',row,specs,trajectory,{**plan,**({'answer':answer} if answer is not None else {})})
            if answer is not None:add('summarizer',row,specs,trajectory,{'answer':answer},plan)
    finally:
        for f in handles.values():f.close()
    manifest={'split':assignments,'counts':dict(counts),'excluded':[],
        'train_episodes':1300,'validation_episodes':245,'split_sha256':digest(a.split_file),'train_sha256':TRAIN_SHA,
        'model':a.model,'revision':a.revision,'rationale_authors':['Codex'],'rationales_sha256':digest(Path(__file__).with_name('english_labels.py')),
        'rationale_provenance':'English evidence-compiled labels '+VERSION+'; deterministic annotation rules, not independent per-example human/LLM review.',
        'summary_labels':'Deterministic concise facts from final gold tool observation; max 180 answer tokens; not human-written summaries.',
        'zero_action_convention':'Cancellation/acknowledgement heuristic -> summarizer; other zero-call requests -> give_up. Audit ambiguous cases before claiming semantic correctness.',
        'validation_usage':'Held-out labels only for loss evaluation and post-generation scoring; never optimizer updates or inference prompts.',
        'training_transcripts':'Provided clean audio_text; post-training evaluation uses supplied ASR with no oracle fallback.',
        'prompts_sha256':POLICY_AUDIT['static_prompts_sha256'],'policy':POLICY_AUDIT,
        'tokenization':token_stats,'files_sha256':{f.name:digest(f) for f in out.glob('*.jsonl')}}
    write_json(out/'references.json',references);write_json(out/'manifest.json',manifest)
    print(dumps(manifest),flush=True)
if __name__=='__main__':main()
