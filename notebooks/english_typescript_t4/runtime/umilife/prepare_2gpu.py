"""Fixed split, existing Codex rationales, prior-evidence-only training prompts."""
import argparse,hashlib,json
from copy import deepcopy
from collections import Counter
from pathlib import Path
from jsonschema import Draft202012Validator
from umilife.core import read_jsonl,write_json,dumps,digest,allowed_specs,load_catalog,import_toolenv
from umilife.data import annotation_tasks,normalized_row
from umilife.prompt_policy import PROMPTS,build_messages,request_schema,POLICY_AUDIT
from umilife.train import encode

RATIONAL_SHA='d14768df60e155521a57998a5c6d0c4c1cc3b6372b74cce7a955098fd3fdf497'
SPLIT_SHA='631f8cfab61ddde833477f743cf1a65a9012526e0178e97046a36ee864e3d505'
TRAIN_SHA='24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8'

def brief_facts(value,prefix='',budget=8):
    if isinstance(value,dict):
        out=[]
        for key,item in value.items():
            out.extend(brief_facts(item,f'{prefix}.{key}'.strip('.'),budget-len(out)))
            if len(out)>=budget:break
        return out
    if isinstance(value,list):
        out=[f'{prefix}: {len(value)} mục']
        for i,item in enumerate(value[:2]):
            out.extend(brief_facts(item,f'{prefix}[{i}]',budget-len(out)))
            if len(out)>=budget:break
        return out[:budget]
    if budget<=0:return []
    text=str(value) if value is not None else 'null'
    return [prefix+'='+text[:120]]

def terminal(trajectory,tokenizer):
    if not trajectory:
        # Existing dataset convention: zero calls -> give_up. Includes some no-op/cancellation cases.
        return {'thought':'Yêu cầu không cần hoặc không thể xử lý bằng các công cụ được cài.','next':'give_up'},None
    last=trajectory[-1];action=last['action']
    if action['name']=='sys.ask_missing_info':
        answer='Bạn vui lòng bổ sung '+', '.join(action['arguments'].get('params',[]))+' cho '+action['arguments'].get('tool_name','yêu cầu')+'.'
        thought='Đã hỏi thông tin còn thiếu, cần chờ người dùng trả lời.'
    else:
        # Deterministic grounded label, not an LLM teacher and not a full observation dump.
        answer='Kết quả '+action['name']+': '+'; '.join(brief_facts(last['observation']))+'.'
        thought='Đã có kết quả của các bước trong nhãn, chuyển sang tóm tắt bằng chứng.'
    answer=tokenizer.decode(tokenizer.encode(answer[:800],add_special_tokens=False)[:180],skip_special_tokens=True)
    return {'thought':thought,'next':'summarizer'},answer

def main():
    p=argparse.ArgumentParser()
    for name in ('data-root','output','rationales','split-file','model','revision'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(a.data_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert digest(root/'train/train.jsonl')==TRAIN_SHA
    assert digest(a.rationales)==RATIONAL_SHA and digest(a.split_file)==SPLIT_SHA
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    ids=Path(a.split_file).read_text().splitlines();valid=set(ids);assert len(ids)==len(valid)==100
    rows=read_jsonl(root/'train/train.jsonl');assert len(rows)==1545
    assignments={r['id']:'validation' if r['id'] in valid else 'train' for r in rows}
    assert valid=={k for k,v in assignments.items() if v=='validation'}
    records=read_jsonl(a.rationales);assert len(records)==3541 and {r['author'] for r in records}=={'Codex'}
    rationales={r['key']:r['thought'] for r in records};assert len(rationales)==len(records)
    catalog=load_catalog(root,'train');ToolEnv=import_toolenv(root,'train')
    tasks={(t['id'],t['step']):t for t in annotation_tasks(rows,catalog,assignments)}
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
                thought=rationales[tasks[row['id'],step]['key']]
                plan={'thought':thought,'next':'caller','tool':action['name']}
                add('planner',row,specs,trajectory,plan)
                add('caller',row,specs,trajectory,action,plan)
                add('global',row,specs,trajectory,{**plan,'action':action})
                trajectory.append({'thought':thought,'action':action,'observation':observed})
            references.append({'id':row['id'],'tool_calls':gold,'state_hash':env.get_state_hash()})
            plan,answer=terminal(trajectory,tokenizer)
            add('planner',row,specs,trajectory,plan)
            add('global',row,specs,trajectory,{**plan,**({'answer':answer} if answer is not None else {})})
            if answer is not None:add('summarizer',row,specs,trajectory,{'answer':answer},plan)
    finally:
        for f in handles.values():f.close()
    assert counts=={'global.train':4759,'planner.train':4759,'caller.train':3314,'summarizer.train':1334,
                    'global.validation':327,'planner.validation':327,'caller.validation':227,'summarizer.validation':100},dict(counts)
    manifest={'split':assignments,'counts':dict(counts),'excluded':[],
        'train_episodes':1445,'validation_episodes':100,'split_sha256':SPLIT_SHA,'train_sha256':TRAIN_SHA,
        'model':a.model,'revision':a.revision,'rationale_authors':['Codex'],'rationales_sha256':RATIONAL_SHA,
        'rationale_provenance':'Existing Codex-authored tool explanation templates plus episode overrides; unchanged thought strings. Ignore stale partition field.',
        'summary_labels':'Deterministic concise facts from final gold tool observation; max 180 answer tokens; not human-written summaries.',
        'zero_action_convention':'Preserved give_up target for all zero-call gold episodes, including some no-op/cancellation cases.',
        'validation_usage':'Held-out labels only for loss evaluation and post-generation scoring; never optimizer updates or inference prompts.',
        'training_transcripts':'Provided clean audio_text; post-training evaluation uses supplied ASR with no oracle fallback.',
        'prompts_sha256':POLICY_AUDIT['static_prompts_sha256'],'policy':POLICY_AUDIT,
        'tokenization':token_stats,'files_sha256':{f.name:digest(f) for f in out.glob('*.jsonl')}}
    write_json(out/'references.json',references);write_json(out/'manifest.json',manifest)
    print(dumps(manifest),flush=True)
if __name__=='__main__':main()
