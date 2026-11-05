"""Fixed split; train-only evidence/repair supervision and explicit label quarantine."""
import argparse,json,hashlib,copy
from pathlib import Path
from collections import Counter
from jsonschema import Draft202012Validator
from umilife.core import read_jsonl,write_json,dumps,digest,allowed_specs,load_catalog,import_toolenv
from umilife.data import normalized_row
from umilife.train import encode
from umilife.prompt_policy import build_messages,request_schema,POLICY_AUDIT
from umilife.evidence_labels import rationale,terminal,issues,VERSION
ROLES=('planner','caller')
LEAN='Carry out the selected requested operation using the original request, history and observed results. Preserve all conditions, corrections and intent-required fields.'

def prepare(root,out,split_file,tokenizer):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert digest(root/'train/train.jsonl')=='24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8'
    valid=set(Path(split_file).read_text().splitlines());assert len(valid)==245
    rows=read_jsonl(root/'train/train.jsonl');assert len(rows)==1545 and valid<={r['id'] for r in rows}
    zero=json.loads((Path(split_file).parent/'zero_action_train_labels.json').read_text())
    zero=zero['labels'];assert not valid&zero.keys(),'Training annotations must not include validation IDs'
    catalog=load_catalog(root,'train');Env=import_toolenv(root,'train')
    counts=Counter();audit=[];caches={(role,part):[] for role in ROLES for part in ('train','validation')};stats={};references=[]
    handles={key:(out/f'{key[0]}.{key[1]}.jsonl').open('w') for key in caches}
    def add(role,raw,specs,trajectory,target,plan=None,feedback=None,kind='reference'):
        part='validation' if raw['id'] in valid else 'train';key=(role,part)
        schema=request_schema(role,specs,plan);Draft202012Validator(schema).validate(target)
        rec={'id':raw['id'],'role':role,'step':len(trajectory),'kind':kind,
             'messages':build_messages(role,normalized_row(raw),raw['audio_text'],specs,trajectory,plan,feedback),'target':dumps(target)}
        encoded=encode(tokenizer,rec);assert len(encoded['input_ids'])<=8192,('Overlength',raw['id'],role)
        # Targets and diagnostic metadata are separate from message construction.
        handles[key].write(dumps(rec)+'\n');caches[key].append(encoded);counts[f'{role}.{part}']+=1;counts['kind.'+kind+'.'+part]+=1
    try:
        for raw in rows:
            part='validation' if raw['id'] in valid else 'train';specs=allowed_specs(raw,catalog)
            env=Env('train',raw['id'],root=root);trajectory=[];steps=[];problems=[]
            actions=raw['label']['tool_calls'];observations=raw['label']['tool_results'];assert len(actions)==len(observations)
            for index,(action,expected) in enumerate(zip(actions,observations,strict=True)):
                assert env.validate(action['name'],action['arguments']).get('ok')
                observed=env.call(action['name'],action['arguments']);assert observed==expected,(raw['id'],index)
                found=issues(raw,trajectory,action,observed)
                if found:problems.append({'step':index,'issues':found})
                plan={'thought':rationale(raw,action,trajectory),'next':'caller','tool':action['name']}
                steps.append((copy.deepcopy(trajectory),action,plan))
                trajectory.append({'step':index,'thought':plan['thought'],'action':copy.deepcopy(action),'observation':copy.deepcopy(observed)})
            references.append({'id':raw['id'],'tool_calls':actions,'state_hash':env.get_state_hash()})
            if problems:
                audit.append({'id':raw['id'],'split':part,'excluded_from_train':part=='train','problems':problems})
                if part=='train':continue
            if not actions:
                if part=='train':add('planner',raw,specs,[],zero[raw['id']],kind='reviewed_no_action')
                else:audit.append({'id':raw['id'],'split':part,'excluded_from_train':False,'unscored':'No explicit terminal route in contest annotation'})
                continue
            used_repair=False;used_read=False
            for prior,action,plan in steps:
                add('planner',raw,specs,prior,plan)
                caller_plan=plan
                # Half of train Caller records and ALL valid records get a non-oracle generic plan.
                if part=='validation' or int(hashlib.sha256(f"{raw['id']}:{len(prior)}".encode()).hexdigest()[:8],16)%2==0:
                    caller_plan={**plan,'thought':LEAN}
                add('caller',raw,specs,prior,action,caller_plan)
                if part!='train':continue
                if prior and not used_read:
                    last=prior[-1]['action']
                    if specs[last['name']].get('kind')=='read' and last['name']!=action['name']:
                        feedback={'kind':'repeated_read','action':last,'instruction':'Reuse the existing unchanged result and advance the pending objective.'}
                        if int(hashlib.sha256(raw['id'].encode()).hexdigest()[:8],16)%2:
                            feedback={'kind':'completion_review','proposal':{'thought':'The request may be complete.','next':'summarizer'},'instruction':'Check every requested objective against current observations. Continue if an authorized objective remains.'}
                            corrected={**plan,'thought':'An authorized objective remains; do not end after lookup alone. '+plan['thought']};kind='premature_end_recovery'
                        else:corrected={**plan,'thought':'Reuse the previous lookup; it already supplies evidence. '+plan['thought']};kind='read_recovery'
                        add('planner',raw,specs,prior,corrected,feedback=feedback,kind=kind);used_read=True
                if not used_repair:
                    optional=[k for k in action['arguments'] if k not in specs[action['name']]['parameters'].get('required',[])]
                    if optional:
                        proposal=copy.deepcopy(action);proposal['arguments'].pop(optional[0])
                        feedback={'kind':'intent_review','proposal':proposal,'instruction':'Check whether the proposed call omits a field needed by the original request. Restore intent-required information from available evidence, even if the API marks the field optional.'}
                        add('caller',raw,specs,prior,action,{**plan,'thought':LEAN},feedback=feedback,kind='argument_repair');used_repair=True
            add('planner',raw,specs,trajectory,terminal(raw,trajectory,specs),kind='terminal')
    finally:
        for f in handles.values():f.close()
    import torch
    files={p.name:digest(p) for p in out.glob('*.jsonl')}
    for (role,part),items in caches.items():
        stat={'loaded':len(items),'used':len(items),'overlength':0,'max_tokens':max((len(i['input_ids']) for i in items),default=0),
              'max_target_tokens':max((sum(v!=-100 for v in i['labels']) for i in items),default=0)}
        path=out/f'{role}.{part}.jsonl';torch.save({'items':items,'stats':stat},out/f'{role}.{part}.{files[path.name][:16]}.tokens.pt');stats[f'{role}.{part}']=stat
    manifest={'train_episodes':1300,'validation_episodes':245,'split':{r['id']:'validation' if r['id'] in valid else 'train' for r in rows},
        'counts':dict(counts),'prompts_sha256':POLICY_AUDIT['static_prompts_sha256'],'rationale_authors':['Codex'],
        'rationale_provenance':'Codex-authored evidence compiler '+VERSION+' plus100 manually reviewed training-only no-action targets; not independent per-example LLM reasoning.',
        'rationale_sha256':digest(Path(__file__).with_name('evidence_labels.py')),'zero_labels_sha256':digest(Path(split_file).parent/'zero_action_train_labels.json'),
        'train_quarantined_episodes':sum(x.get('excluded_from_train',False) for x in audit),
        'validation_unscored_no_action':sum('unscored' in x for x in audit),'tokenization':stats,'files_sha256':files,
        'split_sha256':digest(split_file),'model_input':'Clean provided train transcripts; no ASR augmentation. Reference prefix only; repair augmentation train-only.',
        'validation_usage':'Reference-only CE and generation metrics. No optimizer update on valid; no validation-derived prompt exceptions.'}
    write_json(out/'manifest.json',manifest);write_json(out/'data_audit.json',audit);write_json(out/'references.json',references)
    print(dumps(manifest),flush=True);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('data-root','output','split-file','model','revision'):p.add_argument('--'+name,required=True)
    a=p.parse_args();from transformers import AutoTokenizer
    prepare(a.data_root,a.output,a.split_file,AutoTokenizer.from_pretrained(a.model,revision=a.revision))
