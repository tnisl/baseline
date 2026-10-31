"""Assistant-only and next-only masks, without truncating targets or gold evidence."""
from __future__ import annotations
import json
from pathlib import Path


def common_prefix_length(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b))


def encode_completion(tokenizer,messages,target,loss_scope='completion'):
    prompt_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    full_ids=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':target}],
                                          tokenize=True,add_generation_prompt=False)
    if isinstance(prompt_ids,dict):prompt_ids=prompt_ids['input_ids']
    if isinstance(full_ids,dict):full_ids=full_ids['input_ids']
    if full_ids[:len(prompt_ids)]!=prompt_ids:
        raise ValueError('Chat template does not preserve the assistant prefix; refusing an incorrect loss mask')
    labels=[-100]*len(prompt_ids)+full_ids[len(prompt_ids):]
    if loss_scope=='next':
        obj=json.loads(target)
        if set(obj) not in ({'next'},{'next','answer'}):raise ValueError('Invalid summarizer target')
        if 'answer' in obj:
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            decision_prefix=json.dumps({'next':obj['next']},ensure_ascii=False,separators=(',',':'))[:-1]
            if not target.startswith(decision_prefix):
                raise ValueError('Summarizer target must serialize next first for the selective loss mask')
            prefix_ids=tokenizer(prompt+decision_prefix,add_special_tokens=False)['input_ids']
            boundary=common_prefix_length(full_ids,prefix_ids)
            if boundary<=len(prompt_ids):raise ValueError('No decision tokens left after template masking')
            labels[boundary:]=[-100]*(len(labels)-boundary)
    elif loss_scope!='completion':raise ValueError('Unknown loss scope')
    if not any(x!=-100 for x in labels):raise ValueError('No supervised tokens')
    return {'input_ids':full_ids,'attention_mask':[1]*len(full_ids),'labels':labels}


class RoleDataset:
    """Tokenize once, skip overlength cases explicitly, never crop assistant labels."""
    def __init__(self,path,tokenizer,max_length=8192,*,use_negatives=True,seed=42):
        import random
        self.items=[]
        self.stats={'loaded':0,'used':0,'overlength':0,'paired':0,'tokens':0,'supervised_tokens':0}
        rng=random.Random(seed)
        for line in Path(path).open(encoding='utf-8'):
            if not line.strip():continue
            record=json.loads(line);self.stats['loaded']+=1
            chosen=encode_completion(tokenizer,record['messages'],record['target'],record['loss_scope'])
            if len(chosen['input_ids'])>max_length:
                self.stats['overlength']+=1;continue
            candidates=list(record.get('rejected') or []) if use_negatives else []
            rng.shuffle(candidates);rejected=None
            for target in candidates:
                encoded=encode_completion(tokenizer,record['messages'],target,record['loss_scope'])
                if len(encoded['input_ids'])<=max_length:
                    rejected=encoded;break
            self.items.append({'chosen':chosen,'rejected':rejected,'case_id':record['case_id']})
            self.stats['used']+=1;self.stats['paired']+=int(rejected is not None)
            self.stats['tokens']+=len(chosen['input_ids'])
            self.stats['supervised_tokens']+=sum(x!=-100 for x in chosen['labels'])
        if not self.items:raise ValueError('No usable examples after length filtering: '+str(path))

    def __len__(self):return len(self.items)
    def __getitem__(self,index):return self.items[index]


class PairCollator:
    def __init__(self,pad_token_id):self.pad_token_id=pad_token_id

    def __call__(self,items):
        import torch
        def pad(records):
            length=max(len(r['input_ids']) for r in records)
            return {key:torch.tensor([r[key]+[fill]*(length-len(r[key])) for r in records],dtype=torch.long)
                    for key,fill in [('input_ids',self.pad_token_id),('attention_mask',0),('labels',-100)]}
        chosen=pad([x['chosen'] for x in items])
        rejected=pad([x['rejected'] or x['chosen'] for x in items])
        return {'chosen':chosen,'rejected':rejected,
                'paired':torch.tensor([x['rejected'] is not None for x in items],dtype=torch.bool)}
