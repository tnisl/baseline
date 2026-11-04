"""Deterministic episode split, retaining all 100 original validation IDs."""
import argparse,json,random,math,hashlib
from pathlib import Path
from collections import defaultdict,Counter

def create(root,output):
 root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(x) for x in (root/'train+public_test/train/train.jsonl').read_text().splitlines()]
 old=set((root/'split.txt').read_text().splitlines());assert len(old)==100
 groups=defaultdict(list)
 for r in rows:
  if r['id'] in old:continue
  calls=r['label']['tool_calls'];app=calls[0]['name'].split('.')[0] if calls else 'no_action'
  groups[app,min(len(calls),3),'nếu' in r['audio_text'].lower()].append(r['id'])
 quotas={k:len(v)*145/1445 for k,v in groups.items()};take={k:math.floor(v) for k,v in quotas.items()}
 for k in sorted(groups,key=lambda k:(-(quotas[k]-take[k]),str(k)))[:145-sum(take.values())]:take[k]+=1
 added=[]
 for k in sorted(groups,key=str):
  values=sorted(groups[k]);random.Random('english-split-20261104'+str(k)).shuffle(values);added+=values[:take[k]]
 valid=old|set(added);train={r['id'] for r in rows}-valid
 assert len(valid)==245 and len(train)==1300 and not train&valid
 for name,ids in [('split.txt',valid),('train_ids.txt',train)]: (out/name).write_text('\n'.join(sorted(ids))+'\n')
 report={'seed':'english-split-20261104','train':1300,'validation':245,'original_valid_retained':100,'additional_valid':145,'strata':'first-tool app, capped call count, conditional keyword; proportional largest remainder','split_sha256':hashlib.sha256((out/'split.txt').read_bytes()).hexdigest(),'requires_new_global':True,'note':'ID-disjoint split. Semantic paraphrase groups have not been manually audited; old 100 are a development set.'}
 (out/'split_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();create(a.root,a.output)
