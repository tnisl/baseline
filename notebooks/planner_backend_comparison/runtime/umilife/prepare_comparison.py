import argparse,json,hashlib
from pathlib import Path
from umilife.core import read_jsonl,allowed_specs,load_catalog,import_toolenv,write_json
from umilife.data import normalized_row
from umilife.english_labels import rationale,terminal
from umilife.prompt_policy import PROMPTS,digest,build_messages,request_schema
from umilife.comparison_common import fingerprint,training_probes

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--adapters',required=True);p.add_argument('--work',required=True);a=p.parse_args()
 root=Path(a.data_root);work=Path(a.work);work.mkdir(exist_ok=True,parents=True);adapters=Path(a.adapters)
 rows=read_jsonl(root/'train/train.jsonl');valid=set(Path('split.txt').read_text().splitlines());train=set(Path('train_ids.txt').read_text().splitlines())
 assert len(valid)==245 and len(train)==1300 and not train&valid and valid|train=={r['id'] for r in rows}
 manifest=json.loads((adapters/'checkpoint_manifest.json').read_text())
 for name,sha in manifest['files'].items():assert hashlib.sha256((adapters/name).read_bytes()).hexdigest()==sha,name
 lineage=json.loads((adapters/'planner/lineage.json').read_text());assert lineage['prompt_sha256']==digest(PROMPTS)
 specs_catalog=load_catalog(root,'train');ToolEnv=import_toolenv(root,'train');requests=[];references=[]
 def add(row,specs,trajectory,target,kind):
  key=f"{row['id']}:{len(trajectory)}:{kind}"
  requests.append({'probe':key,'kind':kind,'messages':build_messages('planner',normalized_row(row),row['audio_text'],specs,trajectory),'schema':request_schema('planner',specs)})
  references.append({'probe':key,'next':target['next'],**({'tool':target['tool']} if 'tool'in target else {})})
 selected=training_probes(rows,valid,8)
 for row in selected:
  assert row['id'] in train
  actions=row['label']['tool_calls'];specs=allowed_specs(row,specs_catalog)
  target={'next':'caller','tool':actions[0]['name']} if actions else terminal(row,[])[0]
  add(row,specs,[],target,'initial')
 for row in [r for r in selected if len(r['label']['tool_calls'])>=2][:2]:
  env=ToolEnv('train',row['id'],root=root);act=row['label']['tool_calls'][0];obs=env.call(act['name'],act['arguments']);assert obs==row['label']['tool_results'][0]
  trajectory=[{'thought':rationale(row,act,[]),'action':act,'observation':obs}]
  add(row,allowed_specs(row,specs_catalog),trajectory,{'next':'caller','tool':row['label']['tool_calls'][1]['name']},'gold_prefix')
 for row in [r for r in selected if r['label']['tool_calls']][:2]:
  env=ToolEnv('train',row['id'],root=root);trajectory=[]
  for act,expected in zip(row['label']['tool_calls'],row['label']['tool_results']):
   obs=env.call(act['name'],act['arguments']);assert obs==expected
   trajectory.append({'thought':rationale(row,act,trajectory),'action':act,'observation':obs})
  add(row,allowed_specs(row,specs_catalog),trajectory,terminal(row,trajectory)[0],'terminal')
 assert len(requests)==12 and len({r['probe'] for r in requests})==12
 write_json(work/'requests.json',requests);write_json(work/'references.json',references)
 write_json(work/'probe_manifest.json',{'samples':len(requests),'training_episode_ids':[r['id'] for r in selected],'held_out_used':False,'prompts_sha256':digest(PROMPTS),'requests_sha256':fingerprint(requests),'planner_lineage':lineage,'purpose':'Controlled train-only repetition diagnosis, not validation accuracy'})
 print('PREPARED',len(requests),'train-only probes',flush=True)
if __name__=='__main__':main()
