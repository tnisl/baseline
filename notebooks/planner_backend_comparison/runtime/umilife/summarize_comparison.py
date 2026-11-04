import argparse,json,csv,statistics
from pathlib import Path

def summarize(work):
 work=Path(work);rows=[]
 for p in work.glob('*.jsonl'):
  if p.name in ('transformers.jsonl','vllm.jsonl','vllm_graph.jsonl'):rows.extend(json.loads(l) for l in p.read_text().splitlines())
 expected={r['probe'] for r in json.loads((work/'requests.json').read_text())};groups={}
 for r in rows:groups.setdefault((r['backend'],r['cuda_graphs'],r['mode']),[]).append(r)
 assert len(groups)==7,'Missing backend/mode outputs'
 for key,rs in groups.items():assert len(rs)==len(expected) and {r['probe'] for r in rs}==expected,key
 for probe in expected:
  rs=[r for r in rows if r['probe']==probe]
  for field in ('prompt_sha256','input_ids_sha256','schema_sha256','max_new_tokens'):
   assert len({r[field] for r in rs})==1,(probe,field)
 summaries=[]
 for (backend,graph,mode),rs in groups.items():
  summaries.append({'backend':backend,'cuda_graphs':graph,'mode':mode,'samples':len(rs),
   'json_valid_rate':sum(r['schema_valid'] for r in rs)/len(rs),'repetitive_rate':sum(r['repeats_6gram_3plus'] for r in rs)/len(rs),
   'raw_repetitive_rate':sum(r['raw_repeats_6gram_3plus'] for r in rs)/len(rs),'raw_words_mean':statistics.mean(r['raw_words'] for r in rs),'thought_words_valid_mean':statistics.mean(r['thought_words'] for r in rs if r['schema_valid']) if any(r['schema_valid'] for r in rs) else None,'thought_1000chars_rate':sum(r['thought_at_1000chars'] for r in rs)/len(rs),
   'tool_reference_match':sum(r['tool_match'] for r in rs)/len(rs),'seconds_total':sum(r['seconds'] for r in rs),
   'note':'Train-only diagnostic; not validation accuracy. First-request JIT/warmup included; batch1. Different attention kernels can differ numerically.'})
 result={'samples_per_condition':len(expected),'conditions':summaries,'prompt_and_tokenizer_parity':'pass','training':False}
 (work/'report.json').write_text(json.dumps(result,indent=2))
 with (work/'summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',required=True);summarize(p.parse_args().work)
