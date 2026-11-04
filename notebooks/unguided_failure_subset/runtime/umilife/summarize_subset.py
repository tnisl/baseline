"""Post-generation comparison against historical v5 outcomes; never a model input."""
import argparse,json
from collections import Counter
from pathlib import Path


def generation_stats(trace):
    stats={}
    for role in ('planner','caller'):
        rows=[r for r in trace['trace'] if r.get('role')==role and 'raw' in r]
        words=[];repeats=0
        for row in rows:
            try:thought=json.loads(row['raw']).get('thought','')
            except (ValueError,AttributeError):thought=''
            words.append(len(thought.split()))
            tokens=row['raw'].split()
            grams=Counter(tuple(tokens[i:i+6]) for i in range(len(tokens)-5))
            repeats+=int(max(grams.values(),default=0)>=3)
        stats[role]={'generations':len(rows),'raw_repeat_6gram_3plus':repeats,
                     'mean_thought_words':sum(words)/len(words) if words else 0}
    return stats


def summarize(work,baseline):
    work=Path(work);old=json.loads(Path(baseline).read_text())
    report=json.loads((work/'report.json').read_text())
    traces={r['id']:r for r in map(json.loads,(work/'traces.jsonl').read_text().splitlines())}
    predictions={r['id']:r for r in map(json.loads,(work/'submission.jsonl').read_text().splitlines())}
    scores={r['id']:r for r in report['metrics']['per_sample']}
    assert set(scores)==set(old['samples'])==set(traces)==set(predictions)
    result={'baseline_source':old['source'],'baseline_version':old['version'],
            'note':'Four selected historical failures, not representative validation. Historical batch/GPU concurrency differ; no speedup claim.',
            'samples':[]}
    lines=['# Historical LMFE vs unguided diagnostic','',result['note'],'',
           '| ID | Score old/new | Tool F1 old/new | State old/new | Planner repeats old/new | JSON invalid new | Ending new |',
           '|---|---|---|---|---|---|---|']
    for sid,previous in old['samples'].items():
        trace=traces[sid];current={'metrics':scores[sid],'reason':trace['reason'],
            'schema_invalid':sum(r.get('schema_valid') is False for r in trace['trace']),
            'generation':generation_stats(trace),'tool_calls':predictions[sid]['tool_calls']}
        result['samples'].append({'id':sid,'guided_v5':previous,'unguided':current})
        a=previous['metrics'];b=current['metrics']
        lines.append(f"| {sid} | {a['score']:.3f}/{b['score']:.3f} | {a['tool_f1']:.3f}/{b['tool_f1']:.3f} | {a['hash_match']}/{b['hash_match']} | {previous['generation']['planner']['raw_repeat_6gram_3plus']}/{current['generation']['planner']['raw_repeat_6gram_3plus']} | {current['schema_invalid']} | {current['reason']} |")
    (work/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (work/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--baseline',required=True)
    a=p.parse_args();summarize(a.work,a.baseline)
