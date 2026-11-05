import argparse,json
from pathlib import Path
from collections import defaultdict
from umilife.core import read_jsonl,write_json

def summarize_probe(root,source):
    rows=[r for backend in ('unsloth','vllm') for variant in ('base','step75') for r in read_jsonl(root/f'{backend}_{variant}.jsonl')]
    selected=json.loads((source/'probe_manifest.json').read_text());keys={(r['id'],r['step']) for r in selected['samples']}
    report={'adapter_step':75,'historical_step':100,'exact_step100_reproduction':False,'cases':{},'paired':[],
            'limitations':['Six failure-selected/control prefixes, not a validation score.','Unsloth NF4 versus vLLM FP16 confounds backend and base precision.','Warmup/compilation/cache and case order affect batch timings.','Step75 weights cannot reproduce the lost step100 weights exactly.']}
    for backend in ('unsloth','vllm'):
        for variant in ('base','step75'):
            for batch in (1,4):
                group=[r for r in rows if (r['backend'],r['variant'],r['batch_size'])==(backend,variant,batch)]
                assert len(group)==len(keys) and {(r['id'],r['step']) for r in group}==keys
                times={r['batch_index']:r['batch_seconds'] for r in group}
                report['cases'][f'{backend}/{variant}/batch{batch}']={'samples':len(group),'schema_valid':sum(r['schema_valid'] for r in group),
                    'matches':sum(r['correct'] for r in group),'repeated_6gram_ge4':sum(r['max_repeated_6gram']>=4 for r in group),
                    'length_limited':sum(r['finish_reason']=='length' for r in group),'generated_tokens':sum(r['generated_tokens'] for r in group),
                    'generation_seconds':sum(times.values())}
    for key in sorted(keys):
        group=[r for r in rows if (r['id'],r['step'])==key]
        assert len({r['input_ids_sha256'] for r in group})==1,'Prompt/tokenizer parity failed'
        by={(r['backend'],r['variant'],r['batch_size']):r for r in group}
        report['paired'].append({'id':key[0],'step':key[1],'prompt_token_ids_identical':True,
            'batch1_vs4_identical':{f'{b}/{v}':by[b,v,1]['raw']==by[b,v,4]['raw'] for b in ('unsloth','vllm') for v in ('base','step75')},
            'unsloth_vs_vllm_identical':{f'{v}/batch{n}':by['unsloth',v,n]['raw']==by['vllm',v,n]['raw'] for v in ('base','step75') for n in (1,4)}})
    write_json(root/'report.json',report)
    text=['# Evaluation failure probe','', 'Step75 adapter; historical failures were at step100. No training or enforcer. NF4/FP16 precision differs across backends.','',
        '| Case | Valid JSON/schema | Match | Repetition | Length limit | Generation seconds |','|---|---:|---:|---:|---:|---:|']
    for name,r in report['cases'].items():text.append(f"| {name} | {r['schema_valid']}/6 | {r['matches']}/6 | {r['repeated_6gram_ge4']} | {r['length_limited']} | {r['generation_seconds']:.2f} |")
    text+=['','Read report.json for paired comparisons; raw JSONL files include exact output token IDs and finish/stop reasons. Historical outputs are in the input dataset. Settings and CE-restoration probes are in per-case JSON/log files.','',*report['limitations']]
    (root/'report.md').write_text('\n'.join(text)+'\n');print('\n'.join(text),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--input',required=True);a=p.parse_args();summarize_probe(Path(a.work),Path(a.input))
