"""Free generation on fixed gold-prefix inputs; reference labels are scorer-only."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def readl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)

def input_request(record):
    # Neither id/step nor the current reference thought/route/tool is a model input.
    assert set(record) == {'id', 'role', 'step', 'messages', 'schema'}
    assert record['role'] == 'planner'
    return {'role': 'planner', 'messages': record['messages'], 'schema': record['schema']}

def generate_all(records, backend, work, deadline=None):
    """One sample per input, batches of 4, safe batch resume without executing tools."""
    work = Path(work); batches = work/'batches'; batches.mkdir(parents=True, exist_ok=True)
    outputs = []; started = time.monotonic(); measured = []; generated = reused = 0
    with (work/'predictions.jsonl').open('w') as export:
        for start in range(0, len(records), 4):
            group = records[start:start+4]
            fingerprints = [hashlib.sha256(canonical(r).encode()).hexdigest() for r in group]
            batch_file = batches/f'{start:06}.json'
            if batch_file.exists():
                saved = json.loads(batch_file.read_text())
                assert saved['input_sha256'] == fingerprints, 'Cached batch belongs to other inputs'
                responses = saved['outputs']; reused += len(group)
            else:
                if deadline is not None and time.monotonic() >= deadline:
                    atomic(work/'progress.json', {'paused': True, 'completed': len(outputs), 'total': len(records)})
                    raise SystemExit(75)
                assert backend is not None
                tick = time.monotonic()
                requests = [input_request(r) for r in group]
                responses = backend.generate(requests)
                assert len(responses) == len(group)
                for response in responses: assert 'raw' in response and 'schema_valid' in response
                elapsed = time.monotonic()-tick; measured.append(elapsed); generated += len(group)
                atomic(batch_file, {'input_sha256': fingerprints, 'outputs': responses, 'elapsed_s': elapsed})
            assert len(responses) == len(group)
            for record, response in zip(group, responses):
                output = {'id': record['id'], 'step': record['step'], **response}
                outputs.append(output)
                export.write(json.dumps(output, ensure_ascii=False)+'\n')
            export.flush()
            remaining = (len(records)-len(outputs)+3)//4
            print(json.dumps({'completed': len(outputs), 'total': len(records),
                'generated': generated, 'reused': reused,
                'eta_seconds': sum(measured)/len(measured)*remaining if measured else None}), flush=True)
    if (work/'progress.json').exists(): (work/'progress.json').unlink()
    return outputs, {'generated': generated, 'reused': reused, 'generation_wall_s': time.monotonic()-started}

def score_outputs(outputs, labels):
    """Called only after generation finishes; no token/rationale exact-match accuracy."""
    assert len(outputs) == len(labels) > 0
    details = []; episode = defaultdict(list); confusion = Counter(); routes = Counter()
    seen = set()
    for output, label in zip(outputs, labels):
        key = (output['id'], output['step'])
        assert key == (label['id'], label['step']) and key not in seen
        seen.add(key)
        target = json.loads(label['target']); predicted = {}
        if output['schema_valid']:
            try:
                predicted = json.loads(output['raw'])
                assert isinstance(predicted, dict)
            except (ValueError, AssertionError): predicted = {}
        ref_next = target['next']; next_value = predicted.get('next')
        routes[ref_next] += 1
        route_ok = bool(output['schema_valid'] and next_value == ref_next)
        decision_ok = route_ok and (ref_next != 'caller' or predicted.get('tool') == target['tool'])
        confusion[(ref_next, str(next_value))] += 1
        row = {'id': key[0], 'step': key[1], 'expected_next': ref_next,
            'predicted_next': next_value, 'expected_tool': target.get('tool'),
            'predicted_tool': predicted.get('tool'), 'route_correct': route_ok,
            'decision_correct': decision_ok, 'schema_valid': output['schema_valid'],
            'finish_reason': output.get('finish_reason'), 'schema_error': output.get('schema_error'),
            'predicted_thought': predicted.get('thought'), 'raw': output['raw']}
        details.append(row); episode[key[0]].append(decision_ok)
    n = len(details); caller = [r for r in details if r['expected_next'] == 'caller']
    terminal = [r for r in details if r['expected_next'] != 'caller']
    first = [r for r in details if r['step'] == 0]
    def accuracy(group, field): return sum(r[field] for r in group)/len(group) if group else None
    metrics = {'samples': n, 'episodes': len(episode), 'gold_routes': dict(routes),
        'route_accuracy': accuracy(details, 'route_correct'),
        'route_and_tool_accuracy': accuracy(details, 'decision_correct'),
        'caller_steps': len(caller), 'caller_tool_accuracy': accuracy(caller, 'decision_correct'),
        'terminal_steps': len(terminal), 'terminal_route_accuracy': accuracy(terminal, 'route_correct'),
        'first_steps': len(first), 'first_route_accuracy': accuracy(first, 'route_correct'),
        'first_route_and_tool_accuracy': accuracy(first, 'decision_correct'),
        'all_reference_steps_correct_episodes': sum(all(v) for v in episode.values()),
        'all_reference_steps_correct_episode_rate': sum(all(v) for v in episode.values())/len(episode),
        'schema_valid': sum(r['schema_valid'] for r in details),
        'schema_invalid': sum(not r['schema_valid'] for r in details),
        'finish_reasons': dict(Counter(r.get('finish_reason') for r in details)),
        'majority_route_baseline': max(routes.values())/n,
        'route_confusion': [{'expected': x, 'predicted': y, 'count': c} for (x, y), c in sorted(confusion.items())],
        'metric_note': 'Strict next-route/tool agreement on fixed gold prefixes, not end-to-end task success. Alternative valid tool orders are not adjudicated. Rationale semantics are not scored.'}
    return metrics, details

def main():
    p = argparse.ArgumentParser()
    for name in ('inputs', 'adapters', 'work', 'model', 'revision', 'source-sha'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--deadline-monotonic', type=float)
    a = p.parse_args(); inputs = Path(a.inputs); adapters = Path(a.adapters); work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((inputs/'evaluation_manifest.json').read_text())
    for name, expected in provenance['files'].items(): assert sha(inputs/name) == expected, name
    assert provenance['samples'] == 327 and provenance['episodes'] == 100
    assert a.model == provenance['model'] and a.revision == provenance['revision']
    from umilife.prompt_policy import PROMPTS, digest
    assert digest(PROMPTS) == provenance['prompt_sha256']
    assert sha(adapters/'planner/adapter_model.safetensors') == provenance['planner_adapter_sha256']
    assert sha(adapters/'training_manifest.json') == provenance['training_manifest_sha256']
    records = readl(inputs/'planner_inputs.jsonl')
    assert len(records) == 327 and len({r['id'] for r in records}) == 100
    assert len({(r['id'], r['step']) for r in records}) == 327
    for record in records:
        input_request(record)
        payload = json.loads(record['messages'][-1]['content'])
        assert set(payload) == {'context', 'reference_clock', 'history', 'trajectory', 'tools', 'request'}
        assert len(payload['trajectory']) == record['step']
    config = {'source_sha256': a.source_sha, 'inputs_manifest_sha256': sha(inputs/'evaluation_manifest.json'),
        'planner_adapter_sha256': provenance['planner_adapter_sha256'], 'model': a.model, 'revision': a.revision,
        'prompt_sha256': provenance['prompt_sha256'], 'batch_size': 4, 'max_tokens': 320,
        'temperature': 0, 'seed': 42, 'backend': 'vllm==0.9.2', 'dtype': 'float16',
        'guided_decoding': 'lm-format-enforcer', 'completion_review': False, 'retries': 0,
        'transcripts': 'clean', 'previous_trajectory': 'gold', 'scorer_labels_are_model_inputs': False}
    if (work/'config.json').exists(): assert json.loads((work/'config.json').read_text()) == config
    else: atomic(work/'config.json', config)
    pending = any(not (work/'batches'/f'{i:06}.json').exists() for i in range(0, len(records), 4))
    backend = None
    if pending:
        from umilife.infer_2gpu import Backend
        backend = Backend(a.model, a.revision, str(adapters))
        # Synthetic engine/grammar probe only; no validation labels or gold route are provided.
        probes = [{'role': 'planner', 'probe': True, 'schema': {'type': 'object',
            'properties': {'probe': {'type': 'string', 'enum': [f'planner-{i}']}},
            'required': ['probe'], 'additionalProperties': False},
            'messages': [{'role': 'user', 'content': 'Say hello, no JSON.'}]} for i in range(4)]
        probe_responses = backend.generate(probes)
        assert all(r['schema_valid'] and json.loads(r['raw']) == {'probe': f'planner-{i}'}
                   for i, r in enumerate(probe_responses))
        lengths = []
        for record in records:
            request = input_request(record); backend.validate(request)
            prompt = backend.tokenizer.apply_chat_template(request['messages'], tokenize=False, add_generation_prompt=True)
            lengths.append(len(backend.tokenizer.encode(prompt, add_special_tokens=False)))
        atomic(work/'preflight.json', {'synthetic_probes': 4, 'batch_size': 4, 'status': 'passed',
            'input_count': len(records), 'max_prompt_tokens': max(lengths), 't4_compatibility': backend.t4_compatibility})
    outputs, timing = generate_all(records, backend, work, a.deadline_monotonic)
    assert len(outputs) == 327
    # Labels are parsed for the first time only after all 327 predictions have been saved.
    metrics, details = score_outputs(outputs, readl(inputs/'planner_labels.jsonl'))
    (work/'details.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in details))
    with (work/'details.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    result = {'status': 'complete', 'config': config, 'metrics': metrics, 'timing': timing,
        'inputs_provenance': provenance, 'training_run': False, 'tool_calls_executed': 0}
    atomic(work/'report.json', result)
    print('PLANNER_VALIDATION_327_COMPLETE', json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__': main()
