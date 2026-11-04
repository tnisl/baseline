"""Package exact saved Planner validation examples into private inputs and labels."""
import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('original_policy', HERE/'runtime/umilife/prompt_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def readl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def prepare(prepared, data_root, split_file, adapters, output):
    prepared, data_root, adapters, output = map(Path, (prepared, data_root, adapters, output))
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prepared/'manifest.json').read_text())
    records = readl(prepared/'planner.validation.jsonl')
    assert sha(prepared/'planner.validation.jsonl') == manifest['files_sha256']['planner.validation.jsonl']
    assert sha(split_file) == manifest['split_sha256']
    assert sha(data_root/'train/train.jsonl') == manifest['train_sha256']
    assert policy.digest(policy.PROMPTS) == manifest['prompts_sha256']
    lineage = json.loads((adapters/'planner/lineage.json').read_text())
    assert sha(prepared/'manifest.json') == lineage['dataset_manifest_sha256']
    assert lineage['prompt_sha256'] == manifest['prompts_sha256']
    ids = Path(split_file).read_text().splitlines()
    assert len(ids) == len(set(ids)) == 100
    gold = {r['id']: r for r in readl(data_root/'train/train.jsonl') if r['id'] in ids}
    catalog_raw = json.loads((data_root/'train/train_tools_catalog.json').read_text())
    catalog = {tool['name']: tool for app in catalog_raw['apps'] for tool in app['tools']}
    inputs, labels, seen, routes = [], [], set(), Counter()
    by_id = Counter()
    for record in records:
        sid, step = record['id'], record['step']
        assert sid in gold and record['role'] == 'planner'
        assert (sid, step) not in seen
        seen.add((sid, step)); by_id[sid] += 1
        row = gold[sid]
        payload = json.loads(record['messages'][-1]['content'])
        assert set(payload) == {'context', 'reference_clock', 'history', 'trajectory', 'tools', 'request'}
        assert payload['request'] == row['audio_text']
        assert record['messages'][0] == {'role': 'system', 'content': policy.PROMPTS['planner']}
        assert len(payload['trajectory']) == step <= len(row['label']['tool_calls'])
        for index, previous in enumerate(payload['trajectory']):
            assert previous['action'] == row['label']['tool_calls'][index]
            assert previous['observation'] == row['label']['tool_results'][index]
        installed = {n for app in row['context']['installed_apps'].values() for n in app['tools']}
        installed.add('sys.ask_missing_info')
        specs = {n: catalog[n] for n in sorted(installed) if n in catalog}
        assert payload['tools'] == policy.compact_catalog(specs)
        target = json.loads(record['target'])
        expected = 'caller' if step < len(row['label']['tool_calls']) else 'summarizer'
        assert target['next'] == expected
        if expected == 'caller': assert target['tool'] == row['label']['tool_calls'][step]['name']
        schema = policy.request_schema('planner', specs)
        inputs.append({'id': sid, 'role': 'planner', 'step': step,
                       'messages': record['messages'], 'schema': schema})
        labels.append({'id': sid, 'step': step, 'target': record['target']})
        routes[target['next']] += 1
    assert len(records) == 327 and set(by_id) == set(ids)
    assert all(by_id[sid] == len(gold[sid]['label']['tool_calls']) + 1 for sid in ids)
    # Prove that splitting input/label files preserves the original serialized records.
    rebuilt = ''.join(dumps({'id': x['id'], 'role': x['role'], 'step': x['step'],
        'messages': x['messages'], 'target': y['target']}) + '\n' for x, y in zip(inputs, labels))
    assert hashlib.sha256(rebuilt.encode()).hexdigest() == sha(prepared/'planner.validation.jsonl')
    for name, rows in [('planner_inputs.jsonl', inputs), ('planner_labels.jsonl', labels)]:
        (output/name).write_text(''.join(dumps(r)+'\n' for r in rows))
    (output/'split.txt').write_bytes(Path(split_file).read_bytes())
    (output/'training_manifest.json').write_bytes((prepared/'manifest.json').read_bytes())
    provenance = {'source_notebook': 'cymeu4l0t/lifemate-qwen3-glpft-finetune-2gpu',
        'source_version': 6, 'samples': 327, 'episodes': 100, 'gold_routes': dict(routes),
        'input_kind': 'Exact saved validation messages: clean transcripts and previous gold trajectory only',
        'model': lineage['model'], 'revision': lineage['revision'],
        'prompt_sha256': lineage['prompt_sha256'],
        'training_manifest_sha256': sha(prepared/'manifest.json'),
        'original_planner_validation_sha256': sha(prepared/'planner.validation.jsonl'),
        'planner_adapter_sha256': sha(adapters/'planner/adapter_model.safetensors'),
        'files': {p.name: sha(p) for p in output.iterdir() if p.name in
                  ('planner_inputs.jsonl', 'planner_labels.jsonl', 'split.txt', 'training_manifest.json')}}
    (output/'evaluation_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (output/'dataset-metadata.json').write_text(json.dumps({
        'id': 'cymeu4l0t/lifemate-v6-planner-validation-inputs',
        'title': 'LifeMate v6 Planner validation inputs', 'licenses': [{'name': 'other'}]}))
    return provenance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('prepared', 'data-root', 'split-file', 'adapters', 'output'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    print(json.dumps(prepare(a.prepared, a.data_root, a.split_file, a.adapters, a.output), indent=2))
