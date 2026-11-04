"""Pure configuration/lineage checks, independent of Torch and GPUs."""
import json
from pathlib import Path

LOCAL_STAGES = ('planner', 'caller', 'summarizer')

def stages_for(mode, selected=LOCAL_STAGES):
    assert mode in ('global', 'local', 'global_to_local'), 'Unknown TRAIN_MODE'
    assert selected and len(set(selected)) == len(selected) and set(selected) <= set(LOCAL_STAGES)
    if mode == 'global': return ('global',)
    if mode == 'local': return tuple(selected)
    return ('global', *selected)

def validate_initialization(stage, init_from, global_adapter):
    assert stage in ('global', *LOCAL_STAGES)
    assert init_from in ('base', 'global')
    if stage == 'global': assert init_from == 'base' and not global_adapter
    assert bool(global_adapter) == (init_from == 'global'), 'Global parent is required only for init-from=global'
    return init_from

def parent_for(stage, mode, local_init, external_parent, output):
    assert local_init in ('base', 'global')
    if stage == 'global': return None
    if mode == 'global_to_local': return Path(output)/'adapters/global'
    if local_init == 'base':
        assert external_parent is None, 'Remove GLOBAL_ADAPTER when LOCAL_INIT=base'
        return None
    assert external_parent is not None, 'Set GLOBAL_ADAPTER for LOCAL_INIT=global'
    return Path(external_parent)

def verify_parent(path, model, revision, manifest_sha, prompt_sha, smoke):
    path = Path(path)
    meta = json.loads((path/'lineage.json').read_text())
    assert meta['stage'] == 'global' and meta['model'] == model and meta['revision'] == revision
    assert meta['dataset_manifest_sha256'] == manifest_sha, 'Parent Global was trained with a different data split/labels'
    assert meta['prompt_sha256'] == prompt_sha and meta['final_adapter']['finite']
    assert smoke or not meta.get('smoke', False), 'Full training cannot initialize from a smoke adapter'
    assert (path/'adapter_model.safetensors').is_file()
    return meta
