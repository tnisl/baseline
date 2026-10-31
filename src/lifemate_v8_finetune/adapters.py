"""Explicit role routing for independent adapters; model outputs never choose adapters."""
from pathlib import Path
import hashlib,json
from .prompts import PLANNER_SYSTEM,CALL_SYSTEM,SUMMARIZER_SYSTEM

ROLE_SYSTEMS={'planner':PLANNER_SYSTEM,'caller':CALL_SYSTEM,'summarizer':SUMMARIZER_SYSTEM}


def role_for_messages(messages):
    if not messages or messages[0].get('role')!='system':raise ValueError('Missing role system prompt')
    for role,system in ROLE_SYSTEMS.items():
        if messages[0].get('content')==system:return role
    raise ValueError('No adapter matches this exact system prompt')


def resolve_adapters(root,base_model):
    if root is None:return {}
    root=Path(root).expanduser().resolve()
    paths={}
    for role,system in ROLE_SYSTEMS.items():
        path=root/role
        cfg_path=path/'adapter_config.json';meta_path=path/'role_metadata.json'
        if not cfg_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f'Missing adapter/config metadata for {role}: {path}')
        cfg=json.loads(cfg_path.read_text());meta=json.loads(meta_path.read_text())
        if not any((path/name).is_file() for name in ('adapter_model.safetensors','adapter_model.bin')):
            raise FileNotFoundError(f'Missing adapter weights: {path}')
        if meta.get('role')!=role or meta.get('base_model')!=base_model:
            raise ValueError('Adapter role/base model mismatch: '+str(path))
        if meta.get('prompt_sha256')!=hashlib.sha256(system.encode()).hexdigest():
            raise ValueError('Adapter was trained with a different prompt: '+str(path))
        if meta.get('global_finetune') is not False or cfg.get('peft_type')!='LORA':
            raise ValueError('Expected a local-only LoRA adapter: '+str(path))
        paths[role]=str(path)
    return paths


def adapter_rank(paths):
    return max((int(json.loads((Path(p)/'adapter_config.json').read_text())['r']) for p in paths.values()),default=16)
