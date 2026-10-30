"""Per-tool call schemas, compiled with the same grammar backend as inference."""
from functools import lru_cache
import json

from .errors import StructuredOutputError

INSTRUCTION_MAX_CHARS = 240


def obj(properties, required=None):
    return {'type': 'object', 'properties': properties,
            'required': list(properties) if required is None else required,
            'additionalProperties': False}


def clean_schema(schema):
    allowed = {'type', 'enum', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
               'minItems', 'maxItems', 'minLength', 'maxLength', 'pattern'}
    out = {k: v for k, v in schema.items() if k in allowed}
    if schema.get('format') == 'time':
        out['pattern'] = r'^([01][0-9]|2[0-3]):[0-5][0-9]$'
    elif schema.get('format') in ('date', 'date-time'):
        out['format'] = schema['format']
    if 'items' in schema:
        out['items'] = clean_schema(schema['items'])
    if 'properties' in schema:
        out.update(properties={k: clean_schema(v) for k, v in schema['properties'].items()},
                   required=schema.get('required', []), additionalProperties=False)
    return out


def argument_schema(spec):
    params = spec.get('parameters') or {}
    properties = {k: clean_schema(v) for k, v in (params.get('properties') or {}).items()}
    required = params.get('required') or []
    groups = params.get('x-one-of') or []
    if groups:
        return {'anyOf': [obj(properties, list(dict.fromkeys([*required, *group]))) for group in groups]}
    return obj(properties, required)


def node_schema(phase, specs):
    names = sorted(name for name in specs if name != 'sys.ask_missing_info')
    if phase == 'plan':
        branches = [obj({'next': {'const': 'done'}}), obj({'next': {'const': 'give_up'}})]
        if names:
            branches.append(obj({'next': {'const': 'caller'}, 'instruction': {
                'type': 'string', 'minLength': 1, 'maxLength': INSTRUCTION_MAX_CHARS,
                'pattern': r'^[^\r\n]+$'}}))
        return {'anyOf': branches}
    if phase != 'call':
        raise ValueError('Unknown node phase: ' + str(phase))
    branches = []
    for name in names:
        spec = specs[name]
        branches.append(obj({'name': {'const': name},
                             'arguments': argument_schema(spec)}))
        fields = list(spec.get('parameters', {}).get('properties') or {})
        if fields and 'sys.ask_missing_info' in specs:
            branches.append(obj({'name': {'const': 'sys.ask_missing_info'}, 'arguments': obj({
                'tool_name': {'const': name}, 'params': {'type': 'array', 'minItems': 1,
                                                       'items': {'type': 'string', 'enum': fields}}})}))
    return {'anyOf': branches} if branches else False


@lru_cache(maxsize=512)
def _compile(serialized):
    try:
        from llguidance import JsonCompiler, LLMatcher
        error = LLMatcher.validate_grammar(JsonCompiler().compile(serialized))
        if error:
            raise ValueError(error)
    except Exception as exc:
        raise StructuredOutputError('guidance schema compilation failed: ' + str(exc)) from exc


def validate_decoder_schema(schema):
    _compile(json.dumps(schema, ensure_ascii=False, sort_keys=True))


def preflight_schemas(catalog):
    validate_decoder_schema(node_schema('plan', catalog))
    validate_decoder_schema(node_schema('call', catalog))
    count = 2
    for name, spec in catalog.items():
        if name == 'sys.ask_missing_info':
            continue
        subset = {name: spec}
        if 'sys.ask_missing_info' in catalog:
            subset['sys.ask_missing_info'] = catalog['sys.ask_missing_info']
        validate_decoder_schema(node_schema('call', subset))
        count += 1
    return count
