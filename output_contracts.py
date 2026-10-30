"""Decoder schemas for control decisions only; task text remains valid tool data."""
import json
from functools import lru_cache
from inference_errors import StructuredOutputError


def obj(props, required=None):
    return {'type': 'object', 'properties': props,
            'required': list(props) if required is None else required, 'additionalProperties': False}


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


def node_schema(phase, specs, selected_tool=None, selected_arguments=None, **unused):
    if phase == 'select':
        names = [name for name in specs if name != 'sys.ask_missing_info']
        branches = [obj({'type': {'const': 'done'}})]
        if names:
            branches.append(obj({'type': {'const': 'call'}, 'name': {'enum': names}}))
        for name in names:
            fields = list(specs[name].get('parameters', {}).get('properties') or {})
            if fields:
                branches.append(obj({'type': {'const': 'ask'}, 'tool_name': {'const': name},
                                     'params': {'type': 'array', 'minItems': 1,
                                                'items': {'enum': fields}}}))
        return {'anyOf': branches}
    properties = selected_tool.get('parameters', {}).get('properties') or {}
    if phase == 'select_arguments':
        fields = [k for k in properties if k not in selected_tool.get('parameters', {}).get('required', [])]
        return obj({'selected_arguments': {'type': 'array', 'items': {'enum': fields} if fields else False}})
    return obj({'arguments': obj({k: clean_schema(properties[k]) for k in selected_arguments or []}, [])})


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
    validate_decoder_schema(node_schema('select', catalog))
    count = 1
    for spec in catalog.values():
        validate_decoder_schema(node_schema('select_arguments', catalog, selected_tool=spec))
        properties = spec.get('parameters', {}).get('properties') or {}
        for fields in [[], list(properties), spec.get('parameters', {}).get('required', [])]:
            validate_decoder_schema(node_schema('fill', catalog, selected_tool=spec, selected_arguments=fields))
            count += 1
        count += 1
    return count
