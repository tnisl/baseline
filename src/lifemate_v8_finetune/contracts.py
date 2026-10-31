"""Closed contracts for a fixed ordered plan, scoped caller and summarizer routing."""
from functools import lru_cache
import json

from .errors import StructuredOutputError

ANSWER_MAX_CHARS = 480


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


def summary_routes(step_status, remaining_plan_steps):
    if remaining_plan_steps < 0:
        raise ValueError('remaining_plan_steps must be nonnegative')
    if step_status == 'waiting_for_user':
        return ('wait',)
    if step_status in ('validation_failed', 'tool_failed', 'ask_failed'):
        if remaining_plan_steps == 0:
            raise ValueError('A failed step must remain pending')
        return ('retry', 'give_up')
    if step_status == 'blocked':
        return ('give_up',)
    if step_status in ('executed', 'skipped', 'empty_plan'):
        if step_status == 'empty_plan' and remaining_plan_steps:
            raise ValueError('empty_plan cannot have pending steps')
        return ('caller', 'give_up') if remaining_plan_steps else ('done', 'give_up')
    raise ValueError('Unknown step_status: ' + str(step_status))


def node_schema(phase, specs, *, selected_tool=None, max_plan_steps=24,
                step_status=None, remaining_plan_steps=0):
    names = sorted(name for name in specs if name != 'sys.ask_missing_info')
    if phase == 'plan':
        return obj({'tools': {'type': 'array', 'maxItems': max_plan_steps,
                              'items': {'type': 'string', 'enum': names} if names else False}})
    if phase == 'summarize':
        answer = {'type': 'string', 'minLength': 1, 'maxLength': ANSWER_MAX_CHARS,
                  'pattern': r'^[^\r\n]+$'}
        branches = []
        for route in summary_routes(step_status, remaining_plan_steps):
            fields = {'next': {'const': route}}
            if route in ('done', 'give_up', 'wait'):
                fields['answer'] = answer
            branches.append(obj(fields))
        return {'anyOf': branches}
    if phase != 'call':
        raise ValueError('Unknown node phase: ' + str(phase))
    if selected_tool not in names:
        raise ValueError('Caller schema requires one available selected_tool')
    spec = specs[selected_tool]
    branches = [obj({'type': {'const': 'call'}, 'arguments': argument_schema(spec)}),
                obj({'type': {'const': 'skip'}}), obj({'type': {'const': 'blocked'}})]
    fields = list(spec.get('parameters', {}).get('properties') or {})
    if fields and 'sys.ask_missing_info' in specs:
        branches.append(obj({'type': {'const': 'ask'},
                             'params': {'type': 'array', 'minItems': 1,
                                        'items': {'type': 'string', 'enum': fields}}}))
    return {'anyOf': branches}


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
    variants = [node_schema('plan', catalog), node_schema('plan', {})]
    for name, spec in catalog.items():
        if name == 'sys.ask_missing_info':
            continue
        scoped = {name: spec}
        if 'sys.ask_missing_info' in catalog:
            scoped['sys.ask_missing_info'] = catalog['sys.ask_missing_info']
        variants.append(node_schema('call', scoped, selected_tool=name))
    for status, remaining in [('executed', 1), ('executed', 0), ('skipped', 1), ('skipped', 0),
                              ('validation_failed', 1), ('tool_failed', 1), ('ask_failed', 1),
                              ('blocked', 1), ('empty_plan', 0), ('waiting_for_user', 1)]:
        variants.append(node_schema('summarize', catalog, step_status=status, remaining_plan_steps=remaining))
    for schema in variants:
        validate_decoder_schema(schema)
    return len(variants)
