"""Per-node JSON schemas. Semantic correctness is checked after generation."""

def obj(properties, required=None):
    return {'type': 'object', 'properties': properties,
            'required': list(properties) if required is None else required,
            'additionalProperties': False}


TEXT = {'type': 'string', 'minLength': 1}
STRINGS = {'type': 'array', 'items': TEXT}
GOAL = obj({'id': TEXT, 'task': TEXT, 'when': TEXT,
            'kind': {'type': 'string', 'enum': ['read', 'write']}})
GOALS = {'type': 'array', 'items': GOAL, 'minItems': 1}


def issue_schema(types, fields):
    return obj({'issue': obj({'type': {'type': 'string', 'enum': types},
               'fields': {'type': 'array', 'items': {'type': 'string', 'enum': fields} if fields else False}, 'reason': TEXT})})


def clean_schema(schema):
    allowed = {'type', 'enum', 'format', 'minimum', 'maximum', 'minItems',
               'maxItems', 'minLength', 'maxLength', 'pattern'}
    out = {k: v for k, v in schema.items() if k in allowed}
    if 'items' in schema:
        out['items'] = clean_schema(schema['items'])
    if 'properties' in schema:
        out['properties'] = {k: clean_schema(v) for k, v in schema['properties'].items()}
        out['required'] = schema.get('required', [])
        out['additionalProperties'] = False
    return out


def node_schema(phase, specs, selected_tool=None, selected_arguments=None, goals=None):
    if phase == 'select':
        call = obj({'type': {'const': 'call'}, 'name': {'enum': list(specs)},
                    'tool_reason': TEXT, 'goals': GOALS},
                   ['type', 'name', 'tool_reason'] + ([] if goals is not None else ['goals']))
        ask = obj({'type': {'const': 'ask'}, 'tool_name': {'enum': list(specs)},
                   'params': dict(STRINGS, minItems=1), 'tool_reason': TEXT, 'goals': GOALS},
                  ['type', 'tool_name', 'params', 'tool_reason'] + ([] if goals is not None else ['goals']))
        done = obj({'type': {'const': 'done'},
                    'completed_goals': {'type': 'object', 'additionalProperties': {'type': 'integer', 'minimum': 0}},
                    'skipped_goals': {'type': 'object', 'additionalProperties': TEXT}},
                   ['type', 'completed_goals'])
        revise = obj({'type': {'const': 'revise'}, 'goals': GOALS,
                      'removed_goals': {'type': 'object', 'additionalProperties': TEXT}, 'reason': TEXT})
        empty = obj({'type': {'const': 'done'}, 'goals': {'type': 'array', 'maxItems': 0}})
        return {'anyOf': [call, ask, done, revise] if goals is not None else [call, ask, empty]}
    properties = selected_tool.get('parameters', {}).get('properties') or {}
    if phase == 'select_arguments':
        fields = list(properties)
        return {'anyOf': [obj({'selected_arguments': {'type': 'array',
                               'items': {'type': 'string', 'enum': fields}}}),
                          issue_schema(['unsupported_tool'], fields)]}
    fields = list(selected_arguments or [])
    params = selected_tool.get('parameters') or {}
    mandatory = set(params.get('required') or [])
    for group in params.get('x-one-of') or []:
        mandatory.update(set(group) & set(fields))
    required = sorted(mandatory & set(fields))
    values = obj({k: clean_schema(properties[k]) for k in fields}, required)
    reasons = obj({k: TEXT for k in fields}, required)
    return {'anyOf': [obj({'arguments': values, 'value_reasons': reasons}),
                      issue_schema(['missing_data', 'ambiguous_target'], fields),
                      issue_schema(['argument_selection_error'], [k for k in properties if k not in fields])]}


# Compile with the same backend selected in QwenVLLM; JSON Schema validity alone
# does not imply a decoder implements every keyword. No silent schema stripping.
from functools import lru_cache
import json
from inference_errors import StructuredOutputError


@lru_cache(maxsize=512)
def _compile_guidance(serialized):
    try:
        from llguidance import JsonCompiler, LLMatcher
        grammar = JsonCompiler().compile(serialized)
        error = LLMatcher.validate_grammar(grammar)
        if error:
            raise ValueError(error)
    except Exception as exc:
        raise StructuredOutputError(
            "Structured output schema cannot compile with guidance: " + str(exc)
            + ". Fix the schema or explicitly use --no-structured-outputs for an A/B run."
        ) from exc


def validate_decoder_schema(schema):
    _compile_guidance(json.dumps(schema, ensure_ascii=False, sort_keys=True))


def preflight_schemas(catalog):
    """Compile catalog variants before loading model weights or opening results."""
    count = 0
    for goals in (None, []):
        validate_decoder_schema(node_schema('select', catalog, goals=goals))
        count += 1
    for spec in catalog.values():
        properties = spec.get('parameters', {}).get('properties') or {}
        validate_decoder_schema(node_schema('select_arguments', catalog, selected_tool=spec))
        count += 1
        required = set(spec.get('parameters', {}).get('required') or [])
        choices = [list(properties), sorted(required)]
        choices += [sorted(required | {field}) for field in properties if field not in required]
        for fields in choices:
            validate_decoder_schema(node_schema('fill', catalog, selected_tool=spec,
                                               selected_arguments=fields))
            count += 1
    return count
