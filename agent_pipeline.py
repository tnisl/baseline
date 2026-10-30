"""Three selectors with environment-only feedback; all session state is local."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import time

from agent import TOOL_SYSTEM, ARG_SYSTEM, VAL_SYSTEM
from common import allowed_specs, compact_history, dumps, result_succeeded
from inference_errors import ContextLimitError
from output_contracts import node_schema


class ContractError(ValueError):
    """Local control-contract failure, logged but never fed back to the model."""


def parse_object(text):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ContractError('duplicate_json_key')
            obj[key] = value
        return obj

    def constant(_):
        raise ContractError('non_finite_json')

    def real(raw):
        number = float(raw)
        if not math.isfinite(number):
            raise ContractError('non_finite_json')
        return number

    obj = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=real)
    if not isinstance(obj, dict):
        raise ContractError('expected_object')
    return obj


def string_list(value):
    return (isinstance(value, list) and all(isinstance(x, str) for x in value)
            and len(set(value)) == len(value))


def fingerprint(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def check_action(action, specs):
    kind = action.get('type')
    if kind == 'done' and set(action) == {'type'}:
        return
    if kind == 'call' and set(action) == {'type', 'name'}:
        if isinstance(action['name'], str) and action['name'] in specs and action['name'] != 'sys.ask_missing_info':
            return
    if kind == 'ask' and set(action) == {'type', 'tool_name', 'params'}:
        name, fields = action['tool_name'], action['params']
        if (isinstance(name, str) and name in specs and name != 'sys.ask_missing_info'
                and string_list(fields) and fields
                and set(fields) <= set(specs[name].get('parameters', {}).get('properties') or {})):
            return
    raise ContractError('invalid_tool_selection')


def select_fields(output, spec):
    params = spec.get('parameters') or {}
    required = params.get('required') or []
    extras = output.get('selected_arguments')
    selectable = set(params.get('properties') or {}) - set(required)
    if set(output) != {'selected_arguments'} or not string_list(extras) or not set(extras) <= selectable:
        raise ContractError('invalid_argument_selection')
    return list(dict.fromkeys([*required, *extras]))


def check_values(output, fields):
    if (set(output) != {'arguments'} or not isinstance(output['arguments'], dict)
            or not set(output['arguments']) <= set(fields)):
        raise ContractError('invalid_value_selection')
    return output['arguments']


class Agent:
    def __init__(self, llm, catalog, ToolEnv, root, *, max_tool_calls=10,
                 max_decisions=24, max_repairs=3, max_prompt_tokens=11500,
                 max_new_tokens=512, max_no_progress=2):
        limits = (max_tool_calls, max_decisions, max_repairs, max_prompt_tokens,
                  max_new_tokens, max_no_progress)
        if any(x < 1 for x in limits):
            raise ValueError('Agent limits must be positive')
        self.llm, self.catalog, self.ToolEnv, self.root = llm, catalog, ToolEnv, root
        self.max_tool_calls, self.max_decisions = max_tool_calls, max_decisions
        self.max_repairs, self.max_no_progress = max_repairs, max_no_progress
        self.max_prompt_tokens, self.max_new_tokens = max_prompt_tokens, max_new_tokens

    def messages(self, row, transcript, specs, feedback, remaining, phase,
                 selected_tool=None, selected_arguments=None):
        payload = {
            'request': transcript,
            'context': {k: deepcopy(v) for k, v in row['context'].items() if k != 'installed_apps'},
            'history': compact_history(row.get('history') or []),
            'env_feedback': deepcopy(feedback), 'remaining_tool_calls': remaining,
        }
        if phase == 'select':
            # Preserve property descriptions/enum, but avoid repeating the full JSON schema here.
            payload['available_tools'] = [
                {'name': s['name'], 'kind': s.get('kind'), 'description': s.get('description', ''),
                 'input_fields': {k: {key: value for key, value in p.items()
                                      if key in ('type', 'description', 'enum', 'format')}
                                  for k, p in (s.get('parameters', {}).get('properties') or {}).items()},
                 'required': s.get('parameters', {}).get('required', []),
                 'x-one-of': s.get('parameters', {}).get('x-one-of', []),
                 'returns': s.get('returns', '')}
                for s in specs.values() if s['name'] != 'sys.ask_missing_info'
            ]
            system = TOOL_SYSTEM
        else:
            payload['selected_tool'] = selected_tool
            if phase == 'select_arguments':
                payload['required_arguments'] = selected_tool.get('parameters', {}).get('required', [])
                system = ARG_SYSTEM
            else:
                payload['selected_arguments'] = selected_arguments
                system = VAL_SYSTEM
        trimmed = False
        while True:
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': dumps(payload)}]
            if self.llm.count(messages) <= self.max_prompt_tokens:
                return messages, trimmed
            # Never drop the current request, schema or newest environment result.
            if payload['history']:
                payload['history'].pop(0)
            elif len(payload['env_feedback']) > 1:
                payload['env_feedback'].pop(0)
            else:
                raise ContextLimitError('Current request, schema and latest environment feedback exceed prompt budget')
            trimmed = True

    def run(self, row, transcript):
        env = self.ToolEnv('train', row['id'], root=self.root)
        specs = {s['name']: s for s in allowed_specs(row, self.catalog)}
        feedback, trace, prompt_counts = [], [], []
        generation_tokens = llm_latency = 0
        repairs = invalid = parse_errors = failures = no_progress_events = 0
        repair_streak = no_progress_streak = 0
        seen_reads, seen_invalid, successful_writes = set(), set(), set()
        trimmed = False
        stop = 'decision_limit'
        started = time.perf_counter()

        def generate(phase, event, **kwargs):
            nonlocal generation_tokens, llm_latency, trimmed
            messages, cut = self.messages(row, transcript, specs, feedback,
                                         self.max_tool_calls - len(env.get_tool_calls()), phase, **kwargs)
            trimmed |= cut
            options = {}
            if getattr(self.llm, 'structured_outputs', False):
                options['output_schema'] = node_schema(phase, specs, **kwargs)
            result = self.llm.complete(messages, max_new_tokens=self.max_new_tokens, **options)
            prompt_counts.append(result['prompt_tokens'])
            generation_tokens += result['generation_tokens']
            llm_latency += result['latency_s']
            event.setdefault('node_outputs', {})[phase] = result
            return parse_object(result['text'])

        def record(stage, name, arguments, response):
            # Only these two call sites produce feedback. No model-authored diagnostics.
            feedback.append({'stage': stage, 'tool': name,
                             'arguments': deepcopy(arguments), 'response': deepcopy(response)})

        for decision in range(1, self.max_decisions + 1):
            event = {'decision': decision}
            trace.append(event)
            phase = 'select'
            try:
                action = generate(phase, event)
                check_action(action, specs)
                event['action'] = deepcopy(action)
                event['tool_selection'] = deepcopy(action)
                if action['type'] == 'done':
                    stop = 'done'
                    break
                if len(env.get_tool_calls()) >= self.max_tool_calls:
                    stop = 'tool_limit'
                    break
                if action['type'] == 'ask':
                    name = 'sys.ask_missing_info'
                    arguments = {'tool_name': action['tool_name'], 'params': action['params']}
                else:
                    name = action['name']
                    spec = specs[name]
                    params = spec.get('parameters') or {}
                    phase = 'select_arguments'
                    if set(params.get('properties') or {}) - set(params.get('required') or []):
                        chosen = generate(phase, event, selected_tool=spec)
                        event['argument_selection_output'] = deepcopy(chosen)
                    else:
                        chosen = {'selected_arguments': []}
                        event['argument_selector_skipped'] = True
                    fields = select_fields(chosen, spec)
                    event['selected_arguments'] = fields
                    phase = 'fill'
                    if fields:
                        values = generate(phase, event, selected_tool=spec, selected_arguments=fields)
                        event['argument_output'] = deepcopy(values)
                    else:
                        values = {'arguments': {}}
                        event['value_selector_skipped'] = True
                    arguments = check_values(values, fields)
                    event['action'].update(selected_arguments=fields, arguments=deepcopy(arguments))
            except json.JSONDecodeError:
                parse_errors += 1
                event['local_diagnostic'] = {'code': 'invalid_json', 'stage': phase}
                stop = 'invalid_model_output'
                break
            except ContractError as exc:
                invalid += 1
                event['local_diagnostic'] = {'code': str(exc), 'stage': phase}
                stop = 'invalid_model_output'
                break

            validation = env.validate(name, arguments)
            record('validate', name, arguments, validation)
            event['validation'] = deepcopy(validation)
            if not validation.get('ok'):
                invalid += 1
                repairs += 1
                repair_streak += 1
                signature = fingerprint([name, arguments, validation])
                no_progress_streak = no_progress_streak + 1 if signature in seen_invalid else 0
                seen_invalid.add(signature)
                if no_progress_streak:
                    no_progress_events += 1
                if repair_streak >= self.max_repairs or no_progress_streak >= self.max_no_progress:
                    stop = 'validation_no_progress'
                    break
                continue

            signature = fingerprint([name, arguments])
            if signature in successful_writes:
                # A local stop is never represented as an environment rejection.
                event['local_diagnostic'] = {'code': 'duplicate_successful_write'}
                stop = 'duplicate_write'
                break
            result = env.call(name, arguments)
            record('call', name, arguments, result)
            event['tool_result'] = deepcopy(result)
            event['tool_succeeded'] = result_succeeded(result)
            if action['type'] == 'ask':
                stop = 'ask_missing_info' if result_succeeded(result) else 'ask_failed'
                break
            if result_succeeded(result):
                repair_streak = 0
                if specs[name].get('kind') != 'read':
                    successful_writes.add(signature)
                    # Writes may change read responses, so permit reads again after a write.
                    seen_reads.clear()
                    no_progress_streak = 0
                else:
                    fact = fingerprint([name, arguments, result])
                    no_progress_streak = no_progress_streak + 1 if fact in seen_reads else 0
                    seen_reads.add(fact)
            else:
                failures += 1
                repair_streak += 1
                repairs += 1
                no_progress_streak += 1
            if no_progress_streak:
                no_progress_events += 1
            if repair_streak >= self.max_repairs or no_progress_streak >= self.max_no_progress:
                stop = 'environment_no_progress'
                break
        return {
            'id': row['id'], 'transcript': transcript, 'tool_calls': env.get_tool_calls(),
            'state_hash': env.get_state_hash(), 'stop_reason': stop, 'decisions': len(trace),
            'tool_iterations': len(env.get_tool_calls()), 'repair_attempts': repairs,
            'invalid_schema_proposals': invalid, 'parse_errors': parse_errors,
            'tool_failures': failures, 'no_progress_events': no_progress_events,
            'prompt_tokens_each': prompt_counts, 'prompt_tokens_total': sum(prompt_counts),
            'prompt_tokens_max': max(prompt_counts, default=0),
            'generation_tokens_total': generation_tokens, 'llm_latency_s': llm_latency,
            'agent_latency_s': time.perf_counter() - started, 'context_trimmed': trimmed,
            'trace': trace,
        }
