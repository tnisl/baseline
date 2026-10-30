"""One planner pass, then a caller restricted to that tool set; no repair nodes."""
from copy import deepcopy
import json
import math
import time

from agent import PLANNER_SYSTEM, CALL_SYSTEM
from common import allowed_specs, compact_history, dumps, result_succeeded
from inference_errors import ContextLimitError
from output_contracts import node_schema


class ContractError(ValueError):
    """Local structural failure, never represented as feedback from ToolEnv."""


def parse_object(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ContractError('duplicate_json_key')
            value[key] = item
        return value

    def constant(_):
        raise ContractError('non_finite_json')

    def real(raw):
        value = float(raw)
        if not math.isfinite(value):
            raise ContractError('non_finite_json')
        return value

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=real)
    if not isinstance(value, dict):
        raise ContractError('expected_object')
    return value


def string_set(value):
    return (isinstance(value, list) and all(isinstance(x, str) for x in value)
            and len(value) == len(set(value)))


def fingerprint(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def select_tools(output, specs):
    names = output.get('tools')
    if (set(output) != {'tools'} or not string_set(names)
            or not set(names) <= (set(specs) - {'sys.ask_missing_info'})):
        raise ContractError('invalid_tool_set')
    # This is a set, not a sequence. Canonicalize independently of model output order.
    return sorted(names)


def check_call(output, specs):
    kind = output.get('type')
    if kind == 'done' and set(output) == {'type'}:
        return
    if kind == 'call' and set(output) == {'type', 'name', 'arguments'}:
        name, arguments = output['name'], output['arguments']
        if isinstance(name, str) and name in specs and name != 'sys.ask_missing_info':
            properties = specs[name].get('parameters', {}).get('properties') or {}
            if isinstance(arguments, dict) and set(arguments) <= set(properties):
                return
    if kind == 'ask' and set(output) == {'type', 'tool_name', 'params'}:
        name, fields = output['tool_name'], output['params']
        if (isinstance(name, str) and name in specs and name != 'sys.ask_missing_info'
                and string_set(fields) and fields
                and set(fields) <= set(specs[name].get('parameters', {}).get('properties') or {})):
            return
    raise ContractError('invalid_caller_output')


class Agent:
    def __init__(self, llm, catalog, ToolEnv, root, *, max_tool_calls=10,
                 max_decisions=24, max_prompt_tokens=11500, max_new_tokens=512,
                 max_no_progress=2, save_node_inputs=False):
        if any(x < 1 for x in (max_tool_calls, max_decisions, max_prompt_tokens,
                               max_new_tokens, max_no_progress)):
            raise ValueError('Agent limits must be positive')
        self.llm, self.catalog, self.ToolEnv, self.root = llm, catalog, ToolEnv, root
        self.max_tool_calls, self.max_decisions = max_tool_calls, max_decisions
        self.max_prompt_tokens, self.max_new_tokens = max_prompt_tokens, max_new_tokens
        self.max_no_progress, self.save_node_inputs = max_no_progress, save_node_inputs

    def messages(self, row, transcript, specs, feedback, remaining, phase, selected_tools=None):
        payload = {
            'request': transcript,
            'context': {k: deepcopy(v) for k, v in row['context'].items() if k != 'installed_apps'},
            'history': compact_history(row.get('history') or []),
            'remaining_tool_calls': remaining,
        }
        if phase == 'plan':
            # The planner is invoked before any tool interaction, exactly once.
            payload['available_tools'] = [
                {'name': s['name'], 'kind': s.get('kind'), 'description': s.get('description', ''),
                 'input_fields': {k: {key: value for key, value in p.items()
                                      if key in ('type', 'description', 'enum', 'format', 'x-ref')}
                                  for k, p in (s.get('parameters', {}).get('properties') or {}).items()},
                 'required': s.get('parameters', {}).get('required', []),
                 'x-one-of': s.get('parameters', {}).get('x-one-of', []),
                 'returns': s.get('returns', '')}
                for s in specs.values() if s['name'] != 'sys.ask_missing_info'
            ]
            system = PLANNER_SYSTEM
        elif phase == 'call':
            payload['selected_tools'] = list(selected_tools or [])
            payload['available_tools'] = [deepcopy(specs[name]) for name in payload['selected_tools']]
            payload['env_feedback'] = deepcopy(feedback)
            system = CALL_SYSTEM
        else:
            raise ValueError('Unknown node phase')
        trimmed = False
        while True:
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': dumps(payload)}]
            if self.llm.count(messages) <= self.max_prompt_tokens:
                return messages, trimmed
            if payload['history']:
                payload['history'].pop(0)
            elif len(payload.get('env_feedback', [])) > 1:
                payload['env_feedback'].pop(0)
            else:
                raise ContextLimitError('Request, tool set/schema and latest environment response exceed prompt budget')
            trimmed = True

    def run(self, row, transcript):
        env = self.ToolEnv('train', row['id'], root=self.root)
        specs = {s['name']: s for s in allowed_specs(row, self.catalog)}
        feedback, trace, prompt_counts = [], [], []
        selected_tools = []
        generation_tokens = llm_latency = 0
        invalid = parse_errors = failures = no_progress_events = 0
        caller_calls = no_progress_streak = 0
        successful_writes, seen_reads = set(), set()
        trimmed = False
        stop = 'decision_limit'
        started = time.perf_counter()

        def generate(phase, event):
            nonlocal generation_tokens, llm_latency, trimmed
            messages, cut = self.messages(row, transcript, specs, feedback,
                                         self.max_tool_calls - len(env.get_tool_calls()), phase,
                                         selected_tools=selected_tools)
            trimmed |= cut
            options = {}
            if getattr(self.llm, 'structured_outputs', False):
                scoped = specs if phase == 'plan' else {name: specs[name] for name in selected_tools}
                options['output_schema'] = node_schema(phase, scoped)
            output = self.llm.complete(messages, max_new_tokens=self.max_new_tokens, **options)
            prompt_counts.append(output['prompt_tokens'])
            generation_tokens += output['generation_tokens']
            llm_latency += output['latency_s']
            record = deepcopy(output)
            if self.save_node_inputs:
                record['input_messages'] = deepcopy(messages)
            event['node_outputs'] = {phase: record}
            return parse_object(output['text'])

        def record_env(stage, name, arguments, response):
            feedback.append({'stage': stage, 'tool': name,
                             'arguments': deepcopy(arguments), 'response': deepcopy(response)})

        for decision in range(1, self.max_decisions + 1):
            phase = 'plan' if decision == 1 else 'call'
            event = {'decision': decision, 'event_type': phase}
            trace.append(event)
            try:
                if phase == 'plan':
                    output = generate(phase, event)
                    selected_tools = select_tools(output, specs)
                    event['action'] = deepcopy(output)
                    event['selected_tools'] = selected_tools.copy()
                    if not selected_tools:
                        stop = 'done'
                        break
                    continue
                caller_calls += 1
                action = generate(phase, event)
                scoped = {name: specs[name] for name in selected_tools}
                check_call(action, scoped)
                event['action'] = deepcopy(action)
                if action['type'] == 'done':
                    stop = 'done'
                    break
                if len(env.get_tool_calls()) >= self.max_tool_calls:
                    event['local_diagnostic'] = {'code': 'tool_budget_exhausted'}
                    stop = 'tool_limit'
                    break
                asking = action['type'] == 'ask'
                name = 'sys.ask_missing_info' if asking else action['name']
                arguments = ({'tool_name': action['tool_name'], 'params': action['params']}
                             if asking else action['arguments'])
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
            record_env('validate', name, arguments, validation)
            event['validation'] = deepcopy(validation)
            if not validation.get('ok'):
                invalid += 1
                stop = 'validation_failed'
                break
            signature = fingerprint([name, arguments])
            if signature in successful_writes:
                event['local_diagnostic'] = {'code': 'duplicate_successful_write'}
                stop = 'duplicate_write'
                break
            result = env.call(name, arguments)
            record_env('call', name, arguments, result)
            event['tool_result'] = deepcopy(result)
            event['tool_succeeded'] = result_succeeded(result)
            if asking:
                stop = 'ask_missing_info' if result_succeeded(result) else 'ask_failed'
                break
            if not result_succeeded(result):
                failures += 1
                stop = 'tool_failed'
                break
            if specs[name].get('kind') != 'read':
                successful_writes.add(signature)
                seen_reads.clear()
                no_progress_streak = 0
            else:
                fact = fingerprint([name, arguments, result])
                no_progress_streak = no_progress_streak + 1 if fact in seen_reads else 0
                seen_reads.add(fact)
            if no_progress_streak:
                no_progress_events += 1
            if no_progress_streak >= self.max_no_progress:
                stop = 'environment_no_progress'
                break
        return {
            'id': row['id'], 'transcript': transcript, 'tool_calls': env.get_tool_calls(),
            'state_hash': env.get_state_hash(), 'stop_reason': stop, 'decisions': len(trace),
            'selected_tools': selected_tools, 'planner_calls': 1, 'caller_calls': caller_calls,
            'llm_calls': len(prompt_counts), 'tool_iterations': len(env.get_tool_calls()),
            'invalid_schema_proposals': invalid, 'parse_errors': parse_errors,
            'tool_failures': failures, 'no_progress_events': no_progress_events,
            'prompt_tokens_each': prompt_counts, 'prompt_tokens_total': sum(prompt_counts),
            'prompt_tokens_max': max(prompt_counts, default=0),
            'generation_tokens_total': generation_tokens, 'llm_latency_s': llm_latency,
            'agent_latency_s': time.perf_counter() - started, 'context_trimmed': trimmed,
            'trace': trace,
        }
