"""Step-wise planning and calling; only ToolEnv responses become observations."""
from copy import deepcopy
import json
import math
import time

from .common import allowed_specs, compact_history, dumps, result_succeeded
from .contracts import INSTRUCTION_MAX_CHARS, node_schema
from .errors import ContextLimitError
from .prompts import PLANNER_SYSTEM, CALL_SYSTEM


class ContractError(ValueError):
    """Local structural failure, never presented as environment feedback."""


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


def check_plan(output, specs):
    next_role = output.get('next')
    if next_role in ('done', 'give_up') and set(output) == {'next'}:
        return
    if next_role == 'caller' and set(output) == {'next', 'instruction'}:
        instruction = output['instruction']
        if (set(specs) - {'sys.ask_missing_info'} and isinstance(instruction, str)
                and instruction.strip() and len(instruction) <= INSTRUCTION_MAX_CHARS
                and '\n' not in instruction and '\r' not in instruction):
            return
    raise ContractError('invalid_planner_output')


def check_call(output, specs):
    if set(output) != {'name', 'arguments'}:
        raise ContractError('invalid_caller_output')
    name, args = output['name'], output['arguments']
    if not isinstance(name, str) or name not in specs or not isinstance(args, dict):
        raise ContractError('invalid_caller_output')
    properties = specs[name].get('parameters', {}).get('properties') or {}
    if not set(args) <= set(properties):
        raise ContractError('unknown_argument')
    if name == 'sys.ask_missing_info':
        target, fields = args.get('tool_name'), args.get('params')
        if (set(args) != {'tool_name', 'params'} or not isinstance(target, str)
                or target == name or target not in specs or not string_set(fields) or not fields
                or not set(fields) <= set(specs[target].get('parameters', {}).get('properties') or {})):
            raise ContractError('invalid_ask_target')


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

    def messages(self, row, transcript, specs, feedback, remaining, phase, instruction=None):
        payload = {
            'request': transcript,
            'context': {k: deepcopy(v) for k, v in row['context'].items() if k != 'installed_apps'},
            'history': compact_history(row.get('history') or []),
            'env_feedback': deepcopy(feedback), 'remaining_tool_calls': remaining,
        }
        if phase == 'plan':
            payload['available_tools'] = [
                {'name': s['name'], 'kind': s.get('kind'), 'description': s.get('description', ''),
                 'input_fields': {k: {key: value for key, value in p.items()
                                      if key in ('type', 'description', 'enum', 'format', 'x-ref')}
                                  for k, p in (s.get('parameters', {}).get('properties') or {}).items()},
                 'required': s.get('parameters', {}).get('required', []),
                 'x-one-of': s.get('parameters', {}).get('x-one-of', []),
                 'returns': s.get('returns', '')} for s in specs.values()
            ]
            system = PLANNER_SYSTEM
        elif phase == 'call':
            payload['planner_instruction'] = instruction
            payload['available_tools'] = deepcopy(list(specs.values()))
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
            elif len(payload['env_feedback']) > 1:
                payload['env_feedback'].pop(0)
            else:
                raise ContextLimitError('Request, tool schemas, current instruction and latest observation exceed budget')
            trimmed = True

    def run(self, row, transcript):
        env = self.ToolEnv('train', row['id'], root=self.root)
        specs = {s['name']: s for s in allowed_specs(row, self.catalog)}
        feedback, trace, prompt_counts = [], [], []
        generation_tokens = llm_latency = 0
        invalid = parse_errors = failures = no_progress_events = 0
        planner_calls = caller_calls = no_progress_streak = 0
        successful_writes, seen_reads = set(), set()
        trimmed = False
        stop = 'decision_limit'
        started = time.perf_counter()

        def generate(phase, event, instruction=None):
            nonlocal generation_tokens, llm_latency, trimmed, planner_calls, caller_calls
            messages, cut = self.messages(row, transcript, specs, feedback,
                                         self.max_tool_calls - len(env.get_tool_calls()), phase,
                                         instruction=instruction)
            trimmed |= cut
            options = {}
            if getattr(self.llm, 'structured_outputs', False):
                options['output_schema'] = node_schema(phase, specs)
            if phase == 'plan':
                planner_calls += 1
            else:
                caller_calls += 1
            output = self.llm.complete(messages, max_new_tokens=self.max_new_tokens, **options)
            prompt_counts.append(output['prompt_tokens'])
            generation_tokens += output['generation_tokens']
            llm_latency += output['latency_s']
            record = deepcopy(output)
            if self.save_node_inputs:
                record['input_messages'] = deepcopy(messages)
            event.setdefault('node_outputs', {})[phase] = record
            return parse_object(output['text'])

        def record_env(stage, name, arguments, response):
            feedback.append({'stage': stage, 'tool': name,
                             'arguments': deepcopy(arguments), 'response': deepcopy(response)})

        for decision in range(1, self.max_decisions + 1):
            event = {'decision': decision}
            trace.append(event)
            phase = 'plan'
            try:
                plan = generate(phase, event)
                check_plan(plan, specs)
                event['planner_output'] = deepcopy(plan)
                if plan['next'] in ('done', 'give_up'):
                    stop = plan['next']
                    break
                if len(env.get_tool_calls()) >= self.max_tool_calls:
                    event['local_diagnostic'] = {'code': 'tool_budget_exhausted'}
                    stop = 'tool_limit'
                    break
                phase = 'call'
                action = generate(phase, event, instruction=plan['instruction'])
                check_call(action, specs)
                event['action'] = deepcopy(action)
                name, arguments = action['name'], action['arguments']
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
            if name == 'sys.ask_missing_info':
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
            'planner_calls': planner_calls, 'caller_calls': caller_calls,
            'llm_calls': len(prompt_counts), 'tool_iterations': len(env.get_tool_calls()),
            'invalid_schema_proposals': invalid, 'parse_errors': parse_errors,
            'tool_failures': failures, 'no_progress_events': no_progress_events,
            'prompt_tokens_each': prompt_counts, 'prompt_tokens_total': sum(prompt_counts),
            'prompt_tokens_max': max(prompt_counts, default=0),
            'generation_tokens_total': generation_tokens, 'llm_latency_s': llm_latency,
            'agent_latency_s': time.perf_counter() - started, 'context_trimmed': trimmed,
            'trace': trace,
        }
