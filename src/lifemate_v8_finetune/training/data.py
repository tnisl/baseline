"""Role-local teacher trajectories and verified counterfactual negatives.

Gold labels guide targets only; prompts are assembled by the inference Agent.
Counterfactual calls run in independent ToolEnv sessions, never state files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import random

from ..common import (allowed_specs, dumps, find_root, import_toolenv, leaves,
                      load_catalog, load_train, result_succeeded, write_json)
from ..pipeline import Agent, check_call, check_summary, select_plan
from ..contracts import summary_routes
from ..prompts import PLANNER_SYSTEM, CALL_SYSTEM, SUMMARIZER_SYSTEM

ROLES = ('planner', 'caller', 'summarizer')
PHASES = {'planner': 'plan', 'caller': 'call', 'summarizer': 'summarize'}
TERMINAL_TEXT = {
    'done': 'Đã hoàn tất các thao tác được xác nhận.',
    'give_up': 'Chưa thể hoàn thành đúng yêu cầu với kế hoạch hiện tại.',
    'wait': 'Cần bổ sung thông tin còn thiếu để tiếp tục.',
}


class NoTokenLimit:
    def count(self, messages):
        return 0


def stable_hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def prompt_hashes():
    return {role: hashlib.sha256(text.encode()).hexdigest() for role, text in zip(
        ROLES, (PLANNER_SYSTEM, CALL_SYSTEM, SUMMARIZER_SYSTEM))}


def summary_target(route):
    value = {'next': route}
    if route in TERMINAL_TEXT:
        value['answer'] = TERMINAL_TEXT[route]
    return value


def split_ids(rows, validation_fraction, seed, holdout_ids=()):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be in (0, 1)')
    held = set(holdout_ids)
    known = {r['id'] for r in rows}
    if held - known:
        raise ValueError('Unknown holdout IDs: ' + ', '.join(sorted(held - known)[:5]))
    eligible = sorted(known - held)
    if len(eligible) < 2:
        raise ValueError('At least two non-holdout episodes are required')
    random.Random(seed).shuffle(eligible)
    n_valid = max(1, min(len(eligible)-1, round(len(eligible) * validation_fraction)))
    valid = set(eligible[:n_valid])
    return {sid: 'holdout' if sid in held else 'validation' if sid in valid else 'train' for sid in known}


def get_at(value, path):
    for key in path:
        value = value[key]
    return value


def set_at(value, path, replacement):
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


def scalar_paths(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from scalar_paths(item, (*path, key))
    elif isinstance(value, list):
        for key, item in enumerate(value):
            yield from scalar_paths(item, (*path, key))
    else:
        yield path, value


def mutation_candidates(action, spec, evidence, seed, maximum=12):
    """Plausible wrong values, not copied bad targets or arbitrary control prose."""
    rng = random.Random(seed)
    found = []
    known = defaultdict(set)
    for _, item, record in leaves(evidence):
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                known[key].add(value)
    properties = spec.get('parameters', {}).get('properties') or {}
    for path, value in scalar_paths(action.get('arguments') or {}):
        field = path[-1]
        if not isinstance(field, str):
            continue
        values = []
        schema = properties.get(field, {})
        if isinstance(value, bool):
            values = [not value]
        elif isinstance(value, (int, float)):
            if field in ('amount', 'quantity', 'minutes', 'steps', 'volume', 'temperature', 'brightness', 'delay_minutes'):
                values = [value + (1000 if field in ('amount', 'steps') else 1)]
        elif isinstance(value, str):
            values.extend(v for v in schema.get('enum', []) if v != value)
            if field == 'id' or field.endswith('_id'):
                values.extend(sorted(v for v in known['id'] | known[field] if isinstance(v, str) and v != value))
            if field in ('time', 'start', 'end', 'due', 'schedule_time', 'date'):
                try:
                    parsed = datetime.fromisoformat(value)
                    values.append((parsed + timedelta(days=1)).isoformat())
                except ValueError:
                    try:
                        parsed_time = datetime.strptime(value, '%H:%M')
                        values.append((parsed_time + timedelta(hours=1)).strftime('%H:%M'))
                    except ValueError:
                        pass
        rng.shuffle(values)
        for alternative in values[:3]:
            if alternative == value:
                continue
            candidate = deepcopy(action)
            set_at(candidate['arguments'], path, alternative)
            found.append((candidate, 'value:' + '.'.join(map(str, path))))
    rng.shuffle(found)
    seen = set()
    for candidate, kind in found:
        key = stable_hash(candidate)
        if key not in seen:
            seen.add(key)
            yield candidate, kind
            if len(seen) >= maximum:
                return


class Builder:
    def __init__(self, root, *, seed=42, max_plan_steps=24, max_tool_calls=10,
                 max_decisions=32, negatives_per_example=2, counterfactuals_per_case=2):
        self.root = Path(root)
        self.catalog = load_catalog(root)
        self.ToolEnv = import_toolenv(root)
        self.agent = Agent(NoTokenLimit(), self.catalog, self.ToolEnv, root,
                           max_plan_steps=max_plan_steps, max_tool_calls=max_tool_calls,
                           max_decisions=max_decisions)
        self.seed, self.negatives = seed, negatives_per_example
        self.counterfactuals = counterfactuals_per_case
        self.issues = []

    def environment(self, row, prefix=()):
        env = self.ToolEnv('train', row['id'], root=self.root)
        for action in prefix:
            result = env.call(action['name'], action['arguments'])
            if not result_succeeded(result):
                raise ValueError('Invalid teacher prefix')
        return env

    @staticmethod
    def feedback(action, validation, result=None):
        facts = [{'stage': 'validate', 'tool': action['name'], 'arguments': deepcopy(action['arguments']),
                  'response': deepcopy(validation)}]
        if result is not None:
            facts.append({'stage': 'call', 'tool': action['name'], 'arguments': deepcopy(action['arguments']),
                          'response': deepcopy(result)})
        return facts

    def messages(self, row, specs, phase, feedback=(), plan=(), cursor=0, results=(), current=None,
                 status=None, calls=0, review=None):
        messages, _ = self.agent.messages(row, row['audio_text'], specs, list(feedback),
            self.agent.max_tool_calls-calls, phase, tool_plan=list(plan), step_index=cursor,
            step_results=list(results), current_step=current, step_status=status, previous_review=review)
        return messages

    def example(self, row, role, messages, target, source, rejected=(), **meta):
        target_text = dumps(target)
        rejected = list(dict.fromkeys(dumps(x) for x in rejected if dumps(x) != target_text))[:self.negatives]
        return {'case_id': row['id'], 'role': role, 'messages': deepcopy(messages),
                'target': target_text, 'rejected': rejected, 'source': source,
                'loss_scope': 'next' if role=='summarizer' else 'completion',
                'metadata': meta}

    def summary_example(self, row, specs, plan, cursor, results, feedback, current, status, calls,
                        chosen, source, **meta):
        check_summary(summary_target(chosen), status, len(plan)-cursor)
        rejected = [summary_target(route) for route in summary_routes(status, len(plan)-cursor) if route!=chosen]
        return self.example(row, 'summarizer', self.messages(row, specs, 'summarize', feedback,
            plan, cursor, results, current, status, calls), summary_target(chosen), source, rejected, **meta)

    def build_case(self, row):
        specs = {s['name']:s for s in allowed_specs(row,self.catalog)}
        gold = deepcopy(row['label']['tool_calls'])
        # A gold ask becomes a pending slot for its business target, not a planner sys tool.
        plan = [a['arguments']['tool_name'] if a['name']=='sys.ask_missing_info' else a['name'] for a in gold]
        select_plan({'tools':plan}, specs, self.agent.max_plan_steps)
        if len(gold)>self.agent.max_tool_calls or len(plan)+1>self.agent.max_decisions:
            raise ValueError('Teacher trajectory exceeds inference budget')
        examples = [self.example(row,'planner',self.messages(row,specs,'plan'),{'tools':plan},'gold_plan')]
        if not gold:
            # Empty tool labels do not distinguish smalltalk/no-op from unsupported requests.
            return examples, {'empty_gold_summary_unlabeled':True}
        env = self.environment(row)
        feedback, results, prefix = [], [], []
        case_counterfactuals = 0
        snapshots = []
        for index, action in enumerate(gold):
            name = action['name']
            selected = plan[index]
            asking = name=='sys.ask_missing_info'
            if not asking and name != selected:
                raise ValueError('Teacher tool order mismatch')
            caller_target = ({'type':'ask','params':action['arguments']['params']} if asking
                             else {'type':'call','arguments':action['arguments']})
            check_call(caller_target,specs,selected)
            before_messages = self.messages(row,specs,'call',feedback,plan,index,results,calls=len(prefix))
            validation = env.validate(name,action['arguments'])
            if not validation.get('ok'):
                raise ValueError('Invalid gold validation: '+str(validation))
            # A preference rejection is never used as a positive assistant target.
            candidates=[]
            if not asking:
                evidence={'history':row.get('history') or [], 'feedback':feedback, 'context':{
                    k:v for k,v in row['context'].items() if k!='installed_apps'}}
                candidate_seed=int(stable_hash([self.seed,row['id'],index])[:12],16)
                candidates=list(mutation_candidates(action,specs[selected],evidence,candidate_seed))
                rejected=[{'type':'call','arguments':a['arguments']} for a,_ in candidates[:self.negatives]]
                rejected.append({'type':'skip'})
            else:
                rejected=[{'type':'blocked'}]
            examples.append(self.example(row,'caller',before_messages,caller_target,'gold_call',rejected,step_index=index))
            result=env.call(name,action['arguments'])
            if not result_succeeded(result):
                raise ValueError('Unsuccessful gold tool call: '+str(result))
            new_feedback=feedback+self.feedback(action,validation,result)
            new_results=deepcopy(results)
            cursor=index
            if not asking:
                new_results.append({'step_index':index,'tool':selected,'status':'executed'})
                cursor+=1
            status='waiting_for_user' if asking else 'executed'
            current={'step_index':index,'selected_tool':selected,'caller_output':caller_target,'action':action}
            route='wait' if asking else 'done' if cursor==len(plan) else 'caller'
            examples.append(self.summary_example(row,specs,plan,cursor,new_results,new_feedback,current,status,
                len(prefix)+1,route,'gold_summary',step_index=index))
            gold_after_hash=env.get_state_hash()
            # Build error/recovery inputs only from real, independent env responses.
            for mutated,kind in candidates:
                if case_counterfactuals>=self.counterfactuals:
                    break
                trial=self.environment(row,prefix)
                prefix_hash=trial.get_state_hash()
                wrong_validation=trial.validate(mutated['name'],mutated['arguments'])
                wrong_result=None
                if wrong_validation.get('ok'):
                    wrong_result=trial.call(mutated['name'],mutated['arguments'])
                failed=not wrong_validation.get('ok') or not result_succeeded(wrong_result)
                wrong_output={'type':'call','arguments':mutated['arguments']}
                wrong_current={'step_index':index,'selected_tool':selected,'caller_output':wrong_output,'action':mutated}
                wrong_feedback=feedback+self.feedback(mutated,wrong_validation,wrong_result)
                wrong_calls=len(prefix)+int(wrong_result is not None)
                if failed and wrong_calls<self.agent.max_tool_calls:
                    # A recoverable negative must not have an untracked partial effect.
                    if trial.get_state_hash()!=prefix_hash:
                        continue
                    corrected_validation=trial.validate(action['name'],action['arguments'])
                    if not corrected_validation.get('ok'):
                        continue
                    corrected_result=trial.call(action['name'],action['arguments'])
                    if not result_succeeded(corrected_result) or trial.get_state_hash()!=gold_after_hash:
                        continue
                    wrong_status='validation_failed' if not wrong_validation.get('ok') else 'tool_failed'
                    examples.append(self.summary_example(row,specs,plan,index,results,wrong_feedback,wrong_current,
                        wrong_status,wrong_calls,'retry','counterfactual_failure',mutation=kind,verified_recovery=True))
                    corrected_messages=self.messages(row,specs,'call',wrong_feedback,plan,index,results,
                        calls=wrong_calls,review={'next':'retry'})
                    examples.append(self.example(row,'caller',corrected_messages,caller_target,
                        'counterfactual_recovery',[wrong_output],mutation=kind,step_index=index))
                    case_counterfactuals+=1
                elif (not failed and index==len(plan)-1 and specs[selected].get('kind')!='read'
                      and trial.get_state_hash()!=gold_after_hash):
                    wrong_results=results+[{'step_index':index,'tool':selected,'status':'executed'}]
                    examples.append(self.summary_example(row,specs,plan,index+1,wrong_results,wrong_feedback,wrong_current,
                        'executed',wrong_calls,'give_up','counterfactual_wrong_effect',mutation=kind,
                        verified_state_difference=True))
                    case_counterfactuals+=1
            snapshots.append({'feedback':deepcopy(new_feedback),'results':deepcopy(new_results),
                              'current':deepcopy(current),'calls':len(prefix)+1,'hash':gold_after_hash})
            feedback,results=new_feedback,new_results
            prefix.append(action)
            if asking:
                if index!=len(plan)-1:
                    raise ValueError('Gold calls continue after waiting for user')
                break
        # An incomplete fixed plan is a legitimate give_up decision, not a fake done label.
        missing_dependency=0
        for index in range(1,len(gold)):
            prior,action=gold[index-1],gold[index]
            if ('sys.ask_missing_info' in (prior['name'],action['name'])
                    or prior['name']==action['name']
                    or specs[prior['name']].get('kind')=='read'
                    or specs[action['name']].get('kind')=='read'):
                continue
            trial=self.environment(row,gold[:index-1]);before_hash=trial.get_state_hash()
            validation=trial.validate(action['name'],action['arguments'])
            result=trial.call(action['name'],action['arguments']) if validation.get('ok') else None
            if result_succeeded(result) or trial.get_state_hash()!=before_hash:
                continue
            missing_plan=plan[:index-1]+plan[index:]
            snap=snapshots[index-2] if index>=2 else {'feedback':[],'results':[]}
            facts=snap['feedback']+self.feedback(action,validation,result)
            output={'type':'call','arguments':action['arguments']}
            current={'step_index':index-1,'selected_tool':action['name'],'caller_output':output,'action':action}
            status='tool_failed' if result is not None else 'validation_failed'
            calls=index-1+int(result is not None)
            examples.append(self.summary_example(row,specs,missing_plan,index-1,snap['results'],facts,current,
                status,calls,'give_up','counterfactual_missing_dependency',
                verified_omitted_write=prior['name']))
            examples.append(self.example(row,'caller',self.messages(row,specs,'call',facts,missing_plan,
                index-1,snap['results'],calls=calls,review={'next':'retry'}),{'type':'blocked'},
                'counterfactual_missing_dependency',[output],verified_omitted_write=prior['name']))
            missing_dependency=1
            break
        if len(plan)>1 and gold[-1]['name']!='sys.ask_missing_info':
            cut=len(plan)-1
            if snapshots[cut-1]['hash']!=env.get_state_hash() and specs[plan[-1]].get('kind')!='read':
                snap=snapshots[cut-1]
                examples.append(self.summary_example(row,specs,plan[:cut],cut,snap['results'],snap['feedback'],
                    snap['current'],'executed',snap['calls'],'give_up','counterfactual_incomplete_plan',
                    verified_missing_write=True))
        return examples, {'counterfactuals':case_counterfactuals,'missing_dependency_cases':missing_dependency}


def build_dataset(root, output, *, validation_fraction=.1, seed=42, holdout_ids=(), limit=0,
                  negatives_per_example=2, counterfactuals_per_case=2):
    rows=load_train(root,0,seed)
    assignment=split_ids(rows,validation_fraction,seed,holdout_ids)
    builder=Builder(root,seed=seed,negatives_per_example=negatives_per_example,
                    counterfactuals_per_case=counterfactuals_per_case)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    collected=defaultdict(list);excluded=[];meta=Counter();episode_counts=Counter()
    selected=[r for r in rows if assignment[r['id']]!='holdout']
    if limit:
        # Limit after splitting; keep deterministic representatives from both splits for smoke runs.
        selected=[r for split in ('train','validation') for r in selected if assignment[r['id']]==split][:limit]
        if not any(assignment[r['id']]=='validation' for r in selected):
            selected[-1]=next(r for r in rows if assignment[r['id']]=='validation')
    for row in selected:
        try:
            examples,flags=builder.build_case(row)
        except (ValueError,KeyError,TypeError) as exc:
            excluded.append({'case_id':row['id'],'split':assignment[row['id']],'error':str(exc)})
            continue
        split=assignment[row['id']];episode_counts[split]+=1;meta.update(flags)
        for example in examples:
            example['split']=split
            example['example_id']=stable_hash([row['id'],example['role'],example['source'],example['messages'],example['target']])[:24]
            collected[(example['role'],split)].append(example)
    counts={};file_hashes={}
    for role in ROLES:
        for split in ('train','validation'):
            records=collected[(role,split)]
            with (output/f'{role}.{split}.jsonl').open('w',encoding='utf-8') as f:
                for record in records:f.write(dumps(record)+'\n')
            file_hashes[f'{role}.{split}.jsonl']=hashlib.sha256(
                (output/f'{role}.{split}.jsonl').read_bytes()).hexdigest()
            counts[f'{role}.{split}']={'examples':len(records),'paired':sum(bool(x['rejected']) for x in records),
                'sources':dict(Counter(x['source'] for x in records))}
    write_json(output/'excluded_episodes.json',excluded)
    write_json(output/'split_ids.json',{s:sorted(i for i,v in assignment.items() if v==s) for s in ('train','validation','holdout')})
    write_json(output/'holdout_ids.json',sorted(i for i,v in assignment.items() if v=='holdout'))
    manifest={'schema_version':1,'method':'role-local SFT + reference-free pairwise ranking; no global stage',
              'base_model':'Qwen/Qwen3-4B-Instruct-2507','seed':seed,'validation_fraction':validation_fraction,
              'generation':{'limit':limit,'negatives_per_example':negatives_per_example,
                            'counterfactuals_per_case':counterfactuals_per_case},
              'files_sha256':file_hashes,
              'dataset_sha256':hashlib.sha256((Path(root)/'train/train.jsonl').read_bytes()).hexdigest(),
              'prompt_sha256':prompt_hashes(),'episodes_used':dict(episode_counts),'excluded_episodes':len(excluded),
              'holdout_count':len(set(holdout_ids)),'counts':counts,'flags':dict(meta),
              'negative_policy':'wrong outputs only rejected; env failures produce correct recovery targets; wrong effects/incomplete plans produce give_up',
              'summarizer_loss':'next only; generic terminal answer text is masked',
              'planner_label_limitation':'gold executed tool sequence, not a complete runtime-branch/foreach plan'}
    write_json(output/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root');p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--validation-fraction',type=float,default=.1)
    p.add_argument('--holdout-ids');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--negatives-per-example',type=int,default=2)
    p.add_argument('--counterfactuals-per-case',type=int,default=2)
    a=p.parse_args()
    if a.limit<0 or a.negatives_per_example<0 or a.counterfactuals_per_case<0:raise ValueError('Counts must be nonnegative')
    holdout=json.loads(Path(a.holdout_ids).read_text()) if a.holdout_ids else []
    print(json.dumps(build_dataset(find_root(a.data_root),a.output_dir,seed=a.seed,
        validation_fraction=a.validation_fraction,holdout_ids=holdout,limit=a.limit,
        negatives_per_example=a.negatives_per_example,counterfactuals_per_case=a.counterfactuals_per_case),
        ensure_ascii=False,indent=2))


if __name__=='__main__':main()
