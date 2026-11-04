"""Generic tool policy. No benchmark examples, held-out labels or per-sample rules."""
from .typescript_schema import typescript_catalog
from copy import deepcopy
from datetime import datetime,timedelta
import hashlib,json

POLICY_VERSION='english-evidence-split1300-resume-v1'
COMMON="""You are LifeMate, a Vietnamese personal assistant using installed tools.
Handle only the current request. History resolves references; only the current trajectory proves execution. Preserve the original request's qualifiers, quantities, negation and exact text to send or save. Use provided time and timezone for relative dates. Never invent identifiers, facts or tool results.
Use known context and observations before searching. Read missing identifiers from the appropriate tools; resolve ambiguity rather than selecting arbitrarily. Evaluate conditions before dependent writes. Avoid redundant reads and repeated successful writes. A top-level operation done=false does not confirm execution. An item field such as item.done=false may describe business state rather than operation failure; empty results may require another supported search or clarification.
Before proposing a write, match it to a still-unfulfilled objective in the current request and check earlier actions AND their actual observations. Once an observation confirms the requested effect, mark that objective complete. Do not execute the same effect again to verify success, obtain another acknowledgement, or produce the final answer. Equivalent identifiers or reworded arguments do not make a completed objective new. Repeat an intentional effect only for a different pending objective or an explicit user-requested count; count confirmed executions toward that count. Retry an unsuccessful operation only when evidence shows it did not take effect and the retry is justified. Reuse unchanged read results; do not keep proposing the same lookup after an empty result.
Tool contracts are TypeScript declarations: ? marks optional fields, literal unions define enums, intersection/union types express alternative requirements. Comments preserve identifier sources and constraints. These declarations document interfaces; do not execute TypeScript. Return exactly one JSON object matching your role contract, without markdown.
"""
PROMPTS={
 'planner':COMMON+"""You are Planner. Write a concise English thought linking observed facts, conditions or calculations to the next decision. Quote original names/text unchanged when necessary.  Choose one next step. Before selecting caller, identify an objective not yet fulfilled by the current trajectory. After a successful write, move to the next pending objective; if none remain, select summarizer immediately. Never select a completed write merely because its tool is still available. Empty searches require a supported change in strategy or clarification, not the same search again. Write a concise English thought explaining the evidence and why that step is needed. Return {thought,next,tool} for next=caller; tool must be installed. Return {thought,next} for next=summarizer or give_up. Summarize only after every requested action has been addressed, a condition rules it out, no action is needed, or clarification has been requested. give_up is only for requests beyond the installed tools, not merely a missing ID or empty lookup.""",
 'caller':COMMON+"""You are Caller. Follow the selected tool contract and the original request. Return exactly {name,arguments}, with name equal to the selected tool. Respect types, enums, required fields and alternative argument groups. Omit optional arguments without evidence; do not invent filters. A plan is not evidence of parameter values. If clarification is selected, use exact tool and parameter names from the installed contracts. Use the original requested content, identity and count; do not invent a new value or alter wording to disguise a duplicate effect. Routing belongs to Planner: do not add finish/next fields to your Caller JSON. The runtime executes your JSON afterward.""",
 'summarizer':COMMON+"""You are Summarizer. Return only {answer}. Answer briefly in natural English using observations from the current trajectory. Report only actions actually confirmed by tools. If clarification was requested, state what is missing and wait for the user. Do not dump observation JSON.""",
 'global':COMMON+"""Handle one ReAct step. Write a concise English thought explaining the next action. For next=caller return {thought,next,tool,action}, where action={name,arguments}. For next=summarizer return {thought,next,answer}, with a brief grounded English answer. For next=give_up return only {thought,next}. Track which objectives have already succeeded. After a confirmed write, continue only for a distinct pending objective or an explicitly requested repetition; otherwise return next=summarizer with the grounded answer instead of calling again. Check every request objective against actual observations before ending."""
}
COMPLETION_REVIEW="""Review this proposed ending against every objective in the current request. Has something only been looked up but not executed? Are you confusing history with current execution? Are there unfinished objectives, actual operation failures, or unresolved empty searches? Do not mistake business-state fields for execution failures. Do not propose repeating any confirmed effect just to verify it or generate a reply; summarize when all objectives are satisfied. A different argument spelling does not justify repeating a completed objective. Continue with caller when work remains. Prefer lookup before clarification. Summarize only with sufficient evidence or while waiting for clarification; give_up only beyond installed capabilities."""

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def obj(properties,required):return {'type':'object','properties':properties,'required':required,'additionalProperties':False}

def parameter_schema(schema):
    """Convert documented x-one-of to ordinary JSON Schema; preserve typed nesting."""
    allowed={'type','properties','required','items','enum','format','minimum','maximum',
             'exclusiveMinimum','exclusiveMaximum','minLength','maxLength','minItems','maxItems',
             'anyOf','oneOf','allOf','additionalProperties','description'}
    out={}
    for key,value in schema.items():
        if key not in allowed:continue
        if key=='properties':out[key]={name:parameter_schema(prop) for name,prop in value.items()}
        elif key=='items' and isinstance(value,dict):out[key]=parameter_schema(value)
        elif key in ('anyOf','oneOf','allOf'):out[key]=[parameter_schema(x) for x in value]
        else:out[key]=deepcopy(value)
    if out.get('type')=='object' and 'properties' in out:out['additionalProperties']=False
    if schema.get('x-one-of'):
        # Full object alternatives are supported by both JSON Schema and LMFE.
        base=out
        out={'anyOf':[{**deepcopy(base),'required':list(dict.fromkeys(base.get('required',[])+list(group)))}
                      for group in schema['x-one-of']]}
    return out

def action_schema(specs,name):
    arguments=parameter_schema(specs[name]['parameters'])
    if name=='sys.ask_missing_info':
        # Only public installed schemas provide these choices, never an answer label.
        choices=[n for n,s in specs.items() if n!=name and s['parameters'].get('properties')]
        if choices:arguments['properties']['tool_name']['enum']=sorted(choices)
        arguments['properties']['params']['minItems']=1
        fields=sorted({key for target in choices for key in specs[target]['parameters']['properties']})
        if fields:arguments['properties']['params']['items']['enum']=fields
    return obj({'name':{'type':'string','enum':[name]},'arguments':arguments},['name','arguments'])

def route_schema(specs,global_mode=False):
    thought={'type':'string','maxLength':1024};branches=[]
    if specs:
        if global_mode:
            for name in sorted(specs):
                branches.append(obj({'thought':thought,'next':{'type':'string','enum':['caller']},
                    'tool':{'type':'string','enum':[name]},'action':action_schema(specs,name)},['thought','next','tool','action']))
        else:
            branches.append(obj({'thought':thought,'next':{'type':'string','enum':['caller']},
                'tool':{'type':'string','enum':sorted(specs)}},['thought','next','tool']))
    done={'thought':thought,'next':{'type':'string','enum':['summarizer']}}
    if global_mode:done['answer']={'type':'string','maxLength':1024}
    branches.append(obj(done,list(done)))
    branches.append(obj({'thought':thought,'next':{'type':'string','enum':['give_up']}},['thought','next']))
    return {'anyOf':branches}

def compact_catalog(specs):
    fields=('type','enum','x-ref','format','items','minimum','maximum')
    return [{'name':name,'kind':spec.get('kind'),'description':spec['description'],
             'parameters':{key:{k:deepcopy(v) for k,v in prop.items() if k in fields}
                           for key,prop in spec['parameters'].get('properties',{}).items()},
             'required':spec['parameters'].get('required',[]),
             **({'one_of':spec['parameters']['x-one-of']} if spec['parameters'].get('x-one-of') else {})}
            for name,spec in sorted(specs.items())]

def reference_clock(context):
    result={}
    try:
        now=datetime.fromisoformat(context['current_time'])
        result={'today':now.date().isoformat(),'tomorrow':(now+timedelta(days=1)).date().isoformat(),
                'day_after_tomorrow':(now+timedelta(days=2)).date().isoformat(),
                'weekday_iso':now.isoweekday(),'timezone':context.get('timezone')}
    except (KeyError,TypeError,ValueError):pass
    return result

def build_messages(role,row,text,specs,trajectory,plan=None,feedback=None):
    # Explicit provenance allowlist. Current labels/final state/reference transcript never enter here.
    context={k:deepcopy(row['context'][k]) for k in ('current_time','timezone','location','user_profile') if k in row['context']}
    history=[{k:deepcopy(turn[k]) for k in ('role','text','tool_calls','tool_results') if k in turn}
             for turn in row.get('history',[])]
    content={'context':context,'reference_clock':reference_clock(context),'history':history,
             'request':text,'trajectory':deepcopy(trajectory)}
    if role in ('planner','global'):content['tools_ts']=typescript_catalog(specs)
    elif role=='caller':
        selected=(plan or {}).get('tool');assert selected in specs,'Planner selected an unavailable tool'
        content.update(plan=deepcopy(plan),selected_tool_ts=typescript_catalog({selected:specs[selected]},detailed=True))
        if selected=='sys.ask_missing_info':
            content['missing_information_tools_ts']=typescript_catalog({name:spec for name,spec in specs.items()
                if name!=selected and spec['parameters'].get('properties')})
    else:content['plan']=deepcopy(plan)
    if feedback is not None:content['validation_feedback']=deepcopy(feedback)
    # Keep the original request at the end, after schemas/observations/feedback.
    content.pop('request');content['request']=text
    return [{'role':'system','content':PROMPTS[role]},
            {'role':'user','content':json.dumps(content,ensure_ascii=False,separators=(',',':'))}]

def request_schema(role,specs,plan=None):
    if role=='caller':return action_schema(specs,plan['tool'])
    if role in ('planner','global'):return route_schema(specs,role=='global')
    return obj({'answer':{'type':'string','maxLength':1024}},['answer'])

def validate_meta_action(action,specs):
    if action['name']!='sys.ask_missing_info':return None
    args=action['arguments'];target=args.get('tool_name');params=args.get('params',[])
    if target not in specs or target=='sys.ask_missing_info':return 'Clarification target must be another installed tool'
    # An optional API parameter can still be necessary to express the user's intent.
    fields=set(specs[target]['parameters'].get('properties',{}))
    if not params or len(params)!=len(set(params)) or not set(params)<=fields:
        return 'params must contain unique parameter names from the target tool contract'
    return None

def repeated_read(action,specs,trajectory):
    if specs[action['name']].get('kind')!='read':return False
    for old in reversed(trajectory):
        if specs.get(old['action']['name'],{}).get('kind')=='write':return False
        if canonical(old['action'])==canonical(action):return True
    return False

POLICY_AUDIT={'policy_version':POLICY_VERSION,'static_prompts_sha256':digest(PROMPTS),
    'completion_review_sha256':digest(COMPLETION_REVIEW),'few_shot_examples':0,
    'sample_specific_rules':False,'gold_inputs_to_model':False,
    'allowed_input_sources':['current ASR request','ASR user history','provided context',
        'installed public tool schemas','observations from executed tools','model-generated plan/feedback']}
