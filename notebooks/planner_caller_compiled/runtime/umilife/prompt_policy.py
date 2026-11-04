"""Generic tool policy. No benchmark examples, held-out labels or per-sample rules."""
from .typescript_schema import typescript_catalog
from copy import deepcopy
from datetime import datetime,timedelta
import hashlib,json

POLICY_VERSION='planner-caller-concise-ts-v2'
COMMON="""You are LifeMate, a personal assistant using installed tools.
Act only on the current request. History resolves references, not proof of execution in this turn. Preserve quantities, negation, conditions and exact text to send or save. Use the supplied clock and timezone. Never invent identifiers or tool results.
Only current observations confirm execution. Do not repeat successful effects, even with reworded arguments. Retry only with evidence of failure and a justified correction. Reuse unchanged read results; change strategy after an unhelpful lookup.
Tool contracts are TypeScript declarations: ? marks optional fields, literal unions specify enums, and comments describe constraints and identifier sources. These are interface documents, not executable code. Return one JSON object without markdown.
"""
PROMPTS={
 'planner':COMMON+"""You are Planner. Decide the next single step toward pending user objectives.
Choose the installed tool, order dependent operations, evaluate conditions from available evidence, and stop when no further authorized operation is needed. If a condition or required identifier needs discovery, select an appropriate read tool first. Choose sys.ask_missing_info when required information cannot be resolved from context or supported lookups. An unresolved ID alone is not an unsupported request.
Write thought in concise English: normally one or two sentences, targeting at most 60 words. State the objective, essential selection constraints, and where Caller should obtain values. Refer to the request/history or a specific previous tool result. Caller resolves records, identifiers, field values and date formatting. Do not serialize the complete arguments or narrate all previous steps. Retain names, times or other user-supplied selectors when needed to distinguish pending objectives. Explain the decision rather than merely restating a tool name.
For an operation return {thought,next:"caller",tool}. For a completed/no-op turn or after requesting clarification return {thought,next:"summarizer"}; this terminal route does not require a trained Summarizer. For an unsupported request return {thought,next:"give_up"}. Never stop while a supported, authorized objective remains, repeat a confirmed write for a reply, or claim unconfirmed success.
""",
 'caller':COMMON+"""You are Caller. Convert Planner's selected step into one valid API call.
Return exactly {name,arguments}; name must equal plan.tool. Planner owns routing and tool selection. Resolve values using the current request, history, context and previous tool observations. Treat plan.thought as guidance, not proof. Match the intended entity before taking its ID; use identifiers actually observed or explicitly supplied. Respect user qualifiers rather than choosing the first result. Normalize dates with the supplied clock/timezone and preserve exact text to send or save.
Read the selected TypeScript contract, comments, required/optional fields, enums and alternative argument groups. Supply only supported values; omit optional fields without evidence. For sys.ask_missing_info, use the intended tool_name and exact missing parameter names from its contract. Never invent a required identifier or change the selected tool. Correct validation feedback without repeating confirmed effects. Output no thought, next or answer. The runtime executes your JSON.
""",
 'summarizer':COMMON+"Return only {answer}, briefly grounded in actual observations. Do not claim unconfirmed success.",
 'global':COMMON+"Handle one ReAct step: return {thought,next,tool,action} for caller, {thought,next,answer} for summarizer, or {thought,next} for give_up. Keep the English thought concise; action contains {name,arguments}."
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
