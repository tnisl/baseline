"""English evidence-compiled annotations. Not an LLM teacher or individually reviewed prose."""
import json,re
from datetime import datetime

VERSION='english-evidence-v1'
def compact(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def facts(value,prefix='',limit=12):
    out=[]
    if isinstance(value,dict):
        for k,v in value.items():out.extend(facts(v,f'{prefix}.{k}'.strip('.'),limit))
    elif isinstance(value,list):
        for i,v in enumerate(value[:4]):out.extend(facts(v,f'{prefix}[{i}]',limit))
    else:out.append((prefix,value))
    return out[:limit]

def rationale(row,action,trajectory):
    name,args=action['name'],action['arguments'];evidence=[]
    if name=='sys.ask_missing_info':
        return ('The next operation '+args['tool_name']+' cannot be specified from the available evidence. '
                'Ask for '+', '.join(args['params'])+' instead of inventing these values.')
    # IDs are linked only to previous observations or history, never current/future results.
    sources=[(p['action']['name'],p['observation']) for p in trajectory]
    sources += [('conversation history',t) for t in row.get('history',[])]
    for key,value in args.items():
        if isinstance(value,str) and (key.endswith('_id') or key in ('phone_number','program_id')):
            for source,observation in reversed(sources):
                if any(v==value for _,v in facts(observation,limit=1000)):
                    evidence.append(f'{key}={compact(value)} is present in {source}');break
    # Bind explicit numerical conditions to actual earlier price/balance readings.
    request=row['audio_text']
    threshold=re.search(r'(trên|hơn|dưới|ít hơn)\s+(\d+(?:[.,]\d+)?)\s*(triệu|nghìn|ngàn)?',request.lower())
    if threshold:
        amount=float(threshold[2].replace(',','.'))*({'triệu':1e6,'nghìn':1e3,'ngàn':1e3}.get(threshold[3],1))
        for source,obs in sources:
            for key,value in facts(obs,limit=100):
                if key.split('.')[-1] in ('sell','buy','price','balance') and isinstance(value,(float,int)):
                    relation='below' if value<amount else 'above' if value>amount else 'equal to'
                    evidence.append(f'{source}: {key}={value:.15g}, {relation} the stated threshold {amount:.15g}')
                    break
    # Time offsets are calculated from dates actually observed, not the next tool result.
    if name=='alarm.create' and 'time' in args:
        try:
            alarm=datetime.fromisoformat(args['time'])
            for source,obs in sources:
                for key,value in facts(obs,limit=100):
                    if key.endswith('.start') and isinstance(value,str):
                        minutes=int((datetime.fromisoformat(value)-alarm).total_seconds()/60)
                        if 0<minutes<=1440:evidence.append(f'observed start {value} minus {minutes} minutes gives {args["time"]}')
        except ValueError:pass
    if not evidence and trajectory:
        previous=trajectory[-1]
        evidence.append('Latest '+previous['action']['name']+' returned '+compact(dict(facts(previous['observation'],limit=3))))
    decision='Next use '+name+' with '+compact(args)+'.'
    if evidence:decision='Evidence: '+'; '.join(evidence[:3])+'. '+decision
    else:decision='Use the current request and known context to specify the next operation. '+decision
    return decision[:1000]

def terminal(row,trajectory):
    if not trajectory:
        noop=bool(re.search(r'thôi|khỏi|không cần|đừng|cảm ơn|khen|chào',row['audio_text'].lower()))
        if noop:return {'thought':'The current request does not authorize a new operation. Preserve the existing state and acknowledge without calling a tool.','next':'summarizer'},'No changes were made.'
        return {'thought':'No supported tool execution is available for this request. Stop without inventing a tool or reporting an unperformed action.','next':'give_up'},None
    last=trajectory[-1]
    if last['action']['name']=='sys.ask_missing_info':
        args=last['action']['arguments']
        return {'thought':'Clarification has been requested for '+args['tool_name']+' ('+', '.join(args['params'])+'). Wait for the user before performing the incomplete operation.','next':'summarizer'},'Please provide '+', '.join(args['params'])+' for '+args['tool_name']+'.'
    done=', '.join(dict.fromkeys(p['action']['name'] for p in trajectory))
    thought=f'The recorded sequence contains {len(trajectory)} executed operations: {done}. Use their actual observations to report the result; do not repeat confirmed effects.'
    answer='Result of '+last['action']['name']+': '+ '; '.join(k+'='+str(v) for k,v in facts(last['observation'],limit=6))+'.'
    return {'thought':thought[:1000],'next':'summarizer'},answer[:700]
