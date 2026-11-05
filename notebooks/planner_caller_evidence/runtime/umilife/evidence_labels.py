"""English evidence compiler authored by Codex; not an independent LLM teacher.
Current labels identify the supervised action, but do not supply its future result.
Only values traceable to visible evidence are serialized as facts.
"""
import json,re,unicodedata
from datetime import datetime,timedelta
VERSION='evidence-goals-v3'

def plain(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s.lower().replace('đ','d')) if unicodedata.category(c)!='Mn')

def records(value):
    if isinstance(value,dict):
        yield value
        for v in value.values():yield from records(v)
    elif isinstance(value,list):
        for v in value:yield from records(v)

def scalars(value):
    if isinstance(value,dict):
        for v in value.values():yield from scalars(v)
    elif isinstance(value,list):
        for v in value:yield from scalars(v)
    else:yield value

def visible(row,trajectory):
    return [row.get('context',{}),row.get('history',[]),[p['observation'] for p in trajectory]]

def supported(value,row,trajectory):
    return (isinstance(value,str) and value in row['audio_text']) or any(type(v) is type(value) and v==value for source in visible(row,trajectory) for v in scalars(source))

def number(text):
    text=plain(text).replace(',','.').strip()
    if re.fullmatch(r'\d+(?:\.\d+)?',text):return float(text)
    small={'khong':0,'mot':1,'hai':2,'ba':3,'bon':4,'tu':4,'nam':5,'lam':5,'sau':6,'bay':7,'tam':8,'chin':9,'muoi':10}
    total=chunk=current=0
    for token in text.split():
        if re.fullmatch(r'\d+(?:\.\d+)?',token):current=float(token)
        elif token in small:
            if token=='muoi':current=max(1,current)*10
            else:current+=small[token]
        elif token=='tram':current=max(1,current)*100
        elif token in ('nghin','ngan','trieu','ty'):
            scale={'nghin':1000,'ngan':1000,'trieu':10**6,'ty':10**9}[token]
            chunk+=current;total+=chunk*scale;chunk=current=0
        elif token in ('le','linh'):continue
        else:return None
    return total+chunk+current

def constraints(row):
    text=plain(row['audio_text']);out=[];max_amount=None;days=None
    m=re.search(r'(?:trong|toi han trong)\s+(\d+|mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi)\s+ngay',text)
    if m:days=int(number(m[1]));out.append(f'Use the next {days} days as the due-date window.')
    # Exclusion wording creates an upper bound; do not treat every price threshold as a budget.
    m=re.search(r'(?:qua|tren|toi da|khong qua)\s+(\d+(?:[.,]\d+)?\s*(?:nghin|ngan|trieu))',text)
    if m and ('hoa don' in text or 'cai nao' in text) and any(x in text for x in ('de do','bo qua','khong tra','toi da','khong qua')):
        max_amount=number(m[1]);out.append(f'Exclude bills costing more than {max_amount:,.0f} VND.')
    if any(x in text for x in ('a khong','a hong','a thoi','a lon','a mo','khong phai')):
        out.append('The latest correction overrides the earlier entity, time or action.')
    if any(x in text for x in ('neu','khi','thi thoi','con khong','khong thi','rieng','ngoai tru','qua','duoi','tren')):
        out.append('Check the requested condition before any dependent write; do not add an unrequested operation.')
    return out,days,max_amount

def bill_evidence(row,trajectory,days,limit):
    bills=[]
    for p in trajectory:
        if p['action']['name']=='bill_payment.list_bills':bills=p['observation'].get('bills',[])
    paid={p['action']['arguments'].get('bill_id') for p in trajectory if p['action']['name']=='bill_payment.pay' and p['observation'].get('ok') is True and not p['observation'].get('error')}
    now=datetime.fromisoformat(row['context']['current_time']).date()
    eligible=[];excluded=[]
    for b in bills:
        if b.get('paid') or b['id'] in paid:continue
        due=datetime.fromisoformat(b['due_date']).date()
        reason='over the spending limit' if limit is not None and b['amount']>limit else 'outside the due-date window' if days is not None and not now<=due<=now+timedelta(days=days) else None
        if reason:excluded.append((b,reason))
        else:eligible.append(b)
    return eligible,excluded

GOALS={
 'calendar.update_event':'Reschedule the existing event rather than creating a duplicate.',
 'calendar.cancel_event':'Cancel the identified event only if the current request authorizes cancellation.',
 'calendar.create_event':'Create the requested new event; do not use creation to modify an existing event.',
 'reminders.create':'Create the requested reminder with the intended content and local time.',
 'bill_payment.pay':'Pay one eligible unpaid bill through the requested payment source.',
 'bill_payment.schedule_autopay':'Change recurring payment settings only when explicitly requested.',
 'banking.transfer':'Transfer the requested amount to the intended beneficiary using the permitted funding account.',
 'ewallet.transfer':'Transfer the requested amount to the intended recipient, preserving any balance condition.',
 'smart_home.light_set':'Apply the requested light change to the latest intended room or device.',
 'smart_home.fan_set':'Apply the requested fan state; switching off is not a request to increase speed.',
 'ride_hailing.book':'Book the selected ride with the requested pickup, destination and scheduled time.',
 'home_automation.create_rule':'Create only the requested rule with the correct target, action and local time.',
 'sms.send':'Send the intended message to the resolved recipient; preserve the requested meaning and facts.',
 'email.reply':'Reply to the referenced email with the latest corrected message, without inventing details.',
 'tasks.complete':'Complete the matching pending task; do not substitute another task identifier.'}

GOALS.update({
 'weather.get_forecast':'Resolve the requested future weather and rain condition before deciding any weather-dependent operation.',
 'weather.get_current':'Retrieve current weather for the requested place; do not substitute it for a future forecast.',
 'bill_payment.list_bills':'Identify unpaid bills and inspect their amounts and due dates before selecting eligible payments.',
 'banking.get_balance':'Inspect available accounts and balances before choosing a funding source or evaluating the requested balance condition.',
 'ewallet.get_balance':'Inspect the wallet balance before evaluating a payment or fallback-source condition.',
 'banking.list_beneficiaries':'Resolve the intended beneficiary from the available records; distinguish similar names before selecting an identifier.',
 'contacts.search':'Resolve the person specified by the latest request or correction, rather than selecting the first similar name.',
 'calendar.list_events':'Identify the referenced event and its existing timing before deciding the requested change or conflict condition.',
 'alarm.list':'Resolve the referenced existing alarm and its time before changing or deleting it.',
 'tv.guide':'Resolve the requested channel or program against the available guide instead of guessing an approximate channel name.',
 'gold_price.get_price':'Retrieve the requested gold type and buying or selling price to answer the query or evaluate its threshold.',
 'fuel_price.get_price':'Retrieve the specifically requested fuel type rather than silently using another product.',
 'home_security.get_camera':'Inspect the camera at the requested location before evaluating any dependent security action.',
 'appliances.list':'Find the requested appliance type and available controls before selecting a compatible operation.',
 'learning.list_courses':'Resolve the intended course and enrollment state before scheduling or enrolling.',
 'notes.list':'Locate the referenced note and inspect its contents before answering or modifying it.',
 'tasks.list':'Find the referenced tasks and their deadlines or completion state before applying the requested filter.',
 'food_delivery.get_menu':'Inspect the selected restaurant menu to resolve valid items and user-requested options.',
 'loyalty.get_points':'Inspect the specified program and available points before evaluating redemption eligibility.',
 'loyalty.list_vouchers':'Inspect available rewards and their costs before choosing the requested eligible reward.'})

def rationale(row,action,trajectory):
    name=action['name'];args=action['arguments'];rules,days,limit=constraints(row)
    if name=='sys.ask_missing_info':
        return f"The supported {args['tool_name']} operation still lacks {', '.join(args['params'])}. Request that information instead of inventing values."
    if name=='bill_payment.pay' and (days is not None or limit is not None):
        eligible,excluded=bill_evidence(row,trajectory,days,limit)
        selected=next((b for b in eligible if b['id']==args.get('bill_id')),None)
        if selected:
            other=[b.get('type',b['id']) for b in eligible if b['id']!=selected['id']]
            parts=[f"Pay the observed {selected.get('type','selected')} bill ({selected['id']}) through {args.get('source','the requested source')}: {selected['amount']:,.0f} VND due {selected['due_date']} satisfies the requested filters."]
            if other:parts.append('Other eligible unpaid bills remain: '+', '.join(other)+'.')
            if excluded:parts.append('Leave '+', '.join(b.get('type',b['id'])+' ('+reason+')' for b,reason in excluded)+' unchanged.')
            return ' '.join(parts)
    if name in GOALS:goal=GOALS[name]
    else:
        app,verb=name.split('.',1);words=verb.replace('_',' ')
        goal=(f'Look up {verb.partition("_")[2].replace("_"," ") or words} in {app.replace("_"," ")} to resolve the next requested step.' if verb.startswith(('get','list','search','find','read','track')) else
              f'Perform the requested {app.replace("_"," ")} operation: {words}. Preserve its user-specified scope.')
    facts=[]
    for field,value in args.items():
        if field.endswith('_id'):
            record=next((rec for source in reversed(visible(row,trajectory)) for rec in records(source) if rec.get('id')==value),None)
            if record:
                selector=record.get('name',record.get('title',record.get('content',record.get('type',value))))
                facts.append(f'Use observed {field}={value} for {json.dumps(str(selector)[:75],ensure_ascii=False)}.')
            elif supported(value,row,trajectory):facts.append(f'Use {field}={value} from the visible request/history.')
        elif field in ('time','start','end','schedule_time','date'):
            # Date/time are supervised derived values, not unobserved tool results.
            facts.append(f'Resolve {field} as {value} from the requested time and supplied clock.')
        elif field in ('amount','type','source','power','action','brightness','enabled') and not isinstance(value,(dict,list)):
            facts.append(f'Preserve {field}={json.dumps(value,ensure_ascii=False)} for this requested step.')
    parts=[goal,*rules[:2],*facts[:3]]
    if trajectory and trajectory[-1]['observation'].get('ok') is True:
        parts.append('Reuse prior results and reassess remaining objectives after this step.')
    return ' '.join(parts)

def terminal(row,trajectory,specs=None):
    if not trajectory:return None
    if trajectory[-1]['action']['name']=='sys.ask_missing_info':
        return {'thought':'The missing information was requested. End this turn and wait for the user instead of executing the incomplete operation.','next':'summarizer'}
    effects=[p for p in trajectory if (specs or {}).get(p['action']['name'],{}).get('kind')=='write']
    if effects:
        descriptions=[]
        for p in effects[-4:]:
            action=p['action'];details=[]
            for key,value in action['arguments'].items():
                if key.endswith('_id') or key in ('time','start','end','amount','power','action','brightness'):
                    details.append(f'{key}={value}')
            descriptions.append(action['name']+(' ['+', '.join(details[:2])+']' if details else ''))
        thought='The observed results confirm '+ '; '.join(descriptions)+'. Summarize the completed request without repeating these effects or adding a new operation.'
    else:thought='The existing lookup results supply the requested information. Answer from those observations instead of repeating an unchanged read or inventing a write.'
    return {'thought':thought,'next':'summarizer'}

def issues(row,trajectory,action,observation):
    result=[]
    if observation.get('ok') is False or observation.get('error'):result.append('tool_reported_failure')
    for key,value in action['arguments'].items():
        if key.endswith('_id') and isinstance(value,str) and not supported(value,row,trajectory):result.append('unobserved_identifier:'+key)
    _,days,limit=constraints(row)
    if action['name']=='bill_payment.pay' and (days is not None or limit is not None):
        eligible,excluded=bill_evidence(row,trajectory,days,limit)
        if any(b['id']==action['arguments'].get('bill_id') for b,_ in excluded):result.append('bill_violates_request_filter')
    return result
