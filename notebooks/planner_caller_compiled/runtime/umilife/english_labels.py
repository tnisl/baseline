"""Concise field-source plans. Deterministic labels, not teacher-model reasoning."""
import re
VERSION='concise-field-sources-v2'

def leaves(value):
    if isinstance(value,dict):
        for v in value.values():yield from leaves(v)
    elif isinstance(value,list):
        for v in value:yield from leaves(v)
    else:yield value

def contains(value,target):
    return target is not None and target!='' and any(type(v) is type(target) and v==target for v in leaves(value))

def rationale(row,action,trajectory):
    name,args=action['name'],action['arguments']
    if name=='sys.ask_missing_info':
        return ('Request clarification for '+args['tool_name']+': '+', '.join(args['params'])+
                '. Wait for the missing information before performing the incomplete operation.')
    # Gold values determine visible source attribution only; never serialize them into thought.
    groups={};history=row.get('history',[])
    for field,value in args.items():
        source=None
        for prior in reversed(trajectory):
            if contains(prior['observation'],value):
                source='the '+prior['action']['name']+' result';break
        if source is None and contains(history,value):source='conversation history'
        if source is None:source='the current request and context'
        groups.setdefault(source,[]).append(field)
    thought='Perform the next pending '+name.replace('_',' ').replace('.',' / ')+' operation.'
    if groups:
        bindings='; '.join('resolve '+', '.join(fields)+' from '+source for source,fields in groups.items())
        thought+=' '+bindings[0].upper()+bindings[1:]+'.'
    else:thought+=' Use the result to advance the pending request.'
    if re.search(r'\b(nếu|khi|trên|dưới)\b',row['audio_text'].lower()):
        thought+=" Respect the user's condition using available evidence before any dependent write."
    return thought

def terminal(row,trajectory):
    if not trajectory:
        noop=bool(re.search(r'thôi|khỏi|không cần|đừng|cảm ơn|khen|chào',row['audio_text'].lower()))
        return ({'thought':'No tool operation is needed; end this turn without changing state.' if noop else
                 'The request cannot be completed with the supported tools; stop without inventing an operation.',
                 'next':'summarizer' if noop else 'give_up'},'No changes were made.' if noop else None)
    if trajectory[-1]['action']['name']=='sys.ask_missing_info':
        return {'thought':'Clarification was requested; end this turn and wait for the user.','next':'summarizer'},'Please provide the requested missing information.'
    # Preserve annotated terminal routes without claiming every observation confirms success.
    return {'thought':'End this turn using the available results; do not repeat completed operations.','next':'summarizer'},'The recorded tool sequence has ended.'
