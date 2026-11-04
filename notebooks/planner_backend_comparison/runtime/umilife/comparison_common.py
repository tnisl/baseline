import hashlib,json,re
from collections import Counter

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def token_hash(tokens):return fingerprint(list(tokens))
def assess(raw,schema,expected):
    from jsonschema import Draft202012Validator
    parsed=None;error=None
    try:
        parsed=json.loads(raw);Draft202012Validator(schema).validate(parsed)
    except Exception as e:error=str(e)[:500]
    thought=parsed.get('thought','') if isinstance(parsed,dict) else ''
    words=thought.lower().split();grams=Counter(tuple(words[i:i+6]) for i in range(max(0,len(words)-5)))
    raw_words=raw.lower().split();raw_grams=Counter(tuple(raw_words[i:i+6]) for i in range(max(0,len(raw_words)-5)))
    return {'raw_words':len(raw_words),'raw_repeats_6gram_3plus':any(n>=3 for n in raw_grams.values()),'schema_valid':error is None,'schema_error':error,'parsed':parsed,
        'thought_chars':len(thought),'thought_words':len(words),'repeats_6gram_3plus':any(n>=3 for n in grams.values()),
        'thought_at_1000chars':len(thought)>=1000,
        'route_match':bool(error is None and parsed.get('next')==expected['next']),
        'tool_match':bool(error is None and parsed.get('next')==expected['next'] and parsed.get('tool')==expected.get('tool'))}

def training_probes(rows,valid_ids,count):
    """Deterministic probes only from training episodes; targets never enter messages."""
    train=[r for r in rows if r['id'] not in valid_ids];chosen=[];seen=set()
    groups=[lambda r:bool(r['label']['tool_calls']) and bool(re.search(r'nếu|khi',r['audio_text'],re.I)),
            lambda r:bool(r['label']['tool_calls']) and bool(r.get('history')),
            lambda r:not r['label']['tool_calls'],lambda r:bool(r['label']['tool_calls'])]
    for predicate in groups:
        for r in sorted(train,key=lambda r:r['id']):
            if r['id'] not in seen and predicate(r):chosen.append(r);seen.add(r['id'])
            if len(chosen)>=(groups.index(predicate)+1)*2:break
    return chosen[:count]
