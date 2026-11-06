"""Local approximation of published hash-gated tool-call F1 (not BTC scorer)."""
from collections import Counter
import re
import unicodedata

def token_f1(a,b):
    a=Counter(re.findall(r"\w+",unicodedata.normalize("NFC",str(a)).lower()))
    b=Counter(re.findall(r"\w+",unicodedata.normalize("NFC",str(b)).lower()))
    if not a and not b:return 1.
    overlap=sum((a&b).values())
    return 2*overlap/(sum(a.values())+sum(b.values())) if overlap else 0.

def matches(a,b,catalog):
    if a.get("name")!=b.get("name"):return False
    x,y=a.get("arguments",{}),b.get("arguments",{})
    if x.keys()!=y.keys():return False
    props=catalog.get(a["name"],{}).get("parameters",{}).get("properties",{})
    for key,value in x.items():
        if props.get(key,{}).get("x-match")=="fuzzy":
            if token_f1(value,y[key])<=.6:return False
        elif value!=y[key] or isinstance(value,bool)!=isinstance(y[key],bool):return False
    return True

def maximum_matches(pred,gold,catalog):
    edges=[[j for j,g in enumerate(gold) if matches(p,g,catalog)] for p in pred]
    assigned={}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in assigned or augment(assigned[j],seen):
                assigned[j]=i;return True
        return False
    return sum(augment(i,set()) for i in range(len(pred)))

def score(pred,gold,catalog,timed_out=False):
    calls,ref=pred["tool_calls"],gold["tool_calls"]
    matched=maximum_matches(calls,ref,catalog)
    hash_ok=pred["state_hash"]==gold["state_hash"]
    f1=2*matched/(len(calls)+len(ref)) if calls or ref else 1.
    value=0. if timed_out or not hash_ok else f1
    return {"id":gold["id"],"score":value,"hash_match":hash_ok,"tool_f1":f1,
            "matched":matched,"pred_calls":len(calls),"gold_calls":len(ref),"timed_out":timed_out}
