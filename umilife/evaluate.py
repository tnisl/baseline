"""Local implementation of the published hash-gated tool-call F1; not an official scorer."""
import argparse
from collections import Counter
import json
import re
import unicodedata
from .core import read_jsonl,load_catalog,write_json

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

def evaluate(predictions,references,catalog,traces=()):
    ids=[x["id"] for x in predictions]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate prediction IDs")
    lookup={x["id"]:x for x in references};trace={x["id"]:x for x in traces}
    if set(ids)-lookup.keys():raise ValueError("Predictions absent from validation references")
    rows=[score(p,lookup[p["id"]],catalog,trace.get(p["id"],{}).get("over_deadline",False)) for p in predictions]
    if not rows:raise ValueError("No predictions")
    latencies=sorted(t["elapsed_s"] for t in traces)
    return {"samples":len(rows),"score_100":100*sum(x["score"] for x in rows)/len(rows),
        "state_accuracy":sum(x["hash_match"] for x in rows)/len(rows),
        "tool_f1":sum(x["tool_f1"] for x in rows)/len(rows),
        "timeout_count":sum(x["timed_out"] for x in rows),
        "latency_mean_s":sum(latencies)/len(latencies) if latencies else None,
        "latency_p95_s":latencies[min(len(latencies)-1,int(.95*len(latencies)))] if latencies else None,
        "metric_note":"Local approximation of published metric; public/private test have no gold labels.",
        "per_sample":rows}

def main():
    p=argparse.ArgumentParser();p.add_argument("--predictions",required=True)
    p.add_argument("--references",required=True);p.add_argument("--data-root",required=True)
    p.add_argument("--traces");p.add_argument("--output",required=True)
    a=p.parse_args();references=json.load(open(a.references))
    report=evaluate(read_jsonl(a.predictions),references,load_catalog(a.data_root,"train"),
                    read_jsonl(a.traces) if a.traces else ())
    write_json(a.output,report);print(json.dumps({k:v for k,v in report.items() if k!="per_sample"},indent=2))
if __name__=="__main__":main()
