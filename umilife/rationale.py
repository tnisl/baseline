"""Generate train-only rationales with current gold action and strictly preceding observations."""
import argparse
import json
from pathlib import Path
import random
from .core import read_jsonl,load_catalog,dumps
from .data import annotation_tasks, grouped_split
from .models import Backend, MODEL_ID

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--model",default=MODEL_ID);p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--limit",type=int);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--revision")
    a=p.parse_args();root=Path(a.data_root);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(root/"train/train.jsonl");split=grouped_split(rows,a.seed)
    if a.limit:
        selected=[]
        for part in ("train","validation"):
            group=[r for r in rows if split[r["id"]]==part];random.Random(a.seed).shuffle(group)
            selected+=group[:max(2,a.limit if part=="train" else max(2,a.limit//5))]
        rows=selected
    done={r["key"] for r in read_jsonl(out)} if out.exists() else set()
    tasks=[r for r in annotation_tasks(rows,load_catalog(root,"train"),split) if r["key"] not in done]
    if not tasks: print("Rationales already complete");return
    backend=Backend(a.model,four_bit=True,max_context=16384,revision=a.revision)
    with out.open("a",encoding="utf-8") as f:
        for start in range(0,len(tasks),a.batch_size):
            chunk=tasks[start:start+a.batch_size]
            responses=backend.generate_batch("teacher",[t["messages"] for t in chunk],remaining_s=600,max_new_tokens=96)
            for task,response in zip(chunk,responses):
                thought=response.strip()
                if not thought or thought.startswith(("{","[")): raise ValueError("Invalid teacher rationale")
                f.write(dumps({k:task[k] for k in ("id","step","key","partition")} |
                    {"thought":thought,"teacher":a.model,"revision":a.revision})+"\n")
            f.flush()
            print(dumps({"rationales":start+len(chunk),"total":len(tasks)}),flush=True)
if __name__=="__main__":main()
