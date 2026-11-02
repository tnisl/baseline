"""Audit labels and public input/schema metadata using only the official tool runtime."""
import argparse
from collections import Counter
from pathlib import Path
from umilife.core import *
from umilife.data import grouped_split

def audit(root,verify=False):
    root=Path(root);train=read_jsonl(root/"train/train.jsonl");public=read_jsonl(root/"public_test/public_test.jsonl")
    catalogs={s:load_catalog(root,s) for s in ("train","public_test")}
    split=grouped_split(train)
    report={"train_samples":len(train),"public_samples":len(public),
        "train_sha256":digest(root/"train/train.jsonl"),"public_sha256":digest(root/"public_test/public_test.jsonl"),
        "train_with_history":sum(bool(r["history"]) for r in train),
        "public_with_history":sum(bool(r["history"]) for r in public),
        "train_call_length_histogram":dict(Counter(len(r["label"]["tool_calls"]) for r in train)),
        "train_calls":sum(len(r["label"]["tool_calls"]) for r in train),
        "train_missing_info_samples":sum(any(c["name"]=="sys.ask_missing_info" for c in r["label"]["tool_calls"]) for r in train),
        "train_out_of_scope_samples":sum(not r["label"]["tool_calls"] for r in train),
        "catalog_tool_counts":{s:len(c) for s,c in catalogs.items()},
        "public_only_tools":sorted(catalogs["public_test"].keys()-catalogs["train"].keys()),
        "train_only_tools":sorted(catalogs["train"].keys()-catalogs["public_test"].keys()),
        "split_counts":dict(Counter(split.values())),
        "grouping":"Equal normalized final request transcripts kept in the same partition",
        "gold_replay":None}
    if verify:
        ToolEnv=import_toolenv(root,"train");valid=0;issues=[]
        for row in train:
            env=ToolEnv("train",row["id"],root=root)
            for action,expected in zip(row["label"]["tool_calls"],row["label"]["tool_results"]):
                check=env.validate(action["name"],action["arguments"])
                if not check.get("ok"):
                    issues.append({"id":row["id"],"type":"schema"});break
                actual=env.call(action["name"],action["arguments"])
                if actual!=expected:
                    issues.append({"id":row["id"],"type":"result"});break
            else:valid+=1
        report["gold_replay"]={"episodes_with_identical_results":valid,"issues":issues}
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--verify-tools",action="store_true");a=p.parse_args()
    result=audit(a.data_root,a.verify_tools);write_json(a.output,result);print(dumps(result))
