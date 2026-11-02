"""Offline single-session inference. Labels are stripped at the input boundary."""
import argparse
import json
from pathlib import Path
import time
from .core import *
from .models import Backend,ASR,MODEL_ID
from .data import normalized_row

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True);p.add_argument("--split",default="public_test")
    p.add_argument("--adapter-dir",required=True);p.add_argument("--output",required=True)
    p.add_argument("--model",default=MODEL_ID);p.add_argument("--revision")
    p.add_argument("--asr-model",default="Systran/faster-whisper-small")
    p.add_argument("--manifest");p.add_argument("--limit",type=int)
    p.add_argument("--oracle-text",action="store_true",help="TRAIN only diagnostic; no ASR")
    p.add_argument("--mode",choices=("multi","single"),default="multi")
    p.add_argument("--fp16",action="store_true");p.add_argument("--deadline",type=float,default=30)
    p.add_argument("--max-context",type=int,default=8192)
    a=p.parse_args()
    if a.oracle_text and a.split!="train":raise ValueError("Oracle transcripts are allowed on train only")
    root=Path(a.data_root);output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(root/a.split/f"{a.split}.jsonl")
    if a.manifest:
        manifest=json.loads(Path(a.manifest).read_text())
        selected=set(manifest["selected_ids"])
        rows=[r for r in rows if manifest["split"].get(r["id"])=="validation" and r["id"] in selected]
    if a.limit:rows=rows[:a.limit]
    catalog=load_catalog(root,a.split);ToolEnv=import_toolenv(root,a.split)
    roles=ROLES if a.mode=="multi" else ("global",)
    backend=Backend(a.model,a.adapter_dir,roles,four_bit=not a.fp16,max_context=a.max_context,revision=a.revision)
    asr=None if a.oracle_text else ASR(a.asr_model)
    if backend.parameter_report["total_llm"]+(asr.parameter_count if asr else 0)>=6_000_000_000:
        raise ValueError("Total parameter budget exceeded")
    agent=Agent(backend,deadline_s=a.deadline,mode=a.mode)
    traces=output.with_suffix(".traces.jsonl")
    with output.open("w",encoding="utf-8") as f,traces.open("w",encoding="utf-8") as tf:
        for raw in rows:
            # Exactly one ToolEnv in the execution path for this sample.
            env=ToolEnv(a.split,raw["id"],root=root)
            start=time.monotonic()
            if a.oracle_text:
                text=raw["audio_text"];row=normalized_row(raw)
            else:
                transcripts={}
                for turn in raw.get("history",[]):
                    if turn["role"]=="user":
                        transcripts[turn["audio"]]=asr.transcribe(root/a.split/turn["audio"])
                row=normalized_row(raw,transcripts)
                text=asr.transcribe(root/a.split/raw["audio"])
            prediction,trace=agent.run(row,text,allowed_specs(row,catalog),env,started=start)
            trace["transcript"]=text
            f.write(dumps(prediction)+"\n");f.flush()
            tf.write(dumps(trace)+"\n");tf.flush()
            print(dumps({"id":row["id"],"elapsed_s":trace["elapsed_s"],"reason":trace["reason"]}),flush=True)
    write_json(output.with_suffix(".runtime.json"),{"model":a.model,"revision":a.revision,
        "mode":a.mode,"oracle_text":a.oracle_text,"parameters":backend.parameter_report,
        "samples":len(rows),"deadline_s":a.deadline,"context":a.max_context})
if __name__=="__main__":main()
