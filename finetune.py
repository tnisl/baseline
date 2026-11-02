"""Run rationale generation, data preparation, both GLPFT stages and validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from umilife.core import write_json,digest,dumps,SYSTEMS

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-root",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--profile",choices=("smoke","pilot","full"),default="pilot")
    p.add_argument("--model",default="Qwen/Qwen3-4B-Instruct-2507");p.add_argument("--revision")
    p.add_argument("--asr-model",default="Systran/faster-whisper-small")
    p.add_argument("--eval-limit",type=int,default=20);p.add_argument("--skip-public",action="store_true")
    a=p.parse_args();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    if not a.revision:
        previous=out/"run_config.json"
        if previous.exists():
            prior=json.loads(previous.read_text())
            if prior["model"]!=a.model or prior["profile"]!=a.profile:
                raise ValueError("Existing run configuration differs")
            a.revision=prior["revision"]
        else:
            from huggingface_hub import HfApi
            a.revision=HfApi().model_info(a.model).sha
    config=vars(a);write_json(out/"run_config.json",config)
    status={"status":"running","profile":a.profile,"stages":[]}
    cuda_libs=sorted((Path(sysconfig.get_path("purelib"))/"nvidia").glob("*/lib"))
    inherited_libs=os.environ.get("LD_LIBRARY_PATH","")
    library_path=":".join([*(str(p) for p in cuda_libs),inherited_libs])
    def run(module,args):
        command=[sys.executable,"-u","-m",module,*map(str,args)]
        print("RUN",command,flush=True)
        subprocess.run(command,check=True,env=os.environ|{"CUDA_VISIBLE_DEVICES":"0",
            "TOKENIZERS_PARALLELISM":"false","LD_LIBRARY_PATH":library_path})
    model=["--model",a.model,"--revision",a.revision]
    rationales=out/"rationales.jsonl";data=out/"prepared"
    limit=12 if a.profile=="smoke" else 160 if a.profile=="pilot" else None
    try:
        annotation=["--data-root",a.data_root,"--output",rationales,*model]
        if limit:annotation+=["--limit",limit]
        run("umilife.rationale",annotation)
        prepared_manifest=data/"manifest.json"
        if prepared_manifest.exists():
            cached=json.loads(prepared_manifest.read_text())
            expected_prompts=hashlib.sha256(dumps(SYSTEMS).encode()).hexdigest()
            if (cached["train_sha256"]!=digest(Path(a.data_root)/"train/train.jsonl") or
                    cached["rationales_sha256"]!=digest(rationales) or
                    cached["prompts_sha256"]!=expected_prompts):
                raise ValueError("Prepared dataset cache belongs to different inputs or prompts")
            for name,checksum in cached["files_sha256"].items():
                if digest(data/name)!=checksum:
                    raise ValueError("Prepared dataset cache checksum mismatch: "+name)
            print("Reuse verified prepared dataset",data,flush=True)
        else:
            prep=["--data-root",a.data_root,"--output",data,"--rationales",rationales]
            if limit:prep+=["--limit",limit]
            run("umilife.data",prep)
        for stage in ("global","planner","caller","summarizer"):
            completed=out/"adapters"/stage/"lineage.json"
            if completed.exists():
                meta=json.loads(completed.read_text())
                if meta["model"]!=a.model or meta["revision"]!=a.revision:
                    raise ValueError("Completed stage belongs to a different base revision")
                print("Reuse completed stage",stage,flush=True)
                status["stages"].append(stage)
                continue
            args=["--stage",stage,"--data-dir",data,"--output-dir",out,*model]
            if stage!="global":args+=["--global-adapter",out/"adapters/global"]
            if a.profile!="full":
                args+=["--max-steps",3 if a.profile=="smoke" else (80 if stage=="global" else 40),
                       "--gradient-accumulation",1 if a.profile=="smoke" else 4]
            checkpoints=out/"checkpoints"/stage
            available=sorted(checkpoints.glob("checkpoint-*"),key=lambda p:int(p.name.split("-")[-1])) if checkpoints.exists() else []
            if available: args+=["--resume-from-checkpoint",available[-1]]
            run("umilife.train",args)
            status["stages"].append(stage);write_json(out/"status.json",status)
        for mode in ("single","multi"):
            for oracle in (True,False):
                name=f"validation_{mode}_{'oracle' if oracle else 'audio'}"
                pred=out/f"{name}.jsonl"
                args=["--data-root",a.data_root,"--split","train","--adapter-dir",out/"adapters",
                      "--output",pred,"--manifest",data/"manifest.json","--limit",a.eval_limit,
                      "--mode",mode,"--asr-model",a.asr_model,*model]
                if oracle:args.append("--oracle-text")
                run("umilife.infer",args)
                run("umilife.evaluate",["--data-root",a.data_root,"--predictions",pred,
                    "--references",data/"references.json","--traces",pred.with_suffix(".traces.jsonl"),
                    "--output",out/f"{name}.metrics.json"])
        if not a.skip_public and a.profile!="smoke":
            # Public inputs are used only for final inference, never annotation, fitting or selection.
            run("umilife.infer",["--data-root",a.data_root,"--split","public_test",
                "--adapter-dir",out/"adapters","--output",out/"submission.jsonl",
                "--asr-model",a.asr_model,*model])
        status["status"]="complete"
    except Exception as exc:
        status.update(status="failed",error=str(exc));raise
    finally:write_json(out/"status.json",status)
if __name__=="__main__":main()
