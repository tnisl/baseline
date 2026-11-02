"""Download an offline inference bundle; execute after training and keep it outside Git."""
import argparse
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);p.add_argument("--output",default="models")
    a=p.parse_args();run=Path(a.run_dir);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    import json
    config=json.loads((run/"run_config.json").read_text())
    snapshot_download(config["model"],revision=config["revision"],local_dir=out/"qwen")
    snapshot_download("Systran/faster-whisper-small",local_dir=out/"whisper")
    for role in ("planner","caller","summarizer"):
        shutil.copytree(run/"adapters"/role,out/"adapters"/role,dirs_exist_ok=True)
    (out/"bundle.json").write_text(json.dumps(config,indent=2))
if __name__=="__main__":main()
