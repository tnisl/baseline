"""Generate a Kaggle notebook and metadata pinned to an exact source commit."""
import argparse
import json
from pathlib import Path
from umilife.core import write_json

def generate(git_ref="solution_paper",profile="pilot"):
    p=Path(__file__).resolve().parent
    cell_source=[
'''from pathlib import Path
import subprocess, sys, json, hashlib, zipfile

REPO_URL = "https://github.com/tnisl/baseline.git"
GIT_REF = __GIT_REF__
PROFILE = __PROFILE__  # smoke: 3 updates/stage; pilot: 80+40+40+40; full: 2/1/1/2 epochs
SOURCE = Path("/kaggle/working/paper_source")
WORK = Path("/kaggle/working/paper_run") / PROFILE
if not SOURCE.exists():
    subprocess.run(["git","clone","--filter=blob:none","--no-checkout",REPO_URL,str(SOURCE)],check=True)
subprocess.run(["git","-C",str(SOURCE),"fetch","origin",GIT_REF],check=True)
subprocess.run(["git","-C",str(SOURCE),"checkout","--detach","FETCH_HEAD"],check=True)
SOURCE_COMMIT=subprocess.check_output(["git","-C",str(SOURCE),"rev-parse","HEAD"],text=True).strip()
print("Source commit:",SOURCE_COMMIT)
''',
'''# Keep Kaggle's CUDA-matched PyTorch.
subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check",
                "-r",str(SOURCE/"requirements.txt")],check=True)
''',
'''# Find manifests without walking thousands of audio/environment directories.
sys.path.insert(0,str(SOURCE))
from umilife.paths import find_data_root
print("Locating competition manifests...",flush=True)
DATA_ROOT,archive_path=find_data_root("/kaggle/input")
if DATA_ROOT is None:
    destination=Path("/tmp/competition_data");destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target=(destination/member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):raise ValueError("Unsafe archive member")
        archive.extractall(destination)
    DATA_ROOT,_=find_data_root(destination)
expected=json.loads((SOURCE/"docs/data_audit.json").read_text())["train_sha256"]
actual=hashlib.sha256((DATA_ROOT/"train/train.jsonl").read_bytes()).hexdigest()
assert actual==expected, "Attached dataset differs from audited release"
WORK.mkdir(parents=True,exist_ok=True)
(WORK/"source.json").write_text(json.dumps({"source_commit":SOURCE_COMMIT,"train_sha256":actual}))
print("Dataset:",DATA_ROOT)
''',
'''# Fresh processes release GPU memory between rationale/global/local/validation phases.
subprocess.run([sys.executable,"-u",str(SOURCE/"finetune.py"),
    "--data-root",str(DATA_ROOT),"--output-dir",str(WORK),"--profile",PROFILE,
    "--model","Qwen/Qwen3-4B-Instruct-2507","--eval-limit","20"],
    cwd=SOURCE,check=True)
''',
'''status=json.loads((WORK/"status.json").read_text())
assert status["status"]=="complete",status
assert status["stages"]==["global","planner","caller","summarizer"]
global_meta=json.loads((WORK/"adapters/global/lineage.json").read_text())
for role in ("planner","caller","summarizer"):
    meta=json.loads((WORK/f"adapters/{role}/lineage.json").read_text())
    assert meta["initial_adapter"]["sha256"]==global_meta["final_adapter"]["sha256"]
    assert meta["final_adapter"]["finite"]
    assert meta["initial_adapter"]["sha256"]!=meta["final_adapter"]["sha256"]
for path in sorted(WORK.glob("*.metrics.json")):
    result=json.loads(path.read_text())
    print(path.name,{k:v for k,v in result.items() if k!="per_sample"})
print("Private notebook artifacts:",WORK)
'''
    ]
    cells=[{"cell_type":"markdown","metadata":{},"source":
        "# LifeMate — α-UMi / Qwen3-4B GLPFT\n"
        "Shared 4B backbone; global QLoRA → three independent local continuations. "
        "Qwen generates train-only rationales with prefix-only observations. "
        "Validation compares global vs local using oracle text and audio. "
        "Public test is inference-only. Pilot is a limited training experiment; full follows the paper's epoch schedule.\n"}]
    for index,source in enumerate(cell_source):
        source=source.replace("__GIT_REF__",json.dumps(git_ref)).replace("__PROFILE__",json.dumps(profile))
        cells.append({"id":f"step-{index}","cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source})
    cells[0]["id"]="overview"
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
              "nbformat":4,"nbformat_minor":5}
    slug="lctrnnguynkhi/lifemate-paper-qwen3-glpft"
    write_json(p/"notebooks/kaggle_glpft.ipynb",notebook)
    write_json(p/"notebooks/kernel-metadata.json",{"id":slug,"title":"LifeMate paper Qwen3 GLPFT",
        "code_file":"kaggle_glpft.ipynb","language":"python","kernel_type":"notebook","is_private":True,
        "enable_gpu":True,"enable_internet":True,"machine_shape":"NvidiaTeslaT4",
        "dataset_sources":["cymeu4l0t/zaloai-dataset"],"competition_sources":[],"kernel_sources":[]})
    return notebook

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--git-ref",default="solution_paper")
    p.add_argument("--profile",choices=("smoke","pilot","full"),default="pilot")
    a=p.parse_args();generate(a.git_ref,a.profile)
