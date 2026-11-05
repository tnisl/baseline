"""Build a standalone source snapshot; no competition data is embedded."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build():
    files = {str(path.relative_to(ROOT)): path.read_bytes()
             for path in sorted((ROOT / "src/twonode").glob("*.py"))}
    files["split.txt"] = (ROOT / "split.txt").read_bytes()
    files["requirements.txt"] = (HERE / "requirements.txt").read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    files["source_manifest.json"] = json.dumps(manifest, indent=2).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    blob = buffer.getvalue()
    cells = []

    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "source": text,
                      "execution_count": None, "outputs": []})

    markdown("""# LifeMate — Two-node Planner / Qwen3.5-4B

Chọn **GPU T4 ×2**, bật **Internet**, gắn dataset cuộc thi riêng tư.
Notebook nhúng source và đúng 200 validation ID của split.txt, không nhúng dữ liệu thi.
**Thinking tắt hoàn toàn**; output chỉ là JSON next-tool hoặc stop, không revise.

Train **Planner**, chưa train Caller: Unsloth DDP, microbatch 2/GPU × 2 GPU × accumulation 2 = 8.
LoRA không quantize 4-bit. Với Qwen3.5/T4, giữ đường FP32 activation của Unsloth
để tránh NaN; phần lớn base weights lưu FP16. Dtype thực tế được ghi trong artifact.
Sau train, process train thoát; hai process **Transformers + PEFT thông thường**
đánh giá các reference prefix, không dùng Unsloth inference, không sửa output.

Mặc định **smoke**: 16 train episodes, 8 valid episodes, 4 optimizer updates,
cùng batch/accumulation như full; kiểm tra gradient, adapter và reload/generation.
Đổi PROFILE thành **full** để train toàn bộ train partition trong 2 epoch và
đánh giá đủ 200 validation episodes. Smoke không chứng minh chất lượng mô hình.
Mỗi run cần tên thư mục mới. Không tự chạy smoke trước full hoặc khởi tạo full bằng adapter smoke.
""")
    code('''from pathlib import Path
import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile

PROFILE = "smoke"  # "smoke" or "full"
RUN_NAME = "planner_smoke_02"  # use a NEW name for each run
DATA_ROOT = None  # parent containing train/train.jsonl; autodetected if None
DATA_ARCHIVE = None  # optional explicit ZIP in the attached dataset
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_LENGTH = 8192  # prompt + output; no silent truncation; contest maximum is 16384
EVAL_BATCH = 2
EPOCHS = 2
LEARNING_RATE = 5e-5
EVALUATE_BASE = True  # paired base vs adapter, same prompts and decoding

assert PROFILE in ("smoke", "full")
assert RUN_NAME and Path(RUN_NAME).name == RUN_NAME
BASE = Path("/kaggle/working")
SOURCE = Path("/tmp/lifemate_two_node_source")
VENV = Path("/tmp/lifemate_two_node_env")
OUTPUT = BASE / "two_node_planner" / RUN_NAME
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise FileExistsError("Choose a fresh RUN_NAME; existing artifacts will not be overwritten")
subprocess.run(["nvidia-smi"], check=True)
''')
    markdown("## Verify embedded source\nNo git clone or separate source dataset is required.")
    code(f'''blob = base64.b64decode("{base64.b64encode(blob).decode()}")
assert hashlib.sha256(blob).hexdigest() == "{hashlib.sha256(blob).hexdigest()}"
SOURCE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    for member in archive.infolist():
        assert (SOURCE / member.filename).resolve().is_relative_to(SOURCE.resolve())
    archive.extractall(SOURCE)
manifest = json.loads((SOURCE / "source_manifest.json").read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest, name
assert hashlib.sha256((SOURCE / "split.txt").read_bytes()).hexdigest() == "e810b418d7018437a16892350e01cbee9526cf43ab4a190101433d4508498d2f"
print("Verified source files:", len(manifest))
''')
    markdown("## Isolated training environment\nUnsloth is imported only in the training subprocess. Validation starts in fresh processes and imports ordinary Transformers/PEFT. Installation and the first Triton compilation can take time.")
    code('''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
uv = [sys.executable, "-m", "uv"]
if not (VENV / "bin/python").exists():
    subprocess.run(uv + ["venv", "--seed", "--python", "3.12", str(VENV)], check=True)
PYTHON = str(VENV / "bin/python")
subprocess.run(uv + ["pip", "install", "--python", PYTHON, "-r", str(SOURCE / "requirements.txt")], check=True)
RUN_ENV = dict(os.environ, PYTHONPATH=str(SOURCE / "src"), TOKENIZERS_PARALLELISM="false",
               HF_HOME="/tmp/lifemate_two_node_hf", PYTHONUNBUFFERED="1")
subprocess.run([PYTHON, "-c", "import torch; print(torch.__version__); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); assert torch.cuda.device_count() == 2, 'Select GPU T4 x2 in Kaggle settings'"], env=RUN_ENV, check=True)
''')
    markdown("## Locate competition training data\nOnly train.jsonl and train_tools_catalog.json are needed. ZIP extraction reads only these two members and never extracts environment state files. If sources are ambiguous, set DATA_ROOT or DATA_ARCHIVE above.")
    code('''if DATA_ROOT is None:
    matches = sorted(Path("/kaggle/input").rglob("train/train.jsonl"))
    if len(matches) == 1:
        DATA_ROOT = matches[0].parent.parent
    elif matches:
        raise ValueError("Multiple training folders; set DATA_ROOT explicitly")
    else:
        archives = [Path(DATA_ARCHIVE)] if DATA_ARCHIVE else sorted(Path("/kaggle/input").rglob("*.zip"))
        candidates = []
        for path in archives:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                for member in sorted(names):
                    if member.endswith("train/train.jsonl") and not member.startswith("__MACOSX/"):
                        catalog = member[:-len("train.jsonl")] + "train_tools_catalog.json"
                        if catalog in names:
                            candidates.append((path, member, catalog))
        if len(candidates) != 1:
            raise ValueError("Attach the private competition dataset and/or set DATA_ROOT or DATA_ARCHIVE")
        path, data_member, catalog_member = candidates[0]
        DATA_ROOT = Path("/tmp/lifemate_two_node_data")
        (DATA_ROOT / "train").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for member in (data_member, catalog_member):
                (DATA_ROOT / "train" / Path(member).name).write_bytes(archive.read(member))
DATA_ROOT = Path(DATA_ROOT)
assert (DATA_ROOT / "train/train.jsonl").is_file()
assert (DATA_ROOT / "train/train_tools_catalog.json").is_file()
print("Training data:", DATA_ROOT)
''')
    markdown("## Train, then validate\nNo periodic validation or best-loss selection. Export the final adapter, exit DDP, then generate once per held-out prefix. Full validation covers all 200 episodes. Invalid JSON/tool choices count as wrong. Base comparison uses the identical frozen backbone with the adapter disabled. An overlength prompt fails explicitly; do not remove observations to make it fit.\n\nStartup prints start/complete markers with PID, rank and RAM/GPU memory for import, model load, LoRA attachment and Trainer initialization. A stage taking over 120 seconds dumps all Python thread stacks repeatedly into OUTPUT/diagnostics/rankN.STAGE.stacks.log. This records evidence; it does not automatically terminate the job or establish the cause of a stall.")
    code('''command = [PYTHON, "-u", "-m", "twonode.run", "--data-root", str(DATA_ROOT),
           "--split-file", str(SOURCE / "split.txt"), "--output", str(OUTPUT),
           "--profile", PROFILE, "--revision", MODEL_REVISION,
           "--max-length", str(MAX_LENGTH), "--eval-batch", str(EVAL_BATCH),
           "--epochs", str(EPOCHS), "--learning-rate", str(LEARNING_RATE)]
if not EVALUATE_BASE:
    command.append("--skip-base")
subprocess.run(command, env=RUN_ENV, check=True)
''')
    markdown("## Results\nReference-prefix decisions are not end-to-end agent/contest scores. Latency is amortized per batch, excludes model load and ASR. Keep notebook and outputs private because prepared examples contain competition data.")
    code('''reports = json.loads((OUTPUT / "validation/report.json").read_text())
for mode, report in reports.items():
    print(mode, json.dumps({key: report[key] for key in (
        "samples", "episodes", "accuracy", "next_tool_accuracy", "call_stop_accuracy",
        "stop_precision", "stop_recall", "stop_f1", "premature_stop_rate",
        "unnecessary_continuation_rate", "json_validity", "contract_validity",
        "episode_macro_accuracy", "all_prefixes_correct_episode_rate")}, indent=2))
print("Adapter:", OUTPUT / "adapter")
print("Raw predictions:", OUTPUT / "validation/adapter.predictions.jsonl")
print("Per-tool/app/step metrics:", OUTPUT / "validation/adapter.metrics.json")
if EVALUATE_BASE:
    print("Paired comparison:", (OUTPUT / "validation/comparison.json").read_text())
# Exclude recomputable compile caches and intermediate checkpoints from the download.
bundle = OUTPUT / "artifacts.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT.rglob("*")):
        relative = path.relative_to(OUTPUT)
        if path.is_file() and path != bundle and relative.parts[0] not in ("compile_cache", "checkpoints"):
            archive.write(path, relative)
print("Download private artifacts:", bundle)
''')
    for i, cell in enumerate(cells):
        cell["id"] = f"two-node-{i:02d}"
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}
    output = HERE / "kaggle_planner.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    build()
