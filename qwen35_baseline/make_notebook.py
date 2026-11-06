"""Build a standalone Kaggle notebook embedding only this baseline's source."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def build():
    payload = {name: (HERE/name).read_text() for name in
               ("agent.py", "run.py", "transcribe.py", "test_agent.py", "requirements.txt", "validation_ids.txt")}
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", """# LifeMate — Qwen3.5-4B baseline
Một model, native tool calling, non-thinking, greedy FP16. Không fine-tune.
Bật Internet và GPU T4 (một GPU được dùng; T4 x2 vẫn chỉ dùng GPU0).
Gắn dataset cuộc thi riêng tư, dạng thư mục hoặc ZIP. ASR cần thêm thư mục
encoder.onnx / decoder.onnx / joiner.onnx / tokens.txt do BTC cung cấp.
Chạy smoke trước. Chất lượng và VRAM trên Kaggle cần được kiểm chứng bằng lần chạy này.
Không công khai notebook có output chứa dữ liệu cuộc thi.
""")
    add("code", '''from pathlib import Path
import json, os, subprocess, sys, zipfile
PROFILE = "smoke"  # smoke: 8 mẫu; validation: 245 IDs cố định; public: toàn bộ public test
MODE = "oracle"    # train: oracle hoặc asr; public bắt buộc asr
DATA_ROOT = ""    # thư mục chứa train/ và public_test/; rỗng = tự tìm
DATA_ZIP = ""     # tùy chọn ZIP cuộc thi; rỗng = tự tìm nếu chưa có thư mục
ASR_MODEL_DIR = ""  # bắt buộc khi MODE=asr
MODEL = "Qwen/Qwen3.5-4B"  # hoặc đường dẫn snapshot đã gắn vào Kaggle
REVISION = "main"  # có thể đặt commit HF; SHA resolved được ghi vào config.json
MAX_INPUT = 16384
MAX_NEW_TOKENS = 768
MAX_STEPS = 8
RUN_NAME = "qwen35_smoke_01"  # đổi tên khi chạy lại; không reset môi trường ngầm
WORK = Path("/kaggle/working/qwen35_baseline")
WORK.mkdir(exist_ok=True)
''')
    add("code", "PAYLOAD = " + repr(payload) + '''
for name, content in PAYLOAD.items():
    (WORK/name).write_text(content)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(WORK/"requirements.txt")], check=True)
subprocess.run([sys.executable, "-B", str(WORK/"test_agent.py")], check=True)
''')
    add("code", '''assert PROFILE in {"smoke", "validation", "public"}
split = "public_test" if PROFILE == "public" else "train"
if split == "public_test":
    assert MODE == "asr", "Set MODE=asr for public test"
assert MODE in {"oracle", "asr"}
input_root = Path("/kaggle/input")
if DATA_ROOT:
    data_root = Path(DATA_ROOT)
else:
    candidates = sorted({p.parent.parent for p in input_root.rglob(f"{split}.jsonl")
                         if p.parent.name == split})
    if not candidates:
        archives = [Path(DATA_ZIP)] if DATA_ZIP else sorted(input_root.rglob("train+public_test.zip"))
        assert len(archives) == 1, "Set DATA_ROOT or DATA_ZIP explicitly"
        extracted = Path("/kaggle/working/lifemate_data")
        assert not extracted.exists(), "Extraction already exists; set DATA_ROOT to it"
        extracted.mkdir()
        with zipfile.ZipFile(archives[0]) as archive:
            for member in archive.infolist():
                target = (extracted/member.filename).resolve()
                assert target.is_relative_to(extracted.resolve()), "Unsafe ZIP path"
            archive.extractall(extracted)
        candidates = sorted({p.parent.parent for p in extracted.rglob(f"{split}.jsonl")
                             if p.parent.name == split})
    assert len(candidates) == 1, f"Ambiguous dataset roots: {candidates}"
    data_root = candidates[0]
assert (data_root/split/"zaic_tools"/"__init__.py").is_file()
output = Path("/kaggle/working")/RUN_NAME
assert not output.exists(), "Change RUN_NAME for a new run"
selection = ["--ids", str(WORK/"validation_ids.txt")] if PROFILE == "validation" else []
if PROFILE == "smoke":
    selection += ["--limit", "8"]
common = ["--data-root", str(data_root), "--split", split] + selection
print("DATA_ROOT:", data_root, "PROFILE:", PROFILE, "MODE:", MODE)
''')
    add("code", '''transcript_args = []
if MODE == "asr":
    assert ASR_MODEL_DIR, "Set ASR_MODEL_DIR to the supplied Zipformer ONNX directory"
    transcript_path = Path("/kaggle/working")/(RUN_NAME + "_transcripts.json")
    subprocess.run([sys.executable, "-B", str(WORK/"transcribe.py"), *common,
                    "--model-dir", ASR_MODEL_DIR, "--output", str(transcript_path)], check=True)
    transcript_args = ["--transcripts", str(transcript_path)]
command = [sys.executable, "-B", str(WORK/"run.py"), *common,
           "--mode", MODE, *transcript_args, "--model", MODEL, "--revision", REVISION,
           "--max-input", str(MAX_INPUT), "--max-new-tokens", str(MAX_NEW_TOKENS),
           "--max-steps", str(MAX_STEPS), "--output", str(output)]
subprocess.run(command, check=True)
''')
    add("code", '''print((output/"report.json").read_text())
# Schema validation of actual submission; labels are never passed to inference.
env = dict(os.environ, PYTHONPATH=str(data_root/split))
subprocess.run([sys.executable, "-m", "zaic_tools", "--split", split,
                "--root", str(data_root), "--validate", str(output/"submission.jsonl")],
               env=env, check=True)
import shutil
archive = shutil.make_archive(str(output), "zip", output)
print("Artifacts:", archive)
from IPython.display import FileLink, display
display(FileLink(str(Path(archive).relative_to("/kaggle/working"))))
''')
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 4}
    return notebook

if __name__ == "__main__":
    (HERE/"kaggle_qwen35_4b.ipynb").write_text(
        json.dumps(build(), ensure_ascii=False, indent=1) + "\n")
