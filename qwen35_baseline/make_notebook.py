"""Build a standalone Kaggle notebook embedding only this baseline's source."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def build(random200=False):
    payload = {name: (HERE/name).read_text() for name in
               ("agent.py", "run.py", "scoring.py", "transcribe.py", "test_agent.py", "test_scoring.py", "test_placement.py", "requirements.txt", "validation_ids.txt")}
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", """# LifeMate — Qwen3.5-4B baseline
Một model, native tool calling, non-thinking, greedy FP16. Không fine-tune.
Bật Internet và GPU T4 x2. Một model FP16 được chia layer lên hai GPU.
Batch inference 4 transcript; mỗi mẫu giữ ToolEnv và history riêng.
Prompt bổ sung điều kiện/ngoại lệ, thời gian, chọn đối tượng, thiếu thông tin và
kết quả chưa thành công; mô tả tool giữ Kind/Returns và schema gốc.
Gắn dataset cuộc thi riêng tư, dạng thư mục hoặc ZIP. ASR cần thêm thư mục
encoder.onnx / decoder.onnx / joiner.onnx / tokens.txt do BTC cung cấp.
Chạy smoke trước. Chất lượng và VRAM trên Kaggle cần được kiểm chứng bằng lần chạy này.
Không công khai notebook có output chứa dữ liệu cuộc thi.
""")
    add("code", '''from pathlib import Path
import json, os, subprocess, sys, zipfile
PROFILE = "smoke"  # smoke: 8 mẫu; validation: 245 IDs cố định; public: toàn bộ public test
SAMPLE_SIZE = 200  # random200: lấy ngẫu nhiên từ toàn bộ train, không thay thế
SEED = 42
MODE = "oracle"    # train: oracle hoặc asr; public bắt buộc asr
DATA_ROOT = ""    # thư mục chứa train/ và public_test/; rỗng = tự tìm
DATA_ZIP = ""     # tùy chọn ZIP cuộc thi; rỗng = tự tìm nếu chưa có thư mục
ASR_MODEL_DIR = ""  # bắt buộc khi MODE=asr
MODEL = "Qwen/Qwen3.5-4B"  # hoặc đường dẫn snapshot đã gắn vào Kaggle
REVISION = "main"  # có thể đặt commit HF; SHA resolved được ghi vào config.json
MAX_INPUT = 16384
MAX_NEW_TOKENS = 768
MAX_STEPS = 8
NUM_GPUS = 2
BATCH_SIZE = 4     # tối đa 4 mẫu đang hoạt động trong mỗi lượt generate
RUN_NAME = "qwen35_smoke_2gpu_batch4_policy_01"  # đổi tên khi chạy lại; không reset môi trường ngầm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
WORK = Path("/kaggle/working/qwen35_baseline")
WORK.mkdir(exist_ok=True)
''')
    add("code", "PAYLOAD = " + repr(payload) + '''
for name, content in PAYLOAD.items():
    (WORK/name).write_text(content)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(WORK/"requirements.txt")], check=True)
subprocess.run([sys.executable, "-B", str(WORK/"test_agent.py")], check=True)
subprocess.run([sys.executable, "-B", str(WORK/"test_scoring.py")], check=True)
subprocess.run([sys.executable, "-B", str(WORK/"test_placement.py")], check=True)
subprocess.run([sys.executable, "-c", "import torch; print('Visible GPUs:', torch.cuda.device_count()); assert torch.cuda.device_count() >= int(__import__('sys').argv[1]), 'Select GPU T4 x2 in Kaggle settings'", str(NUM_GPUS)], check=True)
''')
    add("code", '''assert PROFILE in {"smoke", "validation", "random200", "public"}
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
    if not candidates and PROFILE == "random200" and not DATA_ZIP:
        # KaggleHub attaches/downloads the existing private dataset using the
        # notebook's Kaggle identity; no credentials are embedded in source.
        import kagglehub
        dataset = Path(kagglehub.dataset_download("cymeu4l0t/zaloai-dataset"))
        candidates = sorted({p.parent.parent for p in dataset.rglob(f"{split}.jsonl")
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
if PROFILE == "random200":
    import hashlib, random
    dataset_path = data_root/"train"/"train.jsonl"
    all_ids = sorted(json.loads(line)["id"] for line in dataset_path.read_text().splitlines() if line.strip())
    assert len(all_ids) == len(set(all_ids)), "Duplicate dataset IDs"
    assert 0 < SAMPLE_SIZE <= len(all_ids)
    selected_ids = sorted(random.Random(SEED).sample(all_ids, SAMPLE_SIZE))
    (WORK/"selected_ids.txt").write_text("\\n".join(selected_ids) + "\\n")
    (WORK/"selection.json").write_text(json.dumps({
        "seed": SEED, "sample_size": SAMPLE_SIZE, "population_size": len(all_ids),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "selected_ids": selected_ids}, indent=2))
    selection = ["--ids", str(WORK/"selected_ids.txt")]
    print(f"Selected {SAMPLE_SIZE}/{len(all_ids)} train samples, seed={SEED}")
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
           "--max-steps", str(MAX_STEPS), "--num-gpus", str(NUM_GPUS),
           "--batch-size", str(BATCH_SIZE), "--output", str(output)]
subprocess.run(command, check=True)
''')
    add("code", '''import shutil
if PROFILE == "random200":
    for name in ("selected_ids.txt", "selection.json"):
        shutil.copyfile(WORK/name, output/name)
report = json.loads((output/"report.json").read_text())
if PROFILE == "random200":
    assert report["samples"] == SAMPLE_SIZE
print(json.dumps({k: v for k, v in report.items() if k != "generation"}, indent=2))
# Schema validation of actual submission; labels are never passed to inference.
env = dict(os.environ, PYTHONPATH=str(data_root/split))
subprocess.run([sys.executable, "-m", "zaic_tools", "--split", split,
                "--root", str(data_root), "--validate", str(output/"submission.jsonl")],
               env=env, check=True)
archive = shutil.make_archive(str(output), "zip", output)
print("Artifacts:", archive)
from IPython.display import FileLink, display
display(FileLink(str(Path(archive).relative_to("/kaggle/working"))))
''')
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 4}
    if random200:
        notebook["cells"][0]["source"] = ("# Qwen3.5-4B — Random 200 / transcript only\n\n"
            "Chọn ngẫu nhiên 200 mẫu từ toàn bộ train với seed=42. Dùng audio_text cho user hiện tại và history; không chạy ASR, không fine-tune.\n"
            "Bật Internet và GPU T4 x2, rồi Run All. Một model FP16 chia layer trên hai GPU; không tạo hai bản model. Dataset riêng tư cymeu4l0t/zaloai-dataset được tìm trong Input hoặc nạp bằng KaggleHub nếu chưa gắn.\n"
            "BATCH_SIZE=4: generate cùng lúc tối đa 4 transcript, left padding; mỗi mẫu có history và ToolEnv riêng. Mẫu đã dừng được bỏ khỏi các lượt sau; batch cuối có thể nhỏ hơn 4. VRAM batch 4 chưa được kiểm chứng trên GPU.\n"
            "Prompt mở rộng quy tắc history/sửa lời, điều kiện/ngoại lệ, chọn đối tượng, ngày giờ, schema và missing-info; mô tả tool có Kind/Returns. Chưa đo chất lượng model với prompt mới.\n"
            "Kết quả: score_100 (F1 gated by state hash), hash_accuracy, tool_call_f1, statuses và trace từng mẫu.\n"
            "Scorer local theo công thức công bố; fuzzy tokenizer có thể khác BTC. Không áp latency gate.\n"
            "Danh sách ID, seed và SHA256 dataset được lưu trong artifacts. Mẫu lỗi vẫn tính trong mẫu số 200.\n"
            "Chưa chạy GPU trước khi lưu notebook. Giữ notebook/output riêng tư vì có dữ liệu cuộc thi.\n").splitlines(True)
        config = "".join(notebook["cells"][1]["source"])
        config = config.replace('PROFILE = "smoke"', 'PROFILE = "random200"')
        config = config.replace('RUN_NAME = "qwen35_smoke_2gpu_batch4_policy_01"', 'RUN_NAME = "qwen35_random200_2gpu_batch4_policy_01"')
        # Preserve the input budget selected in the user's current Kaggle v6.
        config = config.replace('MAX_INPUT = 16384', 'MAX_INPUT = 8192')
        notebook["cells"][1]["source"] = config.splitlines(True)
    return notebook

if __name__ == "__main__":
    (HERE/"kaggle_qwen35_4b.ipynb").write_text(
        json.dumps(build(), ensure_ascii=False, indent=1) + "\n")
    (HERE/"kaggle_qwen35_4b_random200.ipynb").write_text(
        json.dumps(build(random200=True), ensure_ascii=False, indent=1) + "\n")
