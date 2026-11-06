"""Transcribe selected current AND history user audio with supplied Zipformer."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from run import read_rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", default="train", choices=["train", "public_test"])
    p.add_argument("--ids", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import soundfile as sf
    import sherpa_onnx
    from scipy.signal import resample_poly
    rows = read_rows(a.data_root, a.split, a.ids, a.limit)
    names = list(dict.fromkeys(
        name for row in rows for name in
        [row["audio"]] + [m["audio"] for m in row["history"] if m["role"] == "user"]))
    if not names:
        raise ValueError("No audio selected")
    paths = {key: a.model_dir/f"{key}.onnx" for key in ("encoder", "decoder", "joiner")}
    paths["tokens"] = a.model_dir/"tokens.txt"
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        **{key: str(path) for key, path in paths.items()}, num_threads=4,
        sample_rate=16000, feature_dim=80, decoding_method="greedy_search", provider="cpu")
    if a.output.exists():
        raise ValueError("Transcript output already exists; choose a fresh path")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    texts = {}
    for index, name in enumerate(names, 1):
        base = (a.data_root/a.split).resolve()
        path = (base/name).resolve()
        if not path.is_relative_to(base):
            raise ValueError("Audio path escapes split root")
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError(f"Invalid audio: {name}")
        if rate != 16000:
            factor = math.gcd(rate, 16000)
            samples = resample_poly(samples, 16000//factor, rate//factor)
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, np.ascontiguousarray(samples, dtype=np.float32))
        recognizer.decode_stream(stream)
        texts[name] = stream.result.text.strip()
        if not texts[name]:
            raise ValueError(f"Empty ASR result: {name}")
        print(f"ASR {index}/{len(names)} {name}", flush=True)
    a.output.write_text(json.dumps(texts, ensure_ascii=False, indent=2))
    a.output.with_suffix(".meta.json").write_text(json.dumps(
        {"split": a.split, "selected_ids": [r["id"] for r in rows],
         "model_sha256": hashes, "sherpa_onnx": sherpa_onnx.__version__,
         "provider": "cpu", "current_and_history": True}, indent=2))

if __name__ == "__main__":
    main()
