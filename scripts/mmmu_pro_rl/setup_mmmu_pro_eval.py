#!/usr/bin/env python3
"""Setup pinned model and strictly isolated evaluation datasets for reproducible local GPU evaluation.

Zero-leakage invariant:
- Evaluation datasets (MMMU-Pro Standard10, MMMU-Pro Vision, MMMU Val) are exported
  exclusively under <MMMU_PRO_ROOT>/eval_data/.
- Training scripts never read from eval_data/.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from io import BytesIO
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("MMMU_PRO_ROOT", "."))
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MMMU_PRO_REPO = "MMMU/MMMU_Pro"
MMMU_PRO_REV = "0d7426df4ccb3d8704a992fe6850abee7c262127"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_training_archives() -> None:
    data_dir = ROOT / "data"
    main_dir = data_dir / "mmmu_train_20261016"
    if not (main_dir / "package/reservoir_candidates.jsonl").exists():
        print("[1/4] Extracting mmmu_train_20261016.tar.zst ...")
        subprocess.run(
            ["tar", "--zstd", "-xf", str(data_dir / "mmmu_train_20261016.tar.zst"), "-C", str(data_dir)],
            check=True,
        )
    else:
        print("[1/4] mmmu_train_20261016 already extracted.")

    exp_dir = data_dir / "mmmu_expansion_mmk12_20261115"
    if not (exp_dir / "package/reservoir_candidates.jsonl").exists():
        print("[2/4] Extracting mmmu_expansion_mmk12_20261115.tar.zst ...")
        subprocess.run(
            ["tar", "--zstd", "-xf", str(data_dir / "mmmu_expansion_mmk12_20261115.tar.zst"), "-C", str(data_dir)],
            check=True,
        )
    else:
        print("[2/4] mmmu_expansion_mmk12_20261115 already extracted.")


def download_base_model() -> Path:
    model_dir = ROOT / "models" / "qwen3-vl-4b-instruct"
    if not (model_dir / "config.json").exists():
        print(f"[3/4] Downloading {MODEL_ID}@{MODEL_REV} ...")
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REV,
            local_dir=str(model_dir),
        )
    else:
        print(f"[3/4] Base model already downloaded at {model_dir}.")
    return model_dir


def save_image_bytes(raw: bytes, img_dir: Path) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(BytesIO(raw)) as im:
        fmt = (im.format or "PNG").upper()
    ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tiff", "GIF": ".gif", "MPO": ".jpg"}.get(fmt, ".png")
    out_path = img_dir / f"{digest}{ext}"
    if not out_path.exists():
        out_path.write_bytes(raw)
    return str(out_path)


def export_mmmu_pro() -> None:
    eval_root = ROOT / "eval_data" / "mmmu_pro"
    std_out = eval_root / "standard10"
    vis_out = eval_root / "vision"

    # Standard 10 options
    if not (std_out / "inputs.jsonl").exists():
        print("[4a/4] Exporting MMMU-Pro Standard (10 options) ...")
        hf_dir = eval_root / "hf_cache"
        img_dir = std_out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        files = [f"standard (10 options)/test-{i:05d}-of-00002.parquet" for i in range(2)]
        inputs, labels = [], []
        for fn in files:
            p = Path(hf_hub_download(repo_id=MMMU_PRO_REPO, filename=fn, repo_type="dataset", revision=MMMU_PRO_REV, local_dir=str(hf_dir)))
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    opts = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]
                    image_paths = {}
                    for idx in range(1, 8):
                        im = row.get(f"image_{idx}")
                        if im is not None and im.get("bytes") is not None:
                            image_paths[str(idx)] = save_image_bytes(im["bytes"], img_dir)
                    inputs.append({
                        "id": row["id"],
                        "subject": row["subject"],
                        "question": row["question"],
                        "options": opts,
                        "image_paths": image_paths,
                    })
                    labels.append({
                        "id": row["id"],
                        "subject": row["subject"],
                        "answer": row["answer"],
                        "options": opts,
                    })
        assert len(inputs) == 1730, f"Expected 1730 Standard10 rows, got {len(inputs)}"
        write_jsonl(std_out / "inputs.jsonl", inputs)
        write_jsonl(std_out / "labels.jsonl", labels)
        print(f"  Exported {len(inputs)} Standard10 rows.")
    else:
        print("[4a/4] MMMU-Pro Standard10 already exported.")

    # Vision
    if not (vis_out / "inputs.jsonl").exists():
        print("[4b/4] Exporting MMMU-Pro Vision ...")
        hf_dir = eval_root / "hf_cache"
        img_dir = vis_out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        files = [f"vision/test-{i:05d}-of-00004.parquet" for i in range(4)]
        inputs, labels = [], []
        for fn in files:
            p = Path(hf_hub_download(repo_id=MMMU_PRO_REPO, filename=fn, repo_type="dataset", revision=MMMU_PRO_REV, local_dir=str(hf_dir)))
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    opts = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]
                    im = row.get("image")
                    assert im is not None and im.get("bytes") is not None, f"Missing vision image for {row['id']}"
                    img_path = save_image_bytes(im["bytes"], img_dir)
                    inputs.append({
                        "id": row["id"],
                        "subject": row["subject"],
                        "image_path": img_path,
                    })
                    labels.append({
                        "id": row["id"],
                        "subject": row["subject"],
                        "answer": row["answer"],
                        "options": opts,
                    })
        assert len(inputs) == 1730, f"Expected 1730 Vision rows, got {len(inputs)}"
        write_jsonl(vis_out / "inputs.jsonl", inputs)
        write_jsonl(vis_out / "labels.jsonl", labels)
        print(f"  Exported {len(inputs)} Vision rows.")
    else:
        print("[4b/4] MMMU-Pro Vision already exported.")


if __name__ == "__main__":
    extract_training_archives()
    download_base_model()
    export_mmmu_pro()
    print("=== Setup Complete ===")
