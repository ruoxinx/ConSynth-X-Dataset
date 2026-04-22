#!/usr/bin/env python3
"""
Extract DINO similarity + merge SSIM for all 3004 snow_strong augmented images.

For each (original, augmented) pair:
  - Load both images
  - Extract DINOv3 CLS token embeddings
  - Compute cosine similarity (structural fidelity)
  - Merge with existing SSIM from meta CSVs

Output: single CSV with image_id, ssim, dino_sim, dino_dist per image
→ enables custom post-hoc filter design.
"""

import io
import csv
import time
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

# ── Paths ──
ORIG_PATH = (_DATA_ROOT / "augmentation_data_arrow/construction_site_test.arrow")
AUG_DIR = (_DATA_ROOT / "output/construction_site_test/diffusion_snow_strong")
OUT_PATH = (_REPO_ROOT / "validation/results/snow_strong_dino_ssim.csv")


def load_arrow_ids_images(path):
    """Load arrow file, return {image_id: PIL.Image}."""
    with open(path, "rb") as f:
        table = pa.ipc.open_stream(f).read_all()
    out = {}
    for i in range(len(table)):
        img_id = str(table.column("image_id")[i].as_py())
        b = table.column("image")[i].as_py()
        if isinstance(b, dict):
            b = b["bytes"]
        img = Image.open(io.BytesIO(b)).convert("RGB")
        out[img_id] = img
    return out


def load_meta_ssim():
    """Merge SSIM values from all meta CSV files."""
    ssim_map = {}
    for csv_file in sorted(AUG_DIR.glob("meta_*.csv")):
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ssim_map[row["image_id"]] = float(row["ssim"])
    return ssim_map


def load_dino():
    """Load DINOv3 ViT-L/16."""
    from transformers import AutoModel, AutoImageProcessor
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().cuda()
    return model, processor


@torch.no_grad()
def extract_cls(model, processor, images, batch_size=16):
    """Extract DINO CLS token embeddings (N, 1024)."""
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to("cuda")
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        feats.append(cls.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f"    {i + len(batch)}/{len(images)}")
    return np.concatenate(feats, axis=0)


def main():
    print("Loading originals (3004 images)...")
    t0 = time.time()
    originals = load_arrow_ids_images(ORIG_PATH)
    print(f"  loaded {len(originals)} in {time.time() - t0:.0f}s")

    print("Loading augmented snow_strong (3004 images, 3 arrow files)...")
    t0 = time.time()
    augmented = {}
    for arrow in sorted(AUG_DIR.glob("batch_*.arrow")):
        augmented.update(load_arrow_ids_images(arrow))
    print(f"  loaded {len(augmented)} in {time.time() - t0:.0f}s")

    # Align by image_id
    common_ids = sorted(set(originals) & set(augmented))
    print(f"  matched pairs: {len(common_ids)}")

    # Load SSIM
    ssim_map = load_meta_ssim()
    print(f"  SSIM entries: {len(ssim_map)}")

    # Load DINO
    print("\nLoading DINOv3 ViT-L/16...")
    model, processor = load_dino()

    # Extract DINO features
    print("\nExtracting DINO features for originals...")
    t0 = time.time()
    orig_images = [originals[i] for i in common_ids]
    orig_feats = extract_cls(model, processor, orig_images)
    print(f"  done in {time.time() - t0:.0f}s, shape={orig_feats.shape}")

    print("\nExtracting DINO features for augmented...")
    t0 = time.time()
    aug_images = [augmented[i] for i in common_ids]
    aug_feats = extract_cls(model, processor, aug_images)
    print(f"  done in {time.time() - t0:.0f}s, shape={aug_feats.shape}")

    # Cosine similarity (L2-normalized → dot product)
    dino_sim = (orig_feats * aug_feats).sum(axis=1)
    dino_dist = 1 - dino_sim

    # Save combined CSV
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "ssim", "dino_sim", "dino_dist"])
        for i, img_id in enumerate(common_ids):
            ssim = ssim_map.get(img_id, -1.0)
            w.writerow([img_id, f"{ssim:.4f}", f"{dino_sim[i]:.4f}", f"{dino_dist[i]:.4f}"])

    # Summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Images: {len(common_ids)}")
    print(f"  SSIM:  mean={np.mean(list(ssim_map.values())):.3f}, "
          f"median={np.median(list(ssim_map.values())):.3f}, "
          f"min={min(ssim_map.values()):.3f}, max={max(ssim_map.values()):.3f}")
    print(f"  DINO similarity:  mean={dino_sim.mean():.3f}, "
          f"median={np.median(dino_sim):.3f}, "
          f"min={dino_sim.min():.3f}, max={dino_sim.max():.3f}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
