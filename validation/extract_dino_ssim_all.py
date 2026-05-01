#!/usr/bin/env python3
"""
Extract DINOv3 similarity + SSIM for ALL augmentation conditions.

Outputs one CSV per condition at:
  validation/results/dino_ssim/{condition}.csv

Columns: image_id, ssim, dino_sim, dino_dist
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
from skimage.metrics import structural_similarity as ssim_fn

# ── Paths ──
CONSTRUCTION = _DATA_ROOT
CONSYNTHX = _REPO_ROOT
ARROW_DATA = CONSTRUCTION / "augmentation_data_arrow"
IP2P_DATA = CONSYNTHX / "augmentation_data" / "construction_site" / "rain_snow" / "diffusion" / "test"
FOG_DATA = CONSYNTHX / "augmentation_data" / "construction_site" / "fog" / "diffusion" / "test"
NIGHT_DATA = CONSYNTHX / "augmentation_data" / "construction_site" / "night" / "test"
NIGHT_WEATHER_DATA = CONSYNTHX / "augmentation_data" / "construction_site" / "night_weather"
OUT_DIR = CONSYNTHX / "validation" / "results" / "dino_ssim"

ORIG_PATH = ARROW_DATA / "construction_site_test.arrow"

ST_DATA = CONSYNTHX / "augmentation_data" / "construction_site" / "rain_snow" / "style_transfer" / "test"

# ── Condition registry ──
CONDITIONS = {
    # Style transfer — single Arrow files
    "st_rain_a":  {"type": "arrow", "path": ST_DATA / "weather_test_style_rain_0.arrow"},
    "st_rain_b":  {"type": "arrow", "path": ST_DATA / "weather_test_style_rain_1.arrow"},
    "st_rain_c":  {"type": "arrow", "path": ST_DATA / "weather_test_style_rain_2.arrow"},
    "st_snow_a":  {"type": "arrow", "path": ST_DATA / "weather_test_style_snow_0.arrow"},
    "st_snow_b":  {"type": "arrow", "path": ST_DATA / "weather_test_style_snow_1.arrow"},
    "st_snow_c":  {"type": "arrow", "path": ST_DATA / "weather_test_style_snow_2.arrow"},
    # IP2P diffusion v4 (IP2P + physics overlay + LPIPS filter) — directories of Arrow batches
    "ip2p_rain":       {"type": "dir", "path": IP2P_DATA / "rain"},
    "ip2p_rain_heavy": {"type": "dir", "path": IP2P_DATA / "rain_heavy"},
    "ip2p_snow_light": {"type": "dir", "path": IP2P_DATA / "snow_light"},
    "ip2p_snow_heavy": {"type": "dir", "path": IP2P_DATA / "snow_heavy"},
    # Night CycleGAN
    "night": {"type": "arrow", "path": NIGHT_DATA / "night_constructionsite_test.arrow"},
    # Fog physics (3 intensities)
    "fog_light":  {"type": "dir", "path": FOG_DATA / "light"},
    "fog_medium": {"type": "dir", "path": FOG_DATA / "medium"},
    "fog_heavy":  {"type": "dir", "path": FOG_DATA / "heavy"},
    # Night + weather (Order B: IP2P -> CycleGAN Night -> physics)
    "night_rain": {"type": "dir", "path": NIGHT_WEATHER_DATA / "rain_night"},
    "night_snow": {"type": "dir", "path": NIGHT_WEATHER_DATA / "snow_night"},
}


def load_arrow_lookup(path):
    with open(path, "rb") as f:
        try:
            table = pa.ipc.open_file(path).read_all()
        except (pa.ArrowInvalid, Exception):
            f.seek(0)
            table = pa.ipc.open_stream(f).read_all()
    out = {}
    for i in range(len(table)):
        img_id = str(table.column("image_id")[i].as_py())
        b = table.column("image")[i].as_py()
        if isinstance(b, dict):
            b = b["bytes"]
        out[img_id] = Image.open(io.BytesIO(b)).convert("RGB")
    return out


def load_dir_lookup(dir_path):
    out = {}
    for af in sorted(Path(dir_path).glob("*.arrow")):
        out.update(load_arrow_lookup(af))
    return out


def load_condition(cfg):
    if cfg["type"] == "arrow":
        return load_arrow_lookup(cfg["path"])
    return load_dir_lookup(cfg["path"])


def load_dino():
    from transformers import AutoModel, AutoImageProcessor
    name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().cuda()
    return model, proc


@torch.no_grad()
def extract_cls(model, proc, images, batch_size=32):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = proc(images=batch, return_tensors="pt").to("cuda")
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        feats.append(cls.cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_ssim_batch(origs, augs):
    """Compute SSIM for each (orig, aug) pair at 256x256."""
    out = []
    for o, a in zip(origs, augs):
        o_np = np.array(o.resize((256, 256)))
        a_np = np.array(a.resize((256, 256)))
        out.append(ssim_fn(o_np, a_np, channel_axis=2))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading originals...")
    t0 = time.time()
    originals = load_arrow_lookup(ORIG_PATH)
    print(f"  {len(originals)} images in {time.time()-t0:.0f}s")

    print("\nLoading DINOv3...")
    model, proc = load_dino()

    # Cache original DINO features by image_id
    print("Extracting DINO features for originals (cached for all conditions)...")
    t0 = time.time()
    orig_ids = sorted(originals.keys())
    orig_imgs_list = [originals[i] for i in orig_ids]
    orig_feats_all = extract_cls(model, proc, orig_imgs_list)
    orig_feat_map = {i: orig_feats_all[k] for k, i in enumerate(orig_ids)}
    print(f"  done in {time.time()-t0:.0f}s")

    summary = {}
    for cond_name, cfg in CONDITIONS.items():
        out_csv = OUT_DIR / f"{cond_name}.csv"
        if out_csv.exists():
            print(f"\n{cond_name}: already exists, skipping")
            # Read for summary
            with open(out_csv) as f:
                rows = list(csv.DictReader(f))
            summary[cond_name] = {
                "n": len(rows),
                "dino_mean": np.mean([float(r["dino_sim"]) for r in rows]),
                "ssim_mean": np.mean([float(r["ssim"]) for r in rows]),
            }
            continue

        print(f"\n=== {cond_name} ===")
        t0 = time.time()
        try:
            aug_lookup = load_condition(cfg)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue
        print(f"  loaded {len(aug_lookup)} augmented in {time.time()-t0:.0f}s")

        # Matched pairs
        common = sorted(set(aug_lookup) & set(originals))
        if not common:
            print("  no matching ids, skipping")
            continue
        print(f"  matched {len(common)} pairs")

        # Extract DINO for augmented only
        t0 = time.time()
        aug_imgs_list = [aug_lookup[i] for i in common]
        aug_feats = extract_cls(model, proc, aug_imgs_list)
        print(f"  DINO extracted in {time.time()-t0:.0f}s")

        # DINO similarity
        orig_feats_matched = np.stack([orig_feat_map[i] for i in common])
        dino_sim = (orig_feats_matched * aug_feats).sum(axis=1)

        # SSIM
        t0 = time.time()
        orig_imgs_matched = [originals[i] for i in common]
        ssims = compute_ssim_batch(orig_imgs_matched, aug_imgs_list)
        print(f"  SSIM computed in {time.time()-t0:.0f}s")

        # Save
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "ssim", "dino_sim", "dino_dist"])
            for i, img_id in enumerate(common):
                w.writerow([img_id, f"{ssims[i]:.4f}",
                            f"{dino_sim[i]:.4f}", f"{1 - dino_sim[i]:.4f}"])

        summary[cond_name] = {
            "n": len(common),
            "dino_mean": float(dino_sim.mean()),
            "ssim_mean": float(np.mean(ssims)),
        }
        print(f"  Saved: {out_csv}")
        print(f"  Mean DINO={dino_sim.mean():.3f}, SSIM={np.mean(ssims):.3f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Condition':<20} {'N':>5} {'DINO mean':>10} {'SSIM mean':>10}")
    print(f"{'='*70}")
    for cond, s in summary.items():
        print(f"{cond:<20} {s['n']:>5} {s['dino_mean']:>10.3f} {s['ssim_mean']:>10.3f}")


if __name__ == "__main__":
    main()
