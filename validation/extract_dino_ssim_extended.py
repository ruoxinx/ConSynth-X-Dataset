#!/usr/bin/env python3
"""
Extract DINOv3 similarity + SSIM for the augmentation conditions that are
NOT yet covered by validation/extract_dino_ssim_all.py — namely:

  • CS train  (rain_light, rain_heavy, snow_light, snow_heavy)
  • SODA-VOC  (rain_light, rain_heavy, snow_light, snow_heavy)
  • SODA-KTSH (rain_light, rain_heavy, snow_light, snow_heavy)

Outputs one CSV per (source, condition) at:
  validation/results/dino_ssim/{source}_{condition}.csv

Columns: image_id, ssim, dino_sim, dino_dist

This implementation streams images in chunks (CHUNK rows at a time) to keep
peak RAM bounded — needed for SODA-VOC snow_light (19k+ rows). DINO features
for originals are cached per-condition so we don't re-extract when ref_ids
overlap between rain_light/rain_heavy/snow_*.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

REPO = Path(__file__).resolve().parents[1]
AUG = REPO / "augmentation_data"
OUT_DIR = REPO / "validation" / "results" / "dino_ssim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Source originals
CS_TRAIN_SHARDS = sorted((Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/LouisChen15___construction_site")
                          ).glob("construction_site-train-*.arrow"))
SODA_VOC_JPEG = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007/JPEGImages")
SODA_KTSH_JPEG = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/soda-ktsh/images")

JOBS = {
    "cs_train": [
        ("rain_light", AUG/"construction_site/rain_snow/diffusion/train/rain_light/rain_light.arrow"),
        ("rain_heavy", AUG/"construction_site/rain_snow/diffusion/train/rain_heavy/rain_heavy.arrow"),
        ("snow_light", AUG/"construction_site/rain_snow/diffusion/train/snow_light/snow_light.arrow"),
        ("snow_heavy", AUG/"construction_site/rain_snow/diffusion/train/snow_heavy/snow_heavy.arrow"),
    ],
    "soda_voc": [
        ("rain_light", AUG/"soda_voc/rain_snow/diffusion/rain_light.arrow"),
        ("rain_heavy", AUG/"soda_voc/rain_snow/diffusion/rain_heavy.arrow"),
        ("snow_light", AUG/"soda_voc/rain_snow/diffusion/snow_light.arrow"),
        ("snow_heavy", AUG/"soda_voc/rain_snow/diffusion/snow_heavy.arrow"),
    ],
    "soda_ktsh": [
        ("rain_light", AUG/"soda_ktsh/rain_snow/diffusion/rain_light.arrow"),
        ("rain_heavy", AUG/"soda_ktsh/rain_snow/diffusion/rain_heavy.arrow"),
        ("snow_light", AUG/"soda_ktsh/rain_snow/diffusion/snow_light.arrow"),
        ("snow_heavy", AUG/"soda_ktsh/rain_snow/diffusion/snow_heavy.arrow"),
    ],
}

CHUNK = 256          # rows per streaming chunk
DINO_BATCH = 32      # GPU batch for DINO extract


def open_arrow(path: Path):
    with pa.memory_map(str(path), "r") as mm:
        try:
            return ipc.RecordBatchFileReader(mm).read_all()
        except Exception:
            mm.seek(0)
            return ipc.RecordBatchStreamReader(mm).read_all()


# ── Lazy original loaders (return PIL on demand, don't preload) ──

class CsTrainLookup:
    """Map image_id → (shard_idx, row_idx); decode JPEG bytes on demand."""

    def __init__(self):
        self._index: dict = {}
        self._tables: list = []
        for si, shard in enumerate(CS_TRAIN_SHARDS):
            t = open_arrow(shard)
            self._tables.append(t)
            ids = t.column("image_id").to_pylist()
            for ri, img_id in enumerate(ids):
                self._index[str(img_id)] = (si, ri)
        print(f"  CS train index: {len(self._index)} originals across {len(self._tables)} shards")

    def __call__(self, img_id: str):
        loc = self._index.get(str(img_id))
        if loc is None:
            return None
        si, ri = loc
        b = self._tables[si].column("image")[ri].as_py()
        if isinstance(b, dict):
            b = b["bytes"]
        return Image.open(io.BytesIO(b)).convert("RGB")


class JpegDirLookup:
    def __init__(self, jpeg_dir: Path):
        self.dir = jpeg_dir

    def __call__(self, img_id: str):
        p = self.dir / f"{img_id}.jpg"
        if not p.exists():
            p = self.dir / f"{img_id}.JPG"
        if not p.exists():
            return None
        return Image.open(p).convert("RGB")


# ── Streaming aug-Arrow iterator: yields (image_id, ref_id, PIL) ──

def iter_aug_rows(arrow_path: Path):
    t = open_arrow(arrow_path)
    n = t.num_rows
    has_ref = "ref_id" in t.column_names
    for i in range(n):
        img_id = str(t.column("image_id")[i].as_py())
        ref = str(t.column("ref_id")[i].as_py()) if has_ref else img_id
        b = t.column("image")[i].as_py()
        if isinstance(b, dict):
            b = b["bytes"]
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            im = None
        yield img_id, ref, im


def load_dino():
    from transformers import AutoModel, AutoImageProcessor
    name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().cuda()
    return model, proc


@torch.no_grad()
def extract_cls(model, proc, images, batch_size=DINO_BATCH):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = proc(images=batch, return_tensors="pt").to("cuda")
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        feats.append(cls.cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_ssim_pairs(orig_imgs, aug_imgs):
    out = []
    for o, a in zip(orig_imgs, aug_imgs):
        o_np = np.array(o.resize((256, 256)))
        a_np = np.array(a.resize((256, 256)))
        out.append(ssim_fn(o_np, a_np, channel_axis=2))
    return out


def process_condition(source: str, cond: str, arrow_path: Path,
                      orig_loader, model, proc):
    out_csv = OUT_DIR / f"{source}_{cond}.csv"
    if out_csv.exists():
        print(f"  [{cond}] already exists, skipping")
        return
    if not arrow_path.exists():
        print(f"  [{cond}] arrow missing: {arrow_path}")
        return

    print(f"  [{cond}] streaming {arrow_path.name}")
    t0 = time.time()

    orig_feat_cache: dict = {}  # ref_id → numpy.ndarray (DINO CLS)
    written = 0
    skipped = 0

    with open(out_csv, "w", newline="", buffering=1) as fout:
        w = csv.writer(fout)
        w.writerow(["image_id", "ssim", "dino_sim", "dino_dist"])

        chunk_ids = []
        chunk_refs = []
        chunk_aug = []
        chunk_orig = []

        def flush():
            nonlocal chunk_ids, chunk_refs, chunk_aug, chunk_orig, written
            if not chunk_ids:
                return
            # Originals needing fresh DINO
            need_idx = [i for i, r in enumerate(chunk_refs) if r not in orig_feat_cache]
            if need_idx:
                feats = extract_cls(model, proc, [chunk_orig[i] for i in need_idx])
                for k, i in enumerate(need_idx):
                    orig_feat_cache[chunk_refs[i]] = feats[k]
            o_feats = np.stack([orig_feat_cache[r] for r in chunk_refs])
            a_feats = extract_cls(model, proc, chunk_aug)
            dsim = (o_feats * a_feats).sum(axis=1)
            ssims = compute_ssim_pairs(chunk_orig, chunk_aug)
            for i in range(len(chunk_ids)):
                w.writerow([chunk_ids[i], f"{ssims[i]:.4f}",
                            f"{dsim[i]:.4f}", f"{1 - dsim[i]:.4f}"])
            written += len(chunk_ids)
            chunk_ids.clear(); chunk_refs.clear(); chunk_aug.clear(); chunk_orig.clear()
            gc.collect()

        for img_id, ref_id, aug_pil in iter_aug_rows(arrow_path):
            if aug_pil is None:
                skipped += 1
                continue
            o = orig_loader(ref_id)
            if o is None:
                skipped += 1
                continue
            chunk_ids.append(img_id)
            chunk_refs.append(ref_id)
            chunk_aug.append(aug_pil)
            chunk_orig.append(o)
            if len(chunk_ids) >= CHUNK:
                flush()
                if written % (CHUNK * 4) == 0:
                    print(f"    {written} rows written ({(time.time()-t0):.0f}s elapsed)")
        flush()

    print(f"  [{cond}] wrote {written} rows in {time.time()-t0:.0f}s "
          f"({skipped} skipped). cache size: {len(orig_feat_cache)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(JOBS.keys()))
    args = ap.parse_args()

    print(f"Source: {args.source}")
    print("Loading DINOv3 ViT-L/16 ...")
    model, proc = load_dino()

    if args.source == "cs_train":
        print("Indexing CS train original shards (lazy decode) ...")
        loader = CsTrainLookup()
    elif args.source == "soda_voc":
        loader = JpegDirLookup(SODA_VOC_JPEG)
    elif args.source == "soda_ktsh":
        loader = JpegDirLookup(SODA_KTSH_JPEG)
    else:
        raise SystemExit(f"unknown source: {args.source}")

    for cond, arrow_path in JOBS[args.source]:
        process_condition(args.source, cond, arrow_path, loader, model, proc)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
