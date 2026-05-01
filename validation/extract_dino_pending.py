"""Compute DINOv3 similarity + SSIM for shards still pending in v1.x release.

Uses the same model (facebook/dinov3-vitl16-pretrain-lvd1689m) and CSV format
(image_id, ssim, dino_sim, dino_dist) as extract_dino_ssim_all.py — so the
runner's _load_quality_csv pulls them in unchanged.

Each entry below maps an output CSV name to (synthetic_arrow_path, source_arrow_path).
The image is matched by image_id; missing matches are warned and skipped.
"""
from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X")
OUT_DIR = ROOT / "validation" / "results" / "dino_ssim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (csv_basename, synthetic_arrow, source_arrow, condition_label_for_log)
PENDING = [
    # cs10k small (outpainting)
    ("cs_small_test",  ROOT / "augmentation_data/construction_site/small/test/small_constructionsite_test.arrow",
                       Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data_arrow/construction_site_test.arrow")),
    ("cs_small_train", ROOT / "augmentation_data/construction_site/small/train/small_constructionsite_train.arrow",
                       Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/LouisChen15___construction_site/construction_site-train-00000-of-00002.arrow")),
    # soda_voc
    ("soda_voc_night",     ROOT / "augmentation_data/soda_voc/night/soda_day2night.arrow",
                            ROOT / "augmentation_data/soda_voc/original/soda_voc_original_first3000.arrow"),
    ("soda_voc_fog_light", ROOT / "augmentation_data/soda_voc/fog/test/fog_light.arrow",
                            ROOT / "augmentation_data/soda_voc/original/soda_voc_original_first3000.arrow"),
    ("soda_voc_fog_medium",ROOT / "augmentation_data/soda_voc/fog/test/fog_medium.arrow",
                            ROOT / "augmentation_data/soda_voc/original/soda_voc_original_first3000.arrow"),
    ("soda_voc_fog_heavy", ROOT / "augmentation_data/soda_voc/fog/test/fog_heavy.arrow",
                            ROOT / "augmentation_data/soda_voc/original/soda_voc_original_first3000.arrow"),
    ("soda_voc_small",     ROOT / "augmentation_data/soda_voc/small/soda_small.arrow",
                            ROOT / "augmentation_data/soda_voc/original/soda_voc_original_first3000.arrow"),
    ("soda_voc_extra2k",   ROOT / "augmentation_data/soda_voc/small/extra2k.arrow",
                            ROOT / "augmentation_data/soda_voc/original/soda_voc_original_first3000.arrow"),
    # soda_ktsh
    ("soda_ktsh_night",    ROOT / "augmentation_data/soda_ktsh/night/soda_ktsh_day2night.arrow",
                            ROOT / "augmentation_data/soda_ktsh/original/soda_ktsh_original.arrow"),
    ("soda_ktsh_fog_light",ROOT / "augmentation_data/soda_ktsh/fog/fog_light.arrow",
                            ROOT / "augmentation_data/soda_ktsh/original/soda_ktsh_original.arrow"),
    ("soda_ktsh_fog_medium",ROOT / "augmentation_data/soda_ktsh/fog/fog_medium.arrow",
                            ROOT / "augmentation_data/soda_ktsh/original/soda_ktsh_original.arrow"),
    ("soda_ktsh_fog_heavy",ROOT / "augmentation_data/soda_ktsh/fog/fog_heavy.arrow",
                            ROOT / "augmentation_data/soda_ktsh/original/soda_ktsh_original.arrow"),
    ("soda_ktsh_small",    ROOT / "augmentation_data/soda_ktsh/small/soda_ktsh_small.arrow",
                            ROOT / "augmentation_data/soda_ktsh/original/soda_ktsh_original.arrow"),
]


def load_arrow_lookup(
    path: Path,
    only_ids: set[str] | None = None,
    *,
    target_size: int | None = 512,
) -> dict[str, Image.Image]:
    """Load arrow shard into {image_id: PIL.Image} (RGB), streaming batch-by-batch.

    `target_size`: if set (default 256), each image is resized to fit within
    target_size x target_size using LANCZOS — bounds memory at ~200KB/image
    (cf. uncompressed 4K which is ~36MB). Both DINO (224 input) and SSIM
    (compute_ssim_batch resizes to 256) work fine at 256.
    Pass `target_size=None` to keep raw resolution.
    """
    out: dict[str, Image.Image] = {}
    with pa.OSFile(str(path), "rb") as f:
        try:
            r = ipc.open_stream(f)
            batches = list(r)
        except pa.ArrowInvalid:
            f.seek(0)
            r = ipc.open_file(f)
            batches = [r.get_batch(i) for i in range(r.num_record_batches)]
        for batch in batches:
            ids_col = batch.column("image_id").to_pylist()
            img_col = batch.column("image").to_pylist()
            for iid, b in zip(ids_col, img_col):
                iid_s = str(iid)
                if only_ids is not None and iid_s not in only_ids:
                    continue
                if isinstance(b, dict):
                    b = b["bytes"]
                if not b or len(b) < 4:
                    # Empty/corrupted upstream defect (e.g. soda_ktsh_night ~365
                    # rows from a failed day2night batch). Skip silently here;
                    # the row simply won't appear in the resulting DINO csv.
                    continue
                try:
                    im = Image.open(io.BytesIO(b)).convert("RGB")
                except (Image.UnidentifiedImageError, OSError):
                    continue
                if target_size is not None:
                    im.thumbnail((target_size, target_size), Image.LANCZOS)
                    im.load()
                out[iid_s] = im
            del ids_col, img_col
    return out


def load_dino():
    from transformers import AutoModel, AutoImageProcessor
    name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    print(f"  loading {name} ...")
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().cuda()
    return model, proc


@torch.no_grad()
def extract_cls(model, proc, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = proc(images=batch, return_tensors="pt").to("cuda")
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        feats.append(cls.cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_ssim_batch(origs: list[Image.Image], augs: list[Image.Image]) -> list[float]:
    out = []
    for o, a in zip(origs, augs):
        o_np = np.array(o.resize((256, 256)))
        a_np = np.array(a.resize((256, 256)))
        out.append(ssim_fn(o_np, a_np, channel_axis=2))
    return out


def main():
    import gc
    print(f"Found {len(PENDING)} pending shards")
    model, proc = load_dino()

    # Memory-bounded cache: only DINO feature arrays per source path (small).
    # Source PIL images are NOT cached across conditions — re-decoded per use.
    src_feat_cache: dict[str, dict[str, "np.ndarray"]] = {}

    summary = {}
    for name, syn_path, src_path in PENDING:
        out_csv = OUT_DIR / f"{name}.csv"
        if out_csv.exists():
            print(f"\n{name}: SKIP (already exists)")
            continue
        if not syn_path.exists():
            print(f"\n{name}: SKIP (synthetic missing: {syn_path})")
            continue
        if not src_path.exists():
            print(f"\n{name}: SKIP (source missing: {src_path})")
            continue
        print(f"\n=== {name} ===")
        print(f"  syn={syn_path}\n  src={src_path}")

        skey = str(src_path)
        if skey not in src_feat_cache:
            t0 = time.time()
            src_imgs = load_arrow_lookup(src_path)
            print(f"  loaded source {len(src_imgs)} in {time.time()-t0:.0f}s")
            src_ids = sorted(src_imgs.keys())
            src_imgs_list = [src_imgs[i] for i in src_ids]
            t0 = time.time()
            src_feats_arr = extract_cls(model, proc, src_imgs_list)
            print(f"  source DINO features in {time.time()-t0:.0f}s")
            src_feat_cache[skey] = {iid: src_feats_arr[k] for k, iid in enumerate(src_ids)}
            # Free PIL images and feature array — only the dict survives
            del src_imgs, src_imgs_list, src_feats_arr, src_ids
            gc.collect()

        src_feat_map = src_feat_cache[skey]

        t0 = time.time()
        syn_imgs = load_arrow_lookup(syn_path)
        print(f"  loaded synthetic {len(syn_imgs)} in {time.time()-t0:.0f}s")

        common = sorted(set(syn_imgs) & set(src_feat_map))
        if not common:
            print(f"  no matching ids, skipping")
            del syn_imgs
            gc.collect()
            continue
        print(f"  matched {len(common)} pairs (synthetic={len(syn_imgs)}, source-only={len(set(src_feat_map) - set(syn_imgs))})")

        t0 = time.time()
        syn_imgs_list = [syn_imgs[i] for i in common]
        syn_feats = extract_cls(model, proc, syn_imgs_list)
        print(f"  synthetic DINO in {time.time()-t0:.0f}s")

        src_feats_matched = np.stack([src_feat_map[i] for i in common])
        dino_sim = (src_feats_matched * syn_feats).sum(axis=1)

        # Re-decode just the matched source images for SSIM (no across-condition cache)
        t0 = time.time()
        src_imgs_for_ssim = load_arrow_lookup(src_path, only_ids=set(common))
        src_imgs_matched = [src_imgs_for_ssim[i] for i in common]
        ssims = compute_ssim_batch(src_imgs_matched, syn_imgs_list)
        print(f"  SSIM in {time.time()-t0:.0f}s")

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "ssim", "dino_sim", "dino_dist"])
            for i, img_id in enumerate(common):
                w.writerow([img_id, f"{ssims[i]:.4f}",
                             f"{dino_sim[i]:.4f}", f"{1 - dino_sim[i]:.4f}"])
        summary[name] = {
            "n": len(common),
            "dino_mean": float(dino_sim.mean()),
            "ssim_mean": float(np.mean(ssims)),
        }
        print(f"  saved {out_csv}  mean DINO={dino_sim.mean():.3f}  SSIM={np.mean(ssims):.3f}")

        # Free per-condition memory before next iteration
        del syn_imgs, syn_imgs_list, syn_feats, src_imgs_for_ssim, src_imgs_matched, ssims
        gc.collect()

    print("\n" + "=" * 70)
    print(f"{'condition':<25} {'N':>6} {'DINO':>8} {'SSIM':>8}")
    print("=" * 70)
    for name, s in summary.items():
        print(f"{name:<25} {s['n']:>6} {s['dino_mean']:>8.3f} {s['ssim_mean']:>8.3f}")


if __name__ == "__main__":
    main()
