#!/usr/bin/env python3
"""
Rebuild augmentation_data_sample/ for Kaggle v5:
  - 300 rows per condition (up from 100)
  - Remove all style_transfer variants (style transfer retired 2026-04-21)
  - Pair rain_light↔rain_heavy and snow_light↔snow_heavy by image_id
  - Repack cs_original / soda_voc_original / soda_ktsh_original to cover new id union
"""

import io
import shutil
import sys
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X")
AUG = ROOT / "augmentation_data"
SAMPLE = ROOT / "augmentation_data_sample"
N_ROWS = 300
SEED = 42

# External source paths (originals, not in augmentation_data)
CS_ORIG_DIR = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/LouisChen15___construction_site")
CS_ORIG_FILES = [
    "construction_site-test.arrow",
    "construction_site-train-00000-of-00002.arrow",
    "construction_site-train-00001-of-00002.arrow",
]
VOC_ROOT = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007")
KTSH_IMG = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/soda-ktsh/images")

STYLE_TRANSFER_FILES = [
    "cs_style_rain_test.arrow",
    "cs_style_rain_train.arrow",
    "cs_style_snow_test.arrow",
    "cs_style_snow_train.arrow",
    "soda_voc_style_rain.arrow",
    "soda_voc_style_snow.arrow",
]


def load_stream(p):
    with open(p, "rb") as f:
        return ipc.open_stream(f).read_all()


def safe_concat(slices):
    if not slices:
        return None
    return pa.concat_tables(slices, promote_options="default")


def sample_n(table, n, seed=SEED):
    """Sample n rows via row-by-row slice to avoid int32 offset overflow."""
    if table.num_rows <= n:
        slices = [table.slice(i, 1) for i in range(table.num_rows)]
    else:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(table.num_rows, size=n, replace=False).tolist())
        slices = [table.slice(i, 1) for i in idx]
    return safe_concat(slices)


def filter_by_ids(table, target_ids):
    ids_str = [str(x) for x in table.column("image_id").to_pylist()]
    target_set = set(str(x) for x in target_ids)
    matching = [i for i, x in enumerate(ids_str) if x in target_set]
    slices = [table.slice(i, 1) for i in matching]
    return safe_concat(slices) if slices else table.slice(0, 0)


def concat_dir(d):
    tables = [load_stream(p) for p in sorted(Path(d).glob("*.arrow"))]
    if not tables:
        return None
    return pa.concat_tables(tables, promote_options="default")


def write_sample(table, name):
    out = SAMPLE / name
    with ipc.new_stream(str(out), table.schema) as w:
        w.write_table(table)
    mb = out.stat().st_size / 1e6
    print(f"  {name}: {table.num_rows} rows, {mb:.1f} MB")


def remove_style_transfer():
    print("=== Removing style_transfer sample files ===")
    for f in STYLE_TRANSFER_FILES:
        p = SAMPLE / f
        if p.exists():
            p.unlink()
            print(f"  removed {f}")


def rebuild_cs_samples():
    print("\n=== CS samples (300 rows each) ===")
    # 1. Fog zones
    for zone in ["heavy", "medium", "light"]:
        t = concat_dir(AUG / f"construction_site/fog/diffusion/test/{zone}")
        sampled = sample_n(t, N_ROWS, seed=SEED + hash(zone) % 1000)
        write_sample(sampled, f"cs_fog_{zone}.arrow")

    # 2. Night
    t = load_stream(AUG / "construction_site/night/test/night_constructionsite_test.arrow")
    write_sample(sample_n(t, N_ROWS), "cs_night.arrow")

    # 3. Small
    t = load_stream(AUG / "construction_site/small/test/small_constructionsite_test.arrow")
    write_sample(sample_n(t, N_ROWS), "cs_small.arrow")

    # 4. Rain light + heavy (paired on image_id)
    rain_light_test = concat_dir(AUG / "construction_site/rain_snow/diffusion/test/rain")
    rain_light_train = load_stream(AUG / "construction_site/rain_snow/diffusion/train/rain/train_rain_v4.arrow")
    rain_heavy_test = concat_dir(AUG / "construction_site/rain_snow/diffusion/test/rain_heavy")
    rain_heavy_train = load_stream(AUG / "construction_site/rain_snow/diffusion/train/rain_heavy/train_rain_heavy.arrow")

    light_test_s = sample_n(rain_light_test, N_ROWS, seed=SEED + 1)
    light_train_s = sample_n(rain_light_train, N_ROWS, seed=SEED + 2)
    write_sample(light_test_s, "cs_diff_rain_test.arrow")
    write_sample(light_train_s, "cs_diff_rain_train.arrow")

    heavy_test_paired = filter_by_ids(rain_heavy_test, light_test_s.column("image_id").to_pylist())
    heavy_train_paired = filter_by_ids(rain_heavy_train, light_train_s.column("image_id").to_pylist())
    write_sample(heavy_test_paired, "cs_diff_rain_heavy_test.arrow")
    write_sample(heavy_train_paired, "cs_diff_rain_heavy_train.arrow")

    # 5. Snow light + heavy (paired) + train
    snow_light = concat_dir(AUG / "construction_site/rain_snow/diffusion/test/snow_light")
    snow_heavy = concat_dir(AUG / "construction_site/rain_snow/diffusion/test/snow_heavy")
    snow_train = load_stream(AUG / "construction_site/rain_snow/diffusion/train/snow/train_snow_v4.arrow")

    snow_light_s = sample_n(snow_light, N_ROWS, seed=SEED + 3)
    write_sample(snow_light_s, "cs_diff_snow_light_test.arrow")
    heavy_snow_paired = filter_by_ids(snow_heavy, snow_light_s.column("image_id").to_pylist())
    write_sample(heavy_snow_paired, "cs_diff_snow_heavy_test.arrow")
    write_sample(sample_n(snow_train, N_ROWS, seed=SEED + 4), "cs_diff_snow_train.arrow")


def rebuild_soda_voc_samples():
    print("\n=== SODA VOC samples (300 rows each) ===")
    # 1. Night
    t = load_stream(AUG / "soda_voc/night/soda_day2night.arrow")
    write_sample(sample_n(t, N_ROWS), "soda_voc_night.arrow")

    # 2. Small
    t = load_stream(AUG / "soda_voc/small/soda_small.arrow")
    write_sample(sample_n(t, N_ROWS), "soda_voc_small.arrow")

    # 3. Rain light + heavy (paired)
    rain_light = load_stream(AUG / "soda_voc/rain_snow/diffusion/rain.arrow")
    rain_heavy = load_stream(AUG / "soda_voc/rain_snow/diffusion/rain_heavy.arrow")
    light_s = sample_n(rain_light, N_ROWS, seed=SEED + 5)
    write_sample(light_s, "soda_voc_diff_rain.arrow")
    heavy_paired = filter_by_ids(rain_heavy, light_s.column("image_id").to_pylist())
    write_sample(heavy_paired, "soda_voc_diff_rain_heavy.arrow")

    # 4. Snow (single level)
    snow = load_stream(AUG / "soda_voc/rain_snow/diffusion/snow.arrow")
    write_sample(sample_n(snow, N_ROWS, seed=SEED + 6), "soda_voc_diff_snow.arrow")


def rebuild_soda_ktsh_samples():
    print("\n=== SODA KTSH samples (300 rows each) ===")
    rain_light = load_stream(AUG / "soda_ktsh/rain_snow/diffusion/rain.arrow")
    rain_heavy = load_stream(AUG / "soda_ktsh/rain_snow/diffusion/rain_heavy.arrow")
    snow = load_stream(AUG / "soda_ktsh/rain_snow/diffusion/snow.arrow")

    light_s = sample_n(rain_light, N_ROWS, seed=SEED + 7)
    write_sample(light_s, "soda_ktsh_diff_rain.arrow")
    heavy_paired = filter_by_ids(rain_heavy, light_s.column("image_id").to_pylist())
    write_sample(heavy_paired, "soda_ktsh_diff_rain_heavy.arrow")
    write_sample(sample_n(snow, N_ROWS, seed=SEED + 8), "soda_ktsh_diff_snow.arrow")


def collect_ids_by_source():
    """Collect image_ids by source from current sample dir."""
    cs_ids, voc_ids, ktsh_ids = set(), set(), set()
    for f in sorted(SAMPLE.iterdir()):
        if not f.name.endswith(".arrow"):
            continue
        t = load_stream(f)
        if "image_id" not in t.column_names:
            continue
        ids = [str(x) for x in t.column("image_id").to_pylist()]
        if f.name.startswith("cs_"):
            cs_ids.update(ids)
        elif f.name.startswith("soda_voc_"):
            voc_ids.update(ids)
        elif f.name.startswith("soda_ktsh_"):
            ktsh_ids.update(ids)
    return cs_ids, voc_ids, ktsh_ids


def repack_cs_original(target_ids):
    print("\n=== Repacking cs_original.arrow ===")
    tables = [load_stream(CS_ORIG_DIR / f) for f in CS_ORIG_FILES]
    full = pa.concat_tables(tables, promote_options="default")
    filt = filter_by_ids(full, target_ids)
    write_sample(filt, "cs_original.arrow")


def repack_soda_voc_original(target_ids):
    print("\n=== Repacking soda_voc_original.arrow ===")
    jpg_dir = VOC_ROOT / "JPEGImages"
    xml_dir = VOC_ROOT / "Annotations"
    ids, blobs, anns = [], [], []
    missing = []
    for stem in sorted(target_ids):
        img = jpg_dir / f"{stem}.jpg"
        xml = xml_dir / f"{stem}.xml"
        if not img.exists():
            missing.append(stem)
            continue
        blobs.append(img.read_bytes())
        anns.append(xml.read_text(encoding="utf-8") if xml.exists() else "")
        ids.append(stem)
    schema = pa.schema([
        pa.field("image_id", pa.string()),
        pa.field("image", pa.binary()),
        pa.field("annotation", pa.string()),
        pa.field("meta", pa.string()),
    ])
    metas = [None] * len(ids)
    tbl = pa.table({"image_id": ids, "image": blobs, "annotation": anns, "meta": metas}, schema=schema)
    write_sample(tbl, "soda_voc_original.arrow")
    if missing:
        print(f"  missing: {len(missing)} (first 5: {missing[:5]})")


def repack_soda_ktsh_original(target_ids):
    print("\n=== Repacking soda_ktsh_original.arrow ===")
    ids, blobs, fnames, caps, refs = [], [], [], [], []
    missing = []
    for stem in sorted(target_ids):
        img = KTSH_IMG / f"{stem}.jpg"
        if not img.exists():
            missing.append(stem)
            continue
        blobs.append(img.read_bytes())
        ids.append(stem)
        fnames.append(img.name)
        caps.append([])
        refs.append(stem)
    tbl = pa.table({
        "image": blobs,
        "image_id": ids,
        "filename": fnames,
        "captions": caps,
        "ref_id": refs,
    })
    write_sample(tbl, "soda_ktsh_original.arrow")
    if missing:
        print(f"  missing: {len(missing)} (first 5: {missing[:5]})")


def main():
    remove_style_transfer()
    rebuild_cs_samples()
    rebuild_soda_voc_samples()
    rebuild_soda_ktsh_samples()

    print("\n=== Collecting image_ids for original repack ===")
    cs_ids, voc_ids, ktsh_ids = collect_ids_by_source()
    print(f"  cs={len(cs_ids)}, voc={len(voc_ids)}, ktsh={len(ktsh_ids)}")

    repack_cs_original(cs_ids)
    repack_soda_voc_original(voc_ids)
    repack_soda_ktsh_original(ktsh_ids)

    print("\n=== Final state ===")
    total_mb = 0
    for p in sorted(SAMPLE.glob("*.arrow")):
        mb = p.stat().st_size / 1e6
        total_mb += mb
        t = load_stream(p)
        print(f"  {p.name:45s} rows={t.num_rows:5d}  size={mb:6.1f} MB")
    print(f"  TOTAL: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
