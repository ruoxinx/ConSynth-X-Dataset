#!/usr/bin/env python3
"""
Rebuild augmentation_data_sample/ for Kaggle v6.

Changes vs v5 (2026-04-21 baseline):
  - Apply DINOv3 cosine-sim filter ``dino_sim > 0.80`` wherever a DINO CSV exists
    under validation/results/dino_ssim/.
  - Add two new CS conditions packaged in augmentation_data/:
        night_weather/rain_night   → cs_diff_rain_night.arrow
        night_weather/snow_night   → cs_diff_snow_night.arrow
  - Add the three SODA-VOC fog zones (heavy/medium/light) — previously absent
    from the Kaggle sample.
  - Pairing: rain_light↔rain_heavy and snow_light↔snow_heavy are sampled from
    the intersection of their DINO-eligible image_ids so paired rows are
    preserved under the stricter filter.

Target: 300 rows per condition (smaller if fewer rows pass the filter).

Run:
    python scripts/rebuild_kaggle_sample_v6.py              # write new arrows
    python scripts/rebuild_kaggle_sample_v6.py --dry-run    # report counts only
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X")
AUG = ROOT / "augmentation_data"
SAMPLE = ROOT / "augmentation_data_sample"
DINO_DIR = ROOT / "validation/results/dino_ssim"
N_ROWS = 300
SEED = 42
DINO_THR = 0.80  # user spec: "DINO > 0.8"

# External source paths for originals (not stored under augmentation_data/)
CS_ORIG_DIR = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/LouisChen15___construction_site")
CS_ORIG_FILES = [
    "construction_site-test.arrow",
    "construction_site-train-00000-of-00002.arrow",
    "construction_site-train-00001-of-00002.arrow",
]
VOC_ROOT = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007")
KTSH_IMG = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/soda-ktsh/images")


# ─── low-level helpers ────────────────────────────────────────────────────────
def load_stream(p):
    with open(p, "rb") as f:
        return ipc.open_stream(f).read_all()


def safe_concat(slices):
    if not slices:
        return None
    return pa.concat_tables(slices, promote_options="default")


def concat_dir(d):
    tables = [load_stream(p) for p in sorted(Path(d).glob("*.arrow"))]
    if not tables:
        return None
    return pa.concat_tables(tables, promote_options="default")


def read_dino_csv(path):
    """Return {image_id: dino_sim} as str→float."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["image_id"])] = float(row["dino_sim"])
    return out


def eligible_ids(dino_map, thr=DINO_THR):
    """Strict inequality per user spec: DINO > 0.80."""
    return {k for k, v in dino_map.items() if v > thr}


def sample_n(table, n, seed=SEED):
    """Row-by-row slice sample to avoid int32 offset overflow on binary cols."""
    if table.num_rows <= n:
        slices = [table.slice(i, 1) for i in range(table.num_rows)]
    else:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(table.num_rows, size=n, replace=False).tolist())
        slices = [table.slice(i, 1) for i in idx]
    return safe_concat(slices)


def filter_by_ids(table, target_ids):
    target_set = {str(x) for x in target_ids}
    ids_str = [str(x) for x in table.column("image_id").to_pylist()]
    matching = [i for i, x in enumerate(ids_str) if x in target_set]
    slices = [table.slice(i, 1) for i in matching]
    return safe_concat(slices) if slices else table.slice(0, 0)


def sample_eligible(table, eligible, n, seed):
    """Filter table to eligible ids, then sample n from survivors."""
    ids = [str(x) for x in table.column("image_id").to_pylist()]
    keep_idx = [i for i, x in enumerate(ids) if x in eligible]
    if len(keep_idx) <= n:
        chosen = keep_idx
    else:
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(keep_idx, size=n, replace=False).tolist())
    return safe_concat([table.slice(i, 1) for i in chosen]), len(keep_idx)


def write_sample(table, name, dry_run=False):
    out = SAMPLE / name
    if dry_run:
        print(f"  [dry-run] would write {name}: {table.num_rows} rows")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_stream(str(out), table.schema) as w:
        w.write_table(table)
    mb = out.stat().st_size / 1e6
    print(f"  {name}: {table.num_rows} rows, {mb:.1f} MB")


# ─── build phases ─────────────────────────────────────────────────────────────
def rebuild_cs_samples(dry_run):
    print("\n=== CS samples (target 300, DINO>0.80 where CSV available) ===")

    # 1. Fog zones — filter by DINO
    for zone in ["heavy", "medium", "light"]:
        t = concat_dir(AUG / f"construction_site/fog/diffusion/test/{zone}")
        dino = read_dino_csv(DINO_DIR / f"fog_{zone}.csv")
        elig = eligible_ids(dino)
        sampled, n_elig = sample_eligible(t, elig, N_ROWS, SEED + hash(zone) % 1000)
        print(f"  fog_{zone}: source={t.num_rows}  eligible={n_elig}  sampled={sampled.num_rows}")
        write_sample(sampled, f"cs_fog_{zone}.arrow", dry_run)

    # 2. Night — filter by DINO
    t = load_stream(AUG / "construction_site/night/test/night_constructionsite_test.arrow")
    dino = read_dino_csv(DINO_DIR / "night.csv")
    elig = eligible_ids(dino)
    sampled, n_elig = sample_eligible(t, elig, N_ROWS, SEED)
    print(f"  night: source={t.num_rows}  eligible={n_elig}  sampled={sampled.num_rows}")
    write_sample(sampled, "cs_night.arrow", dry_run)

    # 3. Small — no DINO CSV
    t = load_stream(AUG / "construction_site/small/test/small_constructionsite_test.arrow")
    sampled = sample_n(t, N_ROWS)
    print(f"  small: source={t.num_rows}  sampled={sampled.num_rows} (no DINO CSV)")
    write_sample(sampled, "cs_small.arrow", dry_run)

    # 4. Rain light + heavy — DINO filter on BOTH, pair via intersection of eligible ids
    rain_light_test = concat_dir(AUG / "construction_site/rain_snow/diffusion/test/rain")
    rain_heavy_test = concat_dir(AUG / "construction_site/rain_snow/diffusion/test/rain_heavy")
    dino_l = read_dino_csv(DINO_DIR / "ip2p_rain.csv")
    dino_h = read_dino_csv(DINO_DIR / "ip2p_rain_heavy.csv")
    elig_pair = eligible_ids(dino_l) & eligible_ids(dino_h)
    print(f"  rain pair: light elig={len(eligible_ids(dino_l))}  heavy elig={len(eligible_ids(dino_h))}  ∩={len(elig_pair)}")
    light_s, _ = sample_eligible(rain_light_test, elig_pair, N_ROWS, SEED + 1)
    heavy_paired = filter_by_ids(rain_heavy_test, light_s.column("image_id").to_pylist())
    write_sample(light_s, "cs_diff_rain_test.arrow", dry_run)
    write_sample(heavy_paired, "cs_diff_rain_heavy_test.arrow", dry_run)

    # Rain TRAIN — no DINO CSV, sample 300 light then pair heavy by id
    rain_light_train = load_stream(AUG / "construction_site/rain_snow/diffusion/train/rain/train_rain_v4.arrow")
    rain_heavy_train = load_stream(AUG / "construction_site/rain_snow/diffusion/train/rain_heavy/train_rain_heavy.arrow")
    train_l = sample_n(rain_light_train, N_ROWS, SEED + 2)
    train_h = filter_by_ids(rain_heavy_train, train_l.column("image_id").to_pylist())
    print(f"  rain train (no DINO): light={train_l.num_rows}  heavy(paired)={train_h.num_rows}")
    write_sample(train_l, "cs_diff_rain_train.arrow", dry_run)
    write_sample(train_h, "cs_diff_rain_heavy_train.arrow", dry_run)

    # 5. Snow light + heavy TEST — DINO filter on both, pair via intersection
    snow_light = concat_dir(AUG / "construction_site/rain_snow/diffusion/test/snow_light")
    snow_heavy = concat_dir(AUG / "construction_site/rain_snow/diffusion/test/snow_heavy")
    dino_sl = read_dino_csv(DINO_DIR / "ip2p_snow_light.csv")
    dino_sh = read_dino_csv(DINO_DIR / "ip2p_snow_heavy.csv")
    elig_snow = eligible_ids(dino_sl) & eligible_ids(dino_sh)
    print(f"  snow pair: light elig={len(eligible_ids(dino_sl))}  heavy elig={len(eligible_ids(dino_sh))}  ∩={len(elig_snow)}")
    snow_l_s, _ = sample_eligible(snow_light, elig_snow, N_ROWS, SEED + 3)
    snow_h_paired = filter_by_ids(snow_heavy, snow_l_s.column("image_id").to_pylist())
    write_sample(snow_l_s, "cs_diff_snow_light_test.arrow", dry_run)
    write_sample(snow_h_paired, "cs_diff_snow_heavy_test.arrow", dry_run)

    # Snow TRAIN — no DINO CSV
    snow_train = load_stream(AUG / "construction_site/rain_snow/diffusion/train/snow/train_snow_v4.arrow")
    write_sample(sample_n(snow_train, N_ROWS, SEED + 4), "cs_diff_snow_train.arrow", dry_run)

    # 6. NEW: night_weather/rain_night and snow_night — DINO filter
    rn = concat_dir(AUG / "construction_site/night_weather/rain_night")
    dino_rn = read_dino_csv(DINO_DIR / "night_rain.csv")
    elig_rn = eligible_ids(dino_rn)
    rn_s, n_elig = sample_eligible(rn, elig_rn, N_ROWS, SEED + 10)
    print(f"  rain_night: source={rn.num_rows}  eligible={n_elig}  sampled={rn_s.num_rows}")
    write_sample(rn_s, "cs_diff_rain_night.arrow", dry_run)

    sn = concat_dir(AUG / "construction_site/night_weather/snow_night")
    dino_sn = read_dino_csv(DINO_DIR / "night_snow.csv")
    elig_sn = eligible_ids(dino_sn)
    sn_s, n_elig = sample_eligible(sn, elig_sn, N_ROWS, SEED + 11)
    print(f"  snow_night: source={sn.num_rows}  eligible={n_elig}  sampled={sn_s.num_rows}")
    write_sample(sn_s, "cs_diff_snow_night.arrow", dry_run)


def rebuild_soda_voc_samples(dry_run):
    print("\n=== SODA VOC samples (target 300, no DINO CSV for SODA splits) ===")

    t = load_stream(AUG / "soda_voc/night/soda_day2night.arrow")
    write_sample(sample_n(t, N_ROWS, SEED + 20), "soda_voc_night.arrow", dry_run)

    t = load_stream(AUG / "soda_voc/small/soda_small.arrow")
    write_sample(sample_n(t, N_ROWS, SEED + 21), "soda_voc_small.arrow", dry_run)

    rain_light = load_stream(AUG / "soda_voc/rain_snow/diffusion/rain.arrow")
    rain_heavy = load_stream(AUG / "soda_voc/rain_snow/diffusion/rain_heavy.arrow")
    light_s = sample_n(rain_light, N_ROWS, SEED + 22)
    write_sample(light_s, "soda_voc_diff_rain.arrow", dry_run)
    heavy_paired = filter_by_ids(rain_heavy, light_s.column("image_id").to_pylist())
    write_sample(heavy_paired, "soda_voc_diff_rain_heavy.arrow", dry_run)

    snow = load_stream(AUG / "soda_voc/rain_snow/diffusion/snow.arrow")
    write_sample(sample_n(snow, N_ROWS, SEED + 23), "soda_voc_diff_snow.arrow", dry_run)

    # NEW: SODA VOC fog 3 zones
    for zone in ["heavy", "medium", "light"]:
        t = concat_dir(AUG / f"soda_voc/fog/diffusion/test/{zone}")
        sampled = sample_n(t, N_ROWS, SEED + 24 + hash(zone) % 100)
        print(f"  soda_voc fog_{zone}: source={t.num_rows}  sampled={sampled.num_rows}")
        write_sample(sampled, f"soda_voc_fog_{zone}.arrow", dry_run)


def rebuild_soda_ktsh_samples(dry_run):
    print("\n=== SODA KTSH samples (target 300, no DINO CSV) ===")
    rain_light = load_stream(AUG / "soda_ktsh/rain_snow/diffusion/rain.arrow")
    rain_heavy = load_stream(AUG / "soda_ktsh/rain_snow/diffusion/rain_heavy.arrow")
    snow = load_stream(AUG / "soda_ktsh/rain_snow/diffusion/snow.arrow")

    light_s = sample_n(rain_light, N_ROWS, SEED + 30)
    write_sample(light_s, "soda_ktsh_diff_rain.arrow", dry_run)
    heavy_paired = filter_by_ids(rain_heavy, light_s.column("image_id").to_pylist())
    write_sample(heavy_paired, "soda_ktsh_diff_rain_heavy.arrow", dry_run)
    write_sample(sample_n(snow, N_ROWS, SEED + 31), "soda_ktsh_diff_snow.arrow", dry_run)


# ─── originals repack ─────────────────────────────────────────────────────────
def collect_ids_by_source():
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


def repack_cs_original(target_ids, dry_run):
    print("\n=== Repacking cs_original.arrow ===")
    tables = [load_stream(CS_ORIG_DIR / f) for f in CS_ORIG_FILES]
    full = pa.concat_tables(tables, promote_options="default")
    filt = filter_by_ids(full, target_ids)
    print(f"  CS originals: ids requested={len(target_ids)}  matched={filt.num_rows}")
    write_sample(filt, "cs_original.arrow", dry_run)


def repack_soda_voc_original(target_ids, dry_run):
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
    write_sample(tbl, "soda_voc_original.arrow", dry_run)
    if missing:
        print(f"  missing: {len(missing)} (first 5: {missing[:5]})")


def repack_soda_ktsh_original(target_ids, dry_run):
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
    write_sample(tbl, "soda_ktsh_original.arrow", dry_run)
    if missing:
        print(f"  missing: {len(missing)} (first 5: {missing[:5]})")


# ─── entrypoint ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-originals", action="store_true",
                    help="Skip the (slow) originals repack — useful while iterating on augmented samples.")
    args = ap.parse_args()

    SAMPLE.mkdir(parents=True, exist_ok=True)
    print(f"Rebuilding Kaggle sample v6 at {SAMPLE} (DINO>{DINO_THR:.2f}, N={N_ROWS})")

    rebuild_cs_samples(args.dry_run)
    rebuild_soda_voc_samples(args.dry_run)
    rebuild_soda_ktsh_samples(args.dry_run)

    if not args.skip_originals:
        print("\n=== Collecting image_ids for original repack ===")
        cs_ids, voc_ids, ktsh_ids = collect_ids_by_source()
        print(f"  cs={len(cs_ids)}, voc={len(voc_ids)}, ktsh={len(ktsh_ids)}")
        repack_cs_original(cs_ids, args.dry_run)
        repack_soda_voc_original(voc_ids, args.dry_run)
        repack_soda_ktsh_original(ktsh_ids, args.dry_run)

    if not args.dry_run:
        print("\n=== Final state ===")
        total_mb = 0
        for p in sorted(SAMPLE.glob("*.arrow")):
            mb = p.stat().st_size / 1e6
            total_mb += mb
            t = load_stream(p)
            print(f"  {p.name:45s} rows={t.num_rows:5d}  size={mb:7.1f} MB")
        print(f"  TOTAL: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
