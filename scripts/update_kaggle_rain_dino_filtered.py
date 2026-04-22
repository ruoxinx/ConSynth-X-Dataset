#!/usr/bin/env python3
"""
Select 300 samples from CS test rain (light) and rain_heavy using DINO thresholds,
write filtered Arrow subsets to augmentation_data_sample/ for Kaggle release.

Thresholds (per user decision 2026-04-21):
  rain light : DINO >= 0.75 → sample 300
  rain heavy : DINO >= 0.70 → sample 300

Overwrites:
  augmentation_data_sample/cs_diff_rain_test.arrow       (replaces existing 100-row sample)
  augmentation_data_sample/cs_diff_rain_heavy_test.arrow (replaces existing 100-row sample)

Other sample files (SODA VOC / KTSH / train) are untouched — no DINO CSV available
for those splits.

Prerequisite:
  - validation/results/dino_ssim/ip2p_rain.csv       (exists, 1,652 rows from 2026-04-20)
  - validation/results/dino_ssim/ip2p_rain_heavy.csv (need to compute via
    jobs/extract_dino_ssim_rain_heavy.sh before running this script)

Usage:
  python scripts/update_kaggle_rain_dino_filtered.py
  python scripts/update_kaggle_rain_dino_filtered.py --dry-run
  python scripts/update_kaggle_rain_dino_filtered.py --n-samples 300 --thr-light 0.75 --thr-heavy 0.70
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

REPO = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X")
SAMPLE_DIR = REPO / "augmentation_data_sample"
DINO_DIR = REPO / "validation/results/dino_ssim"
CS_TEST = REPO / "augmentation_data/construction_site/rain_snow/diffusion/test"

CONFIGS = {
    "light": {
        "source_dir": CS_TEST / "rain",
        "dino_csv": DINO_DIR / "ip2p_rain.csv",
        "threshold": 0.75,
        "output": SAMPLE_DIR / "cs_diff_rain_test.arrow",
    },
    "heavy": {
        "source_dir": CS_TEST / "rain_heavy",
        "dino_csv": DINO_DIR / "ip2p_rain_heavy.csv",
        "threshold": 0.70,
        "output": SAMPLE_DIR / "cs_diff_rain_heavy_test.arrow",
    },
}


def load_stream(path: Path) -> pa.Table:
    with open(path, "rb") as f:
        return pa.ipc.open_stream(f).read_all()


def concat_shards(dir_path: Path) -> pa.Table:
    shards = sorted(dir_path.glob("*.arrow"))
    if not shards:
        raise FileNotFoundError(f"No Arrow shards in {dir_path}")
    return pa.concat_tables([load_stream(p) for p in shards], promote_options="default")


def read_dino_csv(path: Path) -> dict:
    """Return {image_id: dino_sim} as strings/floats."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["image_id"])] = float(row["dino_sim"])
    return out


def filter_and_sample(table: pa.Table, eligible_ids: set, n: int, seed: int = 42) -> pa.Table:
    """Keep rows whose image_id is eligible, then sample up to n with fixed seed."""
    ids = [str(x) for x in table.column("image_id").to_pylist()]
    eligible_idx = [i for i, x in enumerate(ids) if x in eligible_ids]
    if len(eligible_idx) == 0:
        raise RuntimeError("No eligible rows after threshold filter — check DINO CSV overlap")

    if len(eligible_idx) <= n:
        chosen = eligible_idx
        print(f"    only {len(eligible_idx)} rows eligible; keeping all")
    else:
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(eligible_idx, size=n, replace=False).tolist())

    # Row-by-row slice + concat to avoid int32 offset overflow on large binary columns
    slices = [table.slice(i, 1) for i in chosen]
    return pa.concat_tables(slices, promote_options="default")


def write_stream(table: pa.Table, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_stream(str(out), table.schema) as w:
        w.write_table(table)
    print(f"    wrote {out.name}: {table.num_rows} rows, {out.stat().st_size/1e6:.1f} MB")


def summarise_dino(dino: dict, threshold: float):
    vals = np.array(list(dino.values()))
    n_total = len(vals)
    n_pass = int((vals >= threshold).sum())
    print(f"    DINO n={n_total}  mean={vals.mean():.3f}  min={vals.min():.3f}  max={vals.max():.3f}  "
          f"pass DINO>={threshold}: {n_pass} ({100*n_pass/n_total:.1f}%)")
    return n_pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--thr-light", type=float, default=0.75)
    p.add_argument("--thr-heavy", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write output Arrow files, just report counts")
    args = p.parse_args()

    CONFIGS["light"]["threshold"] = args.thr_light
    CONFIGS["heavy"]["threshold"] = args.thr_heavy

    for name, cfg in CONFIGS.items():
        print(f"=== CS rain {name} (DINO >= {cfg['threshold']:.2f}, target {args.n_samples}) ===")
        if not cfg["dino_csv"].exists():
            print(f"  MISSING DINO CSV: {cfg['dino_csv']}")
            if name == "heavy":
                print(f"  → submit: sbatch jobs/extract_dino_ssim_rain_heavy.sh")
            continue

        dino = read_dino_csv(cfg["dino_csv"])
        n_pass = summarise_dino(dino, cfg["threshold"])
        eligible = {k for k, v in dino.items() if v >= cfg["threshold"]}

        if n_pass < args.n_samples:
            print(f"  ⚠ only {n_pass} rows pass threshold (requested {args.n_samples})")
            print(f"    consider lowering threshold or fewer samples")

        print(f"  loading source Arrow shards from {cfg['source_dir']}...")
        table = concat_shards(cfg["source_dir"])
        print(f"    source: {table.num_rows} rows")

        filtered = filter_and_sample(table, eligible, args.n_samples, args.seed)
        if args.dry_run:
            print(f"    [dry-run] would write {filtered.num_rows} rows to {cfg['output'].name}")
        else:
            write_stream(filtered, cfg["output"])

    if not args.dry_run:
        print("\n=== Next step ===")
        print(f"cd {SAMPLE_DIR} && kaggle datasets version -m \"Rain samples filtered by DINO threshold (light>=0.75, heavy>=0.70, n=300 each)\"")


if __name__ == "__main__":
    main()
