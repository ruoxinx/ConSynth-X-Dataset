#!/usr/bin/env python3
"""
Add rain_heavy variants (physics-only on IP2P light) to augmentation_data_sample/.

Pairs each new rain_heavy sample with the SAME image_ids as the existing light
sample, so reviewers can compare light vs heavy on identical source images.

Creates:
  cs_diff_rain_heavy_test.arrow     (100 rows, paired with cs_diff_rain_test.arrow)
  cs_diff_rain_heavy_train.arrow    (100 rows, paired with cs_diff_rain_train.arrow)
  soda_voc_diff_rain.arrow          (100 rows, NEW — light baseline that was missing)
  soda_voc_diff_rain_heavy.arrow    (100 rows, paired with above)
  soda_ktsh_diff_rain_heavy.arrow   (100 rows, paired with soda_ktsh_diff_rain.arrow)
"""

import io
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

SAMPLE_DIR = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/augmentation_data_sample")
AUG_ROOT = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/augmentation_data")
N_SAMPLES = 100
SEED = 42


def load_stream(path):
    with open(path, "rb") as f:
        return pa.ipc.open_stream(f).read_all()


def concat_shards(dir_path: Path):
    tables = [load_stream(p) for p in sorted(dir_path.glob("*.arrow"))]
    return pa.concat_tables(tables, promote_options="default")


def filter_by_ids(table, target_ids):
    ids_col = table.column("image_id").to_pylist()
    ids_str = [str(x) for x in ids_col]
    target_set = set(str(x) for x in target_ids)
    matching_idx = [i for i, x in enumerate(ids_str) if x in target_set]
    # Row-by-row slice to avoid int32 offset overflow
    slices = [table.slice(i, 1) for i in matching_idx]
    if not slices:
        return table.slice(0, 0)
    return pa.concat_tables(slices, promote_options="default")


def sample_n(table, n, seed=SEED):
    if table.num_rows <= n:
        return table
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(table.num_rows, size=n, replace=False).tolist())
    # Build row-by-row to avoid int32 offset overflow on large binary columns
    slices = [table.slice(i, 1) for i in idx]
    return pa.concat_tables(slices, promote_options="default")


def write_sample(table, out_name):
    out = SAMPLE_DIR / out_name
    with ipc.new_stream(str(out), table.schema) as w:
        w.write_table(table)
    size_mb = out.stat().st_size / 1e6
    print(f"  {out_name}: {table.num_rows} rows, {size_mb:.1f} MB")


def pair_with_light(light_name: str, heavy_table: pa.Table, heavy_out: str):
    """Filter heavy table to image_ids in existing light sample."""
    light_path = SAMPLE_DIR / light_name
    if not light_path.exists():
        print(f"  (no existing {light_name}; skip pairing — will sample randomly)")
        write_sample(sample_n(heavy_table, N_SAMPLES), heavy_out)
        return
    light = load_stream(light_path)
    target_ids = light.column("image_id").to_pylist()
    paired = filter_by_ids(heavy_table, target_ids)
    print(f"  paired {paired.num_rows}/{len(target_ids)} with {light_name}")
    write_sample(paired, heavy_out)


def main():
    print("=== CS test rain_heavy ===")
    cs_test_heavy = concat_shards(AUG_ROOT / "construction_site/rain_snow/diffusion/test/rain_heavy")
    pair_with_light("cs_diff_rain_test.arrow", cs_test_heavy, "cs_diff_rain_heavy_test.arrow")

    print("\n=== CS train rain_heavy ===")
    cs_train_heavy = load_stream(AUG_ROOT / "construction_site/rain_snow/diffusion/train/rain_heavy/train_rain_heavy.arrow")
    pair_with_light("cs_diff_rain_train.arrow", cs_train_heavy, "cs_diff_rain_heavy_train.arrow")

    print("\n=== SODA VOC rain_heavy (+ light baseline if missing) ===")
    voc_heavy = load_stream(AUG_ROOT / "soda_voc/rain_snow/diffusion/rain_heavy.arrow")
    voc_light_sample = SAMPLE_DIR / "soda_voc_diff_rain.arrow"
    if not voc_light_sample.exists():
        print("  creating soda_voc_diff_rain.arrow (light baseline) first...")
        voc_light = load_stream(AUG_ROOT / "soda_voc/rain_snow/diffusion/rain.arrow")
        light_sampled = sample_n(voc_light, N_SAMPLES)
        write_sample(light_sampled, "soda_voc_diff_rain.arrow")
    pair_with_light("soda_voc_diff_rain.arrow", voc_heavy, "soda_voc_diff_rain_heavy.arrow")

    print("\n=== SODA KTSH rain_heavy ===")
    ktsh_heavy = load_stream(AUG_ROOT / "soda_ktsh/rain_snow/diffusion/rain_heavy.arrow")
    pair_with_light("soda_ktsh_diff_rain.arrow", ktsh_heavy, "soda_ktsh_diff_rain_heavy.arrow")

    print("\n=== Done ===")
    print(f"Updated: {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
