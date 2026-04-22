#!/usr/bin/env python3
"""
Update Kaggle sample dataset with IP2P snow heavy variant (gs=12, 2026-04-18).

Adds new file: cs_diff_snow_heavy_test.arrow (100 samples from new 3004-image set).
Existing cs_diff_snow_test.arrow = IP2P snow LIGHT (gs=8) — kept for backward compat.

Uploads new version via kaggle CLI.
"""

import io
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

SAMPLE_DIR = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/augmentation_data_sample")
HEAVY_DIR = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/output/construction_site_test/diffusion_snow_heavy")
N_SAMPLES = 100
SEED = 42


def load_stream(path):
    with open(path, "rb") as f:
        return pa.ipc.open_stream(f).read_all()


def main():
    # Load existing cs_diff_snow_test to match schema
    existing_path = SAMPLE_DIR / "cs_diff_snow_test.arrow"
    print(f"Reading existing schema from {existing_path.name}...")
    existing = load_stream(existing_path)
    print(f"  Columns: {existing.column_names}")
    print(f"  Schema: {existing.schema}")

    # Load heavy snow arrow files
    print(f"\nLoading IP2P snow heavy from {HEAVY_DIR}...")
    tables = []
    for af in sorted(HEAVY_DIR.glob("batch_*.arrow")):
        tables.append(load_stream(af))
        print(f"  {af.name}: {tables[-1].num_rows} rows, cols={tables[-1].column_names[:5]}")

    heavy = pa.concat_tables(tables)
    print(f"  Total: {heavy.num_rows} rows")

    # Sample 100 deterministic
    rng = np.random.default_rng(SEED)
    indices = sorted(rng.choice(heavy.num_rows, size=N_SAMPLES, replace=False).tolist())
    print(f"\nSampling {N_SAMPLES} deterministic indices (seed={SEED})")

    # Select rows and reshape to match existing schema
    sampled = heavy.take(indices)
    target_cols = existing.column_names
    available_cols = sampled.column_names
    print(f"  Existing schema columns: {target_cols}")
    print(f"  Heavy source columns:    {available_cols}")

    # Keep only columns that exist in both
    keep_cols = [c for c in target_cols if c in available_cols]
    print(f"  Common columns: {keep_cols}")

    # Add missing columns as empty if needed
    arrays = []
    col_names = []
    for c in target_cols:
        if c in available_cols:
            arrays.append(sampled.column(c))
            col_names.append(c)
        else:
            # Fill with None
            arrays.append(pa.nulls(N_SAMPLES, type=existing.schema.field(c).type))
            col_names.append(c)

    final = pa.Table.from_arrays(arrays, names=col_names)

    # Write
    out_path = SAMPLE_DIR / "cs_diff_snow_heavy_test.arrow"
    with open(out_path, "wb") as f:
        with pa.ipc.new_stream(f, final.schema) as writer:
            writer.write_table(final)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote: {out_path.name} — {N_SAMPLES} rows, {size_mb:.1f} MB")

    # Also rename old for clarity? No — keep backward compat, just add heavy alongside.
    # Log a note
    print("\n" + "=" * 60)
    print("Note: existing cs_diff_snow_test.arrow = LIGHT variant (gs=8)")
    print("New cs_diff_snow_heavy_test.arrow = HEAVY variant (gs=12)")
    print("=" * 60)


if __name__ == "__main__":
    main()
