#!/usr/bin/env python3
"""
Download ACDC dataset and organize per weather condition for FID/KID reference.

ACDC (Adverse Conditions Dataset with Correspondences):
  - 4 conditions: fog, night, rain, snow (~1000 images each)
  - License: CC BY-NC-SA 4.0
  - Paper: Sakaridis et al., ICCV 2021, ArXiv: 2104.13395

Images are saved as individual files per condition for easy FID computation.

Usage:
  python validation/download_acdc.py
  python validation/download_acdc.py --output-dir validation/reference_data/acdc
  python validation/download_acdc.py --resize 640  # resize to 640x640
"""

import argparse
import os
import sys
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

DEFAULT_OUTPUT = (_REPO_ROOT / "validation/reference_data/acdc")


def download_and_organize(output_dir: Path, resize: int = None, max_per_condition: int = None):
    from datasets import load_dataset
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading ACDC dataset from HuggingFace...")
    print("=" * 60)

    ds = load_dataset("mathpluscode/ACDC")
    print(f"Splits available: {list(ds.keys())}")

    # ACDC structure: images are organized by condition in the dataset
    # We need to inspect the structure to find condition labels
    # Check first few examples to understand schema
    for split_name in ds:
        split = ds[split_name]
        print(f"\n  Split '{split_name}': {len(split)} examples")
        if len(split) > 0:
            print(f"  Columns: {split.column_names}")
            # Print first example keys/values (non-image)
            example = split[0]
            for k, v in example.items():
                if isinstance(v, Image.Image):
                    print(f"    {k}: PIL Image {v.size}")
                elif isinstance(v, str):
                    print(f"    {k}: '{v[:100]}'")
                elif isinstance(v, (int, float)):
                    print(f"    {k}: {v}")
                else:
                    print(f"    {k}: {type(v).__name__}")

    # Try to extract condition from file paths or labels
    # ACDC typically has conditions in the path: fog/, night/, rain/, snow/
    condition_dirs = {}
    conditions_found = {"fog": 0, "night": 0, "rain": 0, "snow": 0}

    for split_name in ds:
        split = ds[split_name]
        for i in range(len(split)):
            example = split[i]

            # Try to determine condition from available fields
            condition = None

            # Check if there's a condition/label field
            for field in ["condition", "weather", "label", "category"]:
                if field in example and isinstance(example[field], str):
                    cond_val = example[field].lower()
                    for c in conditions_found:
                        if c in cond_val:
                            condition = c
                            break

            # Check if condition is in file_name/image_path
            if condition is None:
                for field in ["file_name", "image_path", "path", "id", "image_id"]:
                    if field in example and isinstance(example[field], str):
                        path_val = example[field].lower()
                        for c in conditions_found:
                            if f"/{c}/" in path_val or path_val.startswith(c):
                                condition = c
                                break
                    if condition:
                        break

            if condition is None:
                continue

            conditions_found[condition] += 1

            if max_per_condition and conditions_found[condition] > max_per_condition:
                continue

            # Save image
            cond_dir = output_dir / condition
            cond_dir.mkdir(parents=True, exist_ok=True)

            # Get the image
            img = None
            for field in ["image", "rgb", "img", "pixel_values"]:
                if field in example and isinstance(example[field], Image.Image):
                    img = example[field]
                    break

            if img is None:
                continue

            img = img.convert("RGB")
            if resize:
                img = img.resize((resize, resize), Image.LANCZOS)

            img_name = f"{condition}_{conditions_found[condition]:05d}.jpg"
            img.save(cond_dir / img_name, quality=95)

            if conditions_found[condition] % 200 == 0:
                print(f"    {condition}: {conditions_found[condition]} images saved")

    # Summary
    print(f"\n{'=' * 60}")
    print("ACDC Download Summary")
    print(f"{'=' * 60}")
    total = 0
    for condition, count in conditions_found.items():
        saved = min(count, max_per_condition) if max_per_condition else count
        cond_dir = output_dir / condition
        actual = len(list(cond_dir.glob("*.jpg"))) if cond_dir.exists() else 0
        print(f"  {condition}: {actual} images saved to {cond_dir}")
        total += actual
    print(f"  TOTAL: {total} images")
    print(f"  Output: {output_dir}")

    if total == 0:
        print("\n  WARNING: No images extracted!")
        print("  The ACDC HuggingFace dataset structure may differ from expected.")
        print("  Printing raw dataset info for debugging...")
        for split_name in ds:
            split = ds[split_name]
            if len(split) > 0:
                print(f"\n  Split '{split_name}' example[0] keys: {list(split[0].keys())}")
                for k, v in split[0].items():
                    print(f"    {k} = {repr(v)[:200]}")

    return conditions_found


def main():
    parser = argparse.ArgumentParser(description="Download ACDC for FID/KID reference")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resize", type=int, default=None,
                        help="Resize images to NxN (e.g., 640)")
    parser.add_argument("--max-per-condition", type=int, default=None,
                        help="Max images per condition")
    args = parser.parse_args()

    download_and_organize(
        Path(args.output_dir),
        resize=args.resize,
        max_per_condition=args.max_per_condition,
    )


if __name__ == "__main__":
    main()
