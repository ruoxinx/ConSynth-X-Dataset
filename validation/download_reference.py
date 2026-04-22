#!/usr/bin/env python3
"""
Download real weather reference images for FID/KID validation.

Two sources:
  1. WeatherNet-05 (prithivMLmods/WeatherNet-05-18039) — 18K images, 5 weather classes
     - rain (1,927), snow (1,875), fog (1,261), cloudy (6,702), clear (6,274)
     - License: Apache-2.0
     - Fast download (~2GB), no rate limit issues

  2. BDD100K (dgural/bdd100k, streaming) — 100K driving images with weather labels
     - rain, snow, fog, night, clear
     - License: BSD
     - Paper: Yu et al. 2020, ArXiv: 1805.04687
     - Adds NIGHT class (WeatherNet doesn't have it)

Usage:
  python validation/download_reference.py                    # both sources
  python validation/download_reference.py --source weathernet  # WeatherNet only
  python validation/download_reference.py --source bdd100k     # BDD100K only
"""

import argparse
import json
from pathlib import Path
from PIL import Image
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

BASE_OUTPUT = (_REPO_ROOT / "validation/reference_data")


def download_weathernet(output_dir: Path, resize: int = 640, max_per_condition: int = None):
    """Download WeatherNet-05-18039 — fast, small, 5 weather classes."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Source 1: WeatherNet-05-18039")
    print("=" * 60)

    ds = load_dataset("prithivMLmods/WeatherNet-05-18039", split="train")
    label_names = ds.features["label"].names
    print(f"Classes: {label_names}")
    print(f"Total: {len(ds)} images")

    # Map WeatherNet labels → our condition names
    label_to_condition = {
        "rain or strom": "rain",
        "snow or frosty": "snow",
        "foggy or hazy": "fog",
        "cloudy or overcast": "cloudy",
        "sun or clear": "clear",
    }

    counts = {}
    for i in range(len(ds)):
        example = ds[i]
        label_idx = example["label"]
        label_name = label_names[label_idx]
        condition = label_to_condition.get(label_name, label_name)

        if condition not in counts:
            counts[condition] = 0

        if max_per_condition and counts[condition] >= max_per_condition:
            continue

        img = example["image"].convert("RGB")
        if resize:
            img = img.resize((resize, resize), Image.LANCZOS)

        cond_dir = output_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        counts[condition] += 1
        img.save(cond_dir / f"{condition}_{counts[condition]:05d}.jpg", quality=95)

        if counts[condition] % 500 == 0:
            print(f"    {condition}: {counts[condition]}")

    print(f"\n  Results:")
    total = 0
    for cond in sorted(counts):
        print(f"    {cond}: {counts[cond]}")
        total += counts[cond]
    print(f"    TOTAL: {total}")

    meta = {
        "source": "prithivMLmods/WeatherNet-05-18039",
        "license": "Apache-2.0",
        "conditions": counts,
        "resize": resize,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def download_bdd100k_streaming(output_dir: Path, resize: int = 640,
                                max_per_condition: int = 1000):
    """Download BDD100K via streaming — for NIGHT class + driving domain."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Source 2: BDD100K (streaming)")
    print("=" * 60)
    print(f"Target: {max_per_condition} images per condition")

    ds = load_dataset("dgural/bdd100k", split="train", streaming=True)

    counts = {"rain": 0, "snow": 0, "fog": 0, "night": 0, "clear": 0}
    target = max_per_condition
    scanned = 0

    for example in ds:
        scanned += 1
        attrs = example.get("attributes")
        if not isinstance(attrs, dict):
            continue

        weather = str(attrs.get("weather", "")).lower().strip()
        tod = str(attrs.get("timeofday", "")).lower().strip()

        save_as = []
        if weather == "rainy" and counts["rain"] < target:
            save_as.append("rain")
        if weather == "snowy" and counts["snow"] < target:
            save_as.append("snow")
        if weather == "foggy" and counts["fog"] < target:
            save_as.append("fog")
        if tod == "night" and counts["night"] < target:
            save_as.append("night")
        if weather == "clear" and counts["clear"] < target:
            save_as.append("clear")

        if not save_as:
            if all(c >= target for c in counts.values()):
                break
            continue

        img = example.get("image")
        if not isinstance(img, Image.Image):
            continue

        try:
            img = img.convert("RGB")
            if resize:
                img = img.resize((resize, resize), Image.LANCZOS)
        except Exception:
            continue

        for cond in save_as:
            cond_dir = output_dir / cond
            cond_dir.mkdir(parents=True, exist_ok=True)
            counts[cond] += 1
            img.save(cond_dir / f"{cond}_{counts[cond]:05d}.jpg", quality=95)

        if scanned % 5000 == 0:
            status = " | ".join(f"{k}:{v}" for k, v in counts.items())
            print(f"    Scanned {scanned:,}... {status}")

    print(f"\n  Scanned {scanned:,} images")
    print(f"  Results:")
    total = 0
    for cond in sorted(counts):
        print(f"    {cond}: {counts[cond]}")
        total += counts[cond]
    print(f"    TOTAL: {total}")

    meta = {
        "source": "dgural/bdd100k (streaming)",
        "license": "BSD",
        "paper": "Yu et al. 2020, ArXiv: 1805.04687",
        "conditions": counts,
        "resize": resize,
        "scanned": scanned,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download weather reference data")
    parser.add_argument("--source", type=str, default="both",
                        choices=["both", "weathernet", "bdd100k"])
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--max-per-condition", type=int, default=1000,
                        help="Max per condition (BDD100K only; WeatherNet uses all)")
    args = parser.parse_args()

    if args.source in ("both", "weathernet"):
        download_weathernet(
            BASE_OUTPUT / "weathernet",
            resize=args.resize,
        )

    if args.source in ("both", "bdd100k"):
        download_bdd100k_streaming(
            BASE_OUTPUT / "bdd100k",
            resize=args.resize,
            max_per_condition=args.max_per_condition,
        )

    # Final summary
    print(f"\n{'=' * 60}")
    print("All reference data:")
    print(f"{'=' * 60}")
    for src_dir in sorted(BASE_OUTPUT.iterdir()):
        if src_dir.is_dir():
            print(f"\n  [{src_dir.name}]")
            for cond_dir in sorted(src_dir.iterdir()):
                if cond_dir.is_dir():
                    n = len(list(cond_dir.glob("*.jpg")))
                    print(f"    {cond_dir.name}: {n}")


if __name__ == "__main__":
    main()
