"""
Sample images from augmentation_data/ into human_validation/static/images/
for the human perceptual validation study.

Usage:
    python sample_images.py --n 50 --source /path/to/augmentation_data
    python sample_images.py --n 50  # uses default paths

This script:
1. Samples N images per condition from synthetic augmented data
2. Samples N images from real weather reference datasets
3. Copies them into static/images/<condition>/ for the Flask app
"""

import argparse
import os
import random
import shutil
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

# Default paths (adjust to your setup)
DEFAULT_AUG_DATA = str(_DATA_ROOT / "augmentation_data")
DEFAULT_REAL_REF = str(_REPO_ROOT / "validation/reference_data")

# Conditions to sample from augmentation_data
# (style transfer retired 2026-04-21 — moved to experiments/ablation_style_transfer/)
SYNTHETIC_CONDITIONS = [
    "weather_diff_rain", "weather_diff_rain_heavy",
    "weather_diff_snow_light", "weather_diff_snow_heavy",
    "fog_heavy", "fog_medium", "fog_light",
    "night", "night_rain", "night_snow", "small",
]

# Also include original (non-augmented) as control
CONTROL_CONDITIONS = ["original"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_images(directory):
    """Find all image files in a directory (non-recursive for flat dirs, recursive otherwise)."""
    images = []
    if not os.path.isdir(directory):
        return images
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, f))
    return images


def sample_and_copy(src_dir, dest_dir, n, label=""):
    """Sample n images from src_dir and copy to dest_dir."""
    images = find_images(src_dir)
    if not images:
        print(f"  [SKIP] No images found in {src_dir}")
        return 0

    sampled = random.sample(images, min(n, len(images)))
    os.makedirs(dest_dir, exist_ok=True)

    for img_path in sampled:
        fname = os.path.basename(img_path)
        dest = os.path.join(dest_dir, fname)
        # Avoid name collisions
        if os.path.exists(dest):
            base, ext = os.path.splitext(fname)
            fname = f"{base}_{random.randint(1000,9999)}{ext}"
            dest = os.path.join(dest_dir, fname)
        shutil.copy2(img_path, dest)

    print(f"  [{label}] Sampled {len(sampled)} / {len(images)} images -> {dest_dir}")
    return len(sampled)


def main():
    parser = argparse.ArgumentParser(description="Sample images for human validation")
    parser.add_argument("--n", type=int, default=50, help="Number of images per condition")
    parser.add_argument("--source", type=str, default=DEFAULT_AUG_DATA, help="Augmentation data root")
    parser.add_argument("--real-ref", type=str, default=DEFAULT_REAL_REF, help="Real reference data root")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "static", "images")
    print(f"Output directory: {output_dir}")
    print(f"Sampling {args.n} images per condition\n")

    total = 0

    # 1. Sample synthetic conditions
    print("=== Synthetic Augmented Images ===")
    for condition in SYNTHETIC_CONDITIONS:
        src = os.path.join(args.source, condition)
        # Try alternate directory structures
        if not os.path.isdir(src):
            # Try: augmentation_data/construction_site_10k/condition/
            src = os.path.join(args.source, "construction_site_10k", condition)
        if not os.path.isdir(src):
            # Try: augmentation_data/condition/images/
            src = os.path.join(args.source, condition, "images")

        dest = os.path.join(output_dir, condition)
        total += sample_and_copy(src, dest, args.n, label=condition)

    # 2. Sample original (control)
    print("\n=== Control (Original) Images ===")
    for condition in CONTROL_CONDITIONS:
        src = os.path.join(args.source, condition)
        if not os.path.isdir(src):
            src = os.path.join(args.source, "construction_site_10k", condition)
        dest = os.path.join(output_dir, condition)
        total += sample_and_copy(src, dest, args.n, label=condition)

    # 3. Sample real weather reference images
    print("\n=== Real Weather Reference Images ===")
    if os.path.isdir(args.real_ref):
        for ref_name in os.listdir(args.real_ref):
            ref_path = os.path.join(args.real_ref, ref_name)
            if os.path.isdir(ref_path):
                dest = os.path.join(output_dir, "real_weather")
                total += sample_and_copy(ref_path, dest, args.n, label=f"real/{ref_name}")
    else:
        print(f"  [SKIP] Real reference dir not found: {args.real_ref}")

    print(f"\nTotal images sampled: {total}")
    print(f"\nNext steps:")
    print(f"  1. Start the app:  cd human_validation && python app.py")
    print(f"  2. Login as admin (username: admin)")
    print(f"  3. Go to Dashboard -> Click 'Load Images'")


if __name__ == "__main__":
    main()
