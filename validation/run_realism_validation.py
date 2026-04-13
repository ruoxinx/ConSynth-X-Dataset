#!/usr/bin/env python3
"""
Run AI-generated image detection on ConSynth-X augmented data.

Evaluates realism of augmented images by measuring how often an AI-generated
image detector classifies them as "real". Higher fooling rate = more realistic.

Compares:
  - Original (real) images as baseline
  - Each augmentation condition (weather, night, night_weather, small)

Outputs:
  - Per-condition fooling rates and score distributions
  - Score histograms (real vs augmented overlay)
  - Summary JSON for paper tables
  - Per-image CSV for detailed analysis

Usage:
  # Full run on all conditions (Construction Site dataset)
  python validation/run_realism_validation.py --dataset construction_site

  # Quick test with 50 samples
  python validation/run_realism_validation.py --dataset construction_site --max-samples 50

  # With UnivFD weights (recommended)
  python validation/run_realism_validation.py --weights path/to/fc_weights.pth

  # CLIP zero-shot (no extra download needed)
  python validation/run_realism_validation.py --mode clip_zeroshot
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Data Paths ──────────────────────────────────────────────────
# Mirrors infrastructure.md data locations
DATA_ROOT = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite")
LMUDATA = Path.home() / "LMUData"

# All augmented data is in Arrow format at this location
ARROW_DATA = DATA_ROOT / "augmentation_data_arrow"

# Condition → data source mapping
CONDITION_DIRS = {
    # Original images (from HuggingFace Arrow)
    "original": {
        "type": "arrow",
        "path": ARROW_DATA / "construction_site_test.arrow",
        "description": "Original unmodified construction site images (test set)",
    },
    # Weather augmentation (Style Transfer) — rain
    "weather_style_rain_0": {
        "type": "arrow",
        "path": ARROW_DATA / "weather_test_style_rain_0.arrow",
        "description": "Style transfer rain intensity 0 (light)",
    },
    "weather_style_rain_1": {
        "type": "arrow",
        "path": ARROW_DATA / "weather_test_style_rain_1.arrow",
        "description": "Style transfer rain intensity 1 (moderate)",
    },
    "weather_style_rain_2": {
        "type": "arrow",
        "path": ARROW_DATA / "weather_test_style_rain_2.arrow",
        "description": "Style transfer rain intensity 2 (heavy)",
    },
    # Weather augmentation (Style Transfer) — snow
    "weather_style_snow_0": {
        "type": "arrow",
        "path": ARROW_DATA / "weather_test_style_snow_0.arrow",
        "description": "Style transfer snow intensity 0 (light)",
    },
    "weather_style_snow_1": {
        "type": "arrow",
        "path": ARROW_DATA / "weather_test_style_snow_1.arrow",
        "description": "Style transfer snow intensity 1 (moderate)",
    },
    "weather_style_snow_2": {
        "type": "arrow",
        "path": ARROW_DATA / "weather_test_style_snow_2.arrow",
        "description": "Style transfer snow intensity 2 (heavy)",
    },
    # Day-to-Night (CycleGAN-Turbo)
    "night": {
        "type": "arrow",
        "path": ARROW_DATA / "night.arrow",
        "description": "CycleGAN-Turbo day-to-night conversion",
    },
    # Outpainting (FLUX.1-Fill-dev)
    "small": {
        "type": "arrow",
        "path": ARROW_DATA / "small.arrow",
        "description": "FLUX outpainting scale augmentation",
    },
    # Weather augmentation (IP2P Diffusion) — batched Arrow files
    "diffusion_rain": {
        "type": "arrow_dir",
        "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_rain_heavy",
        "description": "IP2P diffusion rain augmentation (heavy)",
    },
    "diffusion_snow": {
        "type": "arrow_dir",
        "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_snow_heavy",
        "description": "IP2P diffusion snow augmentation (heavy)",
    },
}

OUT_DIR = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/validation/results")


def load_images_from_arrow(arrow_path: Path, max_samples: int = None) -> list:
    """Load PIL images from a single Arrow file (HuggingFace format).

    Arrow files store images as structs with 'bytes' and 'path' fields.
    """
    import io
    import pyarrow as pa

    images = []

    if not arrow_path.exists():
        print(f"  Arrow file not found: {arrow_path}")
        return []

    try:
        # HuggingFace Arrow files use IPC stream format
        reader = pa.ipc.open_stream(str(arrow_path))
        table = reader.read_all()
    except Exception:
        try:
            # Fallback to IPC file format
            reader = pa.ipc.open_file(str(arrow_path))
            table = reader.read_all()
        except Exception as e:
            print(f"  Error reading Arrow file: {e}")
            return []

    # Find image column
    img_col = None
    for col_name in ["image", "img", "pixel_values"]:
        if col_name in table.column_names:
            img_col = col_name
            break

    if img_col is None:
        print(f"  No image column found. Columns: {table.column_names}")
        return []

    n_total = len(table)
    n_load = min(n_total, max_samples) if max_samples else n_total

    # Random sample if max_samples < total (for unbiased evaluation)
    if max_samples and max_samples < n_total:
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(n_total, size=n_load, replace=False))
    else:
        indices = range(n_load)

    for i in indices:
        try:
            img_data = table.column(img_col)[i].as_py()
            if isinstance(img_data, bytes):
                # Raw binary JPEG/PNG bytes
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                images.append(img)
            elif isinstance(img_data, dict) and "bytes" in img_data:
                # HuggingFace struct format: {"bytes": b"...", "path": "..."}
                img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                images.append(img)
            elif isinstance(img_data, Image.Image):
                images.append(img_data.convert("RGB"))
        except Exception:
            pass  # Skip corrupted images silently

    print(f"  Loaded {len(images)}/{n_total} images from {arrow_path.name}")
    return images


def load_images_from_arrow_dir(dir_path: Path, max_samples: int = None) -> list:
    """Load PIL images from a directory of batched Arrow files."""
    import io
    import pyarrow as pa

    arrow_files = sorted(dir_path.glob("*.arrow"))
    if not arrow_files:
        print(f"  No .arrow files in {dir_path}")
        return []

    # First pass: count total rows
    total_rows = 0
    file_row_counts = []
    for af in arrow_files:
        try:
            reader = pa.ipc.open_stream(str(af))
            t = reader.read_all()
            file_row_counts.append((af, len(t)))
            total_rows += len(t)
        except Exception:
            try:
                reader = pa.ipc.open_file(str(af))
                t = reader.read_all()
                file_row_counts.append((af, len(t)))
                total_rows += len(t)
            except Exception:
                continue

    n_load = min(total_rows, max_samples) if max_samples else total_rows

    # Random sample indices across all files
    if max_samples and max_samples < total_rows:
        rng = np.random.default_rng(42)
        sample_indices = set(rng.choice(total_rows, size=n_load, replace=False))
    else:
        sample_indices = None  # load all

    images = []
    global_idx = 0
    for af, n_rows in file_row_counts:
        # Check if any target indices fall in this file
        if sample_indices is not None:
            file_indices = [i - global_idx for i in sample_indices
                           if global_idx <= i < global_idx + n_rows]
            if not file_indices:
                global_idx += n_rows
                continue
        else:
            file_indices = range(n_rows)

        try:
            reader = pa.ipc.open_stream(str(af))
            table = reader.read_all()
        except Exception:
            reader = pa.ipc.open_file(str(af))
            table = reader.read_all()

        img_col = None
        for col_name in ["image", "img"]:
            if col_name in table.column_names:
                img_col = col_name
                break
        if img_col is None:
            global_idx += n_rows
            continue

        for i in file_indices:
            try:
                img_data = table.column(img_col)[i].as_py()
                if isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    images.append(img)
                elif isinstance(img_data, dict) and "bytes" in img_data:
                    img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                    images.append(img)
            except Exception:
                pass

        global_idx += n_rows
        if max_samples and len(images) >= max_samples:
            break

    print(f"  Loaded {len(images)}/{total_rows} images from {len(file_row_counts)} Arrow batches")
    return images


def load_images_from_dir(dir_path: Path, max_samples: int = None) -> list:
    """Load PIL images from a directory of image files."""
    images = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(
        f for f in dir_path.rglob("*") if f.suffix.lower() in extensions
    )
    if max_samples:
        files = files[:max_samples]
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    return images


def load_condition_images(condition: str, config: dict, max_samples: int = None) -> list:
    """Load images for a given condition."""
    print(f"\n  Loading [{condition}]: {config['description']}")
    path = config["path"]

    if not path.exists():
        print(f"  WARNING: Path does not exist: {path}")
        print(f"  Skipping condition '{condition}'")
        return []

    if config["type"] == "arrow":
        images = load_images_from_arrow(path, max_samples=max_samples)
    elif config["type"] == "arrow_dir":
        images = load_images_from_arrow_dir(path, max_samples=max_samples)
    elif config["type"] == "image_dir":
        images = load_images_from_dir(path, max_samples=max_samples)
    else:
        print(f"  Unknown type: {config['type']}")
        return []

    return images


def run_scoring(detector, images: list, batch_size: int = 32,
                desc: str = "") -> np.ndarray:
    """Score images in batches. Returns array of fake probabilities."""
    all_scores = []
    n = len(images)
    for i in range(0, n, batch_size):
        batch = images[i:i + batch_size]
        scores = detector.score_batch(batch)
        all_scores.append(scores)
        done = min(i + batch_size, n)
        if desc:
            print(f"    {desc}: {done}/{n} ({done/n*100:.0f}%)", end="\r")
    if desc:
        print()
    return np.concatenate(all_scores)


def generate_histogram(
    real_scores: np.ndarray,
    aug_scores_dict: dict,
    output_path: Path,
    title: str = "Realism Score Distribution",
):
    """Generate overlay histogram: real vs each augmentation condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_conditions = len(aug_scores_dict)
    fig, axes = plt.subplots(1, n_conditions, figsize=(5 * n_conditions, 4),
                             squeeze=False)
    axes = axes[0]

    bins = np.linspace(0, 1, 50)

    for ax, (condition, scores) in zip(axes, aug_scores_dict.items()):
        ax.hist(real_scores, bins=bins, alpha=0.5, label="Original (real)",
                color="#2ecc71", density=True)
        ax.hist(scores, bins=bins, alpha=0.5, label=f"{condition}",
                color="#e74c3c", density=True)
        ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="Threshold")
        ax.set_xlabel("Fake Probability →", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(condition, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)

        # Annotate fooling rate
        fooling = (scores < 0.5).mean() * 100
        ax.text(0.05, 0.95, f"Fooling: {fooling:.1f}%",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved histogram: {output_path}")


def generate_summary_barplot(summary: dict, output_path: Path):
    """Generate bar plot of fooling rates across conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = []
    fooling_rates = []
    for condition, stats in summary.items():
        if condition == "original":
            continue
        conditions.append(condition)
        fooling_rates.append(stats["fooling_rate"] * 100)

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.2), 5))

    colors = []
    for fr in fooling_rates:
        if fr >= 80:
            colors.append("#2ecc71")  # green = very realistic
        elif fr >= 60:
            colors.append("#f39c12")  # orange = moderate
        else:
            colors.append("#e74c3c")  # red = detectable

    bars = ax.bar(range(len(conditions)), fooling_rates, color=colors, edgecolor="white")

    # Add value labels
    for bar, val in zip(bars, fooling_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Reference lines
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.text(len(conditions) - 0.5, 52, "Random (50%)", fontsize=8, color="gray")

    # Original accuracy
    if "original" in summary:
        orig_acc = summary["original"]["real_accuracy"] * 100
        ax.axhline(y=orig_acc, color="#2ecc71", linestyle=":", alpha=0.7)
        ax.text(len(conditions) - 0.5, orig_acc + 2,
                f"Original: {orig_acc:.0f}%", fontsize=8, color="#2ecc71")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fooling Rate (%)", fontsize=11)
    ax.set_title("AI Detector Fooling Rate per Augmentation Condition\n"
                 "(Higher = More Realistic)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved bar plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Realism validation: AI-generated image detection on ConSynth-X"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to UnivFD fc_weights.pth (if not provided, uses CLIP zero-shot)"
    )
    parser.add_argument(
        "--mode", type=str, default="auto", choices=["auto", "univfd", "clip_zeroshot"],
        help="Detection mode"
    )
    parser.add_argument(
        "--dataset", type=str, default="construction_site",
        choices=["construction_site"],
        help="Which dataset to evaluate"
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Specific conditions to evaluate (default: all available)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max images per condition (for quick testing)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUT_DIR),
        help="Output directory for results"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialize detector ─────────────────────────────────
    print("=" * 60)
    print("ConSynth-X Realism Validation")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).parent))
    from realism_detector import RealismDetector, compute_fooling_rate

    detector = RealismDetector(
        fc_weights_path=args.weights,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mode=args.mode,
    )

    # ── Determine conditions to evaluate ────────────────────
    if args.conditions:
        conditions_to_eval = {
            k: v for k, v in CONDITION_DIRS.items() if k in args.conditions
        }
    else:
        conditions_to_eval = CONDITION_DIRS

    # Always include "original" as baseline
    if "original" not in conditions_to_eval:
        conditions_to_eval = {"original": CONDITION_DIRS["original"], **conditions_to_eval}

    # ── Score all conditions ────────────────────────────────
    print(f"\nEvaluating {len(conditions_to_eval)} conditions...")
    all_scores = {}
    summary = {}

    t_start = time.time()

    for condition, config in conditions_to_eval.items():
        images = load_condition_images(condition, config, args.max_samples)
        if not images:
            print(f"  SKIP: no images loaded for {condition}")
            continue

        print(f"  Scoring {len(images)} images...")
        scores = run_scoring(detector, images, args.batch_size, desc=condition)
        all_scores[condition] = scores

        # Free memory
        del images
        torch.cuda.empty_cache()

    t_elapsed = time.time() - t_start
    print(f"\nScoring complete in {t_elapsed:.1f}s")

    # ── Compute fooling rates ───────────────────────────────
    if "original" not in all_scores:
        print("ERROR: No original images scored. Cannot compute fooling rates.")
        return

    real_scores = all_scores["original"]
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"\nOriginal images: mean_score={real_scores.mean():.3f} "
          f"(std={real_scores.std():.3f}), "
          f"classified_real={(real_scores < 0.5).mean()*100:.1f}%")

    for condition, scores in all_scores.items():
        stats = compute_fooling_rate(real_scores, scores)
        summary[condition] = stats

        if condition == "original":
            continue

        print(f"\n  {condition}:")
        print(f"    Fooling rate:  {stats['fooling_rate']*100:.1f}%  "
              f"(= classified as real)")
        print(f"    Mean score:    {stats['aug_mean_score']:.3f}  "
              f"(gap from real: {stats['score_gap']:+.3f})")
        print(f"    N images:      {stats['n_augmented']}")

    # ── Save results ────────────────────────────────────────
    # Summary JSON
    summary_path = output_dir / "realism_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Per-condition scores CSV
    csv_path = output_dir / "realism_scores.csv"
    with open(csv_path, "w") as f:
        f.write("condition,image_idx,fake_probability\n")
        for condition, scores in all_scores.items():
            for i, s in enumerate(scores):
                f.write(f"{condition},{i},{s:.6f}\n")
    print(f"Saved per-image scores: {csv_path}")

    # ── Generate figures ────────────────────────────────────
    aug_scores_for_hist = {
        k: v for k, v in all_scores.items() if k != "original"
    }

    if aug_scores_for_hist:
        generate_histogram(
            real_scores, aug_scores_for_hist,
            output_dir / "realism_histograms.png",
            title=f"ConSynth-X Realism Validation ({detector.mode})",
        )

        generate_summary_barplot(
            summary,
            output_dir / "realism_fooling_rates.png",
        )

    # ── Print paper-ready table ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("Paper-ready table (LaTeX):")
    print(f"{'=' * 60}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Condition & N & Fooling Rate (\%) & Mean Score & Score Gap \\")
    print(r"\hline")
    for condition, stats in summary.items():
        fr = stats["fooling_rate"] * 100
        ms = stats["aug_mean_score"]
        gap = stats["score_gap"]
        n = stats["n_augmented"]
        label = condition.replace("_", r"\_")
        if condition == "original":
            print(f"{label} & {n} & {stats['real_accuracy']*100:.1f} & {ms:.3f} & --- \\\\")
        else:
            print(f"{label} & {n} & {fr:.1f} & {ms:.3f} & {gap:+.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
