#!/usr/bin/env python3
"""
Weather classification validation using pretrained Weather-Image-Classification.

Model: prithivMLmods/Weather-Image-Classification (SigLIP2, Apache-2.0)
  - Trained on WeatherNet-05-18039 (18K images, 5 classes)
  - Overall accuracy: 85.89%
  - Classes: cloudy/overcast, foggy/hazy, rain/storm, snow/frosty, sun/clear

Validates augmented images by checking if a weather classifier recognizes
the intended weather condition. Higher accuracy = more realistic augmentation.

Expected results:
  - Augmented rain images → classified as "rain/storm" → high = good
  - Augmented snow images → classified as "snow/frosty" → high = good
  - Original clear images → classified as "sun/clear" → baseline
  - Night images → likely "cloudy/overcast" (no night class)

Usage:
  python validation/weather_classifier.py
  python validation/weather_classifier.py --max-samples 100
  python validation/weather_classifier.py --conditions weather_style_rain_0 diffusion_rain
"""

import argparse
import io
import json
import time
from collections import defaultdict
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────
DATA_ROOT = _DATA_ROOT
ARROW_DATA = DATA_ROOT / "augmentation_data_arrow"
OUT_DIR = (_REPO_ROOT / "validation/results/weather_cls")

# ── Model info ───────────────────────────────────────────────────
MODEL_NAME = "prithivMLmods/Weather-Image-Classification"
ID2LABEL = {
    0: "cloudy/overcast",
    1: "foggy/hazy",
    2: "rain/storm",
    3: "snow/frosty",
    4: "sun/clear",
}

# ── Conditions and expected labels ───────────────────────────────
# Maps our condition → expected weather class from the classifier
CONDITION_CONFIG = {
    "original": {
        "source": {"type": "arrow", "path": ARROW_DATA / "construction_site_test.arrow"},
        "expected": "sun/clear",
        "description": "Original construction site images",
    },
    "weather_style_rain_0": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_0.arrow"},
        "expected": "rain/storm",
        "description": "Style transfer rain (light)",
    },
    "weather_style_rain_1": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_1.arrow"},
        "expected": "rain/storm",
        "description": "Style transfer rain (moderate)",
    },
    "weather_style_rain_2": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_2.arrow"},
        "expected": "rain/storm",
        "description": "Style transfer rain (heavy)",
    },
    "weather_style_snow_0": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_0.arrow"},
        "expected": "snow/frosty",
        "description": "Style transfer snow (light)",
    },
    "weather_style_snow_1": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_1.arrow"},
        "expected": "snow/frosty",
        "description": "Style transfer snow (moderate)",
    },
    "weather_style_snow_2": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_2.arrow"},
        "expected": "snow/frosty",
        "description": "Style transfer snow (heavy)",
    },
    "night": {
        "source": {"type": "arrow", "path": ARROW_DATA / "night.arrow"},
        "expected": None,  # No night class — observe distribution
        "description": "CycleGAN-Turbo day-to-night",
    },
    "small": {
        "source": {"type": "arrow", "path": ARROW_DATA / "small.arrow"},
        "expected": "sun/clear",  # Outpainting shouldn't change weather
        "description": "FLUX outpainting",
    },
    "diffusion_rain": {
        "source": {
            "type": "arrow_dir",
            "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_rain_heavy",
        },
        "expected": "rain/storm",
        "description": "IP2P diffusion rain",
    },
    "diffusion_rain_heavy": {
        "source": {
            "type": "arrow_dir",
            "path": _REPO_ROOT / "augmentation_data" / "construction_site" / "rain_snow" / "diffusion" / "test" / "rain_heavy",
        },
        "expected": "rain/storm",
        "description": "IP2P diffusion rain + heavy physics overlay (2026-04-21)",
    },
    "diffusion_snow": {
        "source": {
            "type": "arrow_dir",
            "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_snow_heavy",
        },
        "expected": "snow/frosty",
        "description": "IP2P diffusion snow",
    },
}


# ── Data loading (reuse from run_realism_validation.py) ──────────

def load_images_from_arrow(arrow_path: Path, max_samples: int = None) -> list:
    import pyarrow as pa
    images = []
    try:
        reader = pa.ipc.open_stream(str(arrow_path))
        table = reader.read_all()
    except Exception:
        try:
            reader = pa.ipc.open_file(str(arrow_path))
            table = reader.read_all()
        except Exception:
            return []

    img_col = None
    for col in ["image", "img"]:
        if col in table.column_names:
            img_col = col
            break
    if img_col is None:
        return []

    n = len(table)
    if max_samples and max_samples < n:
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(n, size=max_samples, replace=False))
    else:
        indices = range(n)

    for i in indices:
        try:
            data = table.column(img_col)[i].as_py()
            if isinstance(data, bytes):
                images.append(Image.open(io.BytesIO(data)).convert("RGB"))
            elif isinstance(data, dict) and "bytes" in data:
                images.append(Image.open(io.BytesIO(data["bytes"])).convert("RGB"))
        except Exception:
            pass
    return images


def load_images_from_arrow_dir(dir_path: Path, max_samples: int = None) -> list:
    import pyarrow as pa
    arrow_files = sorted(dir_path.glob("*.arrow"))
    if not arrow_files:
        return []

    file_infos = []
    total = 0
    for af in arrow_files:
        try:
            reader = pa.ipc.open_stream(str(af))
            t = reader.read_all()
            file_infos.append((af, len(t)))
            total += len(t)
        except Exception:
            continue

    n_load = min(total, max_samples) if max_samples else total
    if max_samples and max_samples < total:
        rng = np.random.default_rng(42)
        sample_set = set(rng.choice(total, size=n_load, replace=False))
    else:
        sample_set = None

    images = []
    global_idx = 0
    for af, n_rows in file_infos:
        if sample_set is not None:
            local_indices = [i - global_idx for i in sample_set if global_idx <= i < global_idx + n_rows]
            if not local_indices:
                global_idx += n_rows
                continue
        else:
            local_indices = range(n_rows)

        try:
            reader = pa.ipc.open_stream(str(af))
            table = reader.read_all()
        except Exception:
            global_idx += n_rows
            continue

        img_col = None
        for col in ["image", "img"]:
            if col in table.column_names:
                img_col = col
                break
        if img_col is None:
            global_idx += n_rows
            continue

        for i in local_indices:
            try:
                data = table.column(img_col)[i].as_py()
                if isinstance(data, bytes):
                    images.append(Image.open(io.BytesIO(data)).convert("RGB"))
                elif isinstance(data, dict) and "bytes" in data:
                    images.append(Image.open(io.BytesIO(data["bytes"])).convert("RGB"))
            except Exception:
                pass

        global_idx += n_rows
        if max_samples and len(images) >= max_samples:
            break
    return images


def load_condition_images(config: dict, max_samples: int = None) -> list:
    src = config["source"]
    path = src["path"]
    if not path.exists():
        return []
    if src["type"] == "arrow":
        return load_images_from_arrow(path, max_samples)
    elif src["type"] == "arrow_dir":
        return load_images_from_arrow_dir(path, max_samples)
    return []


# ── Classification ───────────────────────────────────────────────

def classify_batch(model, processor, images: list, batch_size: int = 64) -> list:
    """Classify a list of PIL images. Returns list of (predicted_label, probs_dict)."""
    results = []
    device = next(model.parameters()).device

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()

        for p in probs:
            pred_idx = int(p.argmax())
            pred_label = ID2LABEL[pred_idx]
            prob_dict = {ID2LABEL[j]: float(p[j]) for j in range(len(p))}
            results.append((pred_label, prob_dict))

        done = min(i + batch_size, len(images))
        print(f"      {done}/{len(images)}", end="\r")
    print()
    return results


# ── Visualization ────────────────────────────────────────────────

def generate_accuracy_barplot(summary: dict, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = []
    accuracies = []
    for cond, stats in summary.items():
        if stats.get("expected_accuracy") is not None:
            conditions.append(cond)
            accuracies.append(stats["expected_accuracy"] * 100)

    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.0), 5))

    colors = []
    for acc in accuracies:
        if acc >= 70:
            colors.append("#2ecc71")
        elif acc >= 40:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    bars = ax.bar(range(len(conditions)), accuracies, color=colors, edgecolor="white")
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Weather Classification Accuracy (%)", fontsize=11)
    ax.set_title("Weather Classifier Accuracy on Augmented Images\n"
                 "(Does the classifier recognize the intended weather condition?)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_confusion_heatmap(summary: dict, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [c for c in summary if summary[c].get("class_distribution")]
    if not conditions:
        return

    classes = list(ID2LABEL.values())
    matrix = []
    for cond in conditions:
        dist = summary[cond]["class_distribution"]
        row = [dist.get(cls, 0) * 100 for cls in classes]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, max(4, len(conditions) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.split("/")[0] for c in classes], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=8)

    for i in range(len(conditions)):
        for j in range(len(classes)):
            val = matrix[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Weather Classification Distribution per Augmentation Condition",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Weather Class", fontsize=10)
    plt.colorbar(im, ax=ax, label="% of images")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Weather classification validation on ConSynth-X augmented data"
    )
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Weather Classification Validation")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Load model
    from transformers import AutoImageProcessor, SiglipForImageClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model on {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipForImageClassification.from_pretrained(MODEL_NAME).to(device).eval()
    print("  Model loaded.")

    # Select conditions
    if args.conditions:
        configs = {k: v for k, v in CONDITION_CONFIG.items() if k in args.conditions}
    else:
        configs = CONDITION_CONFIG

    summary = {}
    t_start = time.time()

    for cond_name, config in configs.items():
        print(f"\n  [{cond_name}] {config['description']}")
        images = load_condition_images(config, args.max_samples)
        if not images:
            print(f"    SKIP: no images loaded")
            continue

        print(f"    Classifying {len(images)} images...")
        results = classify_batch(model, processor, images, args.batch_size)

        # Compute stats
        predicted_labels = [r[0] for r in results]
        label_counts = defaultdict(int)
        for lbl in predicted_labels:
            label_counts[lbl] += 1

        n = len(results)
        class_dist = {lbl: count / n for lbl, count in label_counts.items()}

        expected = config["expected"]
        if expected:
            expected_acc = label_counts.get(expected, 0) / n
        else:
            expected_acc = None

        # Mean confidence for expected class
        if expected:
            expected_confs = [r[1].get(expected, 0) for r in results]
            mean_conf = float(np.mean(expected_confs))
        else:
            mean_conf = None

        summary[cond_name] = {
            "n_images": n,
            "expected_class": expected,
            "expected_accuracy": expected_acc,
            "expected_mean_confidence": mean_conf,
            "top_predicted": max(label_counts, key=label_counts.get),
            "class_distribution": class_dist,
            "description": config["description"],
        }

        # Print
        print(f"    Expected: {expected or 'N/A'}")
        if expected_acc is not None:
            print(f"    Accuracy: {expected_acc*100:.1f}%  (confidence: {mean_conf:.3f})")
        print(f"    Distribution: ", end="")
        for lbl in sorted(class_dist, key=class_dist.get, reverse=True):
            print(f"{lbl.split('/')[0]}={class_dist[lbl]*100:.1f}%  ", end="")
        print()

        del images
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # ── Results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"RESULTS (computed in {elapsed:.0f}s)")
    print(f"{'=' * 60}")

    print(f"\n{'Condition':<25} {'Expected':<15} {'Accuracy':>10} {'Top Predicted':<20} {'N':>6}")
    print("-" * 80)
    for cond, stats in summary.items():
        exp = stats["expected_class"] or "N/A"
        acc = f"{stats['expected_accuracy']*100:.1f}%" if stats["expected_accuracy"] is not None else "N/A"
        top = stats["top_predicted"]
        print(f"{cond:<25} {exp:<15} {acc:>10} {top:<20} {stats['n_images']:>6}")

    # Save
    json_path = output_dir / "weather_cls_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Figures
    generate_accuracy_barplot(summary, output_dir / "weather_cls_accuracy.png")
    generate_confusion_heatmap(summary, output_dir / "weather_cls_heatmap.png")

    # LaTeX
    print(f"\n{'=' * 60}")
    print("Paper-ready table (LaTeX):")
    print(f"{'=' * 60}")
    print(r"\begin{tabular}{llccc}")
    print(r"\hline")
    print(r"Condition & Expected & Accuracy (\%) & Confidence & N \\")
    print(r"\hline")
    for cond, stats in summary.items():
        label = cond.replace("_", r"\_")
        exp = (stats["expected_class"] or "---").replace("/", r"/")
        acc = f"{stats['expected_accuracy']*100:.1f}" if stats["expected_accuracy"] is not None else "---"
        conf = f"{stats['expected_mean_confidence']:.3f}" if stats["expected_mean_confidence"] is not None else "---"
        print(f"{label} & {exp} & {acc} & {conf} & {stats['n_images']} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

    print(f"\nDone! Results at {output_dir}/")


if __name__ == "__main__":
    main()
