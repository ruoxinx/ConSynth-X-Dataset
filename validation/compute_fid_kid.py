#!/usr/bin/env python3
"""
Compute FID and KID between ConSynth-X augmented images and real weather
reference images (ACDC dataset).

FID (Fréchet Inception Distance): measures distribution distance using
InceptionV3 features. Lower = more similar to real weather.

KID (Kernel Inception Distance): unbiased alternative to FID, better for
small sample sizes (<5000). Reports mean ± std.

Usage:
  # Compare all conditions against ACDC reference
  python validation/compute_fid_kid.py

  # Compare specific condition
  python validation/compute_fid_kid.py --conditions weather_style_rain_0 diffusion_rain

  # Quick test with fewer images
  python validation/compute_fid_kid.py --max-samples 200

Reference datasets:
  - ACDC: Sakaridis et al., ICCV 2021, ArXiv: 2104.13395
  - FID: Heusel et al., NeurIPS 2017
  - KID: Bińkowski et al., ICLR 2018
"""

import argparse
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ── Paths ────────────────────────────────────────────────────────
DATA_ROOT = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite")
ARROW_DATA = DATA_ROOT / "augmentation_data_arrow"
FOG_DATA = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/augmentation_data/construction_site/fog/diffusion/test")
REFERENCE_BASE = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/validation/reference_data")
OUT_DIR = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/validation/results/fid_kid")

# Available reference datasets (auto-detected at runtime)
REFERENCE_DATASETS = {
    "acdc": REFERENCE_BASE / "acdc" / "rgb_anon",  # ACDC: rain, fog, snow, night (4 conditions, driving scenes)
    "weatherbench": REFERENCE_BASE / "weatherbench",  # WeatherBench: rain, snow, haze (42K real-world paired images)
    "weathernet": REFERENCE_BASE / "weathernet",   # WeatherNet-05: snow, fog (rain/clear/cloudy removed)
}

# ── Condition mapping ────────────────────────────────────────────
# Maps our augmentation conditions to reference conditions
CONDITION_PAIRS = {
    # Our condition → (data source, ACDC reference condition)
    "weather_style_rain_0": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_0.arrow"},
        "reference": "rain",
        "description": "Style transfer rain (light) vs ACDC real rain",
    },
    "weather_style_rain_1": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_1.arrow"},
        "reference": "rain",
        "description": "Style transfer rain (moderate) vs ACDC real rain",
    },
    "weather_style_rain_2": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_2.arrow"},
        "reference": "rain",
        "description": "Style transfer rain (heavy) vs ACDC real rain",
    },
    "weather_style_snow_0": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_0.arrow"},
        "reference": "snow",
        "description": "Style transfer snow (light) vs ACDC real snow",
    },
    "weather_style_snow_1": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_1.arrow"},
        "reference": "snow",
        "description": "Style transfer snow (moderate) vs ACDC real snow",
    },
    "weather_style_snow_2": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_2.arrow"},
        "reference": "snow",
        "description": "Style transfer snow (heavy) vs ACDC real snow",
    },
    "diffusion_rain": {
        "source": {
            "type": "arrow_dir",
            "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_rain_heavy",
        },
        "reference": "rain",
        "description": "IP2P diffusion rain vs ACDC real rain",
    },
    "diffusion_snow": {
        "source": {
            "type": "arrow_dir",
            "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_snow_heavy",
        },
        "reference": "snow",
        "description": "IP2P diffusion snow vs ACDC real snow",
    },
    "diffusion_fog_heavy": {
        "source": {"type": "arrow_dir", "path": FOG_DATA / "heavy"},
        "reference": "fog",
        "description": "IP2P diffusion fog (heavy) vs real fog",
    },
    "diffusion_fog_medium": {
        "source": {"type": "arrow_dir", "path": FOG_DATA / "medium"},
        "reference": "fog",
        "description": "IP2P diffusion fog (medium) vs real fog",
    },
    "diffusion_fog_light": {
        "source": {"type": "arrow_dir", "path": FOG_DATA / "light"},
        "reference": "fog",
        "description": "IP2P diffusion fog (light) vs real fog",
    },
    "night": {
        "source": {"type": "arrow", "path": ARROW_DATA / "night.arrow"},
        "reference": "night",
        "description": "CycleGAN-Turbo night vs ACDC real night",
    },
    # Baseline: original vs ACDC conditions (expect HIGH FID = different domains)
    "original_vs_rain": {
        "source": {"type": "arrow", "path": ARROW_DATA / "construction_site_test.arrow"},
        "reference": "rain",
        "description": "Original (clear) vs ACDC real rain — baseline gap",
    },
    "original_vs_snow": {
        "source": {"type": "arrow", "path": ARROW_DATA / "construction_site_test.arrow"},
        "reference": "snow",
        "description": "Original (clear) vs ACDC real snow — baseline gap",
    },
    "original_vs_night": {
        "source": {"type": "arrow", "path": ARROW_DATA / "construction_site_test.arrow"},
        "reference": "night",
        "description": "Original (clear) vs ACDC real night — baseline gap",
    },
    "original_vs_fog": {
        "source": {"type": "arrow", "path": ARROW_DATA / "construction_site_test.arrow"},
        "reference": "fog",
        "description": "Original (clear) vs real fog — baseline gap",
    },
}


# ── InceptionV3 Feature Extractor ────────────────────────────────

class InceptionV3Features(nn.Module):
    """Extract pool3 (2048-dim) features from InceptionV3 for FID/KID."""

    def __init__(self, device="cuda"):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # Remove the final FC layer — we want pool3 features
        self.blocks = nn.Sequential(
            model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
            model.Mixed_6a, model.Mixed_6b, model.Mixed_6c, model.Mixed_6d, model.Mixed_6e,
            model.Mixed_7a, model.Mixed_7b, model.Mixed_7c,
            nn.AdaptiveAvgPool2d(1),
        )
        self.blocks.eval().to(device)
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_features(self, images: list, batch_size: int = 64) -> np.ndarray:
        """Extract 2048-dim features from a list of PIL images."""
        all_features = []
        for i in range(0, len(images), batch_size):
            batch_pil = images[i:i + batch_size]
            batch_tensor = torch.stack([self.transform(img) for img in batch_pil]).to(self.device)
            features = self.blocks(batch_tensor)
            features = features.squeeze(-1).squeeze(-1)  # (B, 2048)
            all_features.append(features.cpu().numpy())
            done = min(i + batch_size, len(images))
            print(f"      Features: {done}/{len(images)}", end="\r")
        print()
        return np.concatenate(all_features, axis=0)


# ── FID/KID Computation ──────────────────────────────────────────

def compute_fid(features_real: np.ndarray, features_gen: np.ndarray) -> float:
    """Compute Fréchet Inception Distance between two sets of features."""
    from scipy.linalg import sqrtm

    mu_real = features_real.mean(axis=0)
    mu_gen = features_gen.mean(axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_gen = np.cov(features_gen, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid)


def compute_kid(features_real: np.ndarray, features_gen: np.ndarray,
                num_subsets: int = 100, subset_size: int = 500) -> tuple:
    """Compute Kernel Inception Distance (mean ± std).

    Uses polynomial kernel k(x,y) = (x·y/d + 1)^3 as in Bińkowski et al. 2018.
    Returns (mean, std) over random subsets.
    """
    n_real = len(features_real)
    n_gen = len(features_gen)
    d = features_real.shape[1]
    actual_subset = min(subset_size, n_real, n_gen)

    rng = np.random.default_rng(42)
    kid_values = []

    for _ in range(num_subsets):
        idx_real = rng.choice(n_real, size=actual_subset, replace=False)
        idx_gen = rng.choice(n_gen, size=actual_subset, replace=False)

        f_real = features_real[idx_real]
        f_gen = features_gen[idx_gen]

        # Polynomial kernel: k(x,y) = (x·y/d + 1)^3
        k_rr = (f_real @ f_real.T / d + 1) ** 3
        k_gg = (f_gen @ f_gen.T / d + 1) ** 3
        k_rg = (f_real @ f_gen.T / d + 1) ** 3

        # MMD^2 unbiased estimator
        n = actual_subset
        kid = (k_rr.sum() - np.trace(k_rr)) / (n * (n - 1)) \
            + (k_gg.sum() - np.trace(k_gg)) / (n * (n - 1)) \
            - 2 * k_rg.mean()

        kid_values.append(float(kid))

    return float(np.mean(kid_values)), float(np.std(kid_values))


# ── Data Loading ─────────────────────────────────────────────────

def load_images_from_arrow(arrow_path: Path, max_samples: int = None) -> list:
    """Load PIL images from Arrow file (IPC stream format)."""
    import pyarrow as pa
    images = []

    try:
        reader = pa.ipc.open_stream(str(arrow_path))
        table = reader.read_all()
    except Exception:
        try:
            reader = pa.ipc.open_file(str(arrow_path))
            table = reader.read_all()
        except Exception as e:
            print(f"    Error reading {arrow_path.name}: {e}")
            return []

    img_col = None
    for col in ["image", "img", "pixel_values"]:
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
    """Load PIL images from directory of batched Arrow files."""
    import pyarrow as pa
    arrow_files = sorted(dir_path.glob("*.arrow"))
    if not arrow_files:
        return []

    # Count total
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


def load_images_from_dir(dir_path: Path, max_samples: int = None) -> list:
    """Load images from a directory of image files."""
    extensions = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in dir_path.rglob("*") if f.suffix.lower() in extensions)
    if max_samples:
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(len(files), size=min(max_samples, len(files)), replace=False))
        files = [files[i] for i in indices]
    images = []
    for f in files:
        try:
            images.append(Image.open(f).convert("RGB"))
        except Exception:
            pass
    return images


def load_condition_images(config: dict, max_samples: int = None) -> list:
    """Load images for a condition based on config type."""
    src = config["source"]
    path = src["path"]
    if not path.exists():
        print(f"    WARNING: {path} does not exist")
        return []

    if src["type"] == "arrow":
        return load_images_from_arrow(path, max_samples)
    elif src["type"] == "arrow_dir":
        return load_images_from_arrow_dir(path, max_samples)
    return []


# ── Visualization ────────────────────────────────────────────────

def generate_fid_kid_barplot(results: dict, output_path: Path):
    """Generate grouped bar chart: FID per condition, grouped by reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by reference condition
    groups = defaultdict(list)
    for cond, stats in results.items():
        ref = stats["reference"]
        groups[ref].append((cond, stats))

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 6), squeeze=False)
    axes = axes[0]

    for ax, (ref_cond, items) in zip(axes, sorted(groups.items())):
        items.sort(key=lambda x: x[1]["fid"])
        names = [it[0].replace("weather_style_", "style_").replace("original_vs_", "orig→")
                 for it in items]
        fids = [it[1]["fid"] for it in items]

        colors = []
        for name, _ in items:
            if "original" in name:
                colors.append("#95a5a6")  # gray = baseline
            elif "diffusion" in name:
                colors.append("#3498db")  # blue = diffusion
            elif "style" in name:
                colors.append("#e74c3c")  # red = style transfer
            elif "night" in name:
                colors.append("#9b59b6")  # purple = night
            else:
                colors.append("#2ecc71")

        bars = ax.barh(range(len(names)), fids, color=colors, edgecolor="white")
        for bar, val in zip(bars, fids):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=9)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("FID ↓ (lower = more realistic)", fontsize=10)
        ax.set_title(f"vs ACDC {ref_cond}", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle("FID: ConSynth-X Augmented vs Real Weather Reference\n"
                 "(Lower FID = augmented distribution closer to real weather)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────

def run_fid_kid_for_reference(ref_name: str, ref_dir: Path, pairs: dict,
                               extractor, args, output_dir: Path) -> dict:
    """Run FID/KID for one reference dataset."""
    # Discover available conditions
    ref_conditions = {}
    for cond_dir in sorted(ref_dir.iterdir()):
        if cond_dir.is_dir():
            n = len(list(cond_dir.rglob("*.jpg"))) + len(list(cond_dir.rglob("*.png")))
            if n > 0:
                ref_conditions[cond_dir.name] = n

    print(f"\n  Available conditions: {ref_conditions}")
    if not ref_conditions:
        print(f"  No images found in {ref_dir}")
        return {}

    # Condition name aliases (e.g., our "fog" maps to WeatherBench's "haze")
    CONDITION_ALIASES = {"fog": "haze", "haze": "fog"}

    # Filter pairs to those with matching reference (check aliases too)
    def resolve_ref(ref_cond):
        if ref_cond in ref_conditions:
            return ref_cond
        alias = CONDITION_ALIASES.get(ref_cond)
        if alias and alias in ref_conditions:
            return alias
        return None

    active_pairs = {}
    ref_mapping = {}  # maps pair key → actual ref condition name in this dataset
    for k, v in pairs.items():
        resolved = resolve_ref(v["reference"])
        if resolved:
            active_pairs[k] = v
            ref_mapping[k] = resolved
    if not active_pairs:
        print(f"  No matching conditions between augmented data and {ref_name}")
        return {}

    print(f"  Evaluating {len(active_pairs)} condition pairs")

    # Pre-compute reference features (use resolved folder names)
    ref_features = {}
    for actual_cond in set(ref_mapping.values()):
        print(f"\n    [{ref_name}/{actual_cond}] Loading reference images...")
        ref_images = load_images_from_dir(ref_dir / actual_cond, max_samples=args.max_samples)
        if not ref_images:
            continue
        print(f"      Loaded {len(ref_images)} images. Extracting features...")
        ref_features[actual_cond] = extractor.extract_features(ref_images, args.batch_size)
        del ref_images
        torch.cuda.empty_cache()

    # Compute FID/KID per pair
    results = {}
    for cond_name, config in active_pairs.items():
        ref_cond = ref_mapping[cond_name]  # resolved actual folder name
        if ref_cond not in ref_features:
            continue

        print(f"\n    [{cond_name}] {config['description']}")
        images = load_condition_images(config, args.max_samples)
        if not images:
            print(f"      SKIP: no images loaded")
            continue
        print(f"      Loaded {len(images)} images. Extracting features...")

        gen_features = extractor.extract_features(images, args.batch_size)

        fid_value = compute_fid(ref_features[ref_cond], gen_features)
        kid_subset = min(500, len(images), len(ref_features[ref_cond]))
        kid_mean, kid_std = compute_kid(ref_features[ref_cond], gen_features,
                                         subset_size=kid_subset)

        key = f"{cond_name}@{ref_name}"
        results[key] = {
            "fid": fid_value,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
            "reference": config["reference"],  # original name (e.g., "fog"), not alias
            "ref_dataset": ref_name,
            "n_augmented": len(images),
            "n_reference": len(ref_features[ref_cond]),
            "description": config["description"],
        }

        print(f"      FID = {fid_value:.2f}  |  KID = {kid_mean:.4f} ± {kid_std:.4f}")

        del images, gen_features
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="FID/KID: ConSynth-X augmented vs real weather (WeatherNet + BDD100K)"
    )
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Specific conditions to evaluate (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max images per condition (for quick testing)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect available reference datasets
    available_refs = {}
    for name, path in REFERENCE_DATASETS.items():
        if path.exists() and any(path.iterdir()):
            available_refs[name] = path

    if not available_refs:
        print("ERROR: No reference data found.")
        print("Run first: python validation/download_reference.py")
        return

    print(f"{'=' * 60}")
    print(f"FID/KID: ConSynth-X vs Real Weather")
    print(f"{'=' * 60}")
    print(f"Reference datasets: {list(available_refs.keys())}")

    # Select condition pairs
    if args.conditions:
        pairs = {k: v for k, v in CONDITION_PAIRS.items() if k in args.conditions}
    else:
        pairs = CONDITION_PAIRS

    # Initialize feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading InceptionV3 on {device}...")
    extractor = InceptionV3Features(device)

    # Run FID/KID for each reference dataset
    all_results = {}
    t_start = time.time()

    for ref_name, ref_dir in available_refs.items():
        print(f"\n{'=' * 60}")
        print(f"Reference: {ref_name} ({ref_dir})")
        print(f"{'=' * 60}")

        results = run_fid_kid_for_reference(ref_name, ref_dir, pairs, extractor, args, output_dir)
        all_results.update(results)

    elapsed = time.time() - t_start

    # ── Combined Results ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"ALL RESULTS (computed in {elapsed:.0f}s)")
    print(f"{'=' * 60}")

    header = f"{'Condition':<30} {'RefDS':<12} {'RefCond':<8} {'FID ↓':>10} {'KID ↓':>18} {'N_aug':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    for key, stats in sorted(all_results.items(),
                              key=lambda x: (x[1]["ref_dataset"], x[1]["reference"], x[1]["fid"])):
        cond = key.split("@")[0]
        print(f"{cond:<30} {stats['ref_dataset']:<12} {stats['reference']:<8} "
              f"{stats['fid']:>10.2f} "
              f"{stats['kid_mean']:>8.4f}±{stats['kid_std']:.4f} "
              f"{stats['n_augmented']:>7}")

    # Save JSON
    json_path = output_dir / "fid_kid_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Generate figures per reference dataset
    for ref_name in available_refs:
        ref_results = {k.split("@")[0]: v for k, v in all_results.items()
                       if v["ref_dataset"] == ref_name}
        if ref_results:
            generate_fid_kid_barplot(ref_results,
                                     output_dir / f"fid_kid_{ref_name}.png")

    # LaTeX table
    print(f"\n{'=' * 60}")
    print("Paper-ready table (LaTeX):")
    print(f"{'=' * 60}")
    print(r"\begin{tabular}{lllcc}")
    print(r"\hline")
    print(r"Augmentation & Ref. Dataset & Condition & FID $\downarrow$ & KID ($\times 10^{-3}$) $\downarrow$ \\")
    print(r"\hline")
    for key, stats in sorted(all_results.items(),
                              key=lambda x: (x[1]["ref_dataset"], x[1]["reference"], x[1]["fid"])):
        cond = key.split("@")[0].replace("_", r"\_")
        ref_ds = stats["ref_dataset"]
        ref_cond = stats["reference"]
        fid_val = stats["fid"]
        kid_val = stats["kid_mean"] * 1000
        kid_std = stats["kid_std"] * 1000
        print(f"{cond} & {ref_ds} & {ref_cond} & {fid_val:.1f} & {kid_val:.2f}$\\pm${kid_std:.2f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

    print(f"\nDone! Results at {output_dir}/")


if __name__ == "__main__":
    main()
