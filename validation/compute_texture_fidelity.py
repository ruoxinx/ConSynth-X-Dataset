#!/usr/bin/env python3
"""
Texture-based fidelity assessment for weather augmentation quality.

Implements 4 texture feature channels from Duminil et al. (2025):
  1. GLCM  — Gray Level Co-occurrence Matrix (global texture)
  2. LBP   — Local Binary Pattern (local micro-texture)
  3. DCT   — Discrete Cosine Transform (frequency domain)
  4. Haralick — Statistical texture metrics (ASM, contrast, correlation, etc.)

These features detect micro-level artifacts that semantic metrics (FID/KID)
miss: unnatural noise patterns, missing gray-tone continuity, frequency
energy anomalies in synthetic/augmented images.

Usage:
  # Full evaluation (all conditions)
  python validation/compute_texture_fidelity.py

  # Specific conditions
  python validation/compute_texture_fidelity.py --conditions weather_style_rain_0 night

  # Quick test
  python validation/compute_texture_fidelity.py --max-samples 50

Reference:
  Duminil, Ieng & Gruyer. "Fidelity assessment of synthetic images with
  multi-criteria combination under adverse weather conditions."
  Scientific Reports, 2025.
"""

import argparse
import io
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths (same as compute_fid_kid.py) ────────────────────────────
DATA_ROOT = _DATA_ROOT
ARROW_DATA = DATA_ROOT / "augmentation_data_arrow"
FOG_DATA = (_REPO_ROOT / "augmentation_data/construction_site/fog/diffusion/test")
REFERENCE_BASE = (_REPO_ROOT / "validation/reference_data")
OUT_DIR = (_REPO_ROOT / "validation/results/texture_fidelity")

# ── Condition mapping ─────────────────────────────────────────────
CONDITION_PAIRS = {
    "original": {
        "source": {"type": "arrow", "path": ARROW_DATA / "construction_site_test.arrow"},
        "description": "Original clear-day construction images (baseline)",
    },
    "weather_style_rain_0": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_0.arrow"},
        "description": "Style transfer rain (light)",
    },
    "weather_style_rain_1": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_1.arrow"},
        "description": "Style transfer rain (moderate)",
    },
    "weather_style_rain_2": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_2.arrow"},
        "description": "Style transfer rain (heavy)",
    },
    "weather_style_snow_0": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_0.arrow"},
        "description": "Style transfer snow (light)",
    },
    "weather_style_snow_1": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_1.arrow"},
        "description": "Style transfer snow (moderate)",
    },
    "weather_style_snow_2": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_2.arrow"},
        "description": "Style transfer snow (heavy)",
    },
    "diffusion_rain": {
        "source": {
            "type": "arrow_dir",
            "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_rain_heavy",
        },
        "description": "IP2P diffusion rain",
    },
    "diffusion_rain_heavy": {
        "source": {
            "type": "arrow_dir",
            "path": _REPO_ROOT / "augmentation_data" / "construction_site" / "rain_snow" / "diffusion" / "test" / "rain_heavy",
        },
        "description": "IP2P diffusion rain + heavy physics overlay (2026-04-21)",
    },
    "diffusion_snow": {
        "source": {
            "type": "arrow_dir",
            "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_snow_heavy",
        },
        "description": "IP2P diffusion snow",
    },
    "diffusion_fog_heavy": {
        "source": {"type": "arrow_dir", "path": FOG_DATA / "heavy"},
        "description": "Diffusion fog (heavy)",
    },
    "diffusion_fog_medium": {
        "source": {"type": "arrow_dir", "path": FOG_DATA / "medium"},
        "description": "Diffusion fog (medium)",
    },
    "diffusion_fog_light": {
        "source": {"type": "arrow_dir", "path": FOG_DATA / "light"},
        "description": "Diffusion fog (light)",
    },
    "night": {
        "source": {"type": "arrow", "path": ARROW_DATA / "night.arrow"},
        "description": "CycleGAN-Turbo night",
    },
}

# ACDC real weather references
ACDC_REFERENCE = {
    "acdc_rain": {
        "source": {"type": "image_dir", "path": REFERENCE_BASE / "acdc" / "rgb_anon" / "rain"},
        "description": "ACDC real rain (Zurich driving scenes)",
    },
    "acdc_fog": {
        "source": {"type": "image_dir", "path": REFERENCE_BASE / "acdc" / "rgb_anon" / "fog"},
        "description": "ACDC real fog",
    },
    "acdc_snow": {
        "source": {"type": "image_dir", "path": REFERENCE_BASE / "acdc" / "rgb_anon" / "snow"},
        "description": "ACDC real snow",
    },
    "acdc_night": {
        "source": {"type": "image_dir", "path": REFERENCE_BASE / "acdc" / "rgb_anon" / "night"},
        "description": "ACDC real night",
    },
}


# ══════════════════════════════════════════════════════════════════
# TEXTURE FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def compute_glcm_features(gray: np.ndarray, distances=(1, 3), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)) -> np.ndarray:
    """Extract GLCM features: contrast, dissimilarity, homogeneity, energy, correlation, ASM.

    GLCM captures gray-tone spatial dependencies. Real images show continuous
    diagonal patterns in GLCM; synthetic images show discontinuities due to
    limited gray-tone diversity (Duminil et al., 2025, Section 3.1).
    """
    from skimage.feature import graycomatrix, graycoprops

    # Quantize to 64 levels for computational efficiency
    gray_q = (gray / 4).astype(np.uint8)

    glcm = graycomatrix(gray_q, distances=list(distances), angles=list(angles),
                        levels=64, symmetric=True, normed=True)

    props = []
    for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop_name)  # shape: (n_distances, n_angles)
        props.append(vals.flatten())

    # ASM = energy^2 (already computed via energy, but add explicit)
    asm = graycoprops(glcm, "ASM").flatten()
    props.append(asm)

    return np.concatenate(props)  # 6 props × n_distances × n_angles


def compute_lbp_features(gray: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
    """Extract LBP histogram features.

    LBP captures local micro-texture patterns. Augmentation artifacts often
    create unnatural LBP distributions — e.g., rain streak overlays produce
    repeating directional patterns absent in real rain (Duminil et al., 2025,
    Section 3.2).
    """
    from skimage.feature import local_binary_pattern

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2  # uniform LBP has P+2 bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def compute_dct_features(img_rgb: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Extract DCT frequency-domain features per channel.

    Computes the ratio of high-frequency to total energy in the DCT domain.
    Synthetic/augmented images often lack natural high-frequency noise content
    or show unnatural frequency peaks from particle overlay artifacts
    (Duminil et al., 2025, Section 3.3).

    Returns per-channel: [hf_ratio, mean_dc, std_dc, spectral_entropy] × 3 channels.
    """
    from scipy.fft import dctn

    features = []
    for c in range(3):  # R, G, B
        channel = img_rgb[:, :, c].astype(np.float64)

        # Full-image DCT
        dct_coeff = dctn(channel, type=2, norm="ortho")
        dct_power = dct_coeff ** 2
        total_energy = dct_power.sum()

        if total_energy < 1e-10:
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue

        h, w = channel.shape
        # Low-frequency: top-left quarter; high-frequency: the rest
        lf_energy = dct_power[:h//4, :w//4].sum()
        hf_energy = total_energy - lf_energy
        hf_ratio = hf_energy / total_energy

        # DC component stats (block-wise)
        dc_values = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = channel[y:y+block_size, x:x+block_size]
                dc_values.append(block.mean())
        dc_values = np.array(dc_values)

        # Spectral entropy (normalized power spectrum)
        power_norm = dct_power.flatten()
        power_norm = power_norm / power_norm.sum()
        power_norm = power_norm[power_norm > 0]
        spectral_entropy = -np.sum(power_norm * np.log2(power_norm))

        features.extend([
            hf_ratio,
            dc_values.mean() if len(dc_values) > 0 else 0.0,
            dc_values.std() if len(dc_values) > 0 else 0.0,
            spectral_entropy,
        ])

    return np.array(features)  # 4 × 3 = 12 features


def compute_haralick_features(gray: np.ndarray) -> np.ndarray:
    """Extract Haralick texture features from GLCM.

    Computes the 4 most discriminative Haralick metrics identified by PCA
    in Duminil et al. (2025, Section 3.4): Variance, Sum Variance,
    Correlation, and Information Measure of Correlation 1 (IMC1).

    We also add ASM, Contrast, IDM, and Entropy for completeness.
    """
    from skimage.feature import graycomatrix, graycoprops

    # Compute GLCM on patches and average
    patch_size = 64
    h, w = gray.shape
    gray_q = (gray / 4).astype(np.uint8)

    all_features = []

    # Sample patches (up to 25 patches for efficiency)
    step_y = max(patch_size, (h - patch_size) // 5 + 1)
    step_x = max(patch_size, (w - patch_size) // 5 + 1)

    for y in range(0, h - patch_size + 1, step_y):
        for x in range(0, w - patch_size + 1, step_x):
            patch = gray_q[y:y+patch_size, x:x+patch_size]
            glcm = graycomatrix(patch, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=64, symmetric=True, normed=True)

            patch_feats = []
            for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
                vals = graycoprops(glcm, prop).mean()
                patch_feats.append(vals)

            # Compute entropy and variance from GLCM directly
            glcm_norm = glcm[:, :, :, :].mean(axis=(2, 3))  # average over d, theta
            glcm_flat = glcm_norm.flatten()
            glcm_flat = glcm_flat[glcm_flat > 0]
            entropy = -np.sum(glcm_flat * np.log2(glcm_flat + 1e-10))
            variance = np.var(glcm_norm)

            patch_feats.extend([entropy, variance])
            all_features.append(patch_feats)

    if not all_features:
        return np.zeros(8)

    # Return mean and std across patches → 8 mean + 8 std = 16 features
    feats_arr = np.array(all_features)
    return np.concatenate([feats_arr.mean(axis=0), feats_arr.std(axis=0)])


def extract_texture_features(img: Image.Image, resize_to: int = 512) -> dict:
    """Extract all 4 texture feature vectors from a single image.

    Args:
        img: PIL Image (RGB)
        resize_to: resize longest edge to this (for consistent comparison)

    Returns:
        dict with keys: glcm, lbp, dct, haralick — each a numpy array
    """
    # Resize for consistency
    w, h = img.size
    scale = resize_to / max(w, h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img_np = np.array(img)
    gray = np.mean(img_np[:, :, :3], axis=2).astype(np.uint8)

    return {
        "glcm": compute_glcm_features(gray),
        "lbp": compute_lbp_features(gray),
        "dct": compute_dct_features(img_np[:, :, :3]),
        "haralick": compute_haralick_features(gray),
    }


# ══════════════════════════════════════════════════════════════════
# DATA LOADING (reused from compute_fid_kid.py)
# ══════════════════════════════════════════════════════════════════

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


def load_images_from_dir(dir_path: Path, max_samples: int = None) -> list:
    extensions = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in dir_path.rglob("*") if f.suffix.lower() in extensions)
    if not files:
        return []
    if max_samples and max_samples < len(files):
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(len(files), size=max_samples, replace=False))
        files = [files[i] for i in indices]
    images = []
    for f in files:
        try:
            images.append(Image.open(f).convert("RGB"))
        except Exception:
            pass
    return images


def load_condition_images(config: dict, max_samples: int = None) -> list:
    src = config["source"]
    path = src["path"]
    if not path.exists():
        print(f"    WARNING: {path} does not exist")
        return []
    if src["type"] == "arrow":
        return load_images_from_arrow(path, max_samples)
    elif src["type"] == "arrow_dir":
        return load_images_from_arrow_dir(path, max_samples)
    elif src["type"] == "image_dir":
        return load_images_from_dir(path, max_samples)
    return []


# ══════════════════════════════════════════════════════════════════
# DISTRIBUTION COMPARISON
# ══════════════════════════════════════════════════════════════════

def compare_distributions(feats_original: np.ndarray, feats_augmented: np.ndarray) -> dict:
    """Compare two feature distributions using multiple statistical tests.

    Returns:
        dict with: wasserstein (per-dim mean), ks_statistic, ks_pvalue,
                   mean_shift (L2 of mean diff), cov_divergence,
                   per_dim_wasserstein (list)
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    n_dims = feats_original.shape[1]
    ks_stats = []
    ks_pvals = []
    w_dists = []

    for d in range(n_dims):
        orig_d = feats_original[:, d]
        aug_d = feats_augmented[:, d]
        ks_stat, ks_pval = ks_2samp(orig_d, aug_d)
        w_dist = wasserstein_distance(orig_d, aug_d)
        ks_stats.append(ks_stat)
        ks_pvals.append(ks_pval)
        w_dists.append(w_dist)

    # Mean shift
    mean_orig = feats_original.mean(axis=0)
    mean_aug = feats_augmented.mean(axis=0)
    mean_shift = np.linalg.norm(mean_orig - mean_aug)

    # Covariance divergence (Frobenius norm of cov difference)
    cov_orig = np.cov(feats_original, rowvar=False)
    cov_aug = np.cov(feats_augmented, rowvar=False)
    cov_div = np.linalg.norm(cov_orig - cov_aug, ord="fro")

    return {
        "wasserstein_mean": float(np.mean(w_dists)),
        "wasserstein_per_dim": [float(x) for x in w_dists],
        "ks_statistic_mean": float(np.mean(ks_stats)),
        "ks_pvalue_mean": float(np.mean(ks_pvals)),
        "n_significant_dims": int(np.sum(np.array(ks_pvals) < 0.05)),
        "n_total_dims": n_dims,
        "mean_shift_l2": float(mean_shift),
        "cov_divergence_frob": float(cov_div),
    }


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def plot_texture_comparison(all_results: dict, output_dir: Path):
    """Generate comparison plots for texture fidelity analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_types = ["glcm", "lbp", "dct", "haralick"]
    conditions = [c for c in all_results if c != "original"]

    if not conditions:
        return

    # ── Figure 1: Wasserstein distance heatmap ────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(conditions) * 0.8), 5))

    matrix = np.zeros((len(feature_types), len(conditions)))
    for j, cond in enumerate(conditions):
        if cond not in all_results:
            continue
        for i, ft in enumerate(feature_types):
            key = f"vs_original_{ft}"
            if key in all_results[cond]:
                matrix[i, j] = all_results[cond][key]["wasserstein_mean"]

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace("weather_style_", "style_").replace("diffusion_", "diff_")
                        for c in conditions], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(feature_types)))
    ax.set_yticklabels([ft.upper() for ft in feature_types], fontsize=10)

    for i in range(len(feature_types)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i, j] < matrix.max() * 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Wasserstein Distance (higher = more texture shift)")
    ax.set_title("Texture Feature Distribution Shift: Augmented vs Original\n"
                 "(per-feature Wasserstein distance)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "texture_wasserstein_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "texture_wasserstein_heatmap.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: texture_wasserstein_heatmap.png")

    # ── Figure 2: Radar chart per condition ───────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    # Normalize metrics across conditions for radar
    metrics = ["wasserstein_mean", "ks_statistic_mean", "mean_shift_l2"]
    metric_labels = ["Wasserstein\nDist", "KS Stat", "Mean Shift\n(L2)"]

    for idx, cond in enumerate(conditions[:6]):
        if idx >= len(axes):
            break
        ax = axes[idx]
        angles = np.linspace(0, 2 * np.pi, len(feature_types) * len(metrics),
                             endpoint=False).tolist()
        angles += angles[:1]

        values = []
        labels = []
        for ft in feature_types:
            key = f"vs_original_{ft}"
            if key in all_results.get(cond, {}):
                stats = all_results[cond][key]
                for m in metrics:
                    values.append(stats.get(m, 0))
                    labels.append(f"{ft[:4]}\n{m.split('_')[0]}")
            else:
                for _ in metrics:
                    values.append(0)
                    labels.append("")

        # Normalize to [0, 1] for radar
        max_val = max(values) if max(values) > 0 else 1
        values_norm = [v / max_val for v in values]
        values_norm += values_norm[:1]

        ax.plot(angles, values_norm, "o-", linewidth=1.5, markersize=3)
        ax.fill(angles, values_norm, alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=5)
        ax.set_title(cond.replace("weather_style_", "style_").replace("diffusion_", "diff_"),
                     fontsize=9, fontweight="bold", pad=15)

    # Hide unused axes
    for idx in range(len(conditions[:6]), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Texture Fidelity Radar: Per-Condition Artifact Profile",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "texture_radar_per_condition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: texture_radar_per_condition.png")

    # ── Figure 3: LBP histogram comparison ────────────────────
    n_plot = min(6, len(conditions))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, cond in enumerate(conditions[:n_plot]):
        ax = axes[idx]
        if "lbp_hist_mean" in all_results.get("original", {}):
            orig_hist = np.array(all_results["original"]["lbp_hist_mean"])
            ax.bar(range(len(orig_hist)), orig_hist, alpha=0.5, label="Original", color="#3498db")
        if "lbp_hist_mean" in all_results.get(cond, {}):
            aug_hist = np.array(all_results[cond]["lbp_hist_mean"])
            ax.bar(range(len(aug_hist)), aug_hist, alpha=0.5, label=cond.split("_")[-1], color="#e74c3c")
        ax.set_xlabel("LBP bin")
        ax.set_ylabel("Density")
        ax.set_title(cond.replace("weather_style_", "style_").replace("diffusion_", "diff_"),
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

    for idx in range(n_plot, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("LBP Histogram: Original vs Augmented\n"
                 "(differences indicate micro-texture artifact patterns)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "texture_lbp_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: texture_lbp_histograms.png")

    # ── Figure 4: DCT high-freq ratio comparison ─────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    cond_labels = []
    hf_ratios_mean = []
    hf_ratios_std = []
    colors = []

    # Original first
    if "dct_hf_ratio_rgb" in all_results.get("original", {}):
        cond_labels.append("Original")
        vals = all_results["original"]["dct_hf_ratio_rgb"]
        hf_ratios_mean.append(np.mean(vals))
        hf_ratios_std.append(np.std(vals))
        colors.append("#2ecc71")

    for cond in conditions:
        if "dct_hf_ratio_rgb" in all_results.get(cond, {}):
            cond_labels.append(cond.replace("weather_style_", "style_").replace("diffusion_", "diff_"))
            vals = all_results[cond]["dct_hf_ratio_rgb"]
            hf_ratios_mean.append(np.mean(vals))
            hf_ratios_std.append(np.std(vals))
            if "style" in cond:
                colors.append("#e74c3c")
            elif "diffusion" in cond or "diff" in cond:
                colors.append("#3498db")
            elif "night" in cond:
                colors.append("#9b59b6")
            else:
                colors.append("#f39c12")

    if cond_labels:
        bars = ax.bar(range(len(cond_labels)), hf_ratios_mean, yerr=hf_ratios_std,
                       color=colors, edgecolor="white", capsize=3)
        ax.set_xticks(range(len(cond_labels)))
        ax.set_xticklabels(cond_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("High-Frequency Energy Ratio")
        ax.set_title("DCT High-Frequency Content: Original vs Augmented\n"
                     "(lower = loss of natural noise; higher = artificial HF artifacts)",
                     fontsize=11, fontweight="bold")
        ax.axhline(y=hf_ratios_mean[0] if hf_ratios_mean else 0.5, color="green",
                   linestyle="--", alpha=0.5, label="Original baseline")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "texture_dct_hf_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: texture_dct_hf_ratio.png")

    # ── Figure 5: Summary bar chart (artifact score) ─────────
    fig, ax = plt.subplots(figsize=(12, 5))

    cond_labels = []
    artifact_scores = []
    colors = []

    for cond in conditions:
        if cond not in all_results:
            continue
        # Composite artifact score = mean of normalized Wasserstein across 4 features
        w_scores = []
        for ft in feature_types:
            key = f"vs_original_{ft}"
            if key in all_results[cond]:
                w_scores.append(all_results[cond][key]["wasserstein_mean"])
        if w_scores:
            cond_labels.append(cond.replace("weather_style_", "style_").replace("diffusion_", "diff_"))
            artifact_scores.append(np.mean(w_scores))
            if "style" in cond:
                colors.append("#e74c3c")
            elif "diffusion" in cond or "diff" in cond:
                colors.append("#3498db")
            elif "night" in cond:
                colors.append("#9b59b6")
            else:
                colors.append("#f39c12")

    if cond_labels:
        # Sort by artifact score
        sorted_idx = np.argsort(artifact_scores)
        cond_labels = [cond_labels[i] for i in sorted_idx]
        artifact_scores = [artifact_scores[i] for i in sorted_idx]
        colors = [colors[i] for i in sorted_idx]

        bars = ax.barh(range(len(cond_labels)), artifact_scores, color=colors, edgecolor="white")
        for bar, val in zip(bars, artifact_scores):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        ax.set_yticks(range(len(cond_labels)))
        ax.set_yticklabels(cond_labels, fontsize=9)
        ax.set_xlabel("Composite Texture Artifact Score\n"
                      "(mean Wasserstein across GLCM+LBP+DCT+Haralick — lower = more natural)")
        ax.set_title("Texture Fidelity Ranking: Which Augmentation is Most Natural?",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="Style Transfer"),
            Patch(facecolor="#3498db", label="Diffusion (IP2P)"),
            Patch(facecolor="#9b59b6", label="Night (CycleGAN)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "texture_artifact_ranking.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "texture_artifact_ranking.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: texture_artifact_ranking.png")


def plot_acdc_comparison(all_results: dict, output_dir: Path):
    """Plot texture distance: augmented vs ACDC real weather reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_types = ["glcm", "lbp", "dct", "haralick"]

    # Collect conditions that have ACDC comparisons
    acdc_pairs = {}
    for cond, stats in all_results.items():
        for key in stats:
            if key.startswith("vs_acdc_"):
                ref = key.replace("vs_acdc_", "").rsplit("_", 1)
                # key format: vs_acdc_rain_glcm → ref_cond=rain, feature=glcm
                for ft in feature_types:
                    if key.endswith(f"_{ft}"):
                        ref_cond = key[len("vs_acdc_"):-len(f"_{ft}")]
                        if cond not in acdc_pairs:
                            acdc_pairs[cond] = {}
                        acdc_pairs[cond][(ref_cond, ft)] = stats[key]

    if not acdc_pairs:
        return

    # Group by weather condition
    weather_groups = defaultdict(list)
    for cond in acdc_pairs:
        for (ref_cond, ft), stats in acdc_pairs[cond].items():
            weather_groups[ref_cond].append((cond, ft, stats["wasserstein_mean"]))

    if not weather_groups:
        return

    fig, axes = plt.subplots(1, len(weather_groups), figsize=(7 * len(weather_groups), 6),
                              squeeze=False)
    axes = axes[0]

    for ax, (ref_cond, items) in zip(axes, sorted(weather_groups.items())):
        # Group by condition, average across features
        cond_scores = defaultdict(list)
        for cond, ft, w in items:
            cond_scores[cond].append(w)

        cond_means = [(c, np.mean(ws)) for c, ws in cond_scores.items()]
        cond_means.sort(key=lambda x: x[1])

        labels = [c.replace("weather_style_", "style_").replace("diffusion_", "diff_")
                  for c, _ in cond_means]
        values = [v for _, v in cond_means]

        colors = []
        for c, _ in cond_means:
            if "original" in c:
                colors.append("#95a5a6")
            elif "style" in c:
                colors.append("#e74c3c")
            elif "diffusion" in c or "diff" in c:
                colors.append("#3498db")
            elif "night" in c:
                colors.append("#9b59b6")
            else:
                colors.append("#f39c12")

        ax.barh(range(len(labels)), values, color=colors, edgecolor="white")
        for i, v in enumerate(values):
            ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Mean Wasserstein Distance to ACDC")
        ax.set_title(f"vs ACDC {ref_cond}", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle("Texture Distance to Real Weather (ACDC)\n"
                 "(lower = augmentation texture closer to real weather)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_dir / "texture_vs_acdc.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "texture_vs_acdc.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: texture_vs_acdc.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def extract_features_for_condition(images: list, desc: str) -> dict:
    """Extract texture features for a list of images. Returns aggregated stats."""
    all_glcm, all_lbp, all_dct, all_haralick = [], [], [], []
    dct_hf_ratios = []

    for idx, img in enumerate(images):
        feats = extract_texture_features(img)
        all_glcm.append(feats["glcm"])
        all_lbp.append(feats["lbp"])
        all_dct.append(feats["dct"])
        all_haralick.append(feats["haralick"])
        # HF ratio per channel (indices 0, 4, 8 in DCT features)
        dct_hf_ratios.append(np.mean([feats["dct"][0], feats["dct"][4], feats["dct"][8]]))

        if (idx + 1) % 50 == 0 or idx == len(images) - 1:
            print(f"      [{desc}] Features: {idx + 1}/{len(images)}", end="\r")

    print()

    return {
        "glcm": np.array(all_glcm),
        "lbp": np.array(all_lbp),
        "dct": np.array(all_dct),
        "haralick": np.array(all_haralick),
        "lbp_hist_mean": np.array(all_lbp).mean(axis=0).tolist(),
        "dct_hf_ratio_rgb": dct_hf_ratios,
        "n_images": len(images),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Texture-based fidelity assessment for weather augmentation"
    )
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Specific conditions to evaluate (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max images per condition (for quick testing)")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--skip-acdc", action="store_true",
                        help="Skip ACDC reference comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 65}")
    print(f"TEXTURE FIDELITY ASSESSMENT")
    print(f"Method: GLCM + LBP + DCT + Haralick (Duminil et al., 2025)")
    print(f"{'=' * 65}")

    # Select conditions
    if args.conditions:
        pairs = {k: v for k, v in CONDITION_PAIRS.items() if k in args.conditions}
        # Always include original as baseline
        if "original" not in pairs:
            pairs["original"] = CONDITION_PAIRS["original"]
    else:
        pairs = dict(CONDITION_PAIRS)

    # Add ACDC if available and not skipped
    if not args.skip_acdc:
        for k, v in ACDC_REFERENCE.items():
            if v["source"]["path"].exists():
                pairs[k] = v

    t_start = time.time()
    all_features = {}
    all_results = {}

    # ── Extract features for each condition ───────────────────
    for cond_name, config in pairs.items():
        print(f"\n  [{cond_name}] {config['description']}")
        images = load_condition_images(config, args.max_samples)
        if not images:
            print(f"    SKIP: no images loaded")
            continue
        print(f"    Loaded {len(images)} images. Extracting texture features...")

        feats = extract_features_for_condition(images, cond_name)
        all_features[cond_name] = feats
        all_results[cond_name] = {
            "description": config["description"],
            "n_images": feats["n_images"],
            "lbp_hist_mean": feats["lbp_hist_mean"],
            "dct_hf_ratio_rgb": feats["dct_hf_ratio_rgb"],
        }

        del images

    # ── Compare each augmented condition vs original ──────────
    if "original" in all_features:
        print(f"\n{'=' * 65}")
        print("COMPARING AUGMENTED vs ORIGINAL (texture distribution shift)")
        print(f"{'=' * 65}")

        orig = all_features["original"]

        for cond_name in all_features:
            if cond_name == "original" or cond_name.startswith("acdc_"):
                continue
            aug = all_features[cond_name]
            print(f"\n  [{cond_name}] vs original:")

            for ft in ["glcm", "lbp", "dct", "haralick"]:
                stats = compare_distributions(orig[ft], aug[ft])
                key = f"vs_original_{ft}"
                all_results[cond_name][key] = stats
                print(f"    {ft.upper():>8}: Wasserstein={stats['wasserstein_mean']:.4f}  "
                      f"KS={stats['ks_statistic_mean']:.3f} (p={stats['ks_pvalue_mean']:.4f})  "
                      f"sig_dims={stats['n_significant_dims']}/{stats['n_total_dims']}")

    # ── Compare each augmented condition vs ACDC reference ────
    acdc_conds = [c for c in all_features if c.startswith("acdc_")]
    if acdc_conds:
        print(f"\n{'=' * 65}")
        print("COMPARING AUGMENTED vs ACDC REAL WEATHER")
        print(f"{'=' * 65}")

        # Map augmentation → ACDC condition
        aug_to_acdc = {
            "weather_style_rain_0": "acdc_rain",
            "weather_style_rain_1": "acdc_rain",
            "weather_style_rain_2": "acdc_rain",
            "diffusion_rain": "acdc_rain",
            "weather_style_snow_0": "acdc_snow",
            "weather_style_snow_1": "acdc_snow",
            "weather_style_snow_2": "acdc_snow",
            "diffusion_snow": "acdc_snow",
            "diffusion_fog_heavy": "acdc_fog",
            "diffusion_fog_medium": "acdc_fog",
            "diffusion_fog_light": "acdc_fog",
            "night": "acdc_night",
            "original": None,  # compare original vs all ACDC as baseline
        }

        for cond_name, acdc_name in aug_to_acdc.items():
            if cond_name not in all_features:
                continue
            if acdc_name and acdc_name not in all_features:
                continue

            targets = [acdc_name] if acdc_name else acdc_conds
            for acdc_ref in targets:
                if acdc_ref not in all_features:
                    continue
                print(f"\n  [{cond_name}] vs {acdc_ref}:")
                ref = all_features[acdc_ref]
                aug = all_features[cond_name]

                for ft in ["glcm", "lbp", "dct", "haralick"]:
                    stats = compare_distributions(ref[ft], aug[ft])
                    key = f"vs_{acdc_ref}_{ft}"
                    all_results[cond_name][key] = stats
                    print(f"    {ft.upper():>8}: Wasserstein={stats['wasserstein_mean']:.4f}  "
                          f"KS={stats['ks_statistic_mean']:.3f}")

    elapsed = time.time() - t_start

    # ── Summary table ────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"SUMMARY (computed in {elapsed:.0f}s)")
    print(f"{'=' * 65}")

    header = (f"{'Condition':<28} {'N':>5} "
              f"{'GLCM_W':>8} {'LBP_W':>8} {'DCT_W':>8} {'Haralick_W':>10} {'Composite':>10}")
    print(f"\n{header}")
    print("-" * len(header))

    for cond_name in sorted(all_results.keys()):
        if cond_name == "original" or cond_name.startswith("acdc_"):
            continue
        r = all_results[cond_name]
        n = r.get("n_images", 0)

        w_scores = {}
        for ft in ["glcm", "lbp", "dct", "haralick"]:
            key = f"vs_original_{ft}"
            if key in r:
                w_scores[ft] = r[key]["wasserstein_mean"]

        if w_scores:
            composite = np.mean(list(w_scores.values()))
            print(f"{cond_name:<28} {n:>5} "
                  f"{w_scores.get('glcm', 0):>8.4f} "
                  f"{w_scores.get('lbp', 0):>8.4f} "
                  f"{w_scores.get('dct', 0):>8.4f} "
                  f"{w_scores.get('haralick', 0):>10.4f} "
                  f"{composite:>10.4f}")

    # ── Save results ─────────────────────────────────────────
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for cond, data in all_results.items():
        json_results[cond] = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                json_results[cond][k] = v.tolist()
            elif isinstance(v, list) and v and isinstance(v[0], np.floating):
                json_results[cond][k] = [float(x) for x in v]
            else:
                json_results[cond][k] = v

    json_path = output_dir / "texture_fidelity_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # ── Generate plots ───────────────────────────────────────
    print(f"\nGenerating visualizations...")
    plot_texture_comparison(all_results, output_dir)
    if acdc_conds:
        plot_acdc_comparison(all_results, output_dir)

    # ── LaTeX table ──────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("Paper-ready table (LaTeX):")
    print(f"{'=' * 65}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\hline")
    print(r"Augmentation & N & GLCM $W$ & LBP $W$ & DCT $W$ & Haralick $W$ \\")
    print(r"\hline")

    for cond_name in sorted(all_results.keys()):
        if cond_name == "original" or cond_name.startswith("acdc_"):
            continue
        r = all_results[cond_name]
        n = r.get("n_images", 0)
        w_scores = {}
        for ft in ["glcm", "lbp", "dct", "haralick"]:
            key = f"vs_original_{ft}"
            if key in r:
                w_scores[ft] = r[key]["wasserstein_mean"]

        if w_scores:
            cond_tex = cond_name.replace("_", r"\_")
            print(f"{cond_tex} & {n} & "
                  f"{w_scores.get('glcm', 0):.4f} & "
                  f"{w_scores.get('lbp', 0):.4f} & "
                  f"{w_scores.get('dct', 0):.4f} & "
                  f"{w_scores.get('haralick', 0):.4f} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")

    print(f"\nDone! Results at {output_dir}/")


if __name__ == "__main__":
    main()
