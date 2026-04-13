#!/usr/bin/env python3
"""
Relative Mahalanobis Distance for augmentation realism evaluation.

Measures proximity of augmented images to real adverse-condition distributions
in CLIP and DINOv2 embedding spaces, using relative Mahalanobis distance to
cancel shared background features.

Implements the embedding-based distributional analysis from:
  Ruck, Vautravers, Chalkley & Thomas (2026). "Scalable Evaluation of the
  Realism of Synthetic Environmental Augmentations in Images."
  ArXiv: 2603.04325

Key equations:
  d_k(x) = sqrt((x - μ_k)^T Σ_k^{-1} (x - μ_k))    (Eq. 1)
  d_rel  = d_k(x) - d_0(x)                             (Eq. 2)

  where k = target condition, 0 = class-agnostic background.
  Higher -d_rel (closer to 0) = closer to real weather distribution.

Usage:
  python validation/compute_relative_mahalanobis.py
  python validation/compute_relative_mahalanobis.py --max-samples 200
  python validation/compute_relative_mahalanobis.py --embedding clip  # CLIP only
"""

import argparse
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────
DATA_ROOT = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite")
ARROW_DATA = DATA_ROOT / "augmentation_data_arrow"
FOG_DATA = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/augmentation_data/construction_site/fog/diffusion/test")
ACDC_DIR = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/validation/reference_data/acdc/rgb_anon")
OUT_DIR = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/validation/results/relative_mahalanobis")

# ── Condition mapping ─────────────────────────────────────────────
# Maps augmentation → target ACDC condition for d_k
CONDITION_PAIRS = {
    "original": {
        "source": {"type": "arrow", "path": ARROW_DATA / "construction_site_test.arrow"},
        "target_condition": None,  # compute vs ALL conditions as baseline
        "description": "Original clear-day construction images",
    },
    "weather_style_rain_0": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_0.arrow"},
        "target_condition": "rain",
        "description": "Style transfer rain (light)",
    },
    "weather_style_rain_1": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_1.arrow"},
        "target_condition": "rain",
        "description": "Style transfer rain (moderate)",
    },
    "weather_style_rain_2": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_rain_2.arrow"},
        "target_condition": "rain",
        "description": "Style transfer rain (heavy)",
    },
    "weather_style_snow_0": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_0.arrow"},
        "target_condition": "snow",
        "description": "Style transfer snow (light)",
    },
    "weather_style_snow_1": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_1.arrow"},
        "target_condition": "snow",
        "description": "Style transfer snow (moderate)",
    },
    "weather_style_snow_2": {
        "source": {"type": "arrow", "path": ARROW_DATA / "weather_test_style_snow_2.arrow"},
        "target_condition": "snow",
        "description": "Style transfer snow (heavy)",
    },
    "diffusion_rain": {
        "source": {
            "type": "arrow_dir",
            "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_rain_heavy",
        },
        "target_condition": "rain",
        "description": "IP2P diffusion rain",
    },
    "diffusion_snow": {
        "source": {
            "type": "arrow_dir",
            "path": DATA_ROOT / "output" / "construction_site_test" / "diffusion_snow_heavy",
        },
        "target_condition": "snow",
        "description": "IP2P diffusion snow",
    },
    "diffusion_fog_heavy": {
        "source": {"type": "arrow_dir", "path": FOG_DATA / "heavy"},
        "target_condition": "fog",
        "description": "Diffusion fog (heavy)",
    },
    "diffusion_fog_medium": {
        "source": {"type": "arrow_dir", "path": FOG_DATA / "medium"},
        "target_condition": "fog",
        "description": "Diffusion fog (medium)",
    },
    "diffusion_fog_light": {
        "source": {"type": "arrow_dir", "path": FOG_DATA / "light"},
        "target_condition": "fog",
        "description": "Diffusion fog (light)",
    },
    "night": {
        "source": {"type": "arrow", "path": ARROW_DATA / "night.arrow"},
        "target_condition": "night",
        "description": "CycleGAN-Turbo night",
    },
}

ACDC_CONDITIONS = ["fog", "rain", "snow", "night"]


# ══════════════════════════════════════════════════════════════════
# EMBEDDING MODELS
# ══════════════════════════════════════════════════════════════════

class CLIPEmbedder:
    """CLIP ViT-L/14 embeddings (768-dim).

    Paper uses openai/clip-vit-large-patch14 for semantic content aligned
    with natural language. Vision-language training produces tighter
    condition clusters (Ruck et al. 2026, Section 3.4.1).
    """

    def __init__(self, device="cuda"):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model.eval().to(device)
        self.device = device
        self.dim = 768

    @torch.no_grad()
    def extract(self, images: list, batch_size: int = 32) -> np.ndarray:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            feats = self.model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            all_feats.append(feats.cpu().numpy())
            print(f"      CLIP: {min(i + batch_size, len(images))}/{len(images)}", end="\r")
        print()
        return np.concatenate(all_feats, axis=0)


class DINOv2Embedder:
    """DINOv2 ViT-L/14 embeddings (1024-dim).

    Paper uses facebook/dinov2-vitl16-pretrain-lvd1689m for self-supervised
    visual features. Shows greater sensitivity to low-level texture
    differences than CLIP (Ruck et al. 2026, Section 3.4.1).

    Uses transformers AutoModel instead of torch.hub to avoid xformers
    CUDA build dependency issues.
    """

    def __init__(self, device="cuda"):
        from transformers import AutoModel, AutoImageProcessor

        model_name = "facebook/dinov2-large"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(device)
        self.device = device
        self.dim = 1024

    @torch.no_grad()
    def extract(self, images: list, batch_size: int = 32) -> np.ndarray:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            feats = outputs.last_hidden_state[:, 0]  # CLS token
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
            print(f"      DINOv2: {min(i + batch_size, len(images))}/{len(images)}", end="\r")
        print()
        return np.concatenate(all_feats, axis=0)


# ══════════════════════════════════════════════════════════════════
# REFERENCE DISTRIBUTION FITTING
# ══════════════════════════════════════════════════════════════════

def fit_gaussian(features: np.ndarray, reg: float = 1e-5) -> tuple:
    """Fit multivariate Gaussian N(μ, Σ) with regularization.

    Args:
        features: (N, D) array
        reg: regularization added to diagonal for numerical stability

    Returns:
        (mean, precision_matrix) — precision = Σ^{-1} for efficient d_k computation
    """
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    sigma += np.eye(sigma.shape[0]) * reg  # regularize
    precision = np.linalg.inv(sigma)
    return mu, precision


def mahalanobis_distance(x: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance for batch of points.

    d_k(x) = sqrt((x - μ)^T Σ^{-1} (x - μ))    (Eq. 1)

    Args:
        x: (N, D) query points
        mu: (D,) distribution mean
        precision: (D, D) inverse covariance

    Returns:
        (N,) distances
    """
    diff = x - mu  # (N, D)
    left = diff @ precision  # (N, D)
    dist_sq = np.sum(left * diff, axis=1)  # (N,)
    return np.sqrt(np.maximum(dist_sq, 0.0))


def relative_mahalanobis(x: np.ndarray, target_mu: np.ndarray, target_prec: np.ndarray,
                          bg_mu: np.ndarray, bg_prec: np.ndarray) -> np.ndarray:
    """Compute relative Mahalanobis distance.

    d_rel = d_k(x) - d_0(x)    (Eq. 2)

    Subtracts background distance to cancel shared scene features.
    More negative = further from target distribution.
    Values near 0 = close to real adverse-condition imagery.

    Returns:
        (N,) relative distances (report as -d_rel, higher = better)
    """
    d_k = mahalanobis_distance(x, target_mu, target_prec)
    d_0 = mahalanobis_distance(x, bg_mu, bg_prec)
    return d_k - d_0


# ══════════════════════════════════════════════════════════════════
# DATA LOADING (shared with compute_fid_kid.py)
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
            print(f"    Error: {e}")
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
    return []


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def plot_results(all_results: dict, output_dir: Path):
    """Generate visualizations for relative Mahalanobis distance results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    embeddings_used = set()
    for cond, data in all_results.items():
        for key in data:
            if key.startswith("d_rel_"):
                emb = key.split("_")[-1]
                embeddings_used.add(emb)

    embeddings_used = sorted(embeddings_used)
    if not embeddings_used:
        return

    # ── Figure 1: -d_rel bar chart per embedding (like Fig. 3B) ──
    for emb in embeddings_used:
        fig, axes = plt.subplots(1, 4, figsize=(20, 6), squeeze=False)
        axes = axes[0]

        for ax, ref_cond in zip(axes, ACDC_CONDITIONS):
            items = []
            for cond, data in all_results.items():
                key = f"d_rel_{ref_cond}_{emb}"
                if key in data:
                    items.append((cond, data[key]["neg_d_rel_mean"], data[key]["neg_d_rel_std"]))

            # Also add baseline
            baseline_key = f"baseline_{ref_cond}_{emb}"
            for cond, data in all_results.items():
                if baseline_key in data:
                    items.append(("ACDC_baseline", data[baseline_key]["neg_d_rel_mean"],
                                  data[baseline_key]["neg_d_rel_std"]))
                    break

            if not items:
                ax.set_visible(False)
                continue

            items.sort(key=lambda x: x[1], reverse=True)
            labels = [it[0].replace("weather_style_", "style_").replace("diffusion_", "diff_")
                      for it in items]
            means = [it[1] for it in items]
            stds = [it[2] for it in items]

            colors = []
            for name, _, _ in items:
                if "ACDC" in name:
                    colors.append("#95a5a6")
                elif "style" in name:
                    colors.append("#e74c3c")
                elif "diff" in name and "fog" not in name:
                    colors.append("#3498db")
                elif "fog" in name:
                    colors.append("#f39c12")
                elif "night" in name:
                    colors.append("#9b59b6")
                elif "original" in name:
                    colors.append("#2ecc71")
                else:
                    colors.append("#1abc9c")

            bars = ax.barh(range(len(labels)), means, xerr=stds, color=colors,
                           edgecolor="white", capsize=3)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel(f"-d_rel ({emb.upper()}) →\n(higher = closer to real {ref_cond})")
            ax.set_title(f"vs ACDC {ref_cond}", fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)

            # Mark baseline with diamond
            for i, (name, _, _) in enumerate(items):
                if "ACDC" in name:
                    ax.plot(means[i], i, marker="D", color="gray", markersize=10, zorder=10)

        fig.suptitle(f"Relative Mahalanobis Distance ({emb.upper()} embeddings)\n"
                     f"Augmented vs ACDC Real Weather — higher = more realistic\n"
                     f"(grey ◆ = real adverse-condition baseline)",
                     fontsize=13, fontweight="bold", y=1.04)
        plt.tight_layout()
        plt.savefig(output_dir / f"rel_mahalanobis_{emb}.png", dpi=150, bbox_inches="tight")
        plt.savefig(output_dir / f"rel_mahalanobis_{emb}.pdf", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: rel_mahalanobis_{emb}.png")

    # ── Figure 2: Combined summary (all embeddings, target condition only) ──
    fig, ax = plt.subplots(figsize=(14, 6))

    cond_labels = []
    for emb_idx, emb in enumerate(embeddings_used):
        items = []
        for cond, data in all_results.items():
            if cond == "original":
                continue
            target = CONDITION_PAIRS.get(cond, {}).get("target_condition")
            if not target:
                continue
            key = f"d_rel_{target}_{emb}"
            if key in data:
                items.append((cond, data[key]["neg_d_rel_mean"]))

        items.sort(key=lambda x: x[1], reverse=True)

        x = np.arange(len(items))
        width = 0.35
        offset = (emb_idx - (len(embeddings_used) - 1) / 2) * width

        colors = []
        for name, _ in items:
            if "style" in name:
                colors.append("#e74c3c")
            elif "diff" in name and "fog" not in name:
                colors.append("#3498db")
            elif "fog" in name:
                colors.append("#f39c12")
            elif "night" in name:
                colors.append("#9b59b6")
            else:
                colors.append("#1abc9c")

        labels_short = [it[0].replace("weather_style_", "style_").replace("diffusion_", "diff_")
                        for it in items]
        vals = [it[1] for it in items]

        ax.bar(x + offset, vals, width, label=emb.upper(), color=colors, edgecolor="white",
               alpha=0.8 if emb_idx == 0 else 0.6)

        if emb_idx == 0:
            cond_labels = labels_short

    ax.set_xticks(np.arange(len(cond_labels)))
    ax.set_xticklabels(cond_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("-d_rel (higher = closer to target real weather)")
    ax.set_title("Relative Mahalanobis Distance: Augmented → Target ACDC Condition\n"
                 "(each augmentation compared to its matched real weather reference)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor="#e74c3c", label="Style Transfer"),
        Patch(facecolor="#3498db", label="Diffusion (IP2P)"),
        Patch(facecolor="#f39c12", label="Fog"),
        Patch(facecolor="#9b59b6", label="Night"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_dir / "rel_mahalanobis_combined.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "rel_mahalanobis_combined.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: rel_mahalanobis_combined.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Relative Mahalanobis Distance: augmented vs ACDC real weather"
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max images per condition")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--embedding", choices=["clip", "dinov2", "both"], default="both",
                        help="Which embedding model(s) to use")
    parser.add_argument("--holdout", type=int, default=100,
                        help="Held-out ACDC images per condition for baseline (paper: 100)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'=' * 65}")
    print(f"RELATIVE MAHALANOBIS DISTANCE")
    print(f"Method: Ruck et al. (2026), Eq. 1-2")
    print(f"Device: {device}")
    print(f"Embeddings: {args.embedding}")
    print(f"{'=' * 65}")

    # ── Initialize embedding models ──────────────────────────
    embedders = {}
    if args.embedding in ("clip", "both"):
        print(f"\n  Loading CLIP ViT-L/14...")
        embedders["clip"] = CLIPEmbedder(device)
    if args.embedding in ("dinov2", "both"):
        print(f"  Loading DINOv2 ViT-L/14...")
        embedders["dinov2"] = DINOv2Embedder(device)

    # ── Load and embed ACDC reference images ─────────────────
    print(f"\n{'=' * 65}")
    print(f"STEP 1: Build reference distributions from ACDC")
    print(f"{'=' * 65}")

    acdc_features = {}  # {emb_name: {condition: features_array}}
    acdc_holdout = {}   # {emb_name: {condition: features_array}} for baseline

    for emb_name, embedder in embedders.items():
        acdc_features[emb_name] = {}
        acdc_holdout[emb_name] = {}

        for cond in ACDC_CONDITIONS:
            cond_dir = ACDC_DIR / cond
            if not cond_dir.exists():
                print(f"    SKIP: {cond_dir} not found")
                continue

            print(f"\n  [{emb_name}/{cond}] Loading ACDC images...")
            images = load_images_from_dir(cond_dir)
            if not images:
                continue

            print(f"    Loaded {len(images)} images. Extracting features...")
            feats = embedder.extract(images, args.batch_size)

            # Split into reference (for fitting) and holdout (for baseline)
            n_holdout = min(args.holdout, len(feats) // 2)
            rng = np.random.default_rng(42)
            holdout_idx = set(rng.choice(len(feats), size=n_holdout, replace=False))
            ref_idx = [i for i in range(len(feats)) if i not in holdout_idx]

            acdc_features[emb_name][cond] = feats[ref_idx]
            acdc_holdout[emb_name][cond] = feats[list(holdout_idx)]

            print(f"    Reference: {len(ref_idx)} images, Holdout: {n_holdout} images")

            del images
            torch.cuda.empty_cache()

    # ── Fit Gaussian distributions ───────────────────────────
    print(f"\n{'=' * 65}")
    print(f"STEP 2: Fit Gaussian distributions (per-condition + background)")
    print(f"{'=' * 65}")

    gaussians = {}  # {emb_name: {condition: (mu, precision)}}
    bg_gaussians = {}  # {emb_name: (mu_0, precision_0)}

    for emb_name in embedders:
        gaussians[emb_name] = {}

        # Per-condition Gaussians
        for cond in ACDC_CONDITIONS:
            if cond not in acdc_features[emb_name]:
                continue
            feats = acdc_features[emb_name][cond]
            mu, prec = fit_gaussian(feats)
            gaussians[emb_name][cond] = (mu, prec)
            print(f"  [{emb_name}/{cond}] Fitted N(μ, Σ) on {len(feats)} samples, dim={feats.shape[1]}")

        # Background (class-agnostic) Gaussian — fit on ALL conditions pooled
        all_ref = np.concatenate([acdc_features[emb_name][c] for c in ACDC_CONDITIONS
                                  if c in acdc_features[emb_name]], axis=0)
        mu_0, prec_0 = fit_gaussian(all_ref)
        bg_gaussians[emb_name] = (mu_0, prec_0)
        print(f"  [{emb_name}/background] Fitted on {len(all_ref)} pooled samples")

    # ── Compute baselines (held-out real images → near-zero d_rel) ──
    print(f"\n{'=' * 65}")
    print(f"STEP 3: Compute baselines (held-out ACDC → expected near-zero d_rel)")
    print(f"{'=' * 65}")

    baseline_results = {}

    for emb_name in embedders:
        for cond in ACDC_CONDITIONS:
            if cond not in acdc_holdout[emb_name] or cond not in gaussians[emb_name]:
                continue

            holdout_feats = acdc_holdout[emb_name][cond]
            mu_k, prec_k = gaussians[emb_name][cond]
            mu_0, prec_0 = bg_gaussians[emb_name]

            d_rel = relative_mahalanobis(holdout_feats, mu_k, prec_k, mu_0, prec_0)
            neg_d_rel = -d_rel  # paper reports -d_rel (higher = better)

            key = f"baseline_{cond}_{emb_name}"
            baseline_results[key] = {
                "neg_d_rel_mean": float(np.mean(neg_d_rel)),
                "neg_d_rel_std": float(np.std(neg_d_rel)),
                "neg_d_rel_median": float(np.median(neg_d_rel)),
                "n_images": len(holdout_feats),
            }
            print(f"  [{emb_name}/{cond}] Baseline -d_rel = {np.mean(neg_d_rel):.2f} ± {np.std(neg_d_rel):.2f}"
                  f"  (n={len(holdout_feats)}, expected ≈ 0)")

    # ── Compute d_rel for augmented images ───────────────────
    print(f"\n{'=' * 65}")
    print(f"STEP 4: Compute relative Mahalanobis for augmented images")
    print(f"{'=' * 65}")

    all_results = {}

    for cond_name, config in CONDITION_PAIRS.items():
        print(f"\n  [{cond_name}] {config['description']}")
        images = load_condition_images(config, args.max_samples)
        if not images:
            print(f"    SKIP: no images loaded")
            continue
        print(f"    Loaded {len(images)} images")

        cond_results = {
            "description": config["description"],
            "n_images": len(images),
        }

        for emb_name, embedder in embedders.items():
            print(f"    Extracting {emb_name} features...")
            feats = embedder.extract(images, args.batch_size)

            target = config["target_condition"]
            targets = [target] if target else ACDC_CONDITIONS

            for ref_cond in targets:
                if ref_cond not in gaussians[emb_name]:
                    continue

                mu_k, prec_k = gaussians[emb_name][ref_cond]
                mu_0, prec_0 = bg_gaussians[emb_name]

                d_rel = relative_mahalanobis(feats, mu_k, prec_k, mu_0, prec_0)
                neg_d_rel = -d_rel

                key = f"d_rel_{ref_cond}_{emb_name}"
                cond_results[key] = {
                    "neg_d_rel_mean": float(np.mean(neg_d_rel)),
                    "neg_d_rel_std": float(np.std(neg_d_rel)),
                    "neg_d_rel_median": float(np.median(neg_d_rel)),
                    "d_rel_mean": float(np.mean(d_rel)),
                    "n_images": len(feats),
                }
                print(f"      vs ACDC {ref_cond} ({emb_name}): -d_rel = {np.mean(neg_d_rel):.2f} ± {np.std(neg_d_rel):.2f}")

            del feats
            torch.cuda.empty_cache()

        # Store baseline refs in this condition's results for plotting
        for bk, bv in baseline_results.items():
            cond_results[bk] = bv

        all_results[cond_name] = cond_results
        del images
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"SUMMARY")
    print(f"{'=' * 65}")

    for emb_name in embedders:
        print(f"\n  === {emb_name.upper()} ===")
        header = f"{'Condition':<28} {'Target':>6} {'-d_rel mean':>12} {'±std':>8} {'N':>6}"
        print(f"  {header}")
        print(f"  {'-' * len(header)}")

        # Baselines first
        for cond in ACDC_CONDITIONS:
            bk = f"baseline_{cond}_{emb_name}"
            if bk in baseline_results:
                b = baseline_results[bk]
                print(f"  {'ACDC_baseline':<28} {cond:>6} {b['neg_d_rel_mean']:>12.2f} {b['neg_d_rel_std']:>8.2f} {b['n_images']:>6}")
        print(f"  {'---'}")

        for cond_name in sorted(all_results.keys()):
            if cond_name == "original":
                continue
            data = all_results[cond_name]
            target = CONDITION_PAIRS.get(cond_name, {}).get("target_condition")
            if not target:
                continue
            key = f"d_rel_{target}_{emb_name}"
            if key not in data:
                continue
            d = data[key]
            short = cond_name.replace("weather_style_", "style_")
            print(f"  {short:<28} {target:>6} {d['neg_d_rel_mean']:>12.2f} {d['neg_d_rel_std']:>8.2f} {d['n_images']:>6}")

    # ── Save JSON ────────────────────────────────────────────
    json_path = output_dir / "relative_mahalanobis_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Plots ────────────────────────────────────────────────
    print(f"\nGenerating visualizations...")
    plot_results(all_results, output_dir)

    # ── LaTeX ────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("Paper-ready table (LaTeX):")
    print(f"{'=' * 65}")

    for emb_name in embedders:
        print(f"\n% {emb_name.upper()}")
        print(r"\begin{tabular}{llcc}")
        print(r"\hline")
        print(r"Augmentation & Target & $-d_\text{rel}$ $\uparrow$ & N \\")
        print(r"\hline")

        for cond in ACDC_CONDITIONS:
            bk = f"baseline_{cond}_{emb_name}"
            if bk in baseline_results:
                b = baseline_results[bk]
                print(f"ACDC baseline & {cond} & {b['neg_d_rel_mean']:.2f}$\\pm${b['neg_d_rel_std']:.2f} & {b['n_images']} \\\\")

        print(r"\hline")

        for cond_name in sorted(all_results.keys()):
            if cond_name == "original":
                continue
            target = CONDITION_PAIRS.get(cond_name, {}).get("target_condition")
            if not target:
                continue
            key = f"d_rel_{target}_{emb_name}"
            if key not in all_results[cond_name]:
                continue
            d = all_results[cond_name][key]
            cond_tex = cond_name.replace("_", r"\_")
            print(f"{cond_tex} & {target} & {d['neg_d_rel_mean']:.2f}$\\pm${d['neg_d_rel_std']:.2f} & {d['n_images']} \\\\")

        print(r"\hline")
        print(r"\end{tabular}")

    print(f"\nDone! Results at {output_dir}/")


if __name__ == "__main__":
    main()
