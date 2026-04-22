"""
Load image pairs for VLM Jury evaluation.

Loads (original, augmented) pairs from Arrow files and ACDC single images
for baseline calibration.
"""

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import numpy as np
import pyarrow as pa
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────
ARROW_DATA = (_DATA_ROOT / "augmentation_data_arrow")
CONSYNTH_DATA = (_REPO_ROOT / "augmentation_data/construction_site")
ACDC_DIR = (_REPO_ROOT / "validation/reference_data/acdc/rgb_anon")

# ── Condition → data source mapping ────────────────────────────
IP2P_DATA = (_DATA_ROOT / "output/construction_site_test")

SYNTHETIC_CONDITIONS = {
    "st_rain_a": {
        "source": ARROW_DATA / "weather_test_style_rain_0.arrow",
        "weather": "rain",
    },
    "st_rain_b": {
        "source": ARROW_DATA / "weather_test_style_rain_1.arrow",
        "weather": "rain",
    },
    "st_rain_c": {
        "source": ARROW_DATA / "weather_test_style_rain_2.arrow",
        "weather": "rain",
    },
    "st_snow_b": {
        "source": ARROW_DATA / "weather_test_style_snow_1.arrow",
        "weather": "snow",
    },
    "ip2p_rain": {
        "source": IP2P_DATA / "diffusion_rain_heavy",
        "weather": "rain",
        "type": "dir",
    },
    "ip2p_rain_heavy": {
        "source": CONSYNTH_DATA / "rain_snow" / "diffusion" / "test" / "rain_heavy",
        "weather": "rain",
        "type": "dir",
    },
    "ip2p_snow_light": {
        "source": IP2P_DATA / "diffusion_snow_light",
        "weather": "snow",
        "type": "dir",
    },
    "ip2p_snow_heavy": {
        "source": IP2P_DATA / "diffusion_snow_heavy",
        "weather": "snow",
        "type": "dir",
    },
    "night": {
        "source": CONSYNTH_DATA / "night" / "test" / "night_constructionsite_test.arrow",
        "weather": "night",
    },
    "fog_heavy": {
        "source": CONSYNTH_DATA / "fog" / "diffusion" / "test" / "heavy",
        "weather": "fog",
        "type": "dir",
    },
}

ACDC_CONDITIONS = ["fog", "rain", "snow", "night"]


@dataclass
class EvalSample:
    condition: str
    sample_idx: int
    image_id: str
    augmented_img: Image.Image
    original_img: Optional[Image.Image]  # None for ACDC baseline
    eval_mode: str  # "paired" or "baseline"


def _load_arrow_images(arrow_path: Path, max_samples: int = None) -> dict:
    """Load images from Arrow file (IPC stream or IPC file) → {image_id: PIL.Image}."""
    try:
        table = pa.ipc.open_file(arrow_path).read_all()
    except pa.ArrowInvalid:
        with open(arrow_path, "rb") as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
    images = {}
    ids = table.column("image_id").to_pylist()
    img_col = table.column("image")

    indices = list(range(len(ids)))
    if max_samples and max_samples < len(indices):
        rng = np.random.default_rng(42)
        indices = rng.choice(indices, size=max_samples, replace=False).tolist()

    for i in indices:
        img_bytes = img_col[i].as_py()
        if isinstance(img_bytes, dict) and "bytes" in img_bytes:
            img_bytes = img_bytes["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images[str(ids[i])] = img

    return images


def _load_dir_images(dir_path: Path, max_samples: int = None) -> dict:
    """Load images from directory of Arrow files or image files."""
    images = {}

    # Try Arrow files first
    arrow_files = sorted(dir_path.glob("*.arrow"))
    if arrow_files:
        for af in arrow_files:
            chunk = _load_arrow_images(af)
            images.update(chunk)
    else:
        # Try image files
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for f in sorted(dir_path.glob(ext)):
                img = Image.open(f).convert("RGB")
                images[f.stem] = img

    if max_samples and len(images) > max_samples:
        rng = np.random.default_rng(42)
        keys = rng.choice(list(images.keys()), size=max_samples, replace=False)
        images = {k: images[k] for k in keys}

    return images


def load_original_images() -> dict:
    """Load original clear-day construction site images."""
    arrow_path = ARROW_DATA / "construction_site_test.arrow"
    print(f"  Loading originals from {arrow_path.name}...")
    return _load_arrow_images(arrow_path)


def load_synthetic_samples(condition: str, originals: dict,
                           n_samples: int = 50) -> list:
    """Load augmented image pairs for one synthetic condition."""
    cfg = SYNTHETIC_CONDITIONS[condition]
    source = cfg["source"]

    print(f"  Loading {condition} from {source}...")
    if cfg.get("type") == "dir":
        augmented = _load_dir_images(source, max_samples=n_samples * 3)
    else:
        augmented = _load_arrow_images(source, max_samples=n_samples * 3)

    # Pair with originals
    samples = []
    rng = np.random.default_rng(42)
    aug_ids = list(augmented.keys())
    rng.shuffle(aug_ids)

    for idx, aug_id in enumerate(aug_ids):
        if len(samples) >= n_samples:
            break

        # Try to find matching original
        orig_id = aug_id  # Most Arrow files use same image_id
        if orig_id not in originals:
            # Try numeric matching (strip prefix)
            for oid in originals:
                if oid in aug_id or aug_id in oid:
                    orig_id = oid
                    break
            else:
                continue

        if orig_id not in originals:
            continue

        samples.append(EvalSample(
            condition=condition,
            sample_idx=len(samples),
            image_id=aug_id,
            augmented_img=augmented[aug_id],
            original_img=originals[orig_id],
            eval_mode="paired",
        ))

    print(f"    Loaded {len(samples)} pairs for {condition}")
    return samples


def load_acdc_baseline(condition: str, n_samples: int = 40) -> list:
    """Load ACDC real weather images for baseline calibration."""
    cond_dir = ACDC_DIR / condition
    if not cond_dir.exists():
        print(f"  ACDC {condition}: directory not found")
        return []

    # Collect all images from train/val/test
    all_images = []
    for split in ("train", "val", "test"):
        split_dir = cond_dir / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            for img_path in sorted(seq_dir.glob("*.png")):
                all_images.append(img_path)

    rng = np.random.default_rng(42)
    if len(all_images) > n_samples:
        indices = rng.choice(len(all_images), size=n_samples, replace=False)
        all_images = [all_images[i] for i in indices]

    samples = []
    for idx, img_path in enumerate(all_images):
        img = Image.open(img_path).convert("RGB")
        samples.append(EvalSample(
            condition=f"acdc_{condition}",
            sample_idx=idx,
            image_id=img_path.stem,
            augmented_img=img,
            original_img=None,
            eval_mode="baseline",
        ))

    print(f"  ACDC {condition}: {len(samples)} images")
    return samples


def concat_pair(original: Image.Image, augmented: Image.Image,
                target_height: int = 448) -> Image.Image:
    """Horizontally concatenate original (left) and augmented (right)."""
    # Resize both to same height
    w1 = int(original.width * target_height / original.height)
    w2 = int(augmented.width * target_height / augmented.height)

    orig_resized = original.resize((w1, target_height), Image.LANCZOS)
    aug_resized = augmented.resize((w2, target_height), Image.LANCZOS)

    # Concatenate
    combined = Image.new("RGB", (w1 + w2, target_height))
    combined.paste(orig_resized, (0, 0))
    combined.paste(aug_resized, (w1, 0))
    return combined


def load_all_samples(n_synthetic: int = 50, n_acdc: int = 40) -> list:
    """Load all evaluation samples."""
    print("=" * 60)
    print("Loading VLM Jury evaluation samples")
    print("=" * 60)

    originals = load_original_images()
    print(f"  Originals: {len(originals)} images")

    all_samples = []

    # Synthetic conditions
    for condition in SYNTHETIC_CONDITIONS:
        try:
            samples = load_synthetic_samples(condition, originals, n_synthetic)
            all_samples.extend(samples)
        except Exception as e:
            print(f"  WARNING: Failed to load {condition}: {e}")

    # ACDC baselines
    for condition in ACDC_CONDITIONS:
        try:
            samples = load_acdc_baseline(condition, n_acdc)
            all_samples.extend(samples)
        except Exception as e:
            print(f"  WARNING: Failed to load ACDC {condition}: {e}")

    print(f"\nTotal: {len(all_samples)} samples "
          f"({sum(1 for s in all_samples if s.eval_mode == 'paired')} paired, "
          f"{sum(1 for s in all_samples if s.eval_mode == 'baseline')} baseline)")
    return all_samples
