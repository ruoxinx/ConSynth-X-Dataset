#!/usr/bin/env python3
"""
Fog Effect Pipeline - Depth-aware fog synthesis using Koschmieder model.

Reads images from Arrow file, estimates depth with MiDaS, applies
physics-based fog (atmospheric scattering), and saves output samples.

Usage:
    python fog_pipeline.py --arrow /path/to/file.arrow --num-samples 5
    python fog_pipeline.py --arrow /path/to/file.arrow --intensity heavy --use-fake-depth
"""

import sys
import os
import io
import gc
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pyarrow.ipc as ipc

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fog_gen import generate_fog


# ====== HAZE CONFIG ======
# Unified haze range with 3 labeled zones for validation.
# Uniform random sampling across full range [VIS_MIN, VIS_MAX].
# Label assigned based on which zone the sampled visibility falls into.
#
#   heavy (300-500m)  |  medium (500-750m)  |  light (750-1000m)
#   dense fog            moderate haze          light haze
#
HAZE_VIS_MIN = 300    # lower bound (densest, same as haze v1)
HAZE_VIS_MAX = 1000   # upper bound (lightest)
HAZE_FOG_COLOR_RANGE = (210, 245)

HAZE_ZONES = [
    # (vis_lower, vis_upper, label)
    (300,  500,  'heavy'),
    (500,  750,  'medium'),
    (750,  1000, 'light'),
]


def sample_haze():
    """Sample visibility uniformly and return (visibility, fog_color, label)."""
    visibility = np.random.uniform(HAZE_VIS_MIN, HAZE_VIS_MAX)
    fog_color = np.random.randint(*HAZE_FOG_COLOR_RANGE)

    label = 'medium'  # fallback
    for lo, hi, zone_label in HAZE_ZONES:
        if lo <= visibility < hi:
            label = zone_label
            break

    return visibility, fog_color, label


# ====== DEPTH ESTIMATION ======
def load_midas_model(model_name='DPT_Large', device='cpu'):
    dev = torch.device(device)
    print(f'  Loading MiDaS model ({model_name})...')
    model = torch.hub.load('intel-isl/MiDaS', model_name, trust_repo=True)
    model.eval().to(dev)
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    if model_name in ['DPT_Large', 'DPT_Hybrid']:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return model, transform, dev


def estimate_depth(img_array, midas_model, midas_transform, device):
    """Estimate depth from numpy RGB array using pre-loaded MiDaS."""
    input_batch = midas_transform(img_array).to(device)
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_array.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
    disp = prediction.detach().cpu().numpy()
    disp[disp < 0] = 0
    disp = disp + 1e-3
    baseline, focal = 0.54, 721.09
    depth = baseline * focal / disp
    return depth.astype(np.float32)


def fake_depth_map(h, w):
    """Gradient depth: near at top, far at bottom."""
    y = np.linspace(1.0, 80.0, h, dtype=np.float32)[:, None]
    return np.repeat(y, w, axis=1)


# ====== DEPTH ANYTHING V2 ======
def load_depth_anything_model(model_size='Small', device='cpu'):
    """Load Depth Anything V2 via HuggingFace transformers."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_id = f'depth-anything/Depth-Anything-V2-{model_size}-hf'
    print(f'  Loading Depth Anything V2 ({model_size})...')
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval().to(device)
    return model, processor, device


def estimate_depth_anything(pil_img, model, processor, device):
    """Estimate depth from PIL image using Depth Anything V2.

    DA V2 outputs relative disparity (not metric depth). We normalize it
    to a pseudo-metric depth range [near, far] that matches what the
    Koschmieder fog model expects (~1-200m for construction scenes).
    """
    inputs = processor(images=pil_img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    h, w = pil_img.size[1], pil_img.size[0]
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode='bicubic',
        align_corners=False,
    ).squeeze()

    disp = prediction.detach().cpu().numpy()

    # DA V2 outputs disparity-like values: higher = closer.
    # Normalize to pseudo-metric depth: closer → small depth, farther → large depth.
    # Use percentile clipping for robustness against outliers.
    p2, p98 = np.percentile(disp, [2, 98])
    disp_clipped = np.clip(disp, p2, p98)

    # Invert: high disparity (close) → low depth, low disparity (far) → high depth
    # Map to [NEAR_M, FAR_M] range that works well with fog visibility 30-250m
    NEAR_M, FAR_M = 2.0, 200.0
    disp_norm = (disp_clipped - p2) / (p98 - p2 + 1e-8)  # 0=far, 1=close
    depth = NEAR_M + (1.0 - disp_norm) * (FAR_M - NEAR_M)  # invert: close→near, far→far

    return depth.astype(np.float32)


# ====== ARROW FILE READER ======
def read_arrow_images(arrow_path, indices):
    """Read specific images from arrow file by index."""
    with open(arrow_path, 'rb') as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()

    images = []
    for idx in indices:
        row = table['image'][idx].as_py()
        img = Image.open(io.BytesIO(row['bytes'])).convert('RGB')
        image_id = table['image_id'][idx].as_py()
        images.append((image_id, img))

    return images, table.num_rows


# ====== MAIN PROCESSING ======
def apply_fog(img_array, depth_map):
    """Apply Koschmieder fog with uniformly sampled haze.
    Returns (foggy_image, visibility, fog_color, label).
    """
    visibility, fog_color, label = sample_haze()
    foggy = generate_fog(img_array, depth_map, visibility=visibility, fog_color=fog_color)
    return foggy, visibility, fog_color, label


def main():
    parser = argparse.ArgumentParser(description='Fog Effect Pipeline')
    parser.add_argument('--arrow', required=True, help='Path to Arrow file')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    parser.add_argument('--num-samples', '-n', type=int, default=5, help='Number of samples')
    parser.add_argument('--depth-model', default='depth-anything-small',
                        choices=['midas', 'depth-anything-small', 'depth-anything-base',
                                 'depth-anything-large', 'fake'],
                        help='Depth estimation model (default: depth-anything-small)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start index in arrow file')
    parser.add_argument('--save-comparisons', action='store_true',
                        help='Save side-by-side comparison images')
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        DEVICE = args.device

    output_dir = Path(args.output) if args.output else PROJECT_ROOT / 'output_samples'
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_model_name = args.depth_model

    print('=' * 60)
    print('FOG EFFECT PIPELINE (Koschmieder atmospheric scattering)')
    print('=' * 60)
    print(f'Device: {DEVICE}')
    print(f'Depth: {depth_model_name}')
    print(f'Haze range: {HAZE_VIS_MIN}-{HAZE_VIS_MAX}m (uniform)')
    print(f'Zones: {", ".join(f"{z[2]}={z[0]}-{z[1]}m" for z in HAZE_ZONES)}')
    print()

    # Read images
    indices = list(range(args.start_idx, args.start_idx + args.num_samples))
    print(f'Reading {args.num_samples} images from arrow file (indices {indices})...')
    images, total = read_arrow_images(args.arrow, indices)
    print(f'  Total images in file: {total}')

    # Load depth model once
    depth_estimator = None
    if depth_model_name == 'midas':
        midas_model, midas_transform, midas_device = load_midas_model('DPT_Large', DEVICE)
        depth_estimator = ('midas', midas_model, midas_transform, midas_device)
    elif depth_model_name.startswith('depth-anything'):
        size_map = {
            'depth-anything-small': 'Small',
            'depth-anything-base': 'Base',
            'depth-anything-large': 'Large',
        }
        size = size_map[depth_model_name]
        da_model, da_processor, da_device = load_depth_anything_model(size, DEVICE)
        depth_estimator = ('depth-anything', da_model, da_processor, da_device)

    # Metadata log
    metadata = []

    # Process each image
    for i, (image_id, pil_img) in enumerate(images):
        print(f'\n[{i+1}/{len(images)}] Processing {image_id}...')
        img_array = np.array(pil_img)
        h, w = img_array.shape[:2]

        # Depth estimation
        if depth_model_name == 'fake':
            depth_map = fake_depth_map(h, w)
        elif depth_estimator[0] == 'midas':
            depth_map = estimate_depth(img_array, depth_estimator[1], depth_estimator[2], depth_estimator[3])
        elif depth_estimator[0] == 'depth-anything':
            depth_map = estimate_depth_anything(pil_img, depth_estimator[1], depth_estimator[2], depth_estimator[3])

        # Apply fog (uniform random haze)
        foggy, visibility, fog_color, label = apply_fog(img_array, depth_map)
        result = Image.fromarray(foggy)

        # Filename includes label
        out_name = f'{image_id}_fog_{label}.jpg'
        out_path = output_dir / out_name
        result.save(out_path, 'JPEG', quality=95)

        # Optional comparison
        if args.save_comparisons:
            comparison = Image.new('RGB', (w * 2, h))
            comparison.paste(pil_img, (0, 0))
            comparison.paste(result, (w, 0))
            comp_path = output_dir / f'{image_id}_comparison_{label}.jpg'
            comparison.save(comp_path, 'JPEG', quality=90)

        metadata.append({
            'image_id': image_id,
            'filename': out_name,
            'label': label,
            'visibility_m': round(visibility, 1),
            'fog_color': fog_color,
            'width': w,
            'height': h,
        })

        print(f'  {out_name} | {label} | vis={visibility:.0f}m | color={fog_color}')

        del depth_map, foggy, img_array
        gc.collect()

    # Save metadata CSV
    import csv
    csv_path = output_dir / 'metadata.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    # Summary
    from collections import Counter
    counts = Counter(m['label'] for m in metadata)
    print(f'\nDone! {len(images)} fog samples saved to {output_dir}')
    print(f'Label distribution: {dict(counts)}')
    print(f'Metadata: {csv_path}')


if __name__ == '__main__':
    main()
