#!/usr/bin/env python3
"""
Hybrid Depth-Aware Dust Augmentation — 20-sample test.

Approach:
  1. Depth Anything V2 for depth map
  2. Koschmieder atmospheric scattering with warm brown tones + Perlin noise
  3. Ground-level dust concentration (heavier near bottom)
  4. Dust particle overlay (multi-layer, wind-driven)

Output: 20 images × 3 intensities (light/medium/heavy) + comparisons
"""

import os, sys, gc, random
from pathlib import Path
import numpy as np
from PIL import Image
import io, cv2, torch
import pyarrow as pa
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = Path(os.environ.get('CONSYNTH_DATA_ROOT', Path.home() / 'consynth_data'))
OUT_DIR = SCRIPT_DIR / 'output_samples' / 'dust_v2_20'

# Add fog directory to path for Perlin noise + depth utilities
FOG_DIR = BASE_DIR / 'weather_aug' / 'new_method' / 'fog'
sys.path.insert(0, str(FOG_DIR))
from fog_gen import perlin_noise_fast


# ====== DUST CONFIG ======
DUST_ZONES = {
    'heavy':  {'vis_range': (80, 180),   'particle_density': 1.0, 'ground_weight': 0.85},
    'medium': {'vis_range': (180, 350),  'particle_density': 0.6, 'ground_weight': 0.65},
    'light':  {'vis_range': (350, 600),  'particle_density': 0.3, 'ground_weight': 0.45},
}


# ====== DEPTH ESTIMATION ======
def load_depth_model(device='cuda'):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_id = 'depth-anything/Depth-Anything-V2-Small-hf'
    print(f'Loading Depth Anything V2 (Small)...')
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval().to(device)
    return model, processor


def estimate_depth(pil_img, model, processor, device='cuda'):
    """Depth Anything V2 → pseudo-metric depth [2, 200]m."""
    inputs = processor(images=pil_img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.predicted_depth

    h, w = pil_img.size[1], pil_img.size[0]
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False
    ).squeeze()

    disp = pred.cpu().numpy()
    p2, p98 = np.percentile(disp, [2, 98])
    disp_clipped = np.clip(disp, p2, p98)
    disp_norm = (disp_clipped - p2) / (p98 - p2 + 1e-8)
    depth = 2.0 + (1.0 - disp_norm) * 198.0  # [2, 200]m
    return depth.astype(np.float32)


# ====== ATMOSPHERIC DUST (Koschmieder + warm tones) ======
def generate_dust_atmosphere(image, depth, visibility, ground_weight=0.7, seed=None):
    """
    Depth-aware atmospheric dust using Koschmieder scattering.

    Key differences from fog:
    - Warm brown atmospheric light (not neutral gray)
    - Ground-level concentration gradient
    - Perlin noise with multiple octaves for patchy dust clouds
    - Color temperature shift (warmer = more dust)
    """
    h, w = image.shape[:2]
    rng = np.random.RandomState(seed)

    # Warm brown atmospheric light — varied per image
    dust_r = rng.randint(195, 225)
    dust_g = rng.randint(165, 195)
    dust_b = rng.randint(125, 160)
    atm_light = np.array([[[dust_r, dust_g, dust_b]]], dtype=np.float64)

    # Extinction coefficient from visibility
    beta = 3.912 / visibility

    # Perlin noise for patchy, non-uniform dust clouds
    perlin = perlin_noise_fast(h, w, seed=seed)
    perlin_norm = perlin / 255.0  # [0, 1]
    # Boost contrast: make patches more distinct
    perlin_norm = np.clip(perlin_norm * 1.4 - 0.2, 0.15, 1.0)

    # Ground-level concentration: dust is denser near the bottom of the image
    # Use a smooth gradient that peaks at the bottom
    row_weights = np.linspace(1.0 - ground_weight, 1.0, h, dtype=np.float64)
    # Apply slight curve for more natural falloff
    row_weights = row_weights ** 0.7
    ground_mask = row_weights.reshape(h, 1)

    # Combine: extinction modulated by Perlin noise + ground weighting
    effective_beta = beta * perlin_norm * ground_mask

    # Koschmieder attenuation
    transmission = np.exp(-effective_beta * depth)
    transmission = np.clip(transmission, 0.05, 1.0)

    # Compose
    I_attenuated = image.astype(np.float64) * transmission[:, :, None]
    I_dust = (1.0 - transmission)[:, :, None] * atm_light
    result = I_attenuated + I_dust

    return np.clip(result, 0, 255).astype(np.uint8)


# ====== DUST PARTICLE OVERLAY ======
def add_dust_particles(image, density=1.0, seed=None):
    """
    Multi-layer dust particles floating in the air.

    3 layers by distance:
    - Far: many tiny particles, low alpha
    - Mid: medium particles
    - Near: few large particles, high alpha, with motion blur

    All particles are warm-toned (tan/brown/ochre).
    """
    h, w = image.shape[:2]
    rng = random.Random(seed)
    result = image.astype(np.float64)

    # Wind direction (slight horizontal drift)
    wind_angle = rng.uniform(-15, 15)  # degrees from vertical
    wind_rad = np.radians(wind_angle)

    layers = [
        # Far dust: many tiny dots
        {
            'count': int(rng.randint(2000, 4000) * density),
            'size_range': (1, 2),
            'alpha': 0.15,
            'blur_k': 3,
            'color_range': ((200, 225), (180, 205), (150, 175)),
            'motion_len': 0,
        },
        # Mid dust: medium particles
        {
            'count': int(rng.randint(800, 1500) * density),
            'size_range': (2, 4),
            'alpha': 0.25,
            'blur_k': 3,
            'color_range': ((195, 220), (170, 195), (135, 165)),
            'motion_len': rng.randint(2, 5),
        },
        # Near dust: few large particles, visible
        {
            'count': int(rng.randint(150, 400) * density),
            'size_range': (3, 7),
            'alpha': 0.35,
            'blur_k': 5,
            'color_range': ((190, 215), (160, 190), (120, 155)),
            'motion_len': rng.randint(4, 10),
        },
    ]

    for layer in layers:
        canvas = np.zeros((h, w), dtype=np.float64)

        # Pick a single color for this layer
        cr, cg, cb = layer['color_range']
        color = np.array([
            rng.randint(*cr), rng.randint(*cg), rng.randint(*cb)
        ], dtype=np.float64)

        for _ in range(layer['count']):
            y = rng.randint(0, h - 1)
            x = rng.randint(0, w - 1)
            r = rng.randint(*layer['size_range'])

            if layer['motion_len'] > 0:
                # Motion blur: draw short line instead of circle
                ml = rng.randint(1, layer['motion_len'])
                dx = int(ml * np.sin(wind_rad))
                dy = int(ml * np.cos(wind_rad))
                cv2.line(canvas, (x, y), (x + dx, y + dy),
                         255, thickness=r, lineType=cv2.LINE_AA)
            else:
                cv2.circle(canvas, (x, y), r, 255, -1, cv2.LINE_AA)

        # Blur for softness
        k = layer['blur_k']
        if k % 2 == 0:
            k += 1
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

        # Alpha composite
        a = (canvas / 255.0) * layer['alpha']
        a = np.expand_dims(a, axis=2)
        # Screen blend (bright particles on dark bg look better)
        result = result * (1 - a) + color * a

    return np.clip(result, 0, 255).astype(np.uint8)


# ====== COLOR TEMPERATURE SHIFT ======
def warm_color_shift(image, strength=0.1):
    """Subtle warm color shift — dust scatters blue light, making scene warmer."""
    result = image.astype(np.float64)
    # Boost red slightly, reduce blue
    result[:, :, 0] = np.clip(result[:, :, 0] * (1 + strength * 0.3), 0, 255)  # R up
    result[:, :, 1] = np.clip(result[:, :, 1] * (1 + strength * 0.05), 0, 255)  # G slight
    result[:, :, 2] = np.clip(result[:, :, 2] * (1 - strength * 0.25), 0, 255)  # B down
    # Reduce contrast slightly (dust scatters light → lower contrast)
    mean = result.mean()
    result = mean + (result - mean) * (1 - strength * 0.3)
    return np.clip(result, 0, 255).astype(np.uint8)


# ====== FULL DUST PIPELINE ======
def apply_dust(image, depth, intensity='medium', seed=None):
    """
    Full dust effect: atmosphere + particles + color shift.

    Args:
        image: np.ndarray (h, w, 3) RGB uint8
        depth: np.ndarray (h, w) float32, meters
        intensity: 'light', 'medium', or 'heavy'
        seed: int
    Returns:
        np.ndarray (h, w, 3) RGB uint8
    """
    cfg = DUST_ZONES[intensity]
    rng = np.random.RandomState(seed)
    visibility = rng.uniform(*cfg['vis_range'])

    # Step 1: Depth-aware atmospheric dust
    result = generate_dust_atmosphere(
        image, depth,
        visibility=visibility,
        ground_weight=cfg['ground_weight'],
        seed=seed,
    )

    # Step 2: Dust particles
    result = add_dust_particles(
        result,
        density=cfg['particle_density'],
        seed=seed,
    )

    # Step 3: Warm color shift (stronger for heavier dust)
    color_strength = {'light': 0.06, 'medium': 0.12, 'heavy': 0.20}[intensity]
    result = warm_color_shift(result, strength=color_strength)

    return result


# ====== ARROW HELPERS ======
def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()

def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    return Image.open(io.BytesIO(row['bytes'])).convert('RGB')


# ====== MAIN ======
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load data
    arrow_path = BASE_DIR / 'LouisChen15___construction_site' / 'construction_site-test.arrow'
    table = load_arrow(arrow_path)
    print(f'Loaded {len(table)} images from arrow')

    # Select 20 random images
    random.seed(77)
    indices = random.sample(range(len(table)), 20)
    indices.sort()

    # Load depth model
    depth_model, depth_processor = load_depth_model(device)

    for i, idx in enumerate(indices):
        image_id = table.column('image_id')[idx].as_py()
        pil_img = get_image(table, idx)
        img_np = np.array(pil_img)
        h, w = img_np.shape[:2]

        print(f'[{i+1}/20] id={image_id} ({w}x{h})')

        # Estimate depth once
        depth = estimate_depth(pil_img, depth_model, depth_processor, device)

        # Save original
        pil_img.save(OUT_DIR / f'{i+1:02d}_{image_id}_original.jpg', quality=95)

        # Generate 3 intensities
        for intensity in ['light', 'medium', 'heavy']:
            seed = 42 + idx + hash(intensity) % 1000
            dusty = apply_dust(img_np, depth, intensity=intensity, seed=seed)

            # Save dust image
            Image.fromarray(dusty).save(
                OUT_DIR / f'{i+1:02d}_{image_id}_dust_{intensity}.jpg', quality=95
            )

            # Save side-by-side comparison
            comp = Image.new('RGB', (w * 2, h))
            comp.paste(pil_img, (0, 0))
            comp.paste(Image.fromarray(dusty), (w, 0))
            comp.save(
                OUT_DIR / f'{i+1:02d}_{image_id}_compare_{intensity}.jpg', quality=90
            )

        print(f'  Saved light/medium/heavy')

        del depth, img_np
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f'\nDone! Output: {OUT_DIR}')


if __name__ == '__main__':
    main()
