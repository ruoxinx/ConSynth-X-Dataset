#!/usr/bin/env python3
"""
Preview: Compare ordering of Night & Weather augmentation passes.

Two orderings under test:
  Order A (current night_weather pipeline):
      Original -> CycleGAN-Turbo Night -> IP2P weather (light) -> physics
  Order B (proposed):
      Original -> IP2P weather (light, day prompt) -> CycleGAN-Turbo Night -> physics

Output: two 5-column comparison grids (rain & snow), 5 rows each:
  | Original | Night only | IP2P weather only | Order A | Order B |

IP2P "light" config (matches conditions/weather.yaml `rain` (intensity: light) / `snow_light`):
  rain : prompt='a rainy day, wet surfaces, light rain, grey overcast sky',
         guidance_scale=8.0, image_guidance_scale=1.5, physics='light'
  snow : prompt='a cold winter day with snow, frost on surfaces, grey sky, snow on the ground',
         guidance_scale=8.0, image_guidance_scale=1.5, physics='light'
"""

import os
import sys
import gc
import io
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).parent

# --- physics overlays ---
sys.path.insert(0, str(SCRIPT_DIR / 'weather' / 'rain_snow' / 'diffusion'))
from physics import add_natural_rain, add_natural_snow

# --- CycleGAN-Turbo day_to_night (img2img-turbo) ---
IMG2IMG_SRC = SCRIPT_DIR / 'day2night' / 'img2img-turbo' / 'src'
sys.path.insert(0, str(IMG2IMG_SRC))
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from torchvision import transforms

# ---------- IP2P configs — match production rain.arrow / snow.arrow ----------
# Source of truth: generation/weather/rain_snow/diffusion/batch_worker.py defaults
# and jobs/regen_rain_v5.sh (the run that produced the current rain.arrow).
IP2P_LIGHT = {
    'rain': {
        'prompt': 'a rainy day with dark overcast sky, rain falling, grey clouds',
        'guidance_scale': 10.0,
        'image_guidance_scale': 1.5,
        'physics_intensity': 'heavy',  # batch_worker uses default heavy physics
    },
    'snow': {
        'prompt': 'a cold winter day with snow, frost on surfaces, grey sky, snow on the ground',
        'guidance_scale': 8.0,
        'image_guidance_scale': 1.5,
        'physics_intensity': 'heavy',
    },
}

# Night-tuned prompts used by current Order A worker (kept for parity)
IP2P_NIGHT_PROMPT = {
    'rain': 'a rainy night with rain falling, wet reflections on surfaces, dark overcast sky',
    'snow': 'a cold winter night with snow falling, frost on surfaces, snow on the ground',
}

N_SAMPLES = 5
SEED = 42


# ---------- arrow helpers ----------
def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(io.BytesIO(row['bytes'])).convert('RGB')
    return Image.open(io.BytesIO(row)).convert('RGB')


# ---------- model wrappers ----------
def load_ip2p():
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', torch_dtype=torch.float16, safety_checker=None)
    pipe.to('cuda')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_cyclegan():
    model = CycleGAN_Turbo(pretrained_name='day_to_night')
    model.eval()
    try:
        model.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    model.half()
    return model


def run_ip2p(pipe, image, prompt, guidance_scale, image_guidance_scale, seed):
    w, h = image.size
    max_dim = 768
    scale = min(max_dim / max(h, w), 1.0)
    nw, nh = (int(w * scale) // 8 * 8, int(h * scale) // 8 * 8)
    img_resized = image.resize((nw, nh), Image.LANCZOS)
    g = torch.Generator('cuda').manual_seed(seed)
    out = pipe(
        prompt, image=img_resized,
        num_inference_steps=30,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=g,
    ).images[0]
    return out.resize((w, h), Image.LANCZOS)


def run_cyclegan_night(model, image, T_val):
    w, h = image.size
    with torch.no_grad():
        x = T_val(image)
        x = transforms.ToTensor()(x)
        x = transforms.Normalize([0.5], [0.5])(x).unsqueeze(0).cuda().half()
        out = model(x, direction=None, caption=None)
    out_pil = transforms.ToPILImage()(out[0].cpu().float() * 0.5 + 0.5)
    return out_pil.resize((w, h), Image.LANCZOS)


def apply_physics(img_pil, weather, intensity, seed):
    arr = np.array(img_pil)
    random.seed(seed)
    if weather == 'rain':
        arr = add_natural_rain(arr, intensity=intensity)
    else:
        arr = add_natural_snow(arr)  # snow physics is single-tier in physics.py
    return Image.fromarray(arr)


# ---------- grid layout ----------
def build_grid(samples, weather, cell_w=420, cell_h=300, padding=4, label_h=30):
    n_rows = len(samples)
    n_cols = 5

    grid_w = n_cols * cell_w + (n_cols + 1) * padding
    grid_h = n_rows * (cell_h + label_h) + (n_rows + 1) * padding + label_h

    grid = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Bold.ttf', 14)
        font_small = ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Regular.ttf', 11)
    except Exception:
        try:
            font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 14)
            font_small = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 11)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

    headers = [
        'Original',
        'Night only',
        f'IP2P {weather} day only',
        f'A: Night -> IP2P {weather}',
        f'B: IP2P {weather} -> Night',
    ]
    colors = [
        (200, 200, 200), (100, 150, 255), (255, 200, 120),
        (180, 230, 180), (240, 180, 240),
    ]
    for col, (h_text, hcolor) in enumerate(zip(headers, colors)):
        x = padding + col * (cell_w + padding)
        draw.text((x + 10, 8), h_text, fill=hcolor, font=font)

    keys = ['original', 'night', 'ip2p_day', 'order_a', 'order_b']
    for r, sample in enumerate(samples):
        y0 = label_h + padding + r * (cell_h + label_h + padding)
        draw.text((8, y0 - 2), f"ID: {sample['image_id']}", fill=(150, 150, 150), font=font_small)
        for c, key in enumerate(keys):
            x = padding + c * (cell_w + padding)
            y = y0 + label_h - 10
            img = sample[key]
            if img is not None:
                grid.paste(img.resize((cell_w, cell_h), Image.LANCZOS), (x, y))
    return grid


# ---------- main ----------
def main():
    ORIG_PATH = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data_arrow/construction_site_test.arrow')
    NIGHT_PATH = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X/augmentation_data/construction_site/night/test/night_constructionsite_test.arrow')
    OUT_DIR = SCRIPT_DIR / 'preview_grids'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Order comparison: Night<->Weather (rain & snow, IP2P light)')
    print('=' * 60)

    print('\n[1/5] Loading arrows...')
    orig_table = load_arrow(ORIG_PATH)
    night_table = load_arrow(NIGHT_PATH)
    print(f'  Original: {len(orig_table)} | Night: {len(night_table)}')

    night_ids = {night_table.column('image_id')[i].as_py(): i
                 for i in range(len(night_table))}

    # Pick samples that have a precomputed night version available
    rng = random.Random(SEED)
    candidates = list(range(len(orig_table)))
    rng.shuffle(candidates)
    indices = []
    for idx in candidates:
        img_id = orig_table.column('image_id')[idx].as_py()
        if img_id in night_ids:
            indices.append(idx)
        if len(indices) == N_SAMPLES:
            break
    indices.sort()
    print(f'  Selected indices: {indices}')

    print('\n[2/5] Loading IP2P...')
    pipe = load_ip2p()

    print('\n[3/5] Loading CycleGAN-Turbo (day_to_night)...')
    cyclegan = load_cyclegan()
    T_val = build_transform('resize_512x512')

    grids_made = {}
    for weather in ['rain', 'snow']:
        cfg = IP2P_LIGHT[weather]
        print(f'\n[4/5] Processing {weather} (IP2P light: g={cfg["guidance_scale"]}, '
              f'igs={cfg["image_guidance_scale"]})')
        samples = []
        for i, idx in enumerate(indices):
            image_id = orig_table.column('image_id')[idx].as_py()
            print(f'  [{i+1}/{N_SAMPLES}] {image_id}')
            orig_img = get_image(orig_table, idx)
            night_img = get_image(night_table, night_ids[image_id])

            seed_off = SEED + idx + (1000 if weather == 'snow' else 0)

            # IP2P day-only (light prompt) on the original
            ip2p_day_only = run_ip2p(
                pipe, orig_img, cfg['prompt'],
                cfg['guidance_scale'], cfg['image_guidance_scale'], seed_off)
            ip2p_day_only_phys = apply_physics(
                ip2p_day_only, weather, cfg['physics_intensity'], seed_off)

            # Order A: Night -> IP2P (use night-tuned prompt to match current worker)
            order_a_ip2p = run_ip2p(
                pipe, night_img, IP2P_NIGHT_PROMPT[weather],
                cfg['guidance_scale'], cfg['image_guidance_scale'], seed_off)
            order_a = apply_physics(
                order_a_ip2p, weather, cfg['physics_intensity'], seed_off)

            # Order B: IP2P (day light prompt) -> CycleGAN Night -> physics
            ip2p_day = run_ip2p(
                pipe, orig_img, cfg['prompt'],
                cfg['guidance_scale'], cfg['image_guidance_scale'], seed_off + 7)
            order_b_night = run_cyclegan_night(cyclegan, ip2p_day, T_val)
            order_b = apply_physics(
                order_b_night, weather, cfg['physics_intensity'], seed_off)

            samples.append({
                'image_id': image_id,
                'original': orig_img,
                'night': night_img,
                'ip2p_day': ip2p_day_only_phys,
                'order_a': order_a,
                'order_b': order_b,
            })

            # Per-image dump for side-by-side inspection
            for tag, img in [
                (f'{weather}_ip2p_day_only', ip2p_day_only_phys),
                (f'{weather}_order_a', order_a),
                (f'{weather}_order_b', order_b),
            ]:
                img.save(OUT_DIR / f'{image_id}_{tag}.jpg', 'JPEG', quality=92)

            del ip2p_day_only, ip2p_day_only_phys, order_a_ip2p, order_a
            del ip2p_day, order_b_night, order_b
            gc.collect()
            torch.cuda.empty_cache()

        grid = build_grid(samples, weather)
        grid_path = OUT_DIR / f'preview_order_comparison_{weather}.jpg'
        grid.save(grid_path, 'JPEG', quality=95)
        grids_made[weather] = grid_path
        print(f'  Grid saved: {grid_path} ({grid.size[0]}x{grid.size[1]})')

    print('\n[5/5] Done. Grids:')
    for w, p in grids_made.items():
        print(f'  {w}: {p}')


if __name__ == '__main__':
    main()
