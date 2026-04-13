#!/usr/bin/env python3
"""
Preview: IP2P + Physics on Night images → Rain Night & Snow Night grid.

Pipeline:
  Night image (from night.arrow)
    → IP2P diffusion (rain or snow prompt)
    → Physics particle overlay
    → Comparison grid

Produces a 5-row grid:  Original | Night | IP2P Rain+Night | IP2P Snow+Night
"""

import sys
import gc
import random
from pathlib import Path
import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import torch

SCRIPT_DIR = Path(__file__).parent

# Reuse physics particle generators
sys.path.insert(0, str(SCRIPT_DIR / 'weather' / 'rain_snow' / 'diffusion'))
from physics import add_natural_rain, add_natural_snow


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(io.BytesIO(row['bytes'])).convert('RGB')
    elif isinstance(row, bytes):
        return Image.open(io.BytesIO(row)).convert('RGB')
    return None


def create_grid(samples, cell_w=450, cell_h=320, padding=4, label_h=30):
    """
    samples: list of dicts with keys:
      'original', 'night', 'rain_night', 'snow_night', 'image_id'
    """
    n_rows = len(samples)
    n_cols = 4

    grid_w = n_cols * cell_w + (n_cols + 1) * padding
    grid_h = n_rows * (cell_h + label_h) + (n_rows + 1) * padding + label_h

    grid = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font

    headers = ['Original', 'Night (CycleGAN)', 'IP2P Rain + Night', 'IP2P Snow + Night']
    header_colors = [(200, 200, 200), (100, 150, 255), (255, 180, 80), (180, 220, 255)]
    for col, (header, hcolor) in enumerate(zip(headers, header_colors)):
        x = padding + col * (cell_w + padding)
        draw.text((x + cell_w // 2 - len(header) * 4, 8), header, fill=hcolor, font=font)

    keys = ['original', 'night', 'rain_night', 'snow_night']

    for row_idx, sample in enumerate(samples):
        y_offset = label_h + padding + row_idx * (cell_h + label_h + padding)
        img_id = sample.get('image_id', f'#{row_idx}')
        draw.text((8, y_offset - 2), f"ID: {img_id}", fill=(150, 150, 150), font=font_small)

        for col_idx, key in enumerate(keys):
            x = padding + col_idx * (cell_w + padding)
            y = y_offset + label_h - 10
            img = sample[key]
            if img is not None:
                img_resized = img.resize((cell_w, cell_h), Image.LANCZOS)
                grid.paste(img_resized, (x, y))

    return grid


def main():
    BASE_DIR = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data_arrow')
    OUTPUT_DIR = SCRIPT_DIR / 'preview_grids'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = 5

    print("=" * 60)
    print("IP2P + Physics: Rain Night & Snow Night Preview")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading arrow files...")
    orig_table = load_arrow(BASE_DIR / 'construction_site_test.arrow')
    night_table = load_arrow(BASE_DIR / 'night.arrow')
    print(f"  Original: {len(orig_table)}, Night: {len(night_table)}")

    night_ids = {night_table.column('image_id')[i].as_py(): i for i in range(len(night_table))}

    # Sample images
    random.seed(42)
    indices = random.sample(range(len(orig_table)), n_samples)
    indices.sort()

    # Load IP2P model
    print("\n[2/4] Loading InstructPix2Pix model...")
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    print("  Model loaded on CUDA")

    # IP2P prompts — tuned for night scenes
    rain_prompt = "a rainy night with rain falling, wet reflections on surfaces, dark overcast sky"
    snow_prompt = "a cold winter night with snow falling, frost on surfaces, snow on the ground"

    # Process
    print(f"\n[3/4] Processing {n_samples} images (IP2P rain + snow on night)...")
    samples = []

    for i, idx in enumerate(indices):
        image_id = orig_table.column('image_id')[idx].as_py()
        print(f"\n  [{i+1}/{n_samples}] Image {image_id}")

        orig_img = get_image(orig_table, idx)

        night_idx = night_ids.get(image_id)
        if night_idx is None:
            print(f"    Warning: no night version, skipping")
            continue
        night_img = get_image(night_table, night_idx)

        h, w = night_img.height, night_img.width

        # Resize for IP2P (max 768, divisible by 8)
        max_dim = 768
        scale = min(max_dim / max(h, w), 1.0)
        nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
        night_resized = night_img.resize((nw, nh), Image.LANCZOS)
        print(f"    Size: {w}x{h} → IP2P input: {nw}x{nh}")

        # === Rain Night: IP2P + physics ===
        g = torch.Generator("cuda").manual_seed(42 + idx)
        rain_result = pipe(
            rain_prompt, image=night_resized,
            num_inference_steps=30,
            image_guidance_scale=1.5,
            guidance_scale=10.0,
            generator=g
        ).images[0]
        rain_np = np.array(rain_result.resize((w, h), Image.LANCZOS))
        random.seed(42 + idx)
        rain_np = add_natural_rain(rain_np)
        rain_night_img = Image.fromarray(rain_np)
        print(f"    Rain night done")

        # === Snow Night: IP2P + physics ===
        g = torch.Generator("cuda").manual_seed(99 + idx)
        snow_result = pipe(
            snow_prompt, image=night_resized,
            num_inference_steps=30,
            image_guidance_scale=1.5,
            guidance_scale=8.0,
            generator=g
        ).images[0]
        snow_np = np.array(snow_result.resize((w, h), Image.LANCZOS))
        random.seed(99 + idx)
        snow_np = add_natural_snow(snow_np)
        snow_night_img = Image.fromarray(snow_np)
        print(f"    Snow night done")

        samples.append({
            'image_id': image_id,
            'original': orig_img,
            'night': night_img,
            'rain_night': rain_night_img,
            'snow_night': snow_night_img,
        })

        # Save individual images
        rain_night_img.save(OUTPUT_DIR / f'{image_id}_ip2p_rain_night.jpg', 'JPEG', quality=95)
        snow_night_img.save(OUTPUT_DIR / f'{image_id}_ip2p_snow_night.jpg', 'JPEG', quality=95)

        del rain_result, snow_result, rain_np, snow_np
        gc.collect()
        torch.cuda.empty_cache()

    # Create grid
    print(f"\n[4/4] Creating comparison grid ({len(samples)} rows x 4 cols)...")
    grid = create_grid(samples, cell_w=450, cell_h=320)
    grid_path = OUTPUT_DIR / 'preview_ip2p_night_weather_grid.jpg'
    grid.save(grid_path, 'JPEG', quality=95)
    print(f"  Grid saved: {grid_path}")
    print(f"  Grid size: {grid.size[0]}x{grid.size[1]}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
