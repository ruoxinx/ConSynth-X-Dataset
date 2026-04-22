#!/usr/bin/env python3
"""
Test IP2P rain at 2 intensity levels (light, heavy) — matches snow's 2-level scheme.

light : g=8.0,  igs=1.5, mild prompt  + physics intensity='light'
heavy : g=10.0, igs=1.5, strong prompt + physics intensity='heavy' (production default)

Generates 10 samples per config + grid [original | light | heavy] saved as grid.jpg.
"""

import os
import sys
import io
import random
from pathlib import Path

_REPO = Path(os.environ.get("CONSYNTH_REPO_ROOT", Path(__file__).resolve().parents[4]))

import numpy as np
import pyarrow as pa
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from physics import add_natural_rain

OUTPUT_DIR = _REPO / "validation" / "results" / "rain_intensity_test_v5"
ARROW_PATH = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data_arrow/construction_site_test.arrow")

CONFIGS = {
    "light": {
        "guidance_scale": 8.0,
        "image_guidance_scale": 1.5,
        "prompt": "a rainy day, wet surfaces, light rain, grey overcast sky",
        "physics_intensity": "light",
    },
    "heavy_prev": {
        "guidance_scale": 10.0,
        "image_guidance_scale": 1.5,
        "prompt": "a heavy rainy day with dark overcast sky, wet muddy ground, rain falling",
        "physics_intensity": "heavy",
    },
    "heavy": {
        "guidance_scale": 11.0,
        "image_guidance_scale": 1.4,
        "prompt": "a heavy rainstorm with dark overcast sky, wet muddy ground, heavy rain falling, hazy air reducing visibility",
        "physics_intensity": "heavy_fog",
    },
}

N_SAMPLES = 10
SEED = 42


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(io.BytesIO(row['bytes'])).convert('RGB')
    return Image.open(io.BytesIO(row)).convert('RGB')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    table = load_arrow(ARROW_PATH)

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(table), size=N_SAMPLES, replace=False)

    orig_dir = OUTPUT_DIR / "original"
    orig_dir.mkdir(exist_ok=True)
    originals = []
    for i, idx in enumerate(indices):
        img = get_image(table, int(idx))
        img.save(orig_dir / f"{i:03d}.jpg")
        originals.append(img)
    print(f"Saved {N_SAMPLES} originals")

    print("Loading InstructPix2Pix...")
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    all_outputs = {name: [] for name in CONFIGS}

    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}\n{name}  g={cfg['guidance_scale']}  igs={cfg['image_guidance_scale']}  physics={cfg['physics_intensity']}\n  prompt: {cfg['prompt']}\n{'='*60}")
        out_dir = OUTPUT_DIR / name
        out_dir.mkdir(exist_ok=True)

        for i, idx in enumerate(indices):
            orig_pil = get_image(table, int(idx))
            h, w = orig_pil.height, orig_pil.width
            max_dim = 768
            scale = min(max_dim / max(h, w), 1.0)
            nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
            img_resized = orig_pil.resize((nw, nh), Image.LANCZOS)

            random.seed(SEED + int(idx))
            g = torch.Generator("cuda").manual_seed(SEED + int(idx))
            result = pipe(
                cfg["prompt"],
                image=img_resized,
                num_inference_steps=30,
                image_guidance_scale=cfg["image_guidance_scale"],
                guidance_scale=cfg["guidance_scale"],
                generator=g,
            ).images[0]

            aug_np = np.array(result.resize((w, h), Image.LANCZOS))
            aug_np = add_natural_rain(aug_np, intensity=cfg["physics_intensity"])
            aug_pil = Image.fromarray(aug_np)
            aug_pil.save(out_dir / f"{i:03d}.jpg")
            all_outputs[name].append(aug_pil)
            print(f"  [{i+1}/{N_SAMPLES}] done")

    # Grid: cols = [original, light, heavy]
    print("\nBuilding grid...")
    cell_w = 384
    cell_h_list = [int(im.height * cell_w / im.width) for im in originals]
    label_h = 32
    margin = 4
    cols = ["original", "light", "heavy_prev", "heavy"]

    total_w = cell_w * len(cols) + margin * (len(cols) + 1)
    total_h = sum(cell_h_list) + margin * (N_SAMPLES + 1) + label_h

    grid = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for ci, cname in enumerate(cols):
        x = margin + ci * (cell_w + margin)
        draw.text((x + cell_w // 2 - 30, 6), cname, fill=(0, 0, 0), font=font)

    y = label_h + margin
    for ri in range(N_SAMPLES):
        ch = cell_h_list[ri]
        cells = [originals[ri], all_outputs["light"][ri], all_outputs["heavy_prev"][ri], all_outputs["heavy"][ri]]
        for ci, img in enumerate(cells):
            x = margin + ci * (cell_w + margin)
            grid.paste(img.resize((cell_w, ch), Image.LANCZOS), (x, y))
        y += ch + margin

    grid_path = OUTPUT_DIR / "grid.jpg"
    grid.save(grid_path, quality=92)
    print(f"Grid saved: {grid_path}  ({total_w}x{total_h})")


if __name__ == "__main__":
    main()
