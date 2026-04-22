#!/usr/bin/env python3
"""
Quick test: generate 20 rain samples with the NEW prompt (no "wet muddy ground")
and save a comparison grid (original vs new IP2P) for visual review.

Usage:
  python test_new_prompt.py
  python test_new_prompt.py --n 10 --seed 42
"""

import argparse
import os
import sys
import random
from pathlib import Path
from io import BytesIO

_REPO = Path(os.environ.get("CONSYNTH_REPO_ROOT", Path(__file__).resolve().parents[4]))
_DATA = Path(os.environ.get("CONSYNTH_DATA_ROOT", Path.home() / "consynth_data"))

import numpy as np
import pyarrow as pa
from PIL import Image
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from physics import add_natural_rain

ORIG_ARROW = _DATA / 'LouisChen15___construction_site' / 'construction_site-train-00000-of-00002.arrow'
OUT_DIR = _REPO / 'paper' / 'figures' / 'prompt_test'

# Old vs New prompt
OLD_PROMPT = "make it a heavy rainy day, dark overcast sky, wet muddy ground"
NEW_PROMPT = "a rainy day with dark overcast sky, rain falling, grey clouds"


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(BytesIO(row['bytes'])).convert('RGB')
    return Image.open(BytesIO(row)).convert('RGB')


def generate_one(pipe, img_pil, prompt, seed, igs=1.5, guidance=10.0):
    """Run IP2P + physics overlay on one image."""
    h, w = img_pil.height, img_pil.width
    max_dim = 768
    scale = min(max_dim / max(h, w), 1.0)
    nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    img_resized = img_pil.resize((nw, nh), Image.LANCZOS)

    g = torch.Generator("cuda").manual_seed(seed)
    result = pipe(prompt, image=img_resized, num_inference_steps=30,
                  image_guidance_scale=igs, guidance_scale=guidance, generator=g).images[0]
    aug_np = np.array(result.resize((w, h), Image.LANCZOS))
    aug_np = add_natural_rain(aug_np)
    return Image.fromarray(aug_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-old', action='store_true', help='Skip old prompt (faster, 2 cols only)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading Arrow table...')
    table = load_arrow(ORIG_ARROW)
    print(f'  {len(table)} images')

    # Pick N diverse samples (spread across the dataset)
    total = len(table)
    step = total // args.n
    indices = [i * step for i in range(args.n)]

    print('Loading IP2P model...')
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    ncols = 2 if args.skip_old else 3
    col_titles = ['Original', 'NEW prompt'] if args.skip_old else ['Original', 'OLD prompt', 'NEW prompt']

    fig, axes = plt.subplots(args.n, ncols, figsize=(5 * ncols, 4 * args.n))
    if args.n == 1:
        axes = axes.reshape(1, -1)

    for row_i, idx in enumerate(indices):
        image_id = str(table.column('image_id')[idx].as_py())
        orig_pil = get_image(table, idx)
        seed = args.seed + idx

        print(f'  [{row_i+1}/{args.n}] ID={image_id}', end='', flush=True)

        # Original
        col = 0
        axes[row_i, col].imshow(orig_pil)
        axes[row_i, col].axis('off')
        axes[row_i, col].set_ylabel(f'ID: {image_id}', fontsize=8, rotation=0, labelpad=50)

        if not args.skip_old:
            # Old prompt
            print(' | old', end='', flush=True)
            old_img = generate_one(pipe, orig_pil, OLD_PROMPT, seed)
            col = 1
            axes[row_i, col].imshow(old_img)
            axes[row_i, col].axis('off')
            # Save individual
            old_img.save(OUT_DIR / f'{image_id}_old.jpg', quality=95)

        # New prompt
        print(' | new', end='', flush=True)
        new_img = generate_one(pipe, orig_pil, NEW_PROMPT, seed)
        col = ncols - 1
        axes[row_i, col].imshow(new_img)
        axes[row_i, col].axis('off')
        # Save individual
        new_img.save(OUT_DIR / f'{image_id}_new.jpg', quality=95)
        orig_pil.save(OUT_DIR / f'{image_id}_orig.jpg', quality=95)

        print(' done')

        torch.cuda.empty_cache()

    # Column titles
    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.suptitle(
        f'IP2P Rain Prompt Comparison (n={args.n})\n'
        f'OLD: "{OLD_PROMPT}"\n'
        f'NEW: "{NEW_PROMPT}"',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    out_path = OUT_DIR / 'prompt_comparison_grid.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out_path}')
    print(f'Individual images: {OUT_DIR}/')


if __name__ == '__main__':
    main()
