#!/usr/bin/env python3
"""
A/B test: compare IP2P models for rain augmentation quality.
  1. Original IP2P (SD 1.5)  — timbrooks/instruct-pix2pix
  2. SDXL IP2P               — diffusers/sdxl-instructpix2pix-768
  3. CosXL Edit              — stabilityai/cosxl (if available)

Generates 5 samples per model and saves comparison grid.

Usage:
  python test_models.py
  python test_models.py --n 5 --models sd15 sdxl
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from io import BytesIO

_REPO = Path(os.environ.get("CONSYNTH_REPO_ROOT", Path(__file__).resolve().parents[4]))
_DATA = Path(os.environ.get("CONSYNTH_DATA_ROOT", Path.home() / "consynth_data"))

import numpy as np
import pyarrow as pa
from PIL import Image
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from physics import add_natural_rain

ORIG_ARROW = _DATA / 'LouisChen15___construction_site' / 'construction_site-train-00000-of-00002.arrow'
OUT_DIR = _REPO / 'paper' / 'figures' / 'model_test'

NEW_PROMPT = "a rainy day with dark overcast sky, rain falling, grey clouds"

MODEL_CONFIGS = {
    'sd15': {
        'name': 'IP2P (SD 1.5)',
        'model_id': 'timbrooks/instruct-pix2pix',
        'pipeline': 'ip2p',
        'guidance': 10.0,
        'igs': 1.5,
        'steps': 30,
    },
    'sdxl': {
        'name': 'IP2P (SDXL)',
        'model_id': 'diffusers/sdxl-instructpix2pix-768',
        'pipeline': 'ip2p_sdxl',
        'guidance': 7.5,
        'igs': 1.5,
        'steps': 30,
    },
    'cosxl': {
        'name': 'CosXL Edit',
        'model_id': 'stabilityai/cosxl',
        'pipeline': 'cosxl',
        'guidance': 7.0,
        'igs': 1.5,
        'steps': 20,
    },
}


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(BytesIO(row['bytes'])).convert('RGB')
    return Image.open(BytesIO(row)).convert('RGB')


def compute_ssim(orig_pil, aug_pil):
    orig = np.array(orig_pil.resize((256, 256)))
    aug = np.array(aug_pil.resize((256, 256)))
    return ssim(orig, aug, channel_axis=2)


def load_model(config):
    """Load the appropriate pipeline for each model."""
    from diffusers import EulerAncestralDiscreteScheduler

    model_id = config['model_id']
    ptype = config['pipeline']

    print(f'  Loading {config["name"]} from {model_id}...')

    if ptype == 'ip2p':
        from diffusers import StableDiffusionInstructPix2PixPipeline
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None)
        pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        return pipe

    elif ptype == 'ip2p_sdxl':
        from diffusers import StableDiffusionXLInstructPix2PixPipeline
        pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16)
        pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        return pipe

    elif ptype == 'cosxl':
        # CosXL uses the same SDXL IP2P pipeline but different model
        from diffusers import StableDiffusionXLInstructPix2PixPipeline
        pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16)
        pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        return pipe

    else:
        raise ValueError(f'Unknown pipeline type: {ptype}')


def generate_one(pipe, img_pil, prompt, seed, config):
    """Run model + physics overlay."""
    h, w = img_pil.height, img_pil.width
    max_dim = 768
    scale = min(max_dim / max(h, w), 1.0)
    nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    img_resized = img_pil.resize((nw, nh), Image.LANCZOS)

    g = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        prompt,
        image=img_resized,
        num_inference_steps=config['steps'],
        image_guidance_scale=config['igs'],
        guidance_scale=config['guidance'],
        generator=g,
    ).images[0]

    aug_np = np.array(result.resize((w, h), Image.LANCZOS))
    aug_np = add_natural_rain(aug_np)
    return Image.fromarray(aug_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help='Number of samples')
    parser.add_argument('--models', nargs='+', default=['sd15', 'sdxl'],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading Arrow table...')
    table = load_arrow(ORIG_ARROW)
    total = len(table)

    # Pick N diverse samples
    step = total // args.n
    indices = [i * step for i in range(args.n)]

    models_to_test = args.models
    ncols = 1 + len(models_to_test)  # Original + models
    col_titles = ['Original'] + [MODEL_CONFIGS[m]['name'] for m in models_to_test]

    fig, axes = plt.subplots(args.n, ncols, figsize=(5 * ncols, 4 * args.n))
    if args.n == 1:
        axes = axes.reshape(1, -1)

    # Process each model
    for m_i, model_key in enumerate(models_to_test):
        config = MODEL_CONFIGS[model_key]
        print(f'\n{"="*60}')
        print(f'Model: {config["name"]}')
        print(f'{"="*60}')

        try:
            pipe = load_model(config)
        except Exception as e:
            print(f'  FAILED to load: {e}')
            for row_i in range(args.n):
                axes[row_i, 1 + m_i].text(0.5, 0.5, f'Load failed:\n{str(e)[:50]}',
                                            ha='center', va='center', transform=axes[row_i, 1 + m_i].transAxes,
                                            fontsize=8, color='red')
                axes[row_i, 1 + m_i].axis('off')
            continue

        for row_i, idx in enumerate(indices):
            image_id = str(table.column('image_id')[idx].as_py())
            orig_pil = get_image(table, idx)
            seed = args.seed + idx

            print(f'  [{row_i+1}/{args.n}] ID={image_id}', end='', flush=True)

            # Original (only on first model pass)
            if m_i == 0:
                axes[row_i, 0].imshow(orig_pil)
                axes[row_i, 0].axis('off')
                axes[row_i, 0].set_ylabel(f'{image_id}', fontsize=9, rotation=0, labelpad=45)

            # Generate
            try:
                aug_img = generate_one(pipe, orig_pil, NEW_PROMPT, seed, config)
                ss = compute_ssim(orig_pil, aug_img)
                print(f' SSIM={ss:.3f}', flush=True)

                axes[row_i, 1 + m_i].imshow(aug_img)
                axes[row_i, 1 + m_i].set_xlabel(f'SSIM={ss:.2f}', fontsize=9)

                # Save individual
                aug_img.save(OUT_DIR / f'{image_id}_{model_key}.jpg', quality=95)
                if m_i == 0:
                    orig_pil.save(OUT_DIR / f'{image_id}_orig.jpg', quality=95)
            except Exception as e:
                print(f' ERROR: {e}', flush=True)
                axes[row_i, 1 + m_i].text(0.5, 0.5, f'Error:\n{str(e)[:60]}',
                                            ha='center', va='center', transform=axes[row_i, 1 + m_i].transAxes,
                                            fontsize=7, color='red')

            axes[row_i, 1 + m_i].axis('off')
            torch.cuda.empty_cache()

        # Unload model
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        print(f'  Model unloaded.')

    # Titles
    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, fontsize=13, fontweight='bold', pad=10)

    plt.suptitle(
        f'Model Comparison for Rain Augmentation (n={args.n})\n'
        f'Prompt: "{NEW_PROMPT}"',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    out_path = OUT_DIR / 'model_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
