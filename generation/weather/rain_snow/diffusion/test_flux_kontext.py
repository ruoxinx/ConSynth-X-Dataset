#!/usr/bin/env python3
"""
Test FLUX.1-Kontext-dev for rain augmentation.
Compare with SD 1.5 IP2P on the same 5 samples using a safe, conservative prompt.

Usage:
  python test_flux_kontext.py
  python test_flux_kontext.py --n 5 --seed 42
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
OUT_DIR = _REPO / 'paper' / 'figures' / 'flux_test'

# Conservative prompts — focus on atmosphere only, no ground/structure changes
SAFE_PROMPT = "Add overcast grey sky and light rain atmosphere to this photo"
SD15_PROMPT = "a rainy day with dark overcast sky, rain falling, grey clouds"


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


def run_flux_kontext(img_pil, prompt, seed):
    """Generate with FLUX.1-Kontext-dev."""
    from diffusers import FluxKontextPipeline

    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    # Resize to reasonable size for FLUX (max 1024)
    h, w = img_pil.height, img_pil.width
    max_dim = 768
    scale = min(max_dim / max(h, w), 1.0)
    nw, nh = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16  # FLUX needs multiples of 16
    img_resized = img_pil.resize((nw, nh), Image.LANCZOS)

    g = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        image=img_resized,
        prompt=prompt,
        guidance_scale=2.5,
        num_inference_steps=28,
        generator=g,
    ).images[0]

    result = result.resize((w, h), Image.LANCZOS)
    return pipe, result


def run_sd15_ip2p(img_pil, prompt, seed):
    """Generate with SD 1.5 IP2P."""
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    h, w = img_pil.height, img_pil.width
    max_dim = 768
    scale = min(max_dim / max(h, w), 1.0)
    nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    img_resized = img_pil.resize((nw, nh), Image.LANCZOS)

    g = torch.Generator("cuda").manual_seed(seed)
    result = pipe(prompt, image=img_resized, num_inference_steps=30,
                  image_guidance_scale=1.5, guidance_scale=10.0, generator=g).images[0]
    result = result.resize((w, h), Image.LANCZOS)
    return pipe, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading Arrow table...')
    table = load_arrow(ORIG_ARROW)
    total = len(table)
    step = total // args.n
    indices = [i * step for i in range(args.n)]

    # Collect results
    results = {
        'orig': [],
        'sd15': [],
        'flux': [],
        'sd15_ssim': [],
        'flux_ssim': [],
        'ids': [],
    }

    # ── SD 1.5 IP2P ──────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Model 1: IP2P (SD 1.5)')
    print(f'Prompt: "{SD15_PROMPT}"')
    print(f'{"="*60}')

    sd15_pipe = None
    for i, idx in enumerate(indices):
        image_id = str(table.column('image_id')[idx].as_py())
        orig_pil = get_image(table, idx)
        seed = args.seed + idx

        print(f'  [{i+1}/{args.n}] ID={image_id}', end='', flush=True)

        if sd15_pipe is None:
            sd15_pipe, result = run_sd15_ip2p(orig_pil, SD15_PROMPT, seed)
        else:
            h, w = orig_pil.height, orig_pil.width
            max_dim = 768
            scale = min(max_dim / max(h, w), 1.0)
            nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
            img_resized = orig_pil.resize((nw, nh), Image.LANCZOS)
            g = torch.Generator("cuda").manual_seed(seed)
            result = sd15_pipe(SD15_PROMPT, image=img_resized, num_inference_steps=30,
                               image_guidance_scale=1.5, guidance_scale=10.0, generator=g).images[0]
            result = result.resize((w, h), Image.LANCZOS)

        # Apply physics rain
        aug_np = add_natural_rain(np.array(result))
        aug_pil = Image.fromarray(aug_np)

        ss = compute_ssim(orig_pil, aug_pil)
        print(f' SSIM={ss:.3f}')

        results['orig'].append(orig_pil)
        results['sd15'].append(aug_pil)
        results['sd15_ssim'].append(ss)
        results['ids'].append(image_id)

        aug_pil.save(OUT_DIR / f'{image_id}_sd15.jpg', quality=95)
        orig_pil.save(OUT_DIR / f'{image_id}_orig.jpg', quality=95)

        torch.cuda.empty_cache()

    del sd15_pipe
    gc.collect()
    torch.cuda.empty_cache()
    print('  SD 1.5 unloaded.\n')

    # ── FLUX.1 Kontext ────────────────────────────────────────
    print(f'{"="*60}')
    print(f'Model 2: FLUX.1-Kontext-dev')
    print(f'Prompt: "{SAFE_PROMPT}"')
    print(f'{"="*60}')

    flux_pipe = None
    for i, idx in enumerate(indices):
        image_id = results['ids'][i]
        orig_pil = results['orig'][i]
        seed = args.seed + idx

        print(f'  [{i+1}/{args.n}] ID={image_id}', end='', flush=True)

        if flux_pipe is None:
            flux_pipe, result = run_flux_kontext(orig_pil, SAFE_PROMPT, seed)
        else:
            h, w = orig_pil.height, orig_pil.width
            max_dim = 768
            scale = min(max_dim / max(h, w), 1.0)
            nw, nh = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
            img_resized = orig_pil.resize((nw, nh), Image.LANCZOS)
            g = torch.Generator("cuda").manual_seed(seed)
            result = flux_pipe(
                image=img_resized, prompt=SAFE_PROMPT,
                guidance_scale=2.5, num_inference_steps=28, generator=g
            ).images[0]
            result = result.resize((w, h), Image.LANCZOS)

        # Apply physics rain
        aug_np = add_natural_rain(np.array(result))
        aug_pil = Image.fromarray(aug_np)

        ss = compute_ssim(orig_pil, aug_pil)
        print(f' SSIM={ss:.3f}')

        results['flux'].append(aug_pil)
        results['flux_ssim'].append(ss)

        aug_pil.save(OUT_DIR / f'{image_id}_flux.jpg', quality=95)

        torch.cuda.empty_cache()

    del flux_pipe
    gc.collect()
    torch.cuda.empty_cache()
    print('  FLUX unloaded.\n')

    # ── Build comparison grid ─────────────────────────────────
    print('Building comparison grid...')
    n = args.n
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    titles = [
        'Original',
        f'IP2P SD1.5 (new prompt)\n"{SD15_PROMPT}"',
        f'FLUX.1-Kontext (safe prompt)\n"{SAFE_PROMPT}"',
    ]

    for row_i in range(n):
        img_id = results['ids'][row_i]

        axes[row_i, 0].imshow(results['orig'][row_i])
        axes[row_i, 0].axis('off')
        axes[row_i, 0].set_ylabel(f'ID: {img_id}', fontsize=9, rotation=0, labelpad=50)

        axes[row_i, 1].imshow(results['sd15'][row_i])
        axes[row_i, 1].axis('off')
        axes[row_i, 1].set_xlabel(f'SSIM={results["sd15_ssim"][row_i]:.2f}', fontsize=10, fontweight='bold')

        axes[row_i, 2].imshow(results['flux'][row_i])
        axes[row_i, 2].axis('off')
        axes[row_i, 2].set_xlabel(f'SSIM={results["flux_ssim"][row_i]:.2f}', fontsize=10, fontweight='bold')

    for col_i, title in enumerate(titles):
        axes[0, col_i].set_title(title, fontsize=11, fontweight='bold', pad=10)

    # Summary stats
    sd15_mean = np.mean(results['sd15_ssim'])
    flux_mean = np.mean(results['flux_ssim'])
    fig.text(0.5, -0.01,
             f'Mean SSIM — SD1.5: {sd15_mean:.3f} | FLUX Kontext: {flux_mean:.3f}',
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow'))

    plt.suptitle('IP2P (SD1.5) vs FLUX.1-Kontext-dev: Rain Augmentation Comparison',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = OUT_DIR / 'flux_vs_sd15.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.pdf'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    # Print summary
    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    print(f'{"ID":<12} {"SD1.5 SSIM":>12} {"FLUX SSIM":>12} {"Winner":>10}')
    print(f'{"-"*46}')
    for i in range(n):
        s1 = results['sd15_ssim'][i]
        s2 = results['flux_ssim'][i]
        winner = 'SD1.5' if s1 > s2 else 'FLUX'
        print(f'{results["ids"][i]:<12} {s1:>12.3f} {s2:>12.3f} {winner:>10}')
    print(f'{"-"*46}')
    print(f'{"Mean":<12} {sd15_mean:>12.3f} {flux_mean:>12.3f}')


if __name__ == '__main__':
    main()
