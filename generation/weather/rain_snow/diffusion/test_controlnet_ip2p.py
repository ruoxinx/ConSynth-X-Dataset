#!/usr/bin/env python3
"""
Test ControlNet + IP2P for rain augmentation with structure preservation.

Two approaches:
  A) control_v11e_sd15_ip2p — ControlNet trained on IP2P editing, original image as control
  B) Canny ControlNet + SD img2img — edge map locks structure, img2img adds weather

Both compared against vanilla IP2P SD1.5 on same 5 samples.

Usage:
  python test_controlnet_ip2p.py --n 5 --seed 42
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
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from physics import add_natural_rain

ORIG_ARROW = _DATA / 'LouisChen15___construction_site' / 'construction_site-train-00000-of-00002.arrow'
OUT_DIR = _REPO / 'paper' / 'figures' / 'controlnet_test'

RAIN_PROMPT = "a rainy day with dark overcast sky, rain falling, grey clouds"


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


def make_canny(image_pil, low=100, high=200):
    """Extract Canny edges from PIL image."""
    img = np.array(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


def resize_for_sd(img_pil, max_dim=768, divisor=8):
    h, w = img_pil.height, img_pil.width
    scale = min(max_dim / max(h, w), 1.0)
    nw, nh = int(w * scale) // divisor * divisor, int(h * scale) // divisor * divisor
    return img_pil.resize((nw, nh), Image.LANCZOS), (w, h)


# ── Method A: Vanilla IP2P (baseline) ────────────────────────
def run_vanilla_ip2p(images, prompt, seed_base):
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

    print('\n  Loading vanilla IP2P...')
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    results = []
    for i, (img_pil, img_id) in enumerate(images):
        img_resized, (ow, oh) = resize_for_sd(img_pil)
        g = torch.Generator("cuda").manual_seed(seed_base + i)
        result = pipe(prompt, image=img_resized, num_inference_steps=30,
                      image_guidance_scale=1.5, guidance_scale=10.0, generator=g).images[0]
        result = result.resize((ow, oh), Image.LANCZOS)
        aug_np = add_natural_rain(np.array(result))
        results.append(Image.fromarray(aug_np))
        print(f'    [{i+1}/{len(images)}] {img_id} done')
        torch.cuda.empty_cache()

    del pipe; gc.collect(); torch.cuda.empty_cache()
    return results


# ── Method B: ControlNet IP2P (control_v11e_sd15_ip2p) ───────
def run_controlnet_ip2p(images, prompt, seed_base):
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    print('\n  Loading ControlNet IP2P...')
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11e_sd15_ip2p", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
        torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    results = []
    for i, (img_pil, img_id) in enumerate(images):
        img_resized, (ow, oh) = resize_for_sd(img_pil)
        g = torch.Generator("cuda").manual_seed(seed_base + i)
        # Original image is the control signal (structure guide)
        result = pipe(prompt, image=img_resized, num_inference_steps=30,
                      guidance_scale=7.5, generator=g).images[0]
        result = result.resize((ow, oh), Image.LANCZOS)
        aug_np = add_natural_rain(np.array(result))
        results.append(Image.fromarray(aug_np))
        print(f'    [{i+1}/{len(images)}] {img_id} done')
        torch.cuda.empty_cache()

    del pipe, controlnet; gc.collect(); torch.cuda.empty_cache()
    return results


# ── Method C: Canny ControlNet + SD img2img ───────────────────
def run_canny_controlnet_img2img(images, prompt, seed_base):
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
        UniPCMultistepScheduler,
    )

    print('\n  Loading Canny ControlNet + img2img...')
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
        torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    results = []
    for i, (img_pil, img_id) in enumerate(images):
        img_resized, (ow, oh) = resize_for_sd(img_pil)
        canny_img = make_canny(img_resized)
        g = torch.Generator("cuda").manual_seed(seed_base + i)
        # img2img with low strength to preserve content, canny locks edges
        result = pipe(
            prompt, image=img_resized, control_image=canny_img,
            num_inference_steps=30, strength=0.35,
            guidance_scale=7.5, controlnet_conditioning_scale=0.8,
            generator=g,
        ).images[0]
        result = result.resize((ow, oh), Image.LANCZOS)
        aug_np = add_natural_rain(np.array(result))
        results.append(Image.fromarray(aug_np))
        print(f'    [{i+1}/{len(images)}] {img_id} done')
        torch.cuda.empty_cache()

    del pipe, controlnet; gc.collect(); torch.cuda.empty_cache()
    return results


# ── Main ──────────────────────────────────────────────────────
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

    # Collect input images
    images = []
    for idx in indices:
        img_id = str(table.column('image_id')[idx].as_py())
        img_pil = get_image(table, idx)
        images.append((img_pil, img_id))
        img_pil.save(OUT_DIR / f'{img_id}_orig.jpg', quality=95)

    methods = {
        'A) Vanilla IP2P': run_vanilla_ip2p,
        'B) ControlNet IP2P': run_controlnet_ip2p,
        'C) Canny CN + img2img': run_canny_controlnet_img2img,
    }

    all_results = {}
    all_ssims = {}

    for name, func in methods.items():
        print(f'\n{"="*60}')
        print(f'Method: {name}')
        print(f'{"="*60}')
        try:
            results = func(images, RAIN_PROMPT, args.seed)
            ssims = []
            for i, ((orig_pil, img_id), aug_pil) in enumerate(zip(images, results)):
                ss = compute_ssim(orig_pil, aug_pil)
                ssims.append(ss)
                aug_pil.save(OUT_DIR / f'{img_id}_{name[:1].lower()}.jpg', quality=95)
                print(f'    {img_id} SSIM={ss:.3f}')
            all_results[name] = results
            all_ssims[name] = ssims
        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback; traceback.print_exc()
            all_results[name] = [None] * len(images)
            all_ssims[name] = [0.0] * len(images)

    # ── Build comparison grid ─────────────────────────────────
    print('\nBuilding comparison grid...')
    n = args.n
    ncols = 1 + len(methods)
    method_names = list(methods.keys())

    fig, axes = plt.subplots(n, ncols, figsize=(5 * ncols, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    titles = ['Original'] + method_names

    for row_i in range(n):
        img_id = images[row_i][1]

        # Original
        axes[row_i, 0].imshow(images[row_i][0])
        axes[row_i, 0].axis('off')
        axes[row_i, 0].set_ylabel(f'ID: {img_id}', fontsize=9, rotation=0, labelpad=50)

        # Methods
        for col_i, name in enumerate(method_names):
            ax = axes[row_i, 1 + col_i]
            img = all_results[name][row_i]
            if img is not None:
                ax.imshow(img)
                ss = all_ssims[name][row_i]
                color = '#2d8a4e' if ss > 0.7 else '#d4a017' if ss > 0.5 else '#c0392b'
                ax.set_xlabel(f'SSIM={ss:.2f}', fontsize=10, fontweight='bold', color=color)
            else:
                ax.text(0.5, 0.5, 'Failed', ha='center', va='center',
                        transform=ax.transAxes, color='red', fontsize=12)
            ax.axis('off')

    for col_i, title in enumerate(titles):
        axes[0, col_i].set_title(title, fontsize=11, fontweight='bold', pad=10)

    # Summary
    summary_parts = []
    for name in method_names:
        mean_ss = np.mean(all_ssims[name])
        summary_parts.append(f'{name}: {mean_ss:.3f}')
    fig.text(0.5, -0.01, 'Mean SSIM — ' + ' | '.join(summary_parts),
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow'))

    plt.suptitle(
        f'ControlNet + IP2P Rain Augmentation Comparison (n={n})\n'
        f'Prompt: "{RAIN_PROMPT}"',
        fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = OUT_DIR / 'controlnet_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.pdf'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    # Summary table
    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    header = f'{"ID":<12}' + ''.join(f'{name:>22}' for name in method_names)
    print(header)
    print('-' * len(header))
    for row_i in range(n):
        line = f'{images[row_i][1]:<12}'
        for name in method_names:
            line += f'{all_ssims[name][row_i]:>22.3f}'
        print(line)
    print('-' * len(header))
    line = f'{"Mean":<12}'
    for name in method_names:
        line += f'{np.mean(all_ssims[name]):>22.3f}'
    print(line)

    # Winner
    best = max(method_names, key=lambda n: np.mean(all_ssims[n]))
    print(f'\nBest method: {best} (mean SSIM={np.mean(all_ssims[best]):.3f})')


if __name__ == '__main__':
    main()
