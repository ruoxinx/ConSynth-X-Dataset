#!/usr/bin/env python3
"""
Production batch worker for SODA (JPG folder): IP2P + physics + LPIPS/SSIM filter.
Processes a slice of images from a folder, outputs filtered JPGs + metadata CSV.
"""

import argparse
import sys
import gc
import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from physics import add_natural_rain, add_natural_snow

LPIPS_THRESHOLD = 0.35
SSIM_RAIN = (0.6, 0.95)
SSIM_SNOW = (0.5, 0.95)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--weather', type=str, required=True, choices=['rain', 'snow'])
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def compute_lpips(lpips_fn, orig_pil, aug_pil):
    size = (256, 256)
    orig_t = torch.from_numpy(np.array(orig_pil.resize(size))).permute(2, 0, 1).float() / 127.5 - 1.0
    aug_t = torch.from_numpy(np.array(aug_pil.resize(size))).permute(2, 0, 1).float() / 127.5 - 1.0
    with torch.no_grad():
        return lpips_fn(orig_t.unsqueeze(0).cuda(), aug_t.unsqueeze(0).cuda()).item()


def compute_ssim(orig_pil, aug_pil):
    orig = np.array(orig_pil.resize((256, 256)))
    aug = np.array(aug_pil.resize((256, 256)))
    return ssim(orig, aug, channel_axis=2)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    all_images = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    total = len(all_images)
    start = max(0, args.start)
    end = min(args.end, total)
    batch_size = end - start

    print("=" * 60)
    print(f"Diffusion Batch Worker (SODA): {args.weather}")
    print(f"Input: {input_dir} ({total} images)")
    print(f"Range: [{start}, {end}) = {batch_size} images")
    print("=" * 60)

    if batch_size <= 0:
        print("No samples to process!")
        return

    # Load models
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    lpips_fn = None
    if args.weather == 'rain':
        lpips_fn = lpips.LPIPS(net='alex').cuda()

    # Config
    if args.weather == 'rain':
        prompt = "a rainy day with dark overcast sky, rain falling, grey clouds"
        igs, guidance = 1.5, 10.0
        ssim_range = SSIM_RAIN
    else:
        prompt = "a cold winter day with snow, frost on surfaces, grey sky, snow on the ground"
        igs, guidance = 1.5, 8.0
        ssim_range = SSIM_SNOW

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_count = 0
    all_meta = []

    for idx in range(start, end):
        img_path = all_images[idx]
        image_id = img_path.stem
        orig_pil = Image.open(img_path).convert('RGB')
        h, w = orig_pil.height, orig_pil.width

        max_dim = 768
        scale = min(max_dim / max(h, w), 1.0)
        nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
        img_resized = orig_pil.resize((nw, nh), Image.LANCZOS)

        random.seed(args.seed + idx)
        g = torch.Generator("cuda").manual_seed(args.seed + idx)
        result = pipe(prompt, image=img_resized, num_inference_steps=30,
                      image_guidance_scale=igs, guidance_scale=guidance, generator=g).images[0]
        aug_np = np.array(result.resize((w, h), Image.LANCZOS))

        if args.weather == 'rain':
            aug_np = add_natural_rain(aug_np)
        else:
            aug_np = add_natural_snow(aug_np)

        aug_pil = Image.fromarray(aug_np)

        # Compute metrics
        ss = compute_ssim(orig_pil, aug_pil)
        lp = compute_lpips(lpips_fn, orig_pil, aug_pil) if lpips_fn else 0.0

        # Filter
        keep_ssim = ssim_range[0] <= ss <= ssim_range[1]
        keep_lpips = lp < LPIPS_THRESHOLD if args.weather == 'rain' else True
        keep = keep_ssim and keep_lpips

        status = 'KEEP' if keep else 'DROP'
        processed = idx - start + 1
        if args.weather == 'rain':
            print(f"  [{processed}/{batch_size}] {image_id} SSIM={ss:.3f} LPIPS={lp:.3f} {status}")
        else:
            print(f"  [{processed}/{batch_size}] {image_id} SSIM={ss:.3f} {status}")

        all_meta.append(f"{image_id},{ss:.4f},{lp:.4f},{status}")

        if keep:
            aug_pil.save(output_dir / img_path.name, quality=95)
            kept_count += 1

        del result, aug_np, aug_pil, orig_pil
        gc.collect()
        torch.cuda.empty_cache()

    # Save metadata CSV
    csv_path = output_dir / f'meta_{start}-{end}.csv'
    with open(csv_path, 'w') as f:
        f.write("image_id,ssim,lpips,status\n")
        for line in all_meta:
            f.write(line + '\n')
    print(f"Metadata: {csv_path}")

    drop_count = batch_size - kept_count
    print(f"\nSummary: {kept_count} KEEP, {drop_count} DROP out of {batch_size}")


if __name__ == '__main__':
    main()
