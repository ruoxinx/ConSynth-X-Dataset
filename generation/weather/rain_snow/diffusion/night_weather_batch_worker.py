#!/usr/bin/env python3
"""
Night Weather Batch Worker: IP2P + Physics on Night images.

The CycleGAN day-to-night conversion is a separate upstream stage. This worker
expects its resulting night Arrow as input and does not load or run CycleGAN.

Pipeline:
  Night image (from night.arrow)
    → IP2P diffusion (rain/snow prompt tuned for night scenes)
    → Physics particle overlay (rain streaks or snowflakes)
    → SSIM/LPIPS quality filter
    → Output Arrow + metadata CSV

Usage:
    python night_weather_batch_worker.py \
        --night-input night.arrow \
        --orig-input construction_site_test.arrow \
        --output-dir output/night_rain \
        --start 0 --end 500 \
        --weather rain --seed 42
"""

import argparse
import sys
import gc
import random
from pathlib import Path
import io

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Quality filter thresholds (same as day weather pipeline)
LPIPS_THRESHOLD = 0.35
SSIM_RAIN = (0.6, 0.95)
SSIM_SNOW = (0.5, 0.95)

# IP2P config — prompts tuned for night scenes
NIGHT_RAIN_PROMPT = "a rainy night with rain falling, wet reflections on surfaces, dark overcast sky"
NIGHT_SNOW_PROMPT = "a cold winter night with snow falling, frost on surfaces, snow on the ground"

IP2P_CONFIG = {
    'rain': {
        'prompt': NIGHT_RAIN_PROMPT,
        'image_guidance_scale': 1.5,
        'guidance_scale': 10.0,
        'num_inference_steps': 30,
        'ssim_range': SSIM_RAIN,
        'use_lpips': True,
    },
    'snow': {
        'prompt': NIGHT_SNOW_PROMPT,
        'image_guidance_scale': 1.5,
        'guidance_scale': 8.0,
        'num_inference_steps': 30,
        'ssim_range': SSIM_SNOW,
        'use_lpips': False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Night Weather Batch Worker: IP2P + Physics on Night images')
    parser.add_argument('--night-input', type=str, required=True,
                        help='Path to night.arrow (CycleGAN night images)')
    parser.add_argument('--orig-input', type=str, required=True,
                        help='Path to original arrow (for SSIM/LPIPS comparison)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for Arrow + CSV')
    parser.add_argument('--start', type=int, required=True,
                        help='Start index (inclusive)')
    parser.add_argument('--end', type=int, required=True,
                        help='End index (exclusive)')
    parser.add_argument('--weather', type=str, required=True,
                        choices=['rain', 'snow'],
                        help='Weather type: rain or snow')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_arrow(path):
    import pyarrow as pa
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    from PIL import Image
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(io.BytesIO(row['bytes'])).convert('RGB')
    return Image.open(io.BytesIO(row)).convert('RGB')


def image_to_bytes(img, quality=95):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def compute_lpips(lpips_fn, orig_pil, aug_pil):
    import numpy as np
    import torch
    size = (256, 256)
    orig_t = torch.from_numpy(np.array(orig_pil.resize(size))).permute(2, 0, 1).float() / 127.5 - 1.0
    aug_t = torch.from_numpy(np.array(aug_pil.resize(size))).permute(2, 0, 1).float() / 127.5 - 1.0
    with torch.no_grad():
        return lpips_fn(orig_t.unsqueeze(0).cuda(), aug_t.unsqueeze(0).cuda()).item()


def compute_ssim(orig_pil, aug_pil):
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    orig = np.array(orig_pil.resize((256, 256)))
    aug = np.array(aug_pil.resize((256, 256)))
    return ssim(orig, aug, channel_axis=2)


def main():
    args = parse_args()
    cfg = IP2P_CONFIG[args.weather]

    import numpy as np
    import pyarrow as pa
    from PIL import Image
    import torch
    import lpips
    from physics import add_natural_rain, add_natural_snow

    print("=" * 60)
    print(f"NIGHT WEATHER BATCH WORKER: {args.weather.upper()}")
    print(f"Pipeline: Night → IP2P ({args.weather}) → Physics → Filter")
    print(f"Range: [{args.start}, {args.end})")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading arrow files...")
    night_table = load_arrow(args.night_input)
    orig_table = load_arrow(args.orig_input)
    total = len(night_table)
    start = max(0, args.start)
    end = min(args.end, total)
    batch_size = end - start

    if batch_size <= 0:
        print("No samples to process!")
        return

    print(f"  Night: {len(night_table)} images")
    print(f"  Original: {len(orig_table)} images")
    print(f"  Batch: [{start}, {end}) = {batch_size} images")

    # Build image_id lookup for original table (for SSIM/LPIPS against original)
    orig_ids = {orig_table.column('image_id')[i].as_py(): i
                for i in range(len(orig_table))}

    # Load IP2P model
    print("\n[2/4] Loading InstructPix2Pix model...")
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    print("  Model loaded on CUDA")

    # Load LPIPS if needed
    lpips_fn = None
    if cfg['use_lpips']:
        lpips_fn = lpips.LPIPS(net='alex').cuda()
        print("  LPIPS (AlexNet) loaded")

    # Process
    print(f"\n[3/4] Processing {batch_size} images...")
    print(f"  Prompt: \"{cfg['prompt']}\"")
    print(f"  guidance_scale={cfg['guidance_scale']}, image_guidance_scale={cfg['image_guidance_scale']}")
    print(f"  SSIM range: {cfg['ssim_range']}, LPIPS < {LPIPS_THRESHOLD if cfg['use_lpips'] else 'N/A'}")

    kept_images = []
    kept_ref_ids = []
    kept_indices = []
    all_meta = []

    for idx in range(start, end):
        image_id = night_table.column('image_id')[idx].as_py()
        night_img = get_image(night_table, idx)
        h, w = night_img.height, night_img.width

        # Get original for quality comparison
        orig_idx = orig_ids.get(image_id)
        if orig_idx is None:
            print(f"  Warning: no original for {image_id}, skipping")
            all_meta.append(f"{image_id},0.0,0.0,SKIP_NO_ORIG")
            continue
        orig_img = get_image(orig_table, orig_idx)

        # Resize for IP2P (max 768, divisible by 8)
        max_dim = 768
        scale = min(max_dim / max(h, w), 1.0)
        nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
        night_resized = night_img.resize((nw, nh), Image.LANCZOS)

        # IP2P inference
        random.seed(args.seed + idx)
        g = torch.Generator("cuda").manual_seed(args.seed + idx)
        result = pipe(
            cfg['prompt'], image=night_resized,
            num_inference_steps=cfg['num_inference_steps'],
            image_guidance_scale=cfg['image_guidance_scale'],
            guidance_scale=cfg['guidance_scale'],
            generator=g
        ).images[0]
        aug_np = np.array(result.resize((w, h), Image.LANCZOS))

        # Physics particle overlay
        if args.weather == 'rain':
            aug_np = add_natural_rain(aug_np)
        else:
            aug_np = add_natural_snow(aug_np)

        aug_pil = Image.fromarray(aug_np)

        # Quality metrics (compare against ORIGINAL, not night)
        ss = compute_ssim(orig_img, aug_pil)
        lp = compute_lpips(lpips_fn, orig_img, aug_pil) if lpips_fn else 0.0

        # Filter
        ssim_lo, ssim_hi = cfg['ssim_range']
        keep_ssim = ssim_lo <= ss <= ssim_hi
        keep_lpips = lp < LPIPS_THRESHOLD if cfg['use_lpips'] else True
        keep = keep_ssim and keep_lpips
        status = 'KEEP' if keep else 'DROP'

        processed = idx - start + 1
        if cfg['use_lpips']:
            print(f"  [{processed}/{batch_size}] {image_id} SSIM={ss:.3f} LPIPS={lp:.3f} {status}")
        else:
            print(f"  [{processed}/{batch_size}] {image_id} SSIM={ss:.3f} {status}")

        all_meta.append(f"{image_id},{ss:.4f},{lp:.4f},{status}")

        if keep:
            aug_bytes = image_to_bytes(aug_pil)
            kept_images.append(aug_bytes)
            kept_ref_ids.append(str(image_id))
            kept_indices.append(idx)

        del result, aug_np, aug_pil, orig_img, night_img
        gc.collect()
        torch.cuda.empty_cache()

    # Build output Arrow
    print(f"\n[4/4] Saving outputs...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if kept_indices:
        columns = {}
        for name in night_table.column_names:
            if name == 'image':
                columns[name] = kept_images
            else:
                columns[name] = [night_table.column(name)[i].as_py() for i in kept_indices]
        columns['ref_id'] = kept_ref_ids
        columns['condition'] = [f'night_{args.weather}'] * len(kept_indices)

        output_table = pa.table(columns)
        arrow_path = output_dir / f'batch_{start}-{end}.arrow'
        with pa.OSFile(str(arrow_path), 'wb') as f:
            writer = pa.ipc.new_stream(f, output_table.schema)
            writer.write_table(output_table)
            writer.close()
        print(f"  Arrow saved: {arrow_path} ({len(output_table)} kept)")
    else:
        print("  No images kept for this batch!")

    # Save metadata CSV
    csv_path = output_dir / f'meta_{start}-{end}.csv'
    with open(csv_path, 'w') as f:
        f.write("image_id,ssim,lpips,status\n")
        for line in all_meta:
            f.write(line + '\n')
    print(f"  Metadata: {csv_path}")

    kept_count = len(kept_indices)
    drop_count = batch_size - kept_count
    print(f"\n{'='*60}")
    print(f"Summary: {kept_count} KEEP, {drop_count} DROP out of {batch_size}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
