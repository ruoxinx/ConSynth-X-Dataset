#!/usr/bin/env python3
"""
Weather-Night Batch Worker (Order B).

Pipeline (per image):
  Original day image
    -> IP2P diffusion (production day rain/snow prompt)
    -> CycleGAN-Turbo day_to_night
    -> Physics particle overlay (rain streaks or snowflakes)
    -> SSIM / LPIPS quality filter (vs. original)
    -> Output Arrow + metadata CSV

Picked over Order A (Night -> IP2P) by visual inspection: applying night to an
already-augmented daytime weather image preserves IP2P texture cues better.

IP2P config matches production rain.arrow / snow.arrow (batch_worker.py defaults
+ jobs/regen_rain_v5.sh): the "light" / single tier of the dataset.

Usage:
    python weather_night_batch_worker.py \
        --orig-input construction_site_test.arrow \
        --output-dir output/rain_night \
        --start 0 --end 500 \
        --weather rain --seed 42
"""

import argparse
import sys
import gc
import io
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from physics import add_natural_rain, add_natural_snow

# CycleGAN-Turbo (img2img-turbo)
IMG2IMG_SRC = SCRIPT_DIR.parents[2] / 'day2night' / 'img2img-turbo' / 'src'
sys.path.insert(0, str(IMG2IMG_SRC))
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from torchvision import transforms

# ---------- Quality filter thresholds (mirror Order A worker) ----------
LPIPS_THRESHOLD = 0.35
SSIM_RAIN = (0.6, 0.95)
SSIM_SNOW = (0.5, 0.95)

# ---------- Production day prompts (same source-of-truth as rain.arrow / snow.arrow) ----------
IP2P_CONFIG = {
    'rain': {
        'prompt': 'a rainy day with dark overcast sky, rain falling, grey clouds',
        'image_guidance_scale': 1.5,
        'guidance_scale': 10.0,
        'num_inference_steps': 30,
        'ssim_range': SSIM_RAIN,
        'use_lpips': True,
    },
    'snow': {
        'prompt': 'a cold winter day with snow, frost on surfaces, grey sky, snow on the ground',
        'image_guidance_scale': 1.5,
        'guidance_scale': 8.0,
        'num_inference_steps': 30,
        'ssim_range': SSIM_SNOW,
        'use_lpips': False,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='Weather-Night Batch Worker (Order B)')
    p.add_argument('--orig-input', type=str, required=True,
                   help='Path to original day arrow (e.g. construction_site_test.arrow)')
    p.add_argument('--output-dir', type=str, required=True,
                   help='Output directory for Arrow + CSV')
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--weather', type=str, required=True, choices=['rain', 'snow'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-filter', action='store_true',
                   help='Skip SSIM/LPIPS filtering — keep every generated image. '
                        'Order B compounds IP2P + CycleGAN Night, so SSIM-vs-original '
                        'drops well below the Order A thresholds; default thresholds '
                        'would discard most samples.')
    return p.parse_args()


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(io.BytesIO(row['bytes'])).convert('RGB')
    return Image.open(io.BytesIO(row)).convert('RGB')


def image_to_bytes(img, quality=95):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def compute_lpips(lpips_fn, orig_pil, aug_pil):
    size = (256, 256)
    o = torch.from_numpy(np.array(orig_pil.resize(size))).permute(2, 0, 1).float() / 127.5 - 1.0
    a = torch.from_numpy(np.array(aug_pil.resize(size))).permute(2, 0, 1).float() / 127.5 - 1.0
    with torch.no_grad():
        return lpips_fn(o.unsqueeze(0).cuda(), a.unsqueeze(0).cuda()).item()


def compute_ssim(orig_pil, aug_pil):
    o = np.array(orig_pil.resize((256, 256)))
    a = np.array(aug_pil.resize((256, 256)))
    return ssim(o, a, channel_axis=2)


def run_ip2p(pipe, image, cfg, seed):
    w, h = image.size
    max_dim = 768
    scale = min(max_dim / max(h, w), 1.0)
    nw, nh = (int(w * scale) // 8 * 8, int(h * scale) // 8 * 8)
    img_resized = image.resize((nw, nh), Image.LANCZOS)
    g = torch.Generator('cuda').manual_seed(seed)
    out = pipe(
        cfg['prompt'], image=img_resized,
        num_inference_steps=cfg['num_inference_steps'],
        image_guidance_scale=cfg['image_guidance_scale'],
        guidance_scale=cfg['guidance_scale'],
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


def main():
    args = parse_args()
    cfg = IP2P_CONFIG[args.weather]

    print('=' * 60)
    print(f'WEATHER-NIGHT BATCH WORKER (Order B): {args.weather.upper()}')
    print(f'Pipeline: Original -> IP2P ({args.weather}) -> CycleGAN Night -> Physics -> Filter')
    print(f'Range: [{args.start}, {args.end})')
    print('=' * 60)

    print('\n[1/4] Loading arrows...')
    orig_table = load_arrow(args.orig_input)
    total = len(orig_table)
    start = max(0, args.start)
    end = min(args.end, total)
    batch_size = end - start
    if batch_size <= 0:
        print('No samples to process.')
        return
    print(f'  Original: {total} | Batch: [{start}, {end}) = {batch_size}')

    print('\n[2/4] Loading models...')
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', torch_dtype=torch.float16, safety_checker=None)
    pipe.to('cuda')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    print('  IP2P loaded')

    cyclegan = CycleGAN_Turbo(pretrained_name='day_to_night')
    cyclegan.eval()
    try:
        cyclegan.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    cyclegan.half()
    T_val = build_transform('resize_512x512')
    print('  CycleGAN-Turbo (day_to_night) loaded')

    lpips_fn = lpips.LPIPS(net='alex').cuda() if cfg['use_lpips'] else None
    if lpips_fn is not None:
        print('  LPIPS (AlexNet) loaded')

    print(f"\n[3/4] Processing {batch_size} images...")
    print(f"  Prompt: \"{cfg['prompt']}\"")
    print(f"  guidance={cfg['guidance_scale']}, igs={cfg['image_guidance_scale']}")
    print(f"  SSIM range: {cfg['ssim_range']}, LPIPS < {LPIPS_THRESHOLD if cfg['use_lpips'] else 'N/A'}")

    kept_images = []
    kept_ref_ids = []
    kept_indices = []
    all_meta = []

    for idx in range(start, end):
        image_id = orig_table.column('image_id')[idx].as_py()
        orig_img = get_image(orig_table, idx)
        h, w = orig_img.height, orig_img.width

        # 1. IP2P day weather
        ip2p_img = run_ip2p(pipe, orig_img, cfg, args.seed + idx)

        # 2. CycleGAN day_to_night
        night_img = run_cyclegan_night(cyclegan, ip2p_img, T_val)

        # 3. Physics overlay
        random.seed(args.seed + idx)
        arr = np.array(night_img)
        if args.weather == 'rain':
            arr = add_natural_rain(arr)
        else:
            arr = add_natural_snow(arr)
        aug_pil = Image.fromarray(arr)

        # 4. Quality metrics vs original
        ss = compute_ssim(orig_img, aug_pil)
        lp = compute_lpips(lpips_fn, orig_img, aug_pil) if lpips_fn else 0.0

        if args.no_filter:
            keep = True
        else:
            ssim_lo, ssim_hi = cfg['ssim_range']
            keep = (ssim_lo <= ss <= ssim_hi) and (lp < LPIPS_THRESHOLD if cfg['use_lpips'] else True)
        status = 'KEEP' if keep else 'DROP'

        processed = idx - start + 1
        if cfg['use_lpips']:
            print(f'  [{processed}/{batch_size}] {image_id} SSIM={ss:.3f} LPIPS={lp:.3f} {status}')
        else:
            print(f'  [{processed}/{batch_size}] {image_id} SSIM={ss:.3f} {status}')

        all_meta.append(f'{image_id},{ss:.4f},{lp:.4f},{status}')

        if keep:
            kept_images.append(image_to_bytes(aug_pil))
            kept_ref_ids.append(str(image_id))
            kept_indices.append(idx)

        del ip2p_img, night_img, arr, aug_pil, orig_img
        gc.collect()
        torch.cuda.empty_cache()

    print('\n[4/4] Saving outputs...')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if kept_indices:
        columns = {}
        for name in orig_table.column_names:
            if name == 'image':
                columns[name] = kept_images
            else:
                columns[name] = [orig_table.column(name)[i].as_py() for i in kept_indices]
        columns['ref_id'] = kept_ref_ids
        columns['condition'] = [f'night_{args.weather}'] * len(kept_indices)

        out_table = pa.table(columns)
        arrow_path = output_dir / f'batch_{start}-{end}.arrow'
        with pa.OSFile(str(arrow_path), 'wb') as f:
            w = pa.ipc.new_stream(f, out_table.schema)
            w.write_table(out_table)
            w.close()
        print(f'  Arrow saved: {arrow_path} ({len(out_table)} kept)')
    else:
        print('  No images kept for this batch.')

    csv_path = output_dir / f'meta_{start}-{end}.csv'
    with open(csv_path, 'w') as f:
        f.write('image_id,ssim,lpips,status\n')
        for line in all_meta:
            f.write(line + '\n')
    print(f'  Metadata: {csv_path}')

    kept_count = len(kept_indices)
    drop_count = batch_size - kept_count
    print(f"\n{'='*60}")
    print(f'Summary: {kept_count} KEEP, {drop_count} DROP out of {batch_size}')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
