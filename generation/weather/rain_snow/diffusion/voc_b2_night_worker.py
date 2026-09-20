#!/usr/bin/env python3
"""
VOC B2 Night-from-Weather Worker.

B2 variant of Order B: reuse pre-computed IP2P weather images (from voc rain /
snow arrows), apply only CycleGAN day2night + physics. Skips the IP2P step.

Pipeline (per image):
  Weather arrow image (already IP2P'd day rain/snow)
    -> CycleGAN-Turbo day_to_night
    -> Physics particle overlay (rain streaks or snowflakes)
    -> SSIM / LPIPS metrics vs original (always KEEP, no filter)
    -> Output Arrow + metadata CSV

Inputs:
  --weather-input  voc rain or snow arrow (image_id, image, ..., 10 cols)
  --orig-input     voc clear arrow at SAME indices (image_id, image, annotation, meta)
  --output-dir     where to write batch arrow + meta CSV
  --start --end    range of indices to process
  --weather        rain | snow

Output: batch_<start>-<end>.arrow with cols: image_id, image, ssim, lpips, status, ref_id
"""

import argparse
import sys
import gc
import io
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

IMG2IMG_SRC = SCRIPT_DIR.parents[2] / 'day2night' / 'img2img-turbo' / 'src'
sys.path.insert(0, str(IMG2IMG_SRC))


def parse_args():
    p = argparse.ArgumentParser(description='VOC B2 Night-from-Weather Worker')
    p.add_argument('--weather-input', type=str, required=True,
                   help='Path to voc weather arrow (rain/snow sliced)')
    p.add_argument('--orig-input', type=str, required=True,
                   help='Path to voc clear arrow at SAME image_id ordering')
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--weather', type=str, required=True, choices=['rain', 'snow'])
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


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
    o = torch.from_numpy(np.array(orig_pil.resize(size))).permute(2, 0, 1).float() / 127.5 - 1.0
    a = torch.from_numpy(np.array(aug_pil.resize(size))).permute(2, 0, 1).float() / 127.5 - 1.0
    with torch.no_grad():
        return lpips_fn(o.unsqueeze(0).cuda(), a.unsqueeze(0).cuda()).item()


def compute_ssim(orig_pil, aug_pil):
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    o = np.array(orig_pil.resize((256, 256)))
    a = np.array(aug_pil.resize((256, 256)))
    return ssim(o, a, channel_axis=2)


def run_cyclegan_night(model, image, T_val):
    import torch
    from PIL import Image
    from torchvision import transforms
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

    import numpy as np
    import pyarrow as pa
    from PIL import Image
    import torch
    import lpips
    from physics import add_natural_rain, add_natural_snow
    from cyclegan_turbo import CycleGAN_Turbo
    from my_utils.training_utils import build_transform

    print('=' * 60)
    print(f'VOC B2 NIGHT-FROM-WEATHER WORKER: {args.weather.upper()}')
    print(f'Pipeline: WeatherArrow -> CycleGAN Night -> Physics ({args.weather})')
    print(f'Range: [{args.start}, {args.end})')
    print('=' * 60)

    print('\n[1/4] Loading arrows...')
    weather_table = load_arrow(args.weather_input)
    orig_table = load_arrow(args.orig_input)
    if len(weather_table) != len(orig_table):
        raise ValueError(f'weather rows ({len(weather_table)}) != orig rows ({len(orig_table)})')
    total = len(weather_table)
    start = max(0, args.start)
    end = min(args.end, total)
    batch_size = end - start
    if batch_size <= 0:
        print('No samples to process.')
        return
    print(f'  Weather: {total} | Orig: {len(orig_table)} | Batch: [{start}, {end}) = {batch_size}')

    print('\n[2/4] Loading models...')
    cyclegan = CycleGAN_Turbo(pretrained_name='day_to_night')
    cyclegan.eval()
    try:
        cyclegan.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    cyclegan.half()
    T_val = build_transform('resize_512x512')
    print('  CycleGAN-Turbo (day_to_night) loaded')

    lpips_fn = lpips.LPIPS(net='alex').cuda()
    print('  LPIPS (AlexNet) loaded')

    print(f'\n[3/4] Processing {batch_size} images (no filter, all KEEP)...')

    kept_images = []
    kept_ref_ids = []
    kept_ssim = []
    kept_lpips = []
    all_meta = []

    for idx in range(start, end):
        wid = weather_table.column('image_id')[idx].as_py()
        oid = orig_table.column('image_id')[idx].as_py()
        if wid != oid:
            raise ValueError(f'idx {idx}: weather_id {wid} != orig_id {oid} (inputs misaligned)')

        weather_img = get_image(weather_table, idx)
        orig_img = get_image(orig_table, idx)

        night_img = run_cyclegan_night(cyclegan, weather_img, T_val)

        random.seed(args.seed + idx)
        arr = np.array(night_img)
        if args.weather == 'rain':
            arr = add_natural_rain(arr)
        else:
            arr = add_natural_snow(arr)
        aug_pil = Image.fromarray(arr)

        ss = compute_ssim(orig_img, aug_pil)
        lp = compute_lpips(lpips_fn, orig_img, aug_pil)
        status = 'KEEP'

        processed = idx - start + 1
        print(f'  [{processed}/{batch_size}] {wid} SSIM={ss:.3f} LPIPS={lp:.3f} {status}')
        all_meta.append(f'{wid},{ss:.4f},{lp:.4f},{status}')

        kept_images.append(image_to_bytes(aug_pil))
        kept_ref_ids.append(str(wid))
        kept_ssim.append(ss)
        kept_lpips.append(lp)

        del weather_img, night_img, arr, aug_pil, orig_img
        gc.collect()
        torch.cuda.empty_cache()

    print('\n[4/4] Saving outputs...')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_table = pa.table({
        'image_id': kept_ref_ids,
        'image':    kept_images,
        'ssim':     kept_ssim,
        'lpips':    kept_lpips,
        'status':   ['KEEP'] * len(kept_ref_ids),
        'ref_id':   kept_ref_ids,
    })
    arrow_path = output_dir / f'batch_{start}-{end}.arrow'
    with pa.OSFile(str(arrow_path), 'wb') as f:
        w = pa.ipc.new_stream(f, out_table.schema)
        w.write_table(out_table)
        w.close()
    print(f'  Arrow saved: {arrow_path} ({len(out_table)} rows)')

    csv_path = output_dir / f'meta_{start}-{end}.csv'
    with open(csv_path, 'w') as f:
        f.write('image_id,ssim,lpips,status\n')
        for line in all_meta:
            f.write(line + '\n')
    print(f'  Metadata: {csv_path}')

    print(f'\n{"="*60}')
    print(f'Summary: {len(out_table)} processed (all KEEP, no filter)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
