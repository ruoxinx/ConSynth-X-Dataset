#!/usr/bin/env python3
"""
Apply heavy_fog rain physics on a folder of light-rain JPGs.
Pure CPU. Mirrors `apply_heavy_physics_to_light.py` but folder-of-JPGs in/out.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / 'generation/weather/rain_snow/diffusion'))
from physics import add_natural_rain  # noqa: E402

import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg'))
    print(f'Heavy-rain physics: {len(jpgs)} jpgs from {in_dir}')

    for i, p in enumerate(jpgs):
        random.seed(args.seed + i)
        np.random.seed(args.seed + i)
        img = np.array(Image.open(p).convert('RGB'))
        heavy = add_natural_rain(img, intensity='heavy_fog')
        Image.fromarray(heavy).save(out_dir / p.name, quality=95)
        if (i + 1) % 20 == 0:
            print(f'  [{i + 1}/{len(jpgs)}]')
    print(f'Done -> {out_dir}')


if __name__ == '__main__':
    main()
