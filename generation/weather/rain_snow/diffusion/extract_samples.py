#!/usr/bin/env python3
"""
Extract 5 sample comparisons: Original | Rain | Snow
Saves JPG files to output_samples/comparison/
"""

import sys
import pyarrow as pa
from PIL import Image
import io
import random
from pathlib import Path

_GEN_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '_paths.py').exists())
sys.path.insert(0, str(_GEN_ROOT))
from _paths import DATA_ROOT as BASE_DIR
RAIN_DIR = BASE_DIR / 'output' / 'construction_site_test' / 'diffusion_rain_heavy'
SNOW_DIR = BASE_DIR / 'output' / 'construction_site_test' / 'diffusion_snow_heavy'
OUT_DIR = BASE_DIR / 'weather_aug' / 'new_method' / 'output_samples' / 'comparison'


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    return Image.open(io.BytesIO(row['bytes'])).convert('RGB')


def build_lookup(batch_dir):
    lookup = {}
    for bf in sorted(batch_dir.glob('batch_*.arrow')):
        t = load_arrow(bf)
        for i in range(len(t)):
            iid = t.column('image_id')[i].as_py()
            lookup[iid] = (bf, i)
    return lookup


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading original dataset...")
    orig_table = load_arrow(BASE_DIR / 'LouisChen15___construction_site' / 'construction_site-test.arrow')
    print(f"  {len(orig_table)} samples")

    print("Building rain lookup...")
    rain_lookup = build_lookup(RAIN_DIR)
    print(f"  {len(rain_lookup)} images")

    print("Building snow lookup...")
    snow_lookup = build_lookup(SNOW_DIR)
    print(f"  {len(snow_lookup)} images")

    random.seed(99)
    indices = random.sample(range(len(orig_table)), 5)
    indices.sort()

    for i, idx in enumerate(indices):
        image_id = orig_table.column('image_id')[idx].as_py()
        print(f"\n[{i+1}/5] image_id={image_id} (idx={idx})")

        orig = get_image(orig_table, idx)
        orig.save(OUT_DIR / f"{i+1}_{image_id}_original.jpg", quality=95)

        if image_id in rain_lookup:
            bf, li = rain_lookup[image_id]
            rain = get_image(load_arrow(bf), li)
            rain.save(OUT_DIR / f"{i+1}_{image_id}_rain_heavy.jpg", quality=95)

        if image_id in snow_lookup:
            bf, li = snow_lookup[image_id]
            snow = get_image(load_arrow(bf), li)
            snow.save(OUT_DIR / f"{i+1}_{image_id}_snow_heavy.jpg", quality=95)

        print(f"  Saved: original, rain, snow")

    print(f"\nDone! Files at: {OUT_DIR}")


if __name__ == '__main__':
    main()
