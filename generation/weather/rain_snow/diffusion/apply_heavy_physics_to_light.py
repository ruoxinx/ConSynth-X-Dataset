#!/usr/bin/env python3
"""
Production: Apply heavy physics (add_natural_rain intensity='heavy_fog')
on top of existing IP2P 'light' rain arrow files to produce 'heavy' variants.

No diffusion / GPU required — pure CPU physics overlay. Preserves all metadata
columns (image_id, bboxes, ssim/lpips/status from light stage) and only
substitutes the `image` bytes.

Usage:
    python apply_heavy_physics_to_light.py --input <light_arrow> --output <heavy_arrow>
    python apply_heavy_physics_to_light.py --input-dir <dir> --output-dir <dir>
"""

import argparse
import io
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from physics import add_natural_rain


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def image_to_bytes(img, quality=95):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def process_one(in_path: Path, out_path: Path, seed_base: int = 42):
    print(f"[{in_path.name}] loading...")
    table = load_arrow(in_path)
    n = len(table)

    new_image_structs = []
    for i in range(n):
        row = table.column('image')[i].as_py()
        src_bytes = row['bytes'] if isinstance(row, dict) else row
        src_path = row.get('path', '') if isinstance(row, dict) else ''

        img = Image.open(io.BytesIO(src_bytes)).convert('RGB')
        random.seed(seed_base + i)
        aug = add_natural_rain(np.array(img), intensity='heavy_fog')
        new_bytes = image_to_bytes(Image.fromarray(aug))
        new_image_structs.append({'bytes': new_bytes, 'path': src_path})

        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n}")

    columns = {}
    for name in table.column_names:
        if name == 'image':
            columns[name] = new_image_structs
        else:
            columns[name] = table.column(name).to_pylist()

    new_table = pa.table(columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(out_path), 'wb') as f:
        writer = pa.ipc.new_stream(f, new_table.schema)
        writer.write_table(new_table)
        writer.close()
    print(f"  wrote {out_path}  ({len(new_table)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, help='Single light arrow input')
    p.add_argument('--output', type=str, help='Single heavy arrow output')
    p.add_argument('--input-dir', type=str, help='Directory of light arrow files')
    p.add_argument('--output-dir', type=str, help='Directory for heavy arrow outputs')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.input and args.output:
        process_one(Path(args.input), Path(args.output), seed_base=args.seed)
        return

    if args.input_dir and args.output_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        arrows = sorted(in_dir.glob('*.arrow'))
        if not arrows:
            print(f"No arrow files in {in_dir}")
            return
        print(f"Found {len(arrows)} arrow files")
        for shard in arrows:
            process_one(shard, out_dir / shard.name, seed_base=args.seed)
        return

    p.error("Provide either --input/--output or --input-dir/--output-dir")


if __name__ == "__main__":
    main()
