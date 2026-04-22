#!/usr/bin/env python3
"""
Pack augmented JPG folder + original XML annotations into Arrow.
Schema matches soda_day2night.arrow: [image_id, image (binary), annotation (str), meta (str)].

Usage:
    python pack_soda_voc_augmented_jpgs.py \
        --image-dir augmentation_data/soda_voc/rain_snow/diffusion/rain \
        --annotation-dir ".../VOC2007/Annotations" \
        --output augmentation_data/soda_voc/rain_snow/diffusion/rain.arrow \
        --meta '{"weather":"rain","method":"diffusion"}'
"""

import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-dir', required=True)
    p.add_argument('--annotation-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--meta', default=None, help='Meta string applied to all rows')
    args = p.parse_args()

    img_dir = Path(args.image_dir)
    ann_dir = Path(args.annotation_dir)
    images = sorted(img_dir.glob('*.jpg'), key=lambda x: x.stem)
    print(f'Images: {len(images)} in {img_dir}')

    ids, blobs, anns, metas = [], [], [], []
    skipped = 0
    for i, ip in enumerate(images):
        xp = ann_dir / f'{ip.stem}.xml'
        if not xp.exists():
            skipped += 1
            continue
        with open(ip, 'rb') as f:
            blobs.append(f.read())
        with open(xp, 'r', encoding='utf-8') as f:
            anns.append(f.read())
        ids.append(ip.stem)
        metas.append(args.meta)
        if (i + 1) % 1000 == 0:
            print(f'  [{i+1}/{len(images)}]')

    schema = pa.schema([
        pa.field('image_id', pa.string()),
        pa.field('image', pa.binary()),
        pa.field('annotation', pa.string()),
        pa.field('meta', pa.string()),
    ])
    table = pa.table({
        'image_id': ids,
        'image': blobs,
        'annotation': anns,
        'meta': metas,
    }, schema=schema)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_stream(str(out), table.schema) as w:
        w.write_table(table)
    print(f'Saved {out}  rows={len(table)}  skipped={skipped}')


if __name__ == '__main__':
    main()
