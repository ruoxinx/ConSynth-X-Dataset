#!/usr/bin/env python3
"""
Pack original SODA VOC JPEGs + XML annotations into a single Arrow file.
Output schema matches soda_day2night.arrow: [image_id, image (binary), annotation (str), meta (str)].

Usage:
    python pack_soda_voc_original.py \
        --voc-root "$CONSYNTH_DATA_ROOT/SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007" \
        --output ../../../../augmentation_data/soda_voc/original/soda_voc_original_first3000.arrow \
        --limit 3000
"""

import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--voc-root', required=True, help='VOC2007 root containing JPEGImages/ and Annotations/')
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=None, help='Take only the first N images (sorted by filename)')
    args = p.parse_args()

    voc_root = Path(args.voc_root)
    jpg_dir = voc_root / 'JPEGImages'
    xml_dir = voc_root / 'Annotations'

    images = sorted(jpg_dir.glob('*.jpg'), key=lambda x: x.stem)
    if args.limit:
        images = images[:args.limit]

    print(f'Found {len(images)} images under {jpg_dir}')
    ids, blobs, anns = [], [], []
    skipped = 0

    for i, img_path in enumerate(images):
        xml_path = xml_dir / f'{img_path.stem}.xml'
        if not xml_path.exists():
            skipped += 1
            continue
        with open(img_path, 'rb') as f:
            blob = f.read()
        with open(xml_path, 'r', encoding='utf-8') as f:
            ann = f.read()
        ids.append(img_path.stem)
        blobs.append(blob)
        anns.append(ann)
        if (i + 1) % 500 == 0:
            print(f'  [{i+1}/{len(images)}] packed')

    metas = [None] * len(ids)

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
