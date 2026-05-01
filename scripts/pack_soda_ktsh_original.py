#!/usr/bin/env python3
"""Pack soda-ktsh raw JPGs into Arrow with the SODA VOC fog/night schema.

Schema: [image_id, image (binary), annotation (str), meta (str)].
ktsh has no detection XMLs — annotation is left empty.
"""
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--captions-arrow', default=None,
                   help='Optional ktsh arrow with image_id->captions to join (e.g., existing rain_snow arrow).')
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    cap_map = {}
    if args.captions_arrow:
        src = pa.memory_map(args.captions_arrow, 'r')
        r = ipc.open_stream(src)
        for b in r:
            ids_arr = b.column('image_id').to_pylist()
            cap_arr = b.column('captions').to_pylist()
            cap_map.update(dict(zip(ids_arr, cap_arr)))
        print(f'Loaded {len(cap_map)} caption entries from {args.captions_arrow}')

    images = sorted(Path(args.images_dir).glob('*.jpg'), key=lambda x: x.stem)
    if args.limit:
        images = images[:args.limit]
    print(f'Found {len(images)} images')

    ids, blobs, anns, metas, caps = [], [], [], [], []
    missing_cap = 0
    for i, img_path in enumerate(images):
        with open(img_path, 'rb') as f:
            blobs.append(f.read())
        stem = img_path.stem
        ids.append(stem)
        anns.append('')
        metas.append(None)
        c = cap_map.get(stem, [])
        if not c:
            missing_cap += 1
        caps.append(c)
        if (i + 1) % 1000 == 0:
            print(f'  [{i+1}/{len(images)}]')

    if cap_map:
        print(f'Captions: {len(images) - missing_cap}/{len(images)} matched, {missing_cap} missing')

    schema_fields = [
        pa.field('image_id', pa.string()),
        pa.field('image', pa.binary()),
        pa.field('annotation', pa.string()),
        pa.field('meta', pa.string()),
    ]
    cols = {'image_id': ids, 'image': blobs, 'annotation': anns, 'meta': metas}
    if cap_map:
        schema_fields.append(pa.field('captions', pa.list_(pa.string())))
        cols['captions'] = caps
    schema = pa.schema(schema_fields)
    table = pa.table(cols, schema=schema)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_stream(str(out), table.schema) as w:
        w.write_table(table)
    print(f'Saved {out}  rows={len(table)}  size={out.stat().st_size/1e9:.2f}GB')


if __name__ == '__main__':
    main()
