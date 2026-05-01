"""Finalize ktsh outputs after generation jobs complete:
  1. Concat ktsh fog sub-batches per intensity -> 3 final arrows
  2. Pack ktsh night JPGs -> arrow (joins captions from snow_light.arrow)
  3. Pack ktsh small JPGs+annotation JSONs -> arrow (captions from JSONs)
"""
import argparse, io, json, sys
from pathlib import Path
from PIL import Image
import pyarrow as pa
import pyarrow.ipc as ipc


def load_caption_map(arrow_path):
    src = pa.memory_map(str(arrow_path), 'r')
    r = ipc.open_stream(src)
    out = {}
    for b in r:
        for i, c in zip(b.column('image_id').to_pylist(), b.column('captions').to_pylist()):
            out.setdefault(i, c)
    return out


def concat_fog(root: Path):
    base = root / 'augmentation_data/soda_ktsh/fog/test'
    for level in ['light', 'medium', 'heavy']:
        sub = base / level
        parts = sorted(sub.glob(f'fog_{level}_*.arrow'))
        if not parts:
            print(f'[fog/{level}] no parts; skip'); continue
        # Use first part's schema (all should match).
        tables = []
        for p in parts:
            src = pa.memory_map(str(p), 'r')
            r = ipc.open_stream(src)
            tables.append(r.read_all())
        out_table = pa.concat_tables(tables)
        out_path = base.parent / f'fog_{level}.arrow'
        with pa.OSFile(str(out_path), 'wb') as sink:
            with ipc.new_stream(sink, out_table.schema) as w:
                for b in out_table.to_batches():
                    w.write_batch(b)
        print(f'[fog/{level}] {len(parts)} parts -> {out_path.name}  rows={out_table.num_rows}')


def pack_jpg_dir_with_captions(images_dir: Path, output_path: Path, caption_map: dict, weather_label: str = ''):
    images = sorted(images_dir.glob('*.jpg'), key=lambda x: x.stem)
    print(f'  {images_dir}: {len(images)} imgs')
    ids, blobs, caps, filenames, weathers = [], [], [], [], []
    missing = 0
    for i, p in enumerate(images):
        with open(p, 'rb') as f:
            blobs.append(f.read())
        stem = p.stem
        ids.append(stem)
        filenames.append(p.name)
        c = caption_map.get(stem, [])
        if not c: missing += 1
        caps.append(c)
        weathers.append(weather_label)
        if (i + 1) % 500 == 0: print(f'    [{i+1}/{len(images)}]')
    print(f'  captions matched {len(images)-missing}/{len(images)}')

    schema = pa.schema([
        pa.field('image', pa.binary()),
        pa.field('image_id', pa.string()),
        pa.field('filename', pa.string()),
        pa.field('captions', pa.list_(pa.string())),
        pa.field('weather', pa.string()),
    ])
    table = pa.table({'image': blobs, 'image_id': ids, 'filename': filenames,
                      'captions': caps, 'weather': weathers}, schema=schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(output_path), 'wb') as sink:
        with ipc.new_stream(sink, schema) as w:
            for b in table.to_batches():
                w.write_batch(b)
    print(f'  -> {output_path}  rows={table.num_rows}')


def pack_ktsh_small(root: Path, caption_map: dict):
    src_dir = root / 'augmentation_data/soda_ktsh/small/_jpgs'
    images_dir = src_dir / 'images'
    annot_dir = src_dir / 'annotations'
    output = root / 'augmentation_data/soda_ktsh/small/soda_ktsh_small.arrow'

    images = sorted(images_dir.glob('*.jpg'), key=lambda x: x.stem)
    print(f'  ktsh small: {len(images)} imgs')
    ids, blobs, caps, filenames, weathers = [], [], [], [], []
    no_caption = 0
    for i, p in enumerate(images):
        with open(p, 'rb') as f:
            blobs.append(f.read())
        stem = p.stem
        ids.append(stem)
        filenames.append(p.name)
        ann_path = annot_dir / f'{stem}.json'
        captions = []
        if ann_path.exists():
            with open(ann_path) as fa:
                d = json.load(fa)
            captions = d.get('captions') or []
        if not captions:
            captions = caption_map.get(stem, [])
        if not captions: no_caption += 1
        caps.append(captions)
        weathers.append('outpaint')
        if (i + 1) % 500 == 0: print(f'    [{i+1}/{len(images)}]')
    print(f'  captions matched {len(images)-no_caption}/{len(images)}')

    schema = pa.schema([
        pa.field('image', pa.binary()),
        pa.field('image_id', pa.string()),
        pa.field('filename', pa.string()),
        pa.field('captions', pa.list_(pa.string())),
        pa.field('weather', pa.string()),
    ])
    table = pa.table({'image': blobs, 'image_id': ids, 'filename': filenames,
                      'captions': caps, 'weather': weathers}, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(output), 'wb') as sink:
        with ipc.new_stream(sink, schema) as w:
            for b in table.to_batches():
                w.write_batch(b)
    print(f'  -> {output}  rows={table.num_rows}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/users/PGS0407/binben14/VietHuy/ConSynth-X')
    ap.add_argument('--steps', nargs='+', default=['fog', 'night', 'small'])
    args = ap.parse_args()
    root = Path(args.root)

    if 'fog' in args.steps:
        print('=== concat ktsh fog ===')
        concat_fog(root)

    if 'night' in args.steps or 'small' in args.steps:
        print('Loading caption map from snow_light.arrow ...')
        cap = load_caption_map(root / 'augmentation_data/soda_ktsh/rain_snow/diffusion/snow_light.arrow')
        print(f'  {len(cap)} caption entries')

        if 'night' in args.steps:
            print('=== pack ktsh night ===')
            pack_jpg_dir_with_captions(
                root / 'augmentation_data/soda_ktsh/night/_jpgs/images',
                root / 'augmentation_data/soda_ktsh/night/soda_ktsh_day2night.arrow',
                cap, weather_label='night')

        if 'small' in args.steps:
            print('=== pack ktsh small ===')
            pack_ktsh_small(root, cap)


if __name__ == '__main__':
    main()
