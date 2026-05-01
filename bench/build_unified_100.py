#!/usr/bin/env python3
"""
Build a unified 100-image source set for cross-condition zero-shot detection bench.

Picks 100 VOC image_ids deterministically (seed=42) that have at least one
person GT, materialises:

  detection_validation/source_100/
    clear/{id}.jpg              # 100 JPGs (input for IP2P + day2night workers)
    clear.arrow                 # 4-col SODA-VOC schema (input for fog/night_weather)
    manifest.json               # ids + COCO-style GT (person-only)

Run on login node (CPU, no SLURM needed).
"""

import io
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

REPO = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')
VOC = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007')
OUT = REPO / 'detection_validation/source_100'
N = 100
SEED = 42


def parse_voc_persons(image_id: str):
    xml = VOC / 'Annotations' / f'{image_id}.xml'
    if not xml.exists():
        return None, None, None
    tree = ET.parse(xml)
    root = tree.getroot()
    size = root.find('size')
    W = int(size.find('width').text)
    H = int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        if obj.find('name').text.strip() == 'person':
            bb = obj.find('bndbox')
            x1 = float(bb.find('xmin').text)
            y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text)
            y2 = float(bb.find('ymax').text)
            boxes.append([x1, y1, x2, y2])
    return boxes, W, H


def select_ids():
    """Sample N image_ids from VOC ImageSets/Main/test.txt with person GT."""
    all_ids = sorted([p.stem for p in (VOC / 'JPEGImages').glob('*.jpg')])
    print(f'  pool: {len(all_ids)} VOC image_ids')

    rng = random.Random(SEED)
    rng.shuffle(all_ids)
    selected = []
    for iid in all_ids:
        boxes, _, _ = parse_voc_persons(iid)
        if boxes:
            selected.append(iid)
            if len(selected) >= N:
                break
    selected.sort()
    print(f'  selected: {len(selected)} ids with person GT')
    return selected


def write_jpg_folder(ids):
    folder = OUT / 'clear'
    folder.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        src = VOC / 'JPEGImages' / f'{iid}.jpg'
        dst = folder / f'{iid}.jpg'
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f'  wrote {len(ids)} JPGs to {folder}')


def write_arrow(ids):
    """Write SODA-VOC schema arrow: image_id(str), image(binary), annotation(str), meta(str)."""
    schema = pa.schema([
        pa.field('image_id', pa.string()),
        pa.field('image', pa.binary()),
        pa.field('annotation', pa.string()),
        pa.field('meta', pa.string()),
    ])

    rows = []
    for iid in ids:
        jpg = (VOC / 'JPEGImages' / f'{iid}.jpg').read_bytes()
        ann_xml = (VOC / 'Annotations' / f'{iid}.xml').read_text() if (VOC / 'Annotations' / f'{iid}.xml').exists() else ''
        rows.append((iid, jpg, ann_xml, ''))

    table = pa.table({
        'image_id':   pa.array([r[0] for r in rows], type=pa.string()),
        'image':      pa.array([r[1] for r in rows], type=pa.binary()),
        'annotation': pa.array([r[2] for r in rows], type=pa.string()),
        'meta':       pa.array([r[3] for r in rows], type=pa.string()),
    }, schema=schema)
    out = OUT / 'clear.arrow'
    with ipc.new_stream(str(out), schema) as w:
        w.write_table(table)
    print(f'  wrote arrow: {out} ({len(rows)} rows, schema={[f.name for f in schema]})')


def write_manifest(ids):
    coco_images, coco_anns = [], []
    ann_id = 1
    id2int = {iid: i + 1 for i, iid in enumerate(ids)}
    for iid in ids:
        boxes, W, H = parse_voc_persons(iid)
        coco_images.append({'id': id2int[iid], 'image_id': iid, 'file_name': f'{iid}.jpg', 'width': W, 'height': H})
        for b in boxes:
            x1, y1, x2, y2 = b
            coco_anns.append({
                'id': ann_id, 'image_id': id2int[iid], 'category_id': 1,
                'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0,
            })
            ann_id += 1
    manifest = {
        'n': len(ids), 'seed': SEED,
        'image_ids': ids,
        'coco_gt': {
            'images': coco_images, 'annotations': coco_anns,
            'categories': [{'id': 1, 'name': 'person'}],
        },
    }
    out = OUT / 'manifest.json'
    with open(out, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'  wrote manifest: {out} ({len(coco_anns)} person GT boxes)')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('Selecting unified 100-img set...')
    ids = select_ids()
    print('\nMaterialising JPG folder...');  write_jpg_folder(ids)
    print('\nMaterialising Arrow...');        write_arrow(ids)
    print('\nWriting GT manifest...');        write_manifest(ids)
    print(f'\nDone. Source set ready at {OUT}')


if __name__ == '__main__':
    main()
