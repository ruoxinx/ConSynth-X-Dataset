#!/usr/bin/env python3
"""
Build COCO-format person-only GT JSON from SODA-VOC2007 XML annotations.

Output: bench/gt/voc_person_gt.json
Each VOC image_id mapped to integer COCO image_id (deterministic by sorted order).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

VOC = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007')
OUT_DIR = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X/bench/gt')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_one(image_id):
    xml = VOC / 'Annotations' / f'{image_id}.xml'
    tree = ET.parse(xml)
    root = tree.getroot()
    size = root.find('size')
    W = int(size.find('width').text)
    H = int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        if obj.find('name').text.strip() == 'person':
            bb = obj.find('bndbox')
            x1 = float(bb.find('xmin').text); y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text); y2 = float(bb.find('ymax').text)
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # COCO format xywh
    return W, H, boxes


def main():
    ids = sorted(p.stem for p in VOC.glob('Annotations/*.xml'))
    print(f'Found {len(ids)} VOC XML annotations')

    images, annotations = [], []
    iid_to_int = {}
    ann_id = 1
    n_no_person = 0
    n_person_total = 0

    for int_id, vid in enumerate(ids, start=1):
        try:
            W, H, boxes = parse_one(vid)
        except Exception as e:
            print(f'  ERROR parse {vid}: {e}')
            continue
        iid_to_int[vid] = int_id
        images.append({'id': int_id, 'file_name': f'{vid}.jpg', 'width': W, 'height': H})
        if not boxes:
            n_no_person += 1
        for b in boxes:
            annotations.append({
                'id': ann_id, 'image_id': int_id, 'category_id': 1,
                'bbox': b, 'area': b[2] * b[3], 'iscrowd': 0,
            })
            ann_id += 1
            n_person_total += 1
        if int_id % 5000 == 0:
            print(f'  processed {int_id}/{len(ids)}')

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'person'}],
    }

    out_json = OUT_DIR / 'voc_person_gt.json'
    out_json.write_text(json.dumps(coco))
    out_map = OUT_DIR / 'voc_id_to_int.json'
    out_map.write_text(json.dumps(iid_to_int))

    print(f'\nimages:       {len(images)}')
    print(f'with person:  {len(images) - n_no_person}')
    print(f'no person:    {n_no_person}')
    print(f'person boxes: {n_person_total}')
    print(f'avg boxes/img with person: {n_person_total/max(1, len(images)-n_no_person):.2f}')
    print(f'\nsaved: {out_json} ({out_json.stat().st_size/1e6:.1f} MB)')
    print(f'       {out_map}')


if __name__ == '__main__':
    main()
