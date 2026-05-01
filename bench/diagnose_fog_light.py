#!/usr/bin/env python3
"""
Diagnose why fog_light clear baseline = 0.022 (near-zero).

Hypothesis: fog_light's image pool is dominated by tiny-person images.
Check: for the 100 sampled fog_light image_ids, distribution of person bbox sizes
in raw VOC XML, vs other condition pools.
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path
import pyarrow.ipc as ipc

VOC = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007')
REPO = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')

CONDITIONS = {
    'rain':       'augmentation_data/soda_voc/rain_snow/diffusion/rain_light.arrow',
    'fog_light':  'augmentation_data/soda_voc/fog/test/fog_light.arrow',
    'fog_medium': 'augmentation_data/soda_voc/fog/test/fog_medium.arrow',
    'fog_heavy':  'augmentation_data/soda_voc/fog/test/fog_heavy.arrow',
    'snow_heavy': 'augmentation_data/soda_voc/rain_snow/diffusion/snow_heavy.arrow',
}


def person_boxes(image_id):
    """Return list of (x1,y1,x2,y2) person boxes + (W,H)."""
    xml = VOC / 'Annotations' / f'{image_id}.xml'
    if not xml.exists():
        return None, None, None
    tree = ET.parse(xml); root = tree.getroot()
    s = root.find('size')
    W = int(s.find('width').text); H = int(s.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        if obj.find('name').text.strip() == 'person':
            bb = obj.find('bndbox')
            boxes.append([float(bb.find('xmin').text), float(bb.find('ymin').text),
                          float(bb.find('xmax').text), float(bb.find('ymax').text)])
    return boxes, W, H


def sample_with_person(arrow_path, n=100, seed=42):
    rng = random.Random(seed)
    ids = []
    with open(arrow_path, 'rb') as f:
        for batch in ipc.open_stream(f):
            ids.extend(batch.column('image_id').to_pylist())
    rng.shuffle(ids)
    sel = []
    for iid in ids:
        boxes, _, _ = person_boxes(iid)
        if boxes:
            sel.append(iid)
            if len(sel) >= n:
                break
    return sel


def main():
    print(f'{"condition":<14} {"n":>5} {"min_area":>10} {"med_area":>10} {"max_area":>10} '
          f'{"%small":>7} {"%med":>7} {"%large":>7}')
    print('-' * 80)

    for cond, p in CONDITIONS.items():
        sample = sample_with_person(REPO / p, n=100, seed=42)
        areas = []
        for iid in sample:
            boxes, W, H = person_boxes(iid)
            img_area = W * H
            for x1, y1, x2, y2 in boxes:
                bw, bh = x2 - x1, y2 - y1
                # COCO area thresholds: small <32², medium 32²-96², large >96²
                # Normalize by image scale: paper uses raw pixels though
                areas.append(bw * bh)
        areas.sort()
        n = len(areas)
        if n == 0:
            print(f'{cond:<14} {n:>5} (no data)')
            continue
        smallp = sum(1 for a in areas if a < 32**2) / n * 100
        medp   = sum(1 for a in areas if 32**2 <= a < 96**2) / n * 100
        largep = sum(1 for a in areas if a >= 96**2) / n * 100
        print(f'{cond:<14} {n:>5} {areas[0]:>10.0f} {areas[n//2]:>10.0f} {areas[-1]:>10.0f} '
              f'{smallp:>6.1f}% {medp:>6.1f}% {largep:>6.1f}%')

    print('\nLegend: COCO area thresholds — small<32² (1024px²), medium 32²-96², large>96² (9216px²)')


if __name__ == '__main__':
    main()
