#!/usr/bin/env python3
"""
Sanity check: zero-shot YOLOv8m person detection on SODA-VOC, 100 imgs per condition.

For each condition, evaluates COCO mAP@0.5 (person class only) using pycocotools.
Skips night_rain/night_snow (SLURM-pending) and small_object (bbox transform needed).
"""

import io
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

REPO = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')
VOC = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007')
OUT_DIR = REPO / 'bench/sanity_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    'rain':       'augmentation_data/soda_voc/rain_snow/diffusion/rain_light.arrow',
    'rain_heavy': 'augmentation_data/soda_voc/rain_snow/diffusion/rain_heavy.arrow',
    'snow':       'augmentation_data/soda_voc/rain_snow/diffusion/snow_light.arrow',
    'snow_heavy': 'augmentation_data/soda_voc/rain_snow/diffusion/snow_heavy.arrow',
    'fog_light':  'augmentation_data/soda_voc/fog/test/fog_light.arrow',
    'fog_medium': 'augmentation_data/soda_voc/fog/test/fog_medium.arrow',
    'fog_heavy':  'augmentation_data/soda_voc/fog/test/fog_heavy.arrow',
    'night':      'augmentation_data/soda_voc/night/soda_day2night.arrow',
}


def parse_voc_persons(image_id: str):
    """Return list of [x1,y1,x2,y2] person bboxes from VOC XML, with (W,H)."""
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


def stream_collect(arrow_path, ids_set):
    """Stream-read arrow, return dict {image_id: image_bytes} for ids_set."""
    found = {}
    with open(arrow_path, 'rb') as f:
        r = ipc.open_stream(f)
        for batch in r:
            ids = batch.column('image_id').to_pylist()
            for i, iid in enumerate(ids):
                if iid in ids_set and iid not in found:
                    img = batch.column('image')[i].as_py()
                    found[iid] = img['bytes'] if isinstance(img, dict) else img
            if len(found) == len(ids_set):
                break
    return found


def sample_ids_for_condition(arrow_path, n=100, seed=42):
    """Sample n image_ids from this condition arrow that have person GT in VOC XML."""
    rng = random.Random(seed)
    ids = []
    with open(arrow_path, 'rb') as f:
        r = ipc.open_stream(f)
        for batch in r:
            ids.extend(batch.column('image_id').to_pylist())
    # Shuffle, then take first n with person GT
    rng.shuffle(ids)
    selected = []
    for iid in ids:
        boxes, _, _ = parse_voc_persons(iid)
        if boxes:
            selected.append(iid)
            if len(selected) >= n:
                break
    selected.sort()
    return selected


def run_yolov8m_on_images(model, image_dict, sample_ids, conf=0.001):
    """Run YOLOv8m on a {image_id: image_bytes} dict, return dict {image_id: list of [x1,y1,x2,y2,score]} for person class."""
    preds = {}
    for iid in sample_ids:
        if iid not in image_dict:
            preds[iid] = []
            continue
        img = Image.open(io.BytesIO(image_dict[iid])).convert('RGB')
        # YOLOv8m COCO class 0 = person
        result = model.predict(img, classes=[0], conf=conf, verbose=False, device='cpu')[0]
        boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        preds[iid] = [[*b.tolist(), float(s)] for b, s in zip(boxes_xyxy, scores)]
    return preds


def build_coco_gt(sample_ids):
    """Build COCO-format GT for sample_ids (person-only)."""
    images, annotations = [], []
    ann_id = 1
    id2int = {iid: i + 1 for i, iid in enumerate(sample_ids)}
    for iid in sample_ids:
        boxes, W, H = parse_voc_persons(iid)
        images.append({'id': id2int[iid], 'file_name': f'{iid}.jpg', 'width': W, 'height': H})
        for b in boxes:
            x1, y1, x2, y2 = b
            annotations.append({
                'id': ann_id, 'image_id': id2int[iid], 'category_id': 1,
                'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0,
            })
            ann_id += 1
    coco_gt = {'images': images, 'annotations': annotations,
               'categories': [{'id': 1, 'name': 'person'}]}
    return coco_gt, id2int


def preds_to_coco(preds, id2int):
    """Convert {image_id: [[x1,y1,x2,y2,score]]} to COCO predictions list."""
    out = []
    for iid, boxes in preds.items():
        if iid not in id2int:
            continue
        for x1, y1, x2, y2, s in boxes:
            out.append({
                'image_id': id2int[iid], 'category_id': 1,
                'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': s,
            })
    return out


def compute_map50(coco_gt_dict, preds_coco):
    """Run COCOeval on already-built dicts; return mAP@0.5."""
    if not preds_coco:
        return 0.0
    gt_path = OUT_DIR / 'tmp_gt.json'
    pd_path = OUT_DIR / 'tmp_pd.json'
    with open(gt_path, 'w') as f: json.dump(coco_gt_dict, f)
    with open(pd_path, 'w') as f: json.dump(preds_coco, f)
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pd_path))
    e = COCOeval(coco_gt, coco_dt, 'bbox')
    e.params.iouThrs = np.array([0.5])
    e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[0])


def main():
    print('Loading YOLOv8m...')
    model = YOLO('yolov8m.pt')

    rows = []  # condition, mAP_cond, mAP_clear, delta, n_imgs, n_gt_boxes

    for cond, arrow_rel in CONDITIONS.items():
        print(f'\n=== {cond} ===')
        sample = sample_ids_for_condition(REPO / arrow_rel, n=100, seed=42)
        print(f'  sampled {len(sample)} ids with person GT')
        if len(sample) < 20:
            print('  too few samples, skipping')
            continue

        coco_gt_dict, id2int = build_coco_gt(sample)
        n_gt = len(coco_gt_dict['annotations'])
        print(f'  GT: {len(sample)} imgs, {n_gt} person boxes')

        # 1) Augmented condition
        cond_imgs = stream_collect(REPO / arrow_rel, set(sample))
        print(f'  loaded {len(cond_imgs)}/{len(sample)} augmented imgs')
        preds_cond = run_yolov8m_on_images(model, cond_imgs, sample)
        n_pc = sum(len(v) for v in preds_cond.values())
        map_cond = compute_map50(coco_gt_dict, preds_to_coco(preds_cond, id2int))
        print(f'  augmented preds: {n_pc} | mAP@0.5: {map_cond:.4f}')

        # 2) Clear baseline on SAME image_ids (raw VOC)
        clear_imgs = {iid: (VOC / 'JPEGImages' / f'{iid}.jpg').read_bytes() for iid in sample}
        preds_clear = run_yolov8m_on_images(model, clear_imgs, sample)
        n_pcl = sum(len(v) for v in preds_clear.values())
        map_clear = compute_map50(coco_gt_dict, preds_to_coco(preds_clear, id2int))
        delta = map_clear - map_cond
        print(f'  clear preds: {n_pcl} | mAP@0.5 clear: {map_clear:.4f} | ∆clear: {delta:+.4f}')

        rows.append((cond, map_cond, map_clear, delta, len(sample), n_gt))

    print('\n' + '=' * 76)
    print('SUMMARY: zero-shot YOLOv8m person, 100-img sample per condition (seed=42)')
    print('=' * 76)
    print(f'{"condition":<14} {"n_imgs":>7} {"n_GT":>6} {"mAP_clear":>11} {"mAP_cond":>10} {"∆clear":>9}')
    for cond, mc, mcl, d, n, ng in rows:
        print(f'{cond:<14} {n:>7} {ng:>6} {mcl:>11.4f} {mc:>10.4f} {d:>+9.4f}')

    out_csv = OUT_DIR / 'sanity_yolov8m_100.csv'
    with open(out_csv, 'w') as f:
        f.write('condition,n_imgs,n_gt_boxes,mAP_clear,mAP_cond,delta_clear\n')
        for cond, mc, mcl, d, n, ng in rows:
            f.write(f'{cond},{n},{ng},{mcl:.4f},{mc:.4f},{d:.4f}\n')
    print(f'\nSaved: {out_csv}')


if __name__ == '__main__':
    main()
