#!/usr/bin/env python3
"""
Zero-shot person detection on SODA-VOC across weather/fog/night conditions.

Supports:
  --model fasterrcnn    torchvision fasterrcnn_resnet50_fpn_v2 (COCO pretrained)
  --model detr          facebook/detr-resnet-50 (HF transformers, COCO pretrained)

Protocol identical to bench/sanity_yolov8m_100.py:
  - Per-condition independent sample of n image_ids (default 100, seed=42),
    filtered to those that have at least one VOC person GT.
  - mAP@0.5 (person-only) via pycocotools, against matched-pair clear baseline.
"""

import argparse
import io
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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

PERSON_CAT_ID = 1  # both fasterrcnn (torchvision) and DETR use COCO id=1 for person


# ---------- VOC GT ----------

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


def stream_collect(arrow_path, ids_set):
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
    rng = random.Random(seed)
    ids = []
    with open(arrow_path, 'rb') as f:
        r = ipc.open_stream(f)
        for batch in r:
            ids.extend(batch.column('image_id').to_pylist())
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


# ---------- Detector backends ----------

class FasterRCNNDetector:
    name = 'fasterrcnn_resnet50_fpn_v2'

    def __init__(self, device):
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        self.device = device
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.transform = weights.transforms()
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, box_score_thresh=0.0, box_detections_per_img=300,
        ).to(device).eval()

    @torch.no_grad()
    def detect(self, pil_img):
        t = self.transform(pil_img).to(self.device)
        out = self.model([t])[0]
        labels = out['labels'].cpu().numpy()
        boxes = out['boxes'].cpu().numpy()
        scores = out['scores'].cpu().numpy()
        m = labels == PERSON_CAT_ID
        return boxes[m], scores[m]


class DETRDetector:
    name = 'detr_resnet50'

    def __init__(self, device):
        from transformers import DetrImageProcessor, DetrForObjectDetection
        self.device = device
        self.processor = DetrImageProcessor.from_pretrained(
            'facebook/detr-resnet-50', revision='no_timm'
        )
        self.model = DetrForObjectDetection.from_pretrained(
            'facebook/detr-resnet-50', revision='no_timm'
        ).to(device).eval()

    @torch.no_grad()
    def detect(self, pil_img):
        W, H = pil_img.size
        inputs = self.processor(images=pil_img, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([[H, W]], device=self.device)
        # threshold=0 => keep all 100 queries' top-class detections
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.0
        )[0]
        labels = results['labels'].cpu().numpy()
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        m = labels == PERSON_CAT_ID
        return boxes[m], scores[m]


def build_detector(name, device):
    if name == 'fasterrcnn':
        return FasterRCNNDetector(device)
    if name == 'detr':
        return DETRDetector(device)
    raise ValueError(name)


# ---------- COCO eval helpers ----------

def build_coco_gt(sample_ids):
    images, annotations = [], []
    ann_id = 1
    id2int = {iid: i + 1 for i, iid in enumerate(sample_ids)}
    for iid in sample_ids:
        boxes, W, H = parse_voc_persons(iid)
        images.append({'id': id2int[iid], 'file_name': f'{iid}.jpg', 'width': W, 'height': H})
        for b in boxes:
            x1, y1, x2, y2 = b
            annotations.append({
                'id': ann_id, 'image_id': id2int[iid], 'category_id': PERSON_CAT_ID,
                'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0,
            })
            ann_id += 1
    return ({'images': images, 'annotations': annotations,
             'categories': [{'id': PERSON_CAT_ID, 'name': 'person'}]}, id2int)


def preds_to_coco(preds, id2int):
    out = []
    for iid, items in preds.items():
        if iid not in id2int:
            continue
        for x1, y1, x2, y2, s in items:
            out.append({
                'image_id': id2int[iid], 'category_id': PERSON_CAT_ID,
                'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(s),
            })
    return out


def compute_map50(coco_gt_dict, preds_coco, tag):
    if not preds_coco:
        return 0.0
    gt_path = OUT_DIR / f'tmp_gt_{tag}.json'
    pd_path = OUT_DIR / f'tmp_pd_{tag}.json'
    with open(gt_path, 'w') as f: json.dump(coco_gt_dict, f)
    with open(pd_path, 'w') as f: json.dump(preds_coco, f)
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pd_path))
    e = COCOeval(coco_gt, coco_dt, 'bbox')
    e.params.iouThrs = np.array([0.5])
    e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[0])


def run_detector_on_imgs(detector, image_dict, sample_ids):
    preds = {}
    for iid in sample_ids:
        if iid not in image_dict:
            preds[iid] = []
            continue
        img = Image.open(io.BytesIO(image_dict[iid])).convert('RGB')
        boxes, scores = detector.detect(img)
        preds[iid] = [[*b.tolist(), float(s)] for b, s in zip(boxes, scores)]
    return preds


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['fasterrcnn', 'detr'], required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print(f'Loading detector: {args.model}')
    det = build_detector(args.model, device)
    print(f'  {det.name} ready')

    rows = []
    summary = {'model': det.name, 'n_per_condition': args.n, 'seed': args.seed,
               'conditions': {}}
    t0_all = time.time()

    for cond, arrow_rel in CONDITIONS.items():
        t0 = time.time()
        print(f'\n=== {cond} ===')
        sample = sample_ids_for_condition(REPO / arrow_rel, n=args.n, seed=args.seed)
        print(f'  sampled {len(sample)} ids with person GT')
        if len(sample) < 20:
            print('  too few samples, skipping'); continue

        coco_gt_dict, id2int = build_coco_gt(sample)
        n_gt = len(coco_gt_dict['annotations'])
        print(f'  GT: {len(sample)} imgs, {n_gt} person boxes')

        cond_imgs = stream_collect(REPO / arrow_rel, set(sample))
        print(f'  loaded {len(cond_imgs)}/{len(sample)} aug imgs')
        preds_cond = run_detector_on_imgs(det, cond_imgs, sample)
        n_pc = sum(len(v) for v in preds_cond.values())
        map_cond = compute_map50(coco_gt_dict, preds_to_coco(preds_cond, id2int),
                                 tag=f'{args.model}_{cond}_aug')
        print(f'  aug preds: {n_pc} | mAP@0.5: {map_cond:.4f}')

        clear_imgs = {iid: (VOC / 'JPEGImages' / f'{iid}.jpg').read_bytes() for iid in sample}
        preds_clear = run_detector_on_imgs(det, clear_imgs, sample)
        n_pcl = sum(len(v) for v in preds_clear.values())
        map_clear = compute_map50(coco_gt_dict, preds_to_coco(preds_clear, id2int),
                                  tag=f'{args.model}_{cond}_clear')
        delta = map_clear - map_cond
        dt = time.time() - t0
        print(f'  clear preds: {n_pcl} | mAP@0.5 clear: {map_clear:.4f} | ∆clear: {delta:+.4f} | {dt:.1f}s')

        rows.append((cond, map_cond, map_clear, delta, len(sample), n_gt))
        summary['conditions'][cond] = {
            'n_imgs': len(sample), 'n_gt': n_gt,
            'map50_clear': map_clear, 'map50_aug': map_cond, 'delta_clear': delta,
            'sample_ids': sample,
        }

    print('\n' + '=' * 80)
    print(f'SUMMARY: zero-shot {det.name} person, {args.n}-img sample/condition (seed={args.seed})')
    print('=' * 80)
    print(f'{"condition":<14} {"n_imgs":>7} {"n_GT":>6} {"mAP_clear":>11} {"mAP_cond":>10} {"∆clear":>9}')
    for cond, mc, mcl, d, n, ng in rows:
        print(f'{cond:<14} {n:>7} {ng:>6} {mcl:>11.4f} {mc:>10.4f} {d:>+9.4f}')
    print(f'\nTotal: {time.time() - t0_all:.1f}s')

    out_csv = OUT_DIR / f'sanity_{args.model}_{args.n}.csv'
    with open(out_csv, 'w') as f:
        f.write('condition,n_imgs,n_gt_boxes,mAP_clear,mAP_cond,delta_clear\n')
        for cond, mc, mcl, d, n, ng in rows:
            f.write(f'{cond},{n},{ng},{mcl:.4f},{mc:.4f},{d:.4f}\n')
    print(f'Saved CSV: {out_csv}')

    out_json = OUT_DIR / f'sanity_{args.model}_{args.n}.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved JSON: {out_json}')

    # Cleanup temp files
    for p in OUT_DIR.glob(f'tmp_*_{args.model}_*.json'):
        p.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
