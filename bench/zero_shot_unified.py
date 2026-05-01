#!/usr/bin/env python3
"""
Zero-shot person detection on the unified 100-image bench (detection_validation/).

All conditions evaluated against the SAME 100 image_ids (matched-pair),
so ∆mAP across conditions and across detector models are directly comparable.

Conditions (11):
  clear (ref baseline, raw VOC source_100/clear/*.jpg)
  rain_light, rain_heavy, snow_light, snow_heavy   (JPG folders)
  fog_light, fog_medium, fog_heavy                 (Arrow files)
  night                                            (JPG folder under night/images/)
  night_rain, night_snow                           (Arrow files: batch_0-100.arrow)
  small                                            (FLUX outpainting; *new* GT bboxes)

Special case: 'small' uses transformed bboxes from
detection_validation/small/Annotations/*.xml — its mAP_clear baseline is computed
on the small-condition GT (clear baseline = source 100 evaluated against original
manifest GT).

Models: --model {yolov8m, fasterrcnn, detr}
"""

import argparse
import io
import json
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
DV = REPO / 'detection_validation'
SRC = DV / 'source_100'
OUT_DIR = REPO / 'bench/sanity_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERSON_CAT_ID = 1


# ---------- Data sources ----------

def load_jpg_folder(folder, ids):
    out = {}
    for iid in ids:
        p = folder / f'{iid}.jpg'
        if p.exists():
            out[iid] = p.read_bytes()
    return out


def load_arrow_dict(arrow_path, ids):
    """Stream-collect into {image_id: bytes}; handles both binary and struct image columns."""
    found = {}
    ids_set = set(ids)
    with open(arrow_path, 'rb') as f:
        r = ipc.open_stream(f)
        for batch in r:
            iid_col = batch.column('image_id').to_pylist()
            for i, iid in enumerate(iid_col):
                if iid in ids_set and iid not in found:
                    img = batch.column('image')[i].as_py()
                    found[iid] = img['bytes'] if isinstance(img, dict) else img
    return found


CONDITION_SOURCES = {
    'rain_light':  ('folder', DV / 'rain_light'),
    'rain_heavy':  ('folder', DV / 'rain_heavy'),
    'snow_light':  ('folder', DV / 'snow_light'),
    'snow_heavy':  ('folder', DV / 'snow_heavy'),
    'fog_light':   ('arrow',  DV / 'fog_light.arrow'),
    'fog_medium':  ('arrow',  DV / 'fog_medium.arrow'),
    'fog_heavy':   ('arrow',  DV / 'fog_heavy.arrow'),
    'night':       ('folder', DV / 'night' / 'images'),
    'night_rain':  ('arrow',  DV / 'night_rain' / 'batch_0-100.arrow'),
    'night_snow':  ('arrow',  DV / 'night_snow' / 'batch_0-100.arrow'),
    'small':       ('folder', DV / 'small' / 'JPEGImages'),
}


def load_condition_images(cond, ids):
    kind, path = CONDITION_SOURCES[cond]
    if kind == 'folder':
        return load_jpg_folder(path, ids)
    return load_arrow_dict(path, ids)


# ---------- GT ----------

def load_clear_manifest():
    return json.loads((SRC / 'manifest.json').read_text())


def parse_voc_xml_persons(xml_path: Path):
    """Parse one VOC XML, return list of [x1,y1,x2,y2] for person + (W,H)."""
    tree = ET.parse(xml_path)
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


def build_small_gt(ids):
    """Build COCO GT for 'small' condition from FLUX-transferred VOC XMLs."""
    ann_dir = DV / 'small' / 'Annotations'
    images, annotations = [], []
    ann_id = 1
    id2int = {iid: i + 1 for i, iid in enumerate(ids)}
    for iid in ids:
        boxes, W, H = parse_voc_xml_persons(ann_dir / f'{iid}.xml')
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


# ---------- Detector backends ----------

class YOLOv8mDetector:
    name = 'yolov8m'

    def __init__(self, device):
        from ultralytics import YOLO
        self.device = device
        self.model = YOLO('yolov8m.pt')

    def detect(self, pil_img):
        result = self.model.predict(pil_img, classes=[0], conf=0.001, verbose=False,
                                    device=self.device)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4)), np.empty(0)
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()


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
            'facebook/detr-resnet-50', revision='no_timm')
        self.model = DetrForObjectDetection.from_pretrained(
            'facebook/detr-resnet-50', revision='no_timm').to(device).eval()

    @torch.no_grad()
    def detect(self, pil_img):
        W, H = pil_img.size
        inputs = self.processor(images=pil_img, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([[H, W]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.0)[0]
        labels = results['labels'].cpu().numpy()
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        m = labels == PERSON_CAT_ID
        return boxes[m], scores[m]


def build_detector(name, device):
    if name == 'yolov8m':     return YOLOv8mDetector(device)
    if name == 'fasterrcnn':  return FasterRCNNDetector(device)
    if name == 'detr':        return DETRDetector(device)
    raise ValueError(name)


# ---------- COCO eval ----------

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


def run_detector_on_imgs(detector, image_dict, ids):
    preds = {}
    for iid in ids:
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
    ap.add_argument('--model', choices=['yolov8m', 'fasterrcnn', 'detr'], required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print(f'\nLoading detector: {args.model}')
    det = build_detector(args.model, device)
    print(f'  {det.name} ready')

    manifest = load_clear_manifest()
    ids = manifest['image_ids']
    coco_gt_clear = manifest['coco_gt']
    id2int_clear = {im['image_id']: im['id'] for im in coco_gt_clear['images']}
    n_gt_clear = len(coco_gt_clear['annotations'])
    print(f'\nUnified set: {len(ids)} ids | clear GT: {n_gt_clear} person boxes')

    # 1) Clear baseline (reference for all conditions except 'small')
    print('\n=== clear (baseline) ===')
    t0 = time.time()
    clear_imgs = load_jpg_folder(SRC / 'clear', ids)
    preds_clear = run_detector_on_imgs(det, clear_imgs, ids)
    n_pred = sum(len(v) for v in preds_clear.values())
    map_clear = compute_map50(coco_gt_clear,
                              preds_to_coco(preds_clear, id2int_clear),
                              tag=f'{args.model}_clear')
    print(f'  preds={n_pred} | mAP@0.5={map_clear:.4f} | {time.time() - t0:.1f}s')

    rows = [('clear', map_clear, map_clear, 0.0, len(ids), n_gt_clear)]
    summary = {
        'model': det.name, 'unified_n': len(ids),
        'conditions': {
            'clear': {'map50': map_clear, 'n_gt': n_gt_clear, 'baseline_for': 'all except small'},
        },
    }

    # 2) Each augmented condition
    for cond in CONDITION_SOURCES:
        print(f'\n=== {cond} ===')
        t0 = time.time()
        imgs = load_condition_images(cond, ids)
        print(f'  loaded {len(imgs)}/{len(ids)} imgs')

        if cond == 'small':
            gt_dict, id2int = build_small_gt(ids)
            n_gt = len(gt_dict['annotations'])
            print(f'  small GT (FLUX-transferred): {n_gt} person boxes')
            # Compute "clear" reference for small via running clear on same imgs but original gt
            # Actually: for small, the clear baseline isn't directly comparable (same imgs but
            # different scale). We report mAP_small vs its own GT, and ∆ relative to mAP_clear
            # on the unified clear GT (informational).
            preds = run_detector_on_imgs(det, imgs, ids)
            map_cond = compute_map50(gt_dict,
                                     preds_to_coco(preds, id2int),
                                     tag=f'{args.model}_small')
        else:
            n_gt = n_gt_clear
            preds = run_detector_on_imgs(det, imgs, ids)
            map_cond = compute_map50(coco_gt_clear,
                                     preds_to_coco(preds, id2int_clear),
                                     tag=f'{args.model}_{cond}')

        delta = map_clear - map_cond
        print(f'  mAP@0.5={map_cond:.4f} | ∆clear={delta:+.4f} | {time.time() - t0:.1f}s')
        rows.append((cond, map_cond, map_clear, delta, len(ids), n_gt))
        summary['conditions'][cond] = {'map50': map_cond, 'delta_clear': delta, 'n_gt': n_gt}

    # ---------- Print + save ----------
    print('\n' + '=' * 84)
    print(f'UNIFIED BENCH — zero-shot {det.name} person, 100 matched-pair imgs')
    print('=' * 84)
    print(f'{"condition":<13} {"n_imgs":>7} {"n_GT":>5} {"mAP_clear":>11} {"mAP_cond":>10} {"∆clear":>9}')
    for cond, mc, mcl, d, n, ng in rows:
        marker = '   (baseline)' if cond == 'clear' else (' *transformed-gt' if cond == 'small' else '')
        print(f'{cond:<13} {n:>7} {ng:>5} {mcl:>11.4f} {mc:>10.4f} {d:>+9.4f}{marker}')

    csv = OUT_DIR / f'unified_{args.model}.csv'
    with open(csv, 'w') as f:
        f.write('condition,n_imgs,n_gt,mAP_clear,mAP_cond,delta_clear\n')
        for cond, mc, mcl, d, n, ng in rows:
            f.write(f'{cond},{n},{ng},{mcl:.4f},{mc:.4f},{d:.4f}\n')
    print(f'\nCSV: {csv}')

    js = OUT_DIR / f'unified_{args.model}.json'
    with open(js, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'JSON: {js}')

    for p in OUT_DIR.glob(f'tmp_*_{args.model}_*.json'):
        p.unlink(missing_ok=True)
    for p in OUT_DIR.glob(f'tmp_*_{args.model}.json'):
        p.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
