#!/usr/bin/env python3
"""
All-in-one sensitivity analysis: filter Arrow data → export YOLO → train YOLOv8 → save metrics.
Runs as a single SLURM job per threshold config.

Usage:
  python run_sensitivity.py --config current
  python run_sensitivity.py --config loose
"""

import argparse
import json
import shutil
import gc
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
from PIL import Image
from tqdm import tqdm
import torch

ORIG_ARROW_DIR = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/LouisChen15___construction_site')
TRAIN_ARROWS = [
    ORIG_ARROW_DIR / 'construction_site-train-00000-of-00002.arrow',
    ORIG_ARROW_DIR / 'construction_site-train-00001-of-00002.arrow',
]
TEST_ARROW = ORIG_ARROW_DIR / 'construction_site-test.arrow'

AUG = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X/augmentation_data/construction_site/rain_snow')
OUT_BASE = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X/generation/sensitivity/detection_results')

CLASS_NAMES = ['excavator', 'rebar', 'worker_with_white_hard_hat']

CONFIGS = {
    'baseline': {'ssim_rain': 99.0, 'ssim_snow': 99.0, 'lpips': 0.00},  # No augmentation passes
    'loose':    {'ssim_rain': 0.40, 'ssim_snow': 0.40, 'lpips': 0.50},
    'moderate': {'ssim_rain': 0.50, 'ssim_snow': 0.50, 'lpips': 0.40},
    'current':  {'ssim_rain': 0.60, 'ssim_snow': 0.50, 'lpips': 0.35},
    'strict':   {'ssim_rain': 0.65, 'ssim_snow': 0.60, 'lpips': 0.30},
    'tight':    {'ssim_rain': 0.70, 'ssim_snow': 0.65, 'lpips': 0.25},
    'no_lpips': {'ssim_rain': 0.60, 'ssim_snow': 0.50, 'lpips': 1.00},
}


# ── Arrow helpers ──────────────────────────────────────────────────────
def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(BytesIO(row['bytes'])).convert('RGB')
    return Image.open(BytesIO(row)).convert('RGB')


def get_boxes(ann, class_name):
    if ann is None:
        return []
    boxes = ann.get(class_name)
    if boxes is not None:
        return boxes
    objects = ann.get('objects', {})
    return objects.get(class_name, []) or []


# ── Step 1: Determine kept image_ids from diffusion meta CSVs ─────────
def get_kept_ids_diffusion(weather, ssim_lower, ssim_upper, lpips_thresh):
    """Read meta CSVs and return set of image_ids that pass thresholds."""
    kept = set()
    total = 0

    for split_name in ['train']:
        base = AUG / 'diffusion' / split_name / weather
        csvs = list(base.glob('meta_*.csv'))
        for shard in sorted(base.glob('s*')):
            if shard.is_dir():
                csvs.extend(shard.glob('meta_*.csv'))

        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            total += len(df)
            ssim_mask = (df['ssim'] >= ssim_lower) & (df['ssim'] <= ssim_upper)
            has_lpips = 'lpips' in df.columns and (df['lpips'] != 0).any()
            if has_lpips and lpips_thresh < 1.0:
                mask = ssim_mask & (df['lpips'] < lpips_thresh)
            else:
                mask = ssim_mask
            # CSV stores int (4850), Arrow stores zero-padded string ("0004852")
            # Store both formats to match either
            raw_ids = df.loc[mask, 'image_id']
            kept.update(raw_ids.astype(str).values)
            kept.update(raw_ids.apply(lambda x: str(x).zfill(7)).values)

    return kept, total


# ── Step 2: Export Arrow → YOLO format ─────────────────────────────────
def export_arrow_to_yolo(table, output_images, output_labels, prefix='', ann_col=None):
    """Export Arrow table rows to YOLO images + labels."""
    count = 0
    for idx in range(len(table)):
        try:
            img = get_image(table, idx)
        except Exception:
            continue

        image_id = str(table.column('image_id')[idx].as_py())
        fname = f'{prefix}{image_id}.jpg'

        img.save(output_images / fname, 'JPEG', quality=95)

        # Extract annotations
        lines = []
        if ann_col and ann_col in table.column_names:
            ann = table.column(ann_col)[idx].as_py()
        else:
            # Try common annotation columns
            ann = {}
            for col in table.column_names:
                if col in CLASS_NAMES:
                    ann[col] = table.column(col)[idx].as_py()
                elif col == 'objects':
                    ann['objects'] = table.column(col)[idx].as_py()

        for ci, cn in enumerate(CLASS_NAMES):
            boxes = get_boxes(ann, cn)
            for b in (boxes or []):
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    x1, y1, x2, y2 = b[:4]
                    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    if w > 0.001 and h > 0.001:
                        xc = max(0, min(1, xc))
                        yc = max(0, min(1, yc))
                        w = max(0, min(1, w))
                        h = max(0, min(1, h))
                        lines.append(f'{ci} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}')

        label_fname = fname.replace('.jpg', '.txt')
        with open(output_labels / label_fname, 'w') as f:
            f.write('\n'.join(lines))

        count += 1
    return count


def export_filtered_diffusion(weather, kept_ids, output_images, output_labels):
    """Load diffusion Arrow batches and export only kept rows."""
    count = 0
    base = AUG / 'diffusion' / 'train' / weather

    arrow_dirs = [base]
    for shard in sorted(base.glob('s*')):
        if shard.is_dir():
            arrow_dirs.append(shard)

    for d in arrow_dirs:
        for arrow_file in sorted(d.glob('batch_*.arrow')):
            table = load_arrow(arrow_file)
            for idx in range(len(table)):
                img_id = str(table.column('image_id')[idx].as_py())
                if img_id not in kept_ids:
                    continue
                try:
                    img = get_image(table, idx)
                except Exception:
                    continue

                fname = f'diff_{weather}_{img_id}.jpg'
                img.save(output_images / fname, 'JPEG', quality=95)

                # Copy annotations from original columns
                lines = []
                for ci, cn in enumerate(CLASS_NAMES):
                    if cn in table.column_names:
                        boxes = table.column(cn)[idx].as_py()
                        for b in (boxes or []):
                            if isinstance(b, (list, tuple)) and len(b) >= 4:
                                x1, y1, x2, y2 = b[:4]
                                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                                w, h = x2 - x1, y2 - y1
                                if w > 0.001 and h > 0.001:
                                    lines.append(f'{ci} {max(0,min(1,xc)):.6f} {max(0,min(1,yc)):.6f} {max(0,min(1,w)):.6f} {max(0,min(1,h)):.6f}')

                with open(output_labels / (fname.replace('.jpg', '.txt')), 'w') as f:
                    f.write('\n'.join(lines))
                count += 1

            del table
            gc.collect()

    return count


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, choices=list(CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--val-samples', type=int, default=1000)
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    tag = args.config

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    yolo_dir = OUT_BASE / f'yolo_data_{tag}'

    # Clean
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    train_img = yolo_dir / 'images' / 'train'
    train_lbl = yolo_dir / 'labels' / 'train'
    val_img = yolo_dir / 'images' / 'val'
    val_lbl = yolo_dir / 'labels' / 'val'
    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True)

    print('=' * 60)
    print(f'Sensitivity: {tag}')
    print(f'  SSIM rain: [{cfg["ssim_rain"]}, 0.95]')
    print(f'  SSIM snow: [{cfg["ssim_snow"]}, 0.95]')
    print(f'  LPIPS:     < {cfg["lpips"]}' + (' (disabled)' if cfg['lpips'] >= 1 else ''))
    print('=' * 60)

    # ── Step 1: Get kept IDs ──────────────────────────────────────
    print('\n── Step 1: Filter by thresholds ──')
    rain_kept, rain_total = get_kept_ids_diffusion('rain', cfg['ssim_rain'], 0.95, cfg['lpips'])
    snow_kept, snow_total = get_kept_ids_diffusion('snow', cfg['ssim_snow'], 0.95, cfg['lpips'])
    print(f'  Rain: {len(rain_kept)}/{rain_total} kept ({len(rain_kept)/max(rain_total,1)*100:.1f}%)')
    print(f'  Snow: {len(snow_kept)}/{snow_total} kept ({len(snow_kept)/max(snow_total,1)*100:.1f}%)')

    # ── Step 2: Export original train data ────────────────────────
    print('\n── Step 2: Export original training data ──')
    orig_count = 0
    for arrow_path in TRAIN_ARROWS:
        if arrow_path.exists():
            print(f'  Loading {arrow_path.name} ...')
            table = load_arrow(arrow_path)
            n = export_arrow_to_yolo(table, train_img, train_lbl, prefix='orig_')
            orig_count += n
            print(f'    Exported {n} images')
            del table
            gc.collect()
    print(f'  Total original: {orig_count}')

    # ── Step 3: Export filtered diffusion data ────────────────────
    print('\n── Step 3: Export filtered augmentation data ──')
    rain_count = export_filtered_diffusion('rain', rain_kept, train_img, train_lbl)
    print(f'  Rain exported: {rain_count}')
    snow_count = export_filtered_diffusion('snow', snow_kept, train_img, train_lbl)
    print(f'  Snow exported: {snow_count}')

    total_train = orig_count + rain_count + snow_count
    print(f'  Total train: {total_train}')

    # ── Step 4: Export validation data ────────────────────────────
    print('\n── Step 4: Export validation data ──')
    if TEST_ARROW.exists():
        table = load_arrow(TEST_ARROW)
        n_val = min(args.val_samples, len(table))
        # Take first n_val samples
        val_table = table.slice(0, n_val)
        val_count = export_arrow_to_yolo(val_table, val_img, val_lbl, prefix='val_')
        print(f'  Validation: {val_count} images')
        del table, val_table
        gc.collect()
    else:
        print(f'  WARNING: {TEST_ARROW} not found')
        val_count = 0

    # ── Step 5: Create YOLO dataset.yaml ──────────────────────────
    yaml_content = f"""path: {yolo_dir.absolute()}
train: images/train
val: images/val
names:
  0: excavator
  1: rebar
  2: worker_with_white_hard_hat
nc: 3
"""
    with open(yolo_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)

    # ── Step 6: Train YOLOv8 ──────────────────────────────────────
    print(f'\n── Step 6: Train YOLOv8n ({args.epochs} epochs) ──')
    print(f'  Train: {total_train} | Val: {val_count}')

    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    model.train(
        data=str(yolo_dir / 'dataset.yaml'),
        epochs=args.epochs,
        batch=16,
        imgsz=640,
        project=str(OUT_BASE / 'runs'),
        name=f'sens_{tag}',
        exist_ok=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=2,
        patience=10,
        seed=42,
        verbose=True,
    )

    # ── Step 7: Validate & save metrics ───────────────────────────
    print(f'\n── Step 7: Validate ──')
    best = OUT_BASE / 'runs' / f'sens_{tag}' / 'weights' / 'best.pt'
    model = YOLO(best)
    val = model.val(data=str(yolo_dir / 'dataset.yaml'), split='val')

    metrics = {
        'config': f'sens_{tag}',
        'ssim_rain': cfg['ssim_rain'],
        'ssim_snow': cfg['ssim_snow'],
        'lpips': cfg['lpips'],
        'mAP50': float(val.box.map50),
        'mAP50_95': float(val.box.map),
        'precision': float(val.box.mp),
        'recall': float(val.box.mr),
        'train_images': total_train,
        'train_original': orig_count,
        'train_rain': rain_count,
        'train_snow': snow_count,
        'val_images': val_count,
    }
    for i, cn in enumerate(CLASS_NAMES):
        if i < len(val.box.ap50):
            metrics[f'{cn}_AP50'] = float(val.box.ap50[i])

    metrics_path = OUT_BASE / f'metrics_sens_{tag}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n{"=" * 60}')
    print(f'DONE: {tag}')
    print(f'  mAP@0.5:     {metrics["mAP50"]:.4f}')
    print(f'  mAP@0.5:0.95: {metrics["mAP50_95"]:.4f}')
    print(f'  Train: {total_train} (orig={orig_count}, rain={rain_count}, snow={snow_count})')
    print(f'  Metrics: {metrics_path}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
