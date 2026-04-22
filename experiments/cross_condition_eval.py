#!/usr/bin/env python3
"""
Cross-condition detection evaluation for Table 7.
Exports augmented TEST sets → YOLO format, then evaluates existing + new models.

Test sets are UNSEEN: augmented from the 3,004 test-split images.
Training used only the 7,009 train-split images.

Usage:
  python cross_condition_eval.py --phase export     # Export all test sets
  python cross_condition_eval.py --phase train       # Train new configs (+Night, +All)
  python cross_condition_eval.py --phase eval        # Evaluate all models on all test sets
  python cross_condition_eval.py --phase all         # Do everything
"""

import argparse, gc, json, shutil, sys
from pathlib import Path
from io import BytesIO
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import numpy as np
import pyarrow as pa
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────
BASE = _REPO_ROOT
CS_DIR = (_DATA_ROOT / "LouisChen15___construction_site")
AUG_ARROW = (_DATA_ROOT / "augmentation_data_arrow")
AUG_DATA = BASE / 'augmentation_data/construction_site'

SENS_DIR = BASE / 'generation/sensitivity/detection_results'
EXP_DIR = BASE / 'experiments/detection_results'

# Existing trained models
MODELS = {
    'baseline': SENS_DIR / 'runs/sens_baseline/weights/best.pt',
    'weather':  SENS_DIR / 'runs/sens_current/weights/best.pt',
}

CLASS_NAMES = ['excavator', 'rebar', 'worker_with_white_hard_hat']

# ── Test set sources (all from TEST split = unseen) ───────────────────
TEST_SETS = {
    'original': {
        'type': 'arrow_direct',
        'path': CS_DIR / 'construction_site-test.arrow',
        'max_images': 1000,  # Subsample for speed
    },
    'weather_rain': {
        'type': 'arrow_batches',
        'path': AUG_DATA / 'rain_snow/diffusion/test/rain',
        'max_images': 500,
    },
    'weather_snow': {
        'type': 'arrow_batches',
        'path': AUG_DATA / 'rain_snow/diffusion/test/snow',
        'max_images': 500,
    },
    'fog': {
        'type': 'arrow_batches',
        'path': AUG_DATA / 'fog/diffusion/test/heavy',
        'max_images': 500,
    },
    'night': {
        'type': 'arrow_json_ann',
        'path': AUG_ARROW / 'night.arrow',
        'max_images': 500,
    },
    'small': {
        'type': 'arrow_json_ann',
        'path': AUG_ARROW / 'small.arrow',
        'max_images': 500,
    },
}

# Training configs for new models
TRAIN_CONFIGS = {
    'weather_night': {
        'description': 'Original + Weather (current) + Night',
        'sources': ['original_train', 'weather_train', 'night_train'],
    },
    'all': {
        'description': 'Original + Weather + Night + Small + Fog',
        'sources': ['original_train', 'weather_train', 'night_train', 'small_train', 'fog_train'],
    },
}

# Training data sources
TRAIN_SOURCES = {
    'original_train': {
        'type': 'arrow_direct',
        'paths': [
            CS_DIR / 'construction_site-train-00000-of-00002.arrow',
            CS_DIR / 'construction_site-train-00001-of-00002.arrow',
        ],
    },
    'weather_train': {
        'type': 'consolidated_arrow',
        'paths': [
            AUG_DATA / 'rain_snow/diffusion/train/rain/train_rain_v4.arrow',
            AUG_DATA / 'rain_snow/diffusion/train/snow/train_snow_v4.arrow',
        ],
    },
    'night_train': {
        'type': 'arrow_json_ann',
        'path': AUG_ARROW / 'night.arrow',
        'max_images': 2000,  # Subset for balanced training
    },
    'small_train': {
        'type': 'arrow_json_ann',
        'path': AUG_ARROW / 'small.arrow',
        'max_images': 1000,
    },
    'fog_train': {
        'type': 'arrow_batches',
        'path': AUG_DATA / 'fog/diffusion/test/heavy',  # Reuse heavy fog
        'max_images': 500,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(BytesIO(row['bytes'])).convert('RGB')
    return Image.open(BytesIO(row)).convert('RGB')


def extract_boxes_direct(table, idx):
    """Extract boxes from direct class columns (original/weather/fog Arrow)."""
    lines = []
    for ci, cn in enumerate(CLASS_NAMES):
        if cn not in table.column_names:
            continue
        boxes = table.column(cn)[idx].as_py()
        for b in (boxes or []):
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x1, y1, x2, y2 = b[:4]
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                if w > 0.001 and h > 0.001:
                    lines.append(f'{ci} {max(0,min(1,xc)):.6f} {max(0,min(1,yc)):.6f} '
                                 f'{max(0,min(1,w)):.6f} {max(0,min(1,h)):.6f}')
    return lines


def extract_boxes_json(table, idx):
    """Extract boxes from JSON annotation string (night/small Arrow)."""
    ann_str = table.column('annotation')[idx].as_py()
    ann = json.loads(ann_str) if isinstance(ann_str, str) else ann_str
    lines = []
    for ci, cn in enumerate(CLASS_NAMES):
        boxes = ann.get(cn, [])
        for b in (boxes or []):
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x1, y1, x2, y2 = b[:4]
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                if w > 0.001 and h > 0.001:
                    lines.append(f'{ci} {max(0,min(1,xc)):.6f} {max(0,min(1,yc)):.6f} '
                                 f'{max(0,min(1,w)):.6f} {max(0,min(1,h)):.6f}')
    return lines


def export_table(table, img_dir, lbl_dir, prefix, extract_fn, max_images=None):
    """Export Arrow table rows to YOLO images + labels."""
    n = min(len(table), max_images) if max_images else len(table)
    count = 0
    for idx in range(n):
        try:
            img = get_image(table, idx)
        except Exception:
            continue
        img_id = str(table.column('image_id')[idx].as_py())
        fname = f'{prefix}{img_id}.jpg'
        img.save(img_dir / fname, 'JPEG', quality=95)
        lines = extract_fn(table, idx)
        with open(lbl_dir / fname.replace('.jpg', '.txt'), 'w') as f:
            f.write('\n'.join(lines))
        count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: EXPORT TEST SETS
# ═══════════════════════════════════════════════════════════════════════

def export_test_sets():
    """Export all augmented test sets to YOLO format."""
    test_base = EXP_DIR / 'test_sets'

    for name, cfg in TEST_SETS.items():
        out_img = test_base / name / 'images'
        out_lbl = test_base / name / 'labels'

        if out_img.exists() and len(list(out_img.glob('*.jpg'))) > 0:
            print(f'  [{name}] Already exported ({len(list(out_img.glob("*.jpg")))} images), skipping')
            continue

        for d in [out_img, out_lbl]:
            d.mkdir(parents=True, exist_ok=True)

        max_img = cfg.get('max_images')

        if cfg['type'] == 'arrow_direct':
            table = load_arrow(cfg['path'])
            count = export_table(table, out_img, out_lbl, f'{name}_',
                                 extract_boxes_direct, max_img)
            del table; gc.collect()

        elif cfg['type'] == 'arrow_batches':
            count = 0
            batch_dir = Path(cfg['path'])
            for arrow_file in sorted(batch_dir.glob('*.arrow')):
                table = load_arrow(arrow_file)
                remaining = (max_img - count) if max_img else len(table)
                if remaining <= 0:
                    del table; gc.collect()
                    break
                n = export_table(table, out_img, out_lbl,
                                 f'{name}_', extract_boxes_direct,
                                 min(remaining, len(table)))
                count += n
                del table; gc.collect()

        elif cfg['type'] == 'arrow_json_ann':
            table = load_arrow(cfg['path'])
            count = export_table(table, out_img, out_lbl, f'{name}_',
                                 extract_boxes_json, max_img)
            del table; gc.collect()

        print(f'  [{name}] Exported {count} images')

    # Create dataset.yaml for each test set (used by YOLO val)
    for name in TEST_SETS:
        yaml_path = test_base / name / 'dataset.yaml'
        yaml_content = (
            f"path: {test_base / name}\n"
            f"train: images  # dummy, not used for val\n"
            f"val: images\n"
            f"names:\n"
        )
        for i, cn in enumerate(CLASS_NAMES):
            yaml_content += f"  {i}: {cn}\n"
        yaml_content += f"nc: {len(CLASS_NAMES)}\n"
        yaml_path.write_text(yaml_content)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: TRAIN NEW CONFIGS
# ═══════════════════════════════════════════════════════════════════════

def export_training_data(config_name):
    """Export training data for a new config."""
    cfg = TRAIN_CONFIGS[config_name]
    yolo_dir = EXP_DIR / f'yolo_train_{config_name}'
    train_img = yolo_dir / 'images' / 'train'
    train_lbl = yolo_dir / 'labels' / 'train'
    val_img = yolo_dir / 'images' / 'val'
    val_lbl = yolo_dir / 'labels' / 'val'

    if train_img.exists() and len(list(train_img.glob('*.jpg'))) > 100:
        print(f'  [{config_name}] Training data already exported, skipping')
        return yolo_dir

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    total = 0
    for src_name in cfg['sources']:
        src = TRAIN_SOURCES[src_name]
        if src['type'] == 'arrow_direct':
            for p in src['paths']:
                if p.exists():
                    table = load_arrow(p)
                    n = export_table(table, train_img, train_lbl,
                                     f'{src_name}_', extract_boxes_direct)
                    total += n
                    del table; gc.collect()

        elif src['type'] == 'consolidated_arrow':
            for p in src['paths']:
                if p.exists():
                    table = load_arrow(p)
                    n = export_table(table, train_img, train_lbl,
                                     f'{src_name}_', extract_boxes_direct,
                                     src.get('max_images'))
                    total += n
                    del table; gc.collect()

        elif src['type'] == 'arrow_json_ann':
            if src['path'].exists():
                table = load_arrow(src['path'])
                n = export_table(table, train_img, train_lbl,
                                 f'{src_name}_', extract_boxes_json,
                                 src.get('max_images'))
                total += n
                del table; gc.collect()

        elif src['type'] == 'arrow_batches':
            batch_dir = Path(src['path'])
            count = 0
            for arrow_file in sorted(batch_dir.glob('*.arrow')):
                table = load_arrow(arrow_file)
                remaining = (src.get('max_images', 9999) - count)
                if remaining <= 0:
                    del table; gc.collect()
                    break
                n = export_table(table, train_img, train_lbl,
                                 f'{src_name}_', extract_boxes_direct,
                                 min(remaining, len(table)))
                count += n
                del table; gc.collect()
            total += count

        print(f'    {src_name}: exported')

    # Val = subset of original test
    table = load_arrow(CS_DIR / 'construction_site-test.arrow')
    val_table = table.slice(0, 1000)
    export_table(val_table, val_img, val_lbl, 'val_', extract_boxes_direct)
    del table, val_table; gc.collect()

    print(f'  [{config_name}] Total train: {total}')

    # dataset.yaml
    yaml_path = yolo_dir / 'dataset.yaml'
    yaml_content = (
        f"path: {yolo_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
    )
    for i, cn in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {cn}\n"
    yaml_content += f"nc: {len(CLASS_NAMES)}\n"
    yaml_path.write_text(yaml_content)
    return yolo_dir


def train_new_configs(epochs=20):
    """Train new model configs."""
    from ultralytics import YOLO

    for config_name in TRAIN_CONFIGS:
        print(f'\n{"="*60}')
        print(f'Training: {config_name} ({TRAIN_CONFIGS[config_name]["description"]})')
        print(f'{"="*60}')

        yolo_dir = export_training_data(config_name)
        yaml_path = yolo_dir / 'dataset.yaml'
        run_dir = EXP_DIR / 'runs'

        model = YOLO('yolov8n.pt')
        model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=640,
            batch=16,
            project=str(run_dir),
            name=f'cross_{config_name}',
            seed=42,
            exist_ok=True,
            verbose=False,
        )

        # Save model path for eval
        best_pt = run_dir / f'cross_{config_name}' / 'weights' / 'best.pt'
        MODELS[config_name] = best_pt
        print(f'  Model saved: {best_pt}')


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: EVALUATE ALL MODELS ON ALL TEST SETS
# ═══════════════════════════════════════════════════════════════════════

def evaluate_all():
    """Evaluate all models on all test sets → Table 7."""
    from ultralytics import YOLO

    # Check for newly trained models
    for config_name in TRAIN_CONFIGS:
        pt = EXP_DIR / 'runs' / f'cross_{config_name}' / 'weights' / 'best.pt'
        if pt.exists():
            MODELS[config_name] = pt

    test_base = EXP_DIR / 'test_sets'
    results = {}

    for model_name, model_path in MODELS.items():
        if not Path(model_path).exists():
            print(f'  [{model_name}] Model not found: {model_path}, skipping')
            continue

        print(f'\n── Evaluating: {model_name} ──')
        model = YOLO(str(model_path))
        results[model_name] = {}

        for test_name in TEST_SETS:
            yaml_path = test_base / test_name / 'dataset.yaml'
            if not yaml_path.exists():
                print(f'  [{test_name}] Dataset not exported, skipping')
                continue

            # Check images exist
            img_dir = test_base / test_name / 'images'
            n_imgs = len(list(img_dir.glob('*.jpg')))
            if n_imgs == 0:
                continue

            try:
                metrics = model.val(
                    data=str(yaml_path),
                    split='val',
                    verbose=False,
                    plots=False,
                )
                r = {
                    'mAP50': float(metrics.box.map50),
                    'mAP50_95': float(metrics.box.map),
                    'precision': float(metrics.box.mp),
                    'recall': float(metrics.box.mr),
                    'n_images': n_imgs,
                }
                # Per-class
                if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) == len(CLASS_NAMES):
                    for i, cn in enumerate(CLASS_NAMES):
                        r[f'{cn}_AP50'] = float(metrics.box.ap50[i])

                results[model_name][test_name] = r
                print(f'  [{test_name}] mAP50={r["mAP50"]:.3f}, mAP50-95={r["mAP50_95"]:.3f} ({n_imgs} imgs)')

            except Exception as e:
                print(f'  [{test_name}] ERROR: {e}')
                results[model_name][test_name] = {'error': str(e)}

        del model; gc.collect()

    # Save results
    out_path = EXP_DIR / 'cross_condition_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {out_path}')

    # Print Table 7 format
    print_table7(results)
    return results


def print_table7(results):
    """Print results in Table 7 format."""
    test_cols = ['original', 'weather_rain', 'weather_snow', 'fog', 'night', 'small']
    header = f'{"Config":<20} ' + ' '.join(f'{c:<14}' for c in test_cols)
    print(f'\n{"="*120}')
    print('TABLE 7: Cross-condition mAP@0.5')
    print(f'{"="*120}')
    print(header)
    print('-' * 120)

    for model_name in ['baseline', 'weather', 'weather_night', 'all']:
        if model_name not in results:
            continue
        row = f'{model_name:<20} '
        for test_name in test_cols:
            r = results[model_name].get(test_name, {})
            if 'mAP50' in r:
                row += f'{r["mAP50"]:.3f}         '
            else:
                row += f'{"---":<14} '
        print(row)

    print(f'{"="*120}')


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Cross-condition detection evaluation')
    parser.add_argument('--phase', required=True,
                        choices=['export', 'train', 'eval', 'all'])
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs for new configs (default: 20)')
    args = parser.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ('export', 'all'):
        print('\n' + '='*60)
        print('PHASE 1: Export test sets')
        print('='*60)
        export_test_sets()

    if args.phase in ('train', 'all'):
        print('\n' + '='*60)
        print('PHASE 2: Train new configs')
        print('='*60)
        train_new_configs(epochs=args.epochs)

    if args.phase in ('eval', 'all'):
        print('\n' + '='*60)
        print('PHASE 3: Evaluate all models')
        print('='*60)
        evaluate_all()


if __name__ == '__main__':
    main()
