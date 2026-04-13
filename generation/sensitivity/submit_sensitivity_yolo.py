#!/usr/bin/env python3
"""
Submit SLURM jobs for YOLOv8 training across threshold configs.

Workflow:
  1. Run ssim_lpips_sweep.py         → analyze thresholds (CPU)
  2. Run filter_arrow_by_threshold.py → create filtered Arrow per config (CPU)
  3. Run this script                  → submit YOLOv8 training per config (GPU)

Each job:
  a) Exports filtered Arrow → YOLO format (images/ + labels/)
  b) Trains YOLOv8n for 50 epochs
  c) Saves metrics JSON
"""

import argparse
import os
import sys
from pathlib import Path
import json

BASE = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')
AUG  = BASE / 'augmentation_data' / 'construction_site' / 'rain_snow'
SENS = BASE / 'generation' / 'sensitivity'
DETECT_SCRIPT = BASE / 'benchmarks' / 'detection' / 'train_yolo.py'
EXPORT_SCRIPT = BASE / 'generation' / 'utils' / 'export_arrow_to_train.py'

CONFIGS = ['loose', 'moderate', 'current', 'strict', 'tight', 'no_lpips']

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=yolo_sens_{config}
#SBATCH --output={log_dir}/yolo_sens_{config}_%j.out
#SBATCH --error={log_dir}/yolo_sens_{config}_%j.err
#SBATCH --partition=nextgen
#SBATCH --account=pgs0407
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4

echo "=== Sensitivity: {config} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

module load cuda/12.8.1
module load miniconda3/24.1.2-py310
source activate VLM

cd {base_dir}

# Step 1: Filter Arrow files for this threshold config
echo ""
echo "── Step 1: Filter Arrow data ──"
python {filter_script} --config {config} --split train --method diffusion

# Step 2: Export filtered Arrow → YOLO format
echo ""
echo "── Step 2: Export to YOLO format ──"
FILTERED_DIR="{filtered_dir}/{config}"
YOLO_DIR="{output_dir}/yolo_data_{config}"

# Clean previous
rm -rf "$YOLO_DIR"
mkdir -p "$YOLO_DIR/images/train" "$YOLO_DIR/labels/train"

python {export_script} \\
    --input-dir "$FILTERED_DIR" \\
    --output-dir "$YOLO_DIR" \\
    --split train \\
    2>&1 || echo "Export script not available, using direct Arrow loading"

# Also include original data
ORIG_DIR="{orig_data_dir}"
if [ -d "$ORIG_DIR/images" ] && [ -d "$ORIG_DIR/annotations" ]; then
    echo "Adding original data..."
    for img in "$ORIG_DIR/images"/*.jpg; do
        base=$(basename "$img")
        cp "$img" "$YOLO_DIR/images/train/orig_$base"
    done
    for ann in "$ORIG_DIR/annotations"/*.json; do
        base=$(basename "$ann" .json)
        # Convert JSON annotation to YOLO format
        python -c "
import json, sys
ann = json.load(open('$ann'))
classes = ['excavator', 'rebar', 'worker_with_white_hard_hat']
lines = []
for ci, cn in enumerate(classes):
    boxes = ann.get(cn) or ann.get('objects', {{}}).get(cn, [])
    for b in (boxes or []):
        if len(b) >= 4:
            xc, yc = (b[0]+b[2])/2, (b[1]+b[3])/2
            w, h = b[2]-b[0], b[3]-b[1]
            if w > 0.001 and h > 0.001:
                lines.append(f'{{ci}} {{xc:.6f}} {{yc:.6f}} {{w:.6f}} {{h:.6f}}')
with open('$YOLO_DIR/labels/train/orig_$base.txt', 'w') as f:
    f.write('\\n'.join(lines))
" 2>/dev/null
    done
fi

# Prepare val set (use existing val)
VAL_DIR="{val_dir}"
if [ -d "$VAL_DIR" ]; then
    mkdir -p "$YOLO_DIR/images/val" "$YOLO_DIR/labels/val"
    cp -r "$VAL_DIR/images"/* "$YOLO_DIR/images/val/" 2>/dev/null
    cp -r "$VAL_DIR/labels"/* "$YOLO_DIR/labels/val/" 2>/dev/null
    echo "Validation set ready"
fi

# Create dataset.yaml
cat > "$YOLO_DIR/dataset.yaml" << YAMLEOF
path: $YOLO_DIR
train: images/train
val: images/val
names:
  0: excavator
  1: rebar
  2: worker_with_white_hard_hat
nc: 3
YAMLEOF

TRAIN_COUNT=$(ls "$YOLO_DIR/images/train"/*.jpg 2>/dev/null | wc -l)
VAL_COUNT=$(ls "$YOLO_DIR/images/val"/*.jpg 2>/dev/null | wc -l)
echo "Train images: $TRAIN_COUNT"
echo "Val images:   $VAL_COUNT"

# Step 3: Train YOLOv8
echo ""
echo "── Step 3: Train YOLOv8n ──"
python -c "
import torch, json
from ultralytics import YOLO
from pathlib import Path

model = YOLO('yolov8n.pt')
results = model.train(
    data='$YOLO_DIR/dataset.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    project='{output_dir}/runs',
    name='sens_{config}',
    exist_ok=True,
    device=0,
    workers=2,
    patience=10,
    seed=42,
)

best = Path('{output_dir}/runs/sens_{config}/weights/best.pt')
model = YOLO(best)
val = model.val(data='$YOLO_DIR/dataset.yaml', split='val')

metrics = {{
    'config': 'sens_{config}',
    'ssim_rain': {ssim_rain},
    'ssim_snow': {ssim_snow},
    'lpips': {lpips},
    'mAP50': float(val.box.map50),
    'mAP50_95': float(val.box.map),
    'precision': float(val.box.mp),
    'recall': float(val.box.mr),
    'train_images': $TRAIN_COUNT,
    'val_images': $VAL_COUNT,
}}

classes = ['excavator', 'rebar', 'worker_with_white_hard_hat']
for i, cn in enumerate(classes):
    if i < len(val.box.ap50):
        metrics[f'{{cn}}_AP50'] = float(val.box.ap50[i])

out = Path('{output_dir}') / f'metrics_sens_{config}.json'
json.dump(metrics, open(out, 'w'), indent=2)
print(f'Metrics saved: {{out}}')
print(f'mAP@0.5: {{metrics[\"mAP50\"]:.4f}}')
print(f'mAP@0.5:0.95: {{metrics[\"mAP50_95\"]:.4f}}')
"

echo ""
echo "=== Done: {config} ==="
date
"""

CONFIGS_PARAMS = {
    'loose':    {'ssim_rain': 0.40, 'ssim_snow': 0.40, 'lpips': 0.50},
    'moderate': {'ssim_rain': 0.50, 'ssim_snow': 0.50, 'lpips': 0.40},
    'current':  {'ssim_rain': 0.60, 'ssim_snow': 0.50, 'lpips': 0.35},
    'strict':   {'ssim_rain': 0.65, 'ssim_snow': 0.60, 'lpips': 0.30},
    'tight':    {'ssim_rain': 0.70, 'ssim_snow': 0.65, 'lpips': 0.25},
    'no_lpips': {'ssim_rain': 0.60, 'ssim_snow': 0.50, 'lpips': 1.00},
}


def main():
    parser = argparse.ArgumentParser(description='Submit SLURM jobs for sensitivity YOLOv8 training')
    parser.add_argument('--configs', nargs='+', default=CONFIGS,
                        choices=CONFIGS,
                        help='Which configs to run')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate scripts without submitting')
    parser.add_argument('--output-dir', type=str,
                        default=str(BASE / 'generation' / 'sensitivity' / 'detection_results'),
                        help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    jobs_dir = SENS / 'jobs'
    log_dir = SENS / 'logs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_data = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data/construction_site-test')
    val_dir = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/validation_data/downstream_detection/val_from_train')

    submitted = []
    for config in args.configs:
        params = CONFIGS_PARAMS[config]
        script_content = SLURM_TEMPLATE.format(
            config=config,
            base_dir=str(BASE),
            log_dir=str(log_dir),
            filter_script=str(SENS / 'filter_arrow_by_threshold.py'),
            export_script=str(EXPORT_SCRIPT),
            filtered_dir=str(AUG / 'filtered'),
            output_dir=str(output_dir),
            orig_data_dir=str(orig_data),
            val_dir=str(val_dir),
            ssim_rain=params['ssim_rain'],
            ssim_snow=params['ssim_snow'],
            lpips=params['lpips'],
        )

        script_path = jobs_dir / f'yolo_sens_{config}.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)

        if args.dry_run:
            print(f'  [DRY RUN] Generated: {script_path}')
        else:
            os.system(f'sbatch {script_path}')
            print(f'  Submitted: {config} ({script_path})')
        submitted.append(config)

    print(f'\n{"Generated" if args.dry_run else "Submitted"} {len(submitted)} jobs: {submitted}')
    print(f'Logs:    {log_dir}/')
    print(f'Results: {output_dir}/')

    if args.dry_run:
        print('\nRun without --dry-run to submit to SLURM.')


if __name__ == '__main__':
    main()
