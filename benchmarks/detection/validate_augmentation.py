"""
Validation script for evaluating YOLOv8 models on different augmentation types.

This script:
1. Loads trained YOLOv8 models
2. Validates on augmentation test data (2000 samples per type)
3. Produces overall and per-augmentation analysis

Usage:
    python validate_augmentation.py --model all --max-samples 2000
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random

# Ultralytics YOLOv8
from ultralytics import YOLO

# Class definitions
CLASS_NAMES = ['excavator', 'rebar', 'worker_with_white_hard_hat']
NUM_CLASSES = len(CLASS_NAMES)

# Model checkpoints
BASE_DIR = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite")
CHECKPOINTS_DIR = BASE_DIR / "validation_data" / "downstream_detection" / "checkpoints" / "runs"

MODEL_PATHS = {
    'original': CHECKPOINTS_DIR / "original" / "weights" / "best.pt",
    'original_weather': CHECKPOINTS_DIR / "original_weather" / "weights" / "best.pt",
    'original_weather_small': CHECKPOINTS_DIR / "original_weather_small" / "weights" / "best.pt",
    'original_weather_daynight': CHECKPOINTS_DIR / "original_weather_daynight" / "weights" / "best.pt",
    'original_weather_small_daynight': CHECKPOINTS_DIR / "original_weather_small_daynight" / "weights" / "best.pt",
}

# Augmentation data paths
AUG_DATA_DIR = BASE_DIR / "augmentation_data"

AUGMENTATION_SOURCES = {
    'original': {
        'path': AUG_DATA_DIR / "construction_site-test",
        'description': 'Original Test Data'
    },
    'night': {
        'path': AUG_DATA_DIR / "night" / "construction_site-test",
        'description': 'Day2Night Augmentation'
    },
    'small': {
        'path': AUG_DATA_DIR / "small" / "construction_site-test",
        'description': 'Small Object Augmentation'
    },
    'weather_rain_0': {
        'path': AUG_DATA_DIR / "weather" / "construction_site_test" / "day" / "filtered" / "style_rain_0",
        'description': 'Weather: Rain Level 0'
    },
    'weather_rain_1': {
        'path': AUG_DATA_DIR / "weather" / "construction_site_test" / "day" / "filtered" / "style_rain_1",
        'description': 'Weather: Rain Level 1'
    },
    'weather_rain_2': {
        'path': AUG_DATA_DIR / "weather" / "construction_site_test" / "day" / "filtered" / "style_rain_2",
        'description': 'Weather: Rain Level 2'
    },
    'weather_snow_0': {
        'path': AUG_DATA_DIR / "weather" / "construction_site_test" / "day" / "filtered" / "style_snow_0",
        'description': 'Weather: Snow Level 0'
    },
    'weather_snow_1': {
        'path': AUG_DATA_DIR / "weather" / "construction_site_test" / "day" / "filtered" / "style_snow_1",
        'description': 'Weather: Snow Level 1'
    },
    'weather_snow_2': {
        'path': AUG_DATA_DIR / "weather" / "construction_site_test" / "day" / "filtered" / "style_snow_2",
        'description': 'Weather: Snow Level 2'
    },
}


def find_image_annotation_pairs(source_dir):
    """
    Recursively find all (images_dir, annotations_dir) pairs in a source directory.
    """
    source_dir = Path(source_dir)
    pairs = []
    
    # Check if source has direct images/ folder
    if (source_dir / 'images').exists() and (source_dir / 'annotations').exists():
        pairs.append((source_dir / 'images', source_dir / 'annotations'))
    
    # Search for nested images/annotations folders
    for images_dir in source_dir.rglob('images'):
        if images_dir.is_dir():
            annotations_dir = images_dir.parent / 'annotations'
            if annotations_dir.exists():
                pair = (images_dir, annotations_dir)
                if pair not in pairs:
                    pairs.append(pair)
    
    return pairs


def get_boxes_from_annotation(ann, class_name):
    """Extract boxes from annotation, handling different JSON formats."""
    # Try direct format: ann['excavator']
    boxes = ann.get(class_name)
    if boxes is not None:
        return boxes
    
    # Try nested format: ann['objects']['excavator']
    objects = ann.get('objects', {})
    if objects:
        boxes = objects.get(class_name)
        if boxes is not None:
            return boxes
    
    return []


def prepare_yolo_validation_dataset(sources, output_dir, max_samples=2000, seed=42):
    """
    Prepare YOLO validation dataset from multiple sources.
    
    Args:
        sources: Dict of {aug_name: source_path}
        output_dir: Output directory
        max_samples: Maximum samples per augmentation type
        seed: Random seed for reproducibility
    
    Returns:
        Dict with per-augmentation sample counts and image lists
    """
    random.seed(seed)
    
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images' / 'val'
    labels_dir = output_dir / 'labels' / 'val'
    
    # Clean previous
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    aug_stats = {}
    aug_images = {}  # Track which images belong to which augmentation
    
    for aug_name, source_info in sources.items():
        source = Path(source_info['path'])
        
        if not source.exists():
            print(f"Warning: Source not found: {source}")
            aug_stats[aug_name] = 0
            aug_images[aug_name] = []
            continue
        
        # Find all image/annotation folder pairs
        pairs = find_image_annotation_pairs(source)
        
        if not pairs:
            print(f"Warning: No images/annotations found in {source}")
            aug_stats[aug_name] = 0
            aug_images[aug_name] = []
            continue
        
        # Collect all image files
        all_image_files = []
        for images_src, annotations_src in pairs:
            for img_file in images_src.glob('*.jpg'):
                ann_file = annotations_src / f"{img_file.stem}.json"
                if ann_file.exists():
                    all_image_files.append((img_file, ann_file))
            for img_file in images_src.glob('*.png'):
                ann_file = annotations_src / f"{img_file.stem}.json"
                if ann_file.exists():
                    all_image_files.append((img_file, ann_file))
        
        # Sample if needed
        if len(all_image_files) > max_samples:
            all_image_files = random.sample(all_image_files, max_samples)
        
        print(f"Processing {aug_name}: {len(all_image_files)} samples")
        
        aug_images[aug_name] = []
        sample_count = 0
        
        for img_file, ann_file in tqdm(all_image_files, desc=aug_name):
            # Create unique filename with augmentation prefix
            img_filename = f"{aug_name}_{img_file.name}"
            
            # Copy image
            shutil.copy(img_file, images_dir / img_filename)
            
            # Load and convert annotation
            labels = []
            with open(ann_file) as f:
                ann = json.load(f)
            
            for class_id, class_name in enumerate(CLASS_NAMES):
                boxes = get_boxes_from_annotation(ann, class_name)
                if boxes is None:
                    continue
                
                for box in boxes:
                    if len(box) >= 4:
                        # Boxes are already normalized (0-1)
                        x1, y1, x2, y2 = box[:4]
                        
                        # Convert to YOLO format (x_center, y_center, w, h)
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Clamp values
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        if width > 0.001 and height > 0.001:
                            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save labels
            label_filename = img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            with open(labels_dir / label_filename, 'w') as f:
                f.write('\n'.join(labels))
            
            aug_images[aug_name].append(img_filename)
            sample_count += 1
        
        aug_stats[aug_name] = sample_count
    
    # Create dataset yaml
    yaml_content = f"""# YOLOv8 Augmentation Validation Dataset
path: {output_dir.absolute()}
train: images/val  # Using val as train since we only need val
val: images/val

# Classes
names:
  0: excavator
  1: rebar
  2: worker_with_white_hard_hat

nc: {NUM_CLASSES}
"""
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Save augmentation image mapping
    mapping_file = output_dir / 'augmentation_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(aug_images, f, indent=2)
    
    total_samples = sum(aug_stats.values())
    print(f"\nTotal samples prepared: {total_samples}")
    for aug_name, count in aug_stats.items():
        print(f"  {aug_name}: {count}")
    
    return aug_stats, aug_images, yaml_path


def validate_model_on_dataset(model_path, dataset_yaml, model_name):
    """
    Validate model on dataset and return metrics.
    """
    print(f"\nValidating model: {model_name}")
    print(f"Model path: {model_path}")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return None
    
    model = YOLO(model_path)
    results = model.val(data=str(dataset_yaml), split='val', verbose=False)
    
    metrics = {
        'model': model_name,
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }
    
    # Per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(results.box.ap50):
            metrics[f'{class_name}_AP50'] = float(results.box.ap50[i])
    
    return metrics


def validate_model_per_augmentation(model_path, aug_images, base_output_dir, model_name):
    """
    Validate model on each augmentation type separately.
    
    Returns per-augmentation metrics.
    """
    print(f"\n{'='*60}")
    print(f"Per-Augmentation Validation: {model_name}")
    print(f"{'='*60}")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return {}
    
    model = YOLO(model_path)
    all_metrics = {}
    
    for aug_name, images in aug_images.items():
        if not images:
            continue
        
        # Create temp dataset for this augmentation type
        temp_dir = base_output_dir / f'temp_{aug_name}'
        temp_images_dir = temp_dir / 'images' / 'val'
        temp_labels_dir = temp_dir / 'labels' / 'val'
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        temp_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images and labels for this augmentation
        src_images_dir = base_output_dir / 'images' / 'val'
        src_labels_dir = base_output_dir / 'labels' / 'val'
        
        for img_name in images:
            shutil.copy(src_images_dir / img_name, temp_images_dir / img_name)
            label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            label_file = src_labels_dir / label_name
            if label_file.exists():
                shutil.copy(label_file, temp_labels_dir / label_name)
        
        # Create yaml
        yaml_content = f"""path: {temp_dir.absolute()}
train: images/val
val: images/val
names:
  0: excavator
  1: rebar
  2: worker_with_white_hard_hat
nc: {NUM_CLASSES}
"""
        yaml_path = temp_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        # Validate
        try:
            results = model.val(data=str(yaml_path), split='val', verbose=False)
            
            metrics = {
                'augmentation': aug_name,
                'samples': len(images),
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
            }
            
            for i, class_name in enumerate(CLASS_NAMES):
                if i < len(results.box.ap50):
                    metrics[f'{class_name}_AP50'] = float(results.box.ap50[i])
            
            all_metrics[aug_name] = metrics
            print(f"  {aug_name}: mAP50={metrics['mAP50']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        
        except Exception as e:
            print(f"  Error validating {aug_name}: {e}")
            all_metrics[aug_name] = {'augmentation': aug_name, 'samples': len(images), 'error': str(e)}
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
    
    return all_metrics


def generate_analysis_report(all_results, output_dir):
    """
    Generate comprehensive analysis report.
    """
    report_path = output_dir / 'augmentation_analysis_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Augmentation Validation Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall Results Table
        f.write("## 1. Overall Results (Combined Augmentation Test Set)\n\n")
        f.write("| Model | mAP50 | mAP50-95 | Precision | Recall | Excavator AP | Rebar AP | Worker AP |\n")
        f.write("|-------|-------|----------|-----------|--------|--------------|----------|----------|\n")
        
        for model_name, data in all_results.items():
            overall = data.get('overall', {})
            if overall:
                f.write(f"| {model_name} | {overall.get('mAP50', 0):.4f} | {overall.get('mAP50-95', 0):.4f} | ")
                f.write(f"{overall.get('precision', 0):.4f} | {overall.get('recall', 0):.4f} | ")
                f.write(f"{overall.get('excavator_AP50', 0):.4f} | {overall.get('rebar_AP50', 0):.4f} | ")
                f.write(f"{overall.get('worker_with_white_hard_hat_AP50', 0):.4f} |\n")
        
        # Per-Augmentation Analysis
        f.write("\n## 2. Per-Augmentation Analysis\n\n")
        
        for model_name, data in all_results.items():
            f.write(f"\n### {model_name}\n\n")
            
            per_aug = data.get('per_augmentation', {})
            if not per_aug:
                f.write("No per-augmentation data available.\n")
                continue
            
            f.write("| Augmentation | Samples | mAP50 | mAP50-95 | Precision | Recall |\n")
            f.write("|--------------|---------|-------|----------|-----------|--------|\n")
            
            for aug_name, metrics in per_aug.items():
                if 'error' in metrics:
                    f.write(f"| {aug_name} | {metrics.get('samples', 0)} | ERROR | - | - | - |\n")
                else:
                    f.write(f"| {aug_name} | {metrics.get('samples', 0)} | {metrics.get('mAP50', 0):.4f} | ")
                    f.write(f"{metrics.get('mAP50-95', 0):.4f} | {metrics.get('precision', 0):.4f} | ")
                    f.write(f"{metrics.get('recall', 0):.4f} |\n")
        
        # Strengths and Weaknesses Analysis
        f.write("\n## 3. Strengths and Weaknesses Analysis\n\n")
        
        for model_name, data in all_results.items():
            f.write(f"\n### {model_name}\n\n")
            
            per_aug = data.get('per_augmentation', {})
            if not per_aug:
                continue
            
            # Find best and worst augmentations
            aug_scores = [(aug, m.get('mAP50', 0)) for aug, m in per_aug.items() if 'mAP50' in m]
            if not aug_scores:
                continue
            
            aug_scores.sort(key=lambda x: x[1], reverse=True)
            
            f.write("**Strengths (Best Performance):**\n")
            for aug, score in aug_scores[:3]:
                desc = AUGMENTATION_SOURCES.get(aug, {}).get('description', aug)
                f.write(f"- {desc}: mAP50 = {score:.4f}\n")
            
            f.write("\n**Weaknesses (Needs Improvement):**\n")
            for aug, score in aug_scores[-3:]:
                desc = AUGMENTATION_SOURCES.get(aug, {}).get('description', aug)
                f.write(f"- {desc}: mAP50 = {score:.4f}\n")
            
            # Class-level analysis
            f.write("\n**Per-Class Performance:**\n")
            for class_name in CLASS_NAMES:
                class_key = f'{class_name}_AP50'
                class_scores = [(aug, m.get(class_key, 0)) for aug, m in per_aug.items() if class_key in m]
                if class_scores:
                    avg_score = np.mean([s for _, s in class_scores])
                    f.write(f"- {class_name}: avg AP50 = {avg_score:.4f}\n")
        
        # Comparison Matrix
        f.write("\n## 4. Model Comparison by Augmentation Type\n\n")
        
        # Get all augmentation types
        all_augs = set()
        for data in all_results.values():
            all_augs.update(data.get('per_augmentation', {}).keys())
        
        if all_augs:
            # Header
            header = "| Augmentation |"
            for model_name in all_results.keys():
                header += f" {model_name} |"
            f.write(header + "\n")
            f.write("|" + "---|" * (len(all_results) + 1) + "\n")
            
            for aug in sorted(all_augs):
                row = f"| {aug} |"
                for model_name, data in all_results.items():
                    per_aug = data.get('per_augmentation', {})
                    score = per_aug.get(aug, {}).get('mAP50', '-')
                    if isinstance(score, float):
                        row += f" {score:.4f} |"
                    else:
                        row += f" {score} |"
                f.write(row + "\n")
    
    print(f"\nAnalysis report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Validate YOLOv8 models on augmentation data')
    parser.add_argument('--model', type=str, default='all',
                        choices=['original', 'original_weather', 'original_weather_small', 
                                 'original_weather_daynight', 'original_weather_small_daynight', 'all'],
                        help='Model to validate')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Maximum samples per augmentation type')
    parser.add_argument('--output-dir', type=str, 
                        default=str(BASE_DIR / "validation_data" / "downstream_detection" / "augmentation_validation"),
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--skip-preparation', action='store_true',
                        help='Skip data preparation if already done')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Augmentation Validation Analysis")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Max samples per augmentation: {args.max_samples}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Select models to validate
    if args.model == 'all':
        models_to_validate = MODEL_PATHS
    else:
        models_to_validate = {args.model: MODEL_PATHS[args.model]}
    
    # Prepare validation dataset
    dataset_dir = output_dir / 'val_dataset'
    mapping_file = dataset_dir / 'augmentation_mapping.json'
    yaml_path = dataset_dir / 'dataset.yaml'
    
    if args.skip_preparation and mapping_file.exists() and yaml_path.exists():
        print("\nLoading existing dataset preparation...")
        with open(mapping_file) as f:
            aug_images = json.load(f)
        aug_stats = {k: len(v) for k, v in aug_images.items()}
    else:
        print("\nPreparing validation dataset...")
        aug_stats, aug_images, yaml_path = prepare_yolo_validation_dataset(
            AUGMENTATION_SOURCES,
            dataset_dir,
            max_samples=args.max_samples,
            seed=args.seed
        )
    
    # Validate models
    all_results = {}
    
    for model_name, model_path in models_to_validate.items():
        print(f"\n{'='*60}")
        print(f"Validating: {model_name}")
        print(f"{'='*60}")
        
        # Overall validation
        overall_metrics = validate_model_on_dataset(model_path, yaml_path, model_name)
        
        # Per-augmentation validation
        per_aug_metrics = validate_model_per_augmentation(
            model_path, aug_images, dataset_dir, model_name
        )
        
        all_results[model_name] = {
            'overall': overall_metrics,
            'per_augmentation': per_aug_metrics
        }
    
    # Save results
    results_file = output_dir / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate analysis report
    report_path = generate_analysis_report(all_results, output_dir)
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
