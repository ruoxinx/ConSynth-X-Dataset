"""
Training script for Object Detection using YOLOv8.

Trains YOLOv8 models on different data configurations:
1. Original data only
2. Original + Weather augmentation
3. Original + Weather + Small augmentation
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from io import BytesIO
from tqdm import tqdm
import torch

# Ultralytics YOLOv8
from ultralytics import YOLO

# For loading Arrow data
from datasets import Dataset as HFDataset

# Class definitions
CLASS_NAMES = ['excavator', 'rebar', 'worker_with_white_hard_hat']
NUM_CLASSES = len(CLASS_NAMES)


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


def prepare_yolo_dataset(sources, output_dir, split='train', max_samples=None):
    """
    Convert folder data to YOLO format.
    
    YOLO format:
    - images/train/  -> image files
    - labels/train/  -> txt files with: class_id x_center y_center width height (normalized)
    
    Args:
        sources: List of data source paths (folders with images/annotations)
        output_dir: Output directory for YOLO dataset
        split: 'train' or 'val'
        max_samples: Maximum samples to use
    
    Returns:
        Number of samples prepared
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images' / split
    labels_dir = output_dir / 'labels' / split
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    sample_count = 0
    
    for source in sources:
        source = Path(source)
        
        if not source.exists():
            print(f"Warning: Source not found: {source}")
            continue
        
        # Find all image/annotation folder pairs
        pairs = find_image_annotation_pairs(source)
        
        if not pairs:
            print(f"Warning: No images/annotations found in {source}")
            continue
        
        print(f"Found {len(pairs)} data folder(s) in {source}")
        
        for images_src, annotations_src in pairs:
            folder_name = images_src.parent.name
            
            image_files = list(images_src.glob('*.jpg')) + list(images_src.glob('*.png'))
            
            for img_file in tqdm(image_files, desc=f"Processing {folder_name}"):
                if max_samples and sample_count >= max_samples:
                    break
                
                # Copy image
                img_filename = f"{source.name}_{folder_name}_{img_file.name}"
                shutil.copy(img_file, images_dir / img_filename)
                
                # Load and convert annotation
                ann_file = annotations_src / f"{img_file.stem}.json"
                labels = []
                
                if ann_file.exists():
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
                                
                                if width > 0.001 and height > 0.001:  # Filter tiny boxes
                                    labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Save labels
                label_filename = img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
                with open(labels_dir / label_filename, 'w') as f:
                    f.write('\n'.join(labels))
                
                sample_count += 1
            
            if max_samples and sample_count >= max_samples:
                break
        
        if max_samples and sample_count >= max_samples:
            break
    
    print(f"Prepared {sample_count} samples for {split}")
    return sample_count


def create_dataset_yaml(output_dir, config_name):
    """Create YOLO dataset.yaml file."""
    output_dir = Path(output_dir)
    
    yaml_content = f"""# YOLOv8 Dataset Config - {config_name}
path: {output_dir.absolute()}
train: images/train
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
    
    return yaml_path


def train_yolo(config_name, train_sources, eval_source, output_dir, 
               epochs=50, batch_size=16, imgsz=640, model_size='n',
               max_train_samples=None, eval_samples=1000):
    """
    Train YOLOv8 model.
    
    Args:
        config_name: Configuration name
        train_sources: List of training data sources
        eval_source: Evaluation data source
        output_dir: Output directory
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        max_train_samples: Max training samples
        eval_samples: Number of evaluation samples
    
    Returns:
        Training results
    """
    output_dir = Path(output_dir)
    dataset_dir = output_dir / f'yolo_data_{config_name}'
    
    # Clean previous data if exists
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    
    print(f"\n{'='*60}")
    print(f"Preparing dataset: {config_name}")
    print(f"{'='*60}")
    
    # Prepare training data
    print("\nPreparing training data...")
    train_count = prepare_yolo_dataset(
        train_sources, 
        dataset_dir, 
        split='train',
        max_samples=max_train_samples
    )
    
    # Prepare validation data
    print("\nPreparing validation data...")
    val_count = prepare_yolo_dataset(
        [eval_source], 
        dataset_dir, 
        split='val',
        max_samples=eval_samples
    )
    
    # Create dataset yaml
    yaml_path = create_dataset_yaml(dataset_dir, config_name)
    
    print(f"\n{'='*60}")
    print(f"Training YOLOv8{model_size}: {config_name}")
    print(f"{'='*60}")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {imgsz}")
    
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=str(output_dir / 'runs'),
        name=config_name,
        exist_ok=True,
        verbose=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=1,  # Reduced to avoid DataLoader warnings on cluster
        patience=10,  # Early stopping patience
        save=True,
        plots=True,
    )
    
    # Get best model path
    best_model_path = output_dir / 'runs' / config_name / 'weights' / 'best.pt'
    
    # Validate on val set
    print(f"\n{'='*60}")
    print(f"Validation Results: {config_name}")
    print(f"{'='*60}")
    
    model = YOLO(best_model_path)
    val_results = model.val(data=str(yaml_path), split='val')
    
    # Extract metrics
    metrics = {
        'config': config_name,
        'mAP50': float(val_results.box.map50),
        'mAP50-95': float(val_results.box.map),
        'precision': float(val_results.box.mp),
        'recall': float(val_results.box.mr),
        'train_samples': train_count,
        'val_samples': val_count,
        'epochs': epochs,
        'model_size': model_size,
    }
    
    # Per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(val_results.box.ap50):
            metrics[f'{class_name}_AP50'] = float(val_results.box.ap50[i])
    
    print(f"\nmAP@0.5: {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    for class_name in CLASS_NAMES:
        key = f'{class_name}_AP50'
        if key in metrics:
            print(f"  {class_name} AP@0.5: {metrics[key]:.4f}")
    
    # Save metrics
    metrics_file = output_dir / f'metrics_{config_name}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 object detection model')
    parser.add_argument('--config', type=str, required=True,
                        choices=['original', 'original_weather', 'original_weather_small', 
                                 'original_weather_daynight', 'original_weather_small_daynight'],
                        help='Training configuration')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--model-size', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--output-dir', type=str, 
                        default='/users/PGS0407/binben14/VietHuy/construction-site/validation_data/downstream_detection/checkpoints',
                        help='Output directory')
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='Maximum training samples')
    parser.add_argument('--eval-samples', type=int, default=1000,
                        help='Number of evaluation samples')
    args = parser.parse_args()
    
    # Paths
    BASE_DIR = Path("/users/PGS0407/binben14/VietHuy/construction-site")
    AUG_DATA_DIR = BASE_DIR / "augmentation_data"
    
    # Check CUDA
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Build training data sources
    train_sources = []
    
    # Original data
    original_path = AUG_DATA_DIR / "construction_site-test"
    if original_path.exists():
        train_sources.append(original_path)
        print(f"Added: Original data - {original_path}")
    
    if args.config in ['original_weather', 'original_weather_small', 'original_weather_daynight', 'original_weather_small_daynight']:
        weather_path = AUG_DATA_DIR / "weather"
        if weather_path.exists():
            train_sources.append(weather_path)
            print(f"Added: Weather augmentation - {weather_path}")
    
    if args.config in ['original_weather_small', 'original_weather_small_daynight']:
        small_path = AUG_DATA_DIR / "small"
        if small_path.exists():
            train_sources.append(small_path)
            print(f"Added: Small augmentation - {small_path}")
    
    if args.config in ['original_weather_daynight', 'original_weather_small_daynight']:
        night_path = AUG_DATA_DIR / "night"
        if night_path.exists():
            train_sources.append(night_path)
            print(f"Added: Day2Night augmentation - {night_path}")
    
    print(f"\nTotal training sources: {len(train_sources)}")
    
    # Evaluation data - use separate validation set extracted from train arrow file
    # This ensures val data is different from train data
    val_path = BASE_DIR / "validation_data" / "downstream_detection" / "val_from_train"
    if val_path.exists():
        eval_source = val_path
        print(f"Using separate validation set: {val_path}")
    else:
        eval_source = original_path
        print(f"Warning: val_from_train not found, using original_path: {original_path}")
    
    # Train
    metrics = train_yolo(
        config_name=args.config,
        train_sources=train_sources,
        eval_source=eval_source,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        model_size=args.model_size,
        max_train_samples=args.max_train_samples,
        eval_samples=args.eval_samples,
    )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
