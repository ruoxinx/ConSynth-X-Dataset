"""
Training script for Object Detection downstream task validation.

Trains Faster R-CNN models on different data configurations:
1. Original data only
2. Original + Weather augmentation
3. Original + Weather + Small augmentation
"""

import os
import sys
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

# TorchVision imports
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    ConstructionSiteDataset, 
    collate_fn, 
    get_transform,
    NUM_CLASSES,
    CLASS_NAMES
)


def get_model(num_classes: int, pretrained: bool = True):
    """
    Get Faster R-CNN model with ResNet50-FPN backbone.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained backbone
    
    Returns:
        Faster R-CNN model
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)
    
    # Replace box predictor for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        # Move to device
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Skip empty batches
        valid_targets = [t for t in targets if len(t['boxes']) > 0]
        if len(valid_targets) == 0:
            continue
        
        # Filter images and targets
        valid_indices = [i for i, t in enumerate(targets) if len(t['boxes']) > 0]
        images = [images[i] for i in valid_indices]
        targets = [targets[i] for i in valid_indices]
        
        if len(images) == 0:
            continue
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'avg_loss': f"{total_loss / num_batches:.4f}"
        })
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate the model."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        
        # Get predictions
        outputs = model(images)
        
        all_predictions.extend(outputs)
        all_targets.extend(targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    return metrics


def calculate_metrics(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """Calculate detection metrics (mAP, per-class AP).
    
    AP is calculated over ALL predictions (no score filtering).
    P/R/F1 are calculated at the specified score_threshold.
    """
    # For AP calculation: collect all matches without score filtering
    class_matches = {i: [] for i in range(1, NUM_CLASSES)}  # (score, is_tp)
    class_total_gt = {i: 0 for i in range(1, NUM_CLASSES)}
    
    # For P/R/F1 at threshold
    class_tp = {i: 0 for i in range(1, NUM_CLASSES)}
    class_fp = {i: 0 for i in range(1, NUM_CLASSES)}
    class_fn = {i: 0 for i in range(1, NUM_CLASSES)}
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        
        # Count total ground truth per class
        for gt_label in gt_labels:
            if gt_label > 0:
                class_total_gt[gt_label] += 1
        
        # Track matched ground truth (for AP - no threshold)
        matched_gt_ap = set()
        # Track matched ground truth (for P/R/F1 - with threshold)
        matched_gt_thresh = set()
        
        # Sort by score (descending) - process ALL predictions for AP
        sorted_indices = np.argsort(-pred_scores)
        
        for idx in sorted_indices:
            pred_box = pred_boxes[idx]
            pred_label = pred_labels[idx]
            pred_score = pred_scores[idx]
            
            if pred_label == 0:  # Skip background
                continue
            
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx in matched_gt_ap:
                    continue
                if gt_label != pred_label:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Record match for AP (all predictions)
            is_tp = best_iou >= iou_threshold
            if is_tp:
                matched_gt_ap.add(best_gt_idx)
            class_matches[pred_label].append((pred_score, 1 if is_tp else 0))
            
            # Record for P/R/F1 (only predictions above threshold)
            if pred_score >= score_threshold:
                if is_tp and best_gt_idx not in matched_gt_thresh:
                    class_tp[pred_label] += 1
                    matched_gt_thresh.add(best_gt_idx)
                else:
                    class_fp[pred_label] += 1
        
        # Count false negatives for P/R/F1 (unmatched GT at threshold)
        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx not in matched_gt_thresh and gt_label > 0:
                class_fn[gt_label] += 1
    
    # Calculate metrics per class
    results = {}
    aps = []
    
    for class_id in range(1, NUM_CLASSES):
        class_name = CLASS_NAMES[class_id - 1]
        tp = class_tp[class_id]
        fp = class_fp[class_id]
        fn = class_fn[class_id]
        total_gt = class_total_gt[class_id]
        
        # P/R/F1 at score_threshold
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AP using ALL predictions (full PR curve)
        matches = class_matches[class_id]
        
        if len(matches) > 0 and total_gt > 0:
            # Sort by score descending
            matches = sorted(matches, key=lambda x: -x[0])
            
            # Calculate precision-recall at each threshold
            cum_tp = 0
            cum_fp = 0
            precisions = [1.0]  # Start with (recall=0, precision=1)
            recalls = [0.0]
            
            for score, is_tp in matches:
                if is_tp:
                    cum_tp += 1
                else:
                    cum_fp += 1
                
                p = cum_tp / (cum_tp + cum_fp)
                r = cum_tp / total_gt
                precisions.append(p)
                recalls.append(r)
            
            # Make precision monotonically decreasing (interpolation)
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i + 1])
            
            # Calculate AP as Area Under PR Curve
            ap = 0
            for i in range(1, len(recalls)):
                ap += (recalls[i] - recalls[i-1]) * precisions[i]
        else:
            ap = 0
        
        aps.append(ap)
        
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    results['mAP'] = np.mean(aps)
    
    return results


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--config', type=str, required=True,
                        choices=['original', 'original_weather', 'original_weather_small'],
                        help='Training configuration')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='warmup_cosine',
                        choices=['step', 'cosine', 'plateau', 'warmup_cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr-step-size', type=int, default=3,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(os.environ.get('CONSYNTH_DATA_ROOT', Path.home() / 'consynth_data')) / 'validation_data' / 'downstream_detection' / 'checkpoints'),
                        help='Output directory for checkpoints')
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='Maximum training samples (None for all)')
    parser.add_argument('--eval-samples', type=int, default=1000,
                        help='Number of evaluation samples')
    args = parser.parse_args()
    
    # Paths (override via $CONSYNTH_DATA_ROOT)
    BASE_DIR = Path(os.environ.get('CONSYNTH_DATA_ROOT', Path.home() / 'consynth_data'))
    AUG_DATA_DIR = BASE_DIR / "augmentation_data"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Build training data sources based on config
    train_sources = []
    
    # Original data is always included
    original_path = AUG_DATA_DIR / "construction_site-test"
    if original_path.exists():
        train_sources.append(original_path)
        print(f"Added: Original data - {original_path}")
    
    if args.config in ['original_weather', 'original_weather_small']:
        # Add weather augmentations
        weather_base = AUG_DATA_DIR / "weather" / "construction_site_test" / "day" / "filtered"
        if weather_base.exists():
            for style_dir in sorted(weather_base.iterdir()):
                if style_dir.is_dir() and (style_dir / "images").exists():
                    train_sources.append(style_dir)
                    print(f"Added: Weather aug - {style_dir.name}")
    
    if args.config == 'original_weather_small':
        # Add small augmentation
        small_path = AUG_DATA_DIR / "small" / "construction_site-test"
        if small_path.exists():
            train_sources.append(small_path)
            print(f"Added: Small aug - {small_path}")
    
    print(f"\nTotal training sources: {len(train_sources)}")
    
    # Create datasets
    print("\nLoading training data...")
    train_transform = get_transform(train=True)
    train_dataset = ConstructionSiteDataset(
        train_sources,
        transform=train_transform,
        max_samples=args.max_train_samples
    )
    
    # Evaluation data (from train arrow file)
    print("\nLoading evaluation data...")
    eval_transform = get_transform(train=False)
    eval_arrow = BASE_DIR / "LouisChen15___construction_site" / "construction_site-train-00000-of-00002.arrow"
    eval_dataset = ConstructionSiteDataset(
        eval_arrow,
        transform=eval_transform,
        max_samples=args.eval_samples
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = get_model(NUM_CLASSES, pretrained=True)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.lr_scheduler == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    else:  # 'warmup_cosine'
        # Warmup + Cosine Annealing
        warmup_epochs = min(3, args.epochs // 3)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Using {args.lr_scheduler} learning rate scheduler")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'config': args.config,
        'train_sources': [str(s) for s in train_sources],
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'epochs': [],
        'eval_metrics': []
    }
    
    # Training loop
    print("\n" + "="*60)
    print(f"TRAINING: {args.config}")
    print("="*60)
    
    best_mAP = 0
    current_mAP = 0  # Track current mAP for plateau scheduler
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        history['epochs'].append({'epoch': epoch, 'train_loss': train_loss})
        
        # Update learning rate (plateau scheduler is updated after evaluation)
        if args.lr_scheduler != 'plateau':
            lr_scheduler.step()
        
        # Evaluate
        if epoch % 2 == 0 or epoch == args.epochs:
            print("\nEvaluating...")
            metrics = evaluate(model, eval_loader, device)
            current_mAP = metrics['mAP']
            
            print(f"\nEvaluation Results:")
            print(f"  mAP@0.5: {metrics['mAP']:.4f}")
            for class_name in CLASS_NAMES:
                if class_name in metrics:
                    cm = metrics[class_name]
                    print(f"  {class_name}: P={cm['precision']:.3f}, R={cm['recall']:.3f}, F1={cm['f1']:.3f}")
            
            # Update plateau scheduler with mAP metric
            if args.lr_scheduler == 'plateau':
                lr_scheduler.step(current_mAP)
            
            history['eval_metrics'].append({
                'epoch': epoch,
                'metrics': {k: v if not isinstance(v, dict) else v for k, v in metrics.items()}
            })
            
            # Save best model
            if metrics['mAP'] > best_mAP:
                best_mAP = metrics['mAP']
                checkpoint_path = output_dir / f"model_{args.config}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mAP': best_mAP,
                    'config': args.config
                }, checkpoint_path)
                print(f"\nSaved best model (mAP={best_mAP:.4f}) to {checkpoint_path}")
    
    # Save final model
    checkpoint_path = output_dir / f"model_{args.config}_final.pth"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mAP': metrics['mAP'],
        'config': args.config
    }, checkpoint_path)
    print(f"\nSaved final model to {checkpoint_path}")
    
    # Save history
    history_path = output_dir / f"history_{args.config}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Saved training history to {history_path}")
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE: {args.config}")
    print(f"Best mAP@0.5: {best_mAP:.4f}")
    print("="*60)
    
    return best_mAP


if __name__ == "__main__":
    main()
