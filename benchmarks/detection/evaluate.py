"""
Evaluation and comparison script for trained object detection models.
Evaluates all models on the same test set and compares results.
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TorchVision imports
from torchvision.models.detection import fasterrcnn_resnet50_fpn
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
from train import calculate_iou


def get_model(num_classes: int):
    """Get Faster R-CNN model structure."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    model = get_model(NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint.get('config', 'unknown')


@torch.no_grad()
def evaluate_model(model, data_loader, device, score_threshold=0.5):
    """Evaluate model and return detailed metrics."""
    model.eval()
    
    # Per-class statistics
    class_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0, 'scores': []} 
                   for i in range(1, NUM_CLASSES)}
    
    # Store all predictions for analysis
    all_results = []
    
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
        images = list(img.to(device) for img in images)
        outputs = model(images)
        
        for img_idx, (output, target) in enumerate(zip(outputs, targets)):
            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            
            gt_boxes = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()
            
            # Filter by score threshold
            mask = pred_scores >= score_threshold
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            pred_scores = pred_scores[mask]
            
            matched_gt = set()
            
            # Match predictions to ground truth
            sorted_indices = np.argsort(-pred_scores)
            
            for idx in sorted_indices:
                pred_box = pred_boxes[idx]
                pred_label = pred_labels[idx]
                pred_score = pred_scores[idx]
                
                if pred_label == 0:
                    continue
                
                class_stats[pred_label]['scores'].append(pred_score)
                
                # Find best matching GT
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_idx in matched_gt or gt_label != pred_label:
                        continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= 0.5:
                    class_stats[pred_label]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    class_stats[pred_label]['fp'] += 1
            
            # Count FN
            for gt_idx, gt_label in enumerate(gt_labels):
                if gt_idx not in matched_gt and gt_label > 0:
                    class_stats[gt_label]['fn'] += 1
            
            all_results.append({
                'batch_idx': batch_idx,
                'img_idx': img_idx,
                'num_pred': len(pred_boxes),
                'num_gt': len(gt_boxes)
            })
    
    # Calculate metrics
    results = {'per_class': {}, 'overall': {}}
    
    precisions = []
    recalls = []
    f1s = []
    
    for class_id in range(1, NUM_CLASSES):
        class_name = CLASS_NAMES[class_id - 1]
        stats = class_stats[class_id]
        
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        results['per_class'][class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': tp + fn
        }
    
    results['overall'] = {
        'mAP': np.mean([p * r for p, r in zip(precisions, recalls)]),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'mean_f1': np.mean(f1s)
    }
    
    return results


def compare_models(results_dict: dict, output_dir: Path):
    """Compare and visualize results from multiple models."""
    
    # Create comparison DataFrame
    comparison_data = []
    
    for model_name, results in results_dict.items():
        row = {
            'Model': model_name,
            'mAP': results['overall']['mAP'],
            'Mean Precision': results['overall']['mean_precision'],
            'Mean Recall': results['overall']['mean_recall'],
            'Mean F1': results['overall']['mean_f1']
        }
        
        for class_name in CLASS_NAMES:
            if class_name in results['per_class']:
                cls_results = results['per_class'][class_name]
                row[f'{class_name}_F1'] = cls_results['f1']
                row[f'{class_name}_P'] = cls_results['precision']
                row[f'{class_name}_R'] = cls_results['recall']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = output_dir / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison to {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(df[['Model', 'mAP', 'Mean Precision', 'Mean Recall', 'Mean F1']].to_string(index=False))
    
    print("\n" + "-"*80)
    print("PER-CLASS F1 SCORES")
    print("-"*80)
    f1_cols = ['Model'] + [f'{c}_F1' for c in CLASS_NAMES]
    print(df[[c for c in f1_cols if c in df.columns]].to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall metrics comparison
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.2
    
    metrics = ['mAP', 'Mean Precision', 'Mean Recall', 'Mean F1']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax1.bar(x + i * width, df[metric], width, label=metric, color=color)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Metrics Comparison')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1)
    
    # Per-class F1 comparison
    ax2 = axes[1]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    for i, model_name in enumerate(df['Model']):
        f1_values = [df[df['Model'] == model_name][f'{c}_F1'].values[0] 
                    for c in CLASS_NAMES if f'{c}_F1' in df.columns]
        ax2.bar(x + i * width, f1_values, width, label=model_name)
    
    ax2.set_xlabel('Class')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Per-Class F1 Comparison')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([c.replace('_', '\n') for c in CLASS_NAMES], fontsize=8)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    fig_path = output_dir / 'model_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {fig_path}")
    
    plt.close()
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare object detection models')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/users/PGS0407/binben14/VietHuy/construction-site/validation_data/downstream_detection/checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--eval-data', type=str,
                        default='/users/PGS0407/binben14/VietHuy/construction-site/LouisChen15___construction_site/construction_site-train-00000-of-00002.arrow',
                        help='Evaluation data path')
    parser.add_argument('--eval-samples', type=int, default=1000,
                        help='Number of evaluation samples')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str,
                        default='/users/PGS0407/binben14/VietHuy/construction-site/validation_data/downstream_detection/results',
                        help='Output directory for results')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Score threshold for predictions')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("model_*_best.pth"))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        print("Looking for any .pth files...")
        checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp.name}")
    
    # Load evaluation data
    print("\nLoading evaluation data...")
    eval_transform = get_transform(train=False)
    eval_dataset = ConstructionSiteDataset(
        args.eval_data,
        transform=eval_transform,
        max_samples=args.eval_samples
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Evaluate each model
    all_results = {}
    
    for checkpoint_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_path.name}")
        print('='*60)
        
        # Load model
        model, config = load_model(checkpoint_path, device)
        
        # Evaluate
        results = evaluate_model(model, eval_loader, device, args.score_threshold)
        
        # Store results
        model_name = checkpoint_path.stem.replace('model_', '').replace('_best', '').replace('_final', '')
        all_results[model_name] = results
        
        print(f"\nResults for {model_name}:")
        print(f"  mAP: {results['overall']['mAP']:.4f}")
        print(f"  Mean Precision: {results['overall']['mean_precision']:.4f}")
        print(f"  Mean Recall: {results['overall']['mean_recall']:.4f}")
        print(f"  Mean F1: {results['overall']['mean_f1']:.4f}")
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Compare models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(all_results) > 1:
        df = compare_models(all_results, output_dir)
    else:
        print("\nOnly one model found, skipping comparison.")
    
    # Save detailed results
    results_path = output_dir / 'detailed_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'eval_samples': args.eval_samples,
            'score_threshold': args.score_threshold,
            'results': all_results
        }, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
