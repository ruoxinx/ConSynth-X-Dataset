#!/usr/bin/env python3
"""
Worker script để augment SODA VOC dataset với RAIN effect
Output: Augmented images (keeps original directory structure)
"""

import argparse
import sys
import os
from pathlib import Path
from PIL import Image
import tempfile
import torch
import gc
import shutil

# Add Weather_Effect_Generator to path (submodule at generation/weather/libs/Weather_Effect_Generator)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'libs' / 'Weather_Effect_Generator'))

from rain_pipeline import (
    process_image, 
    load_midas_model,
    resolve_vgg_checkpoint,
    PROJECT_ROOT
)
from lib.style_transfer_utils import load_style_transfer_model


def parse_args():
    parser = argparse.ArgumentParser(description='Augment SODA VOC batch with RAIN')
    parser.add_argument('--image-dir', type=str, required=True, 
                        help='Input JPEGImages directory')
    parser.add_argument('--image-list', type=str, required=True, 
                        help='File containing image IDs (one per line)')
    parser.add_argument('--output-dir', type=str, required=True, 
                        help='Output directory for augmented images')
    parser.add_argument('--style', type=str, required=True, 
                        help='Style image path')
    parser.add_argument('--start', type=int, required=True, 
                        help='Start index (inclusive)')
    parser.add_argument('--end', type=int, required=True, 
                        help='End index (exclusive)')
    parser.add_argument('--steps', type=int, default=10, 
                        help='Style transfer steps')
    parser.add_argument('--style-weight', type=float, default=10000, 
                        help='Style weight')
    parser.add_argument('--intensity', type=str, default='light', 
                        choices=['light', 'medium', 'heavy', 'extreme', 'quiet_night'], 
                        help='Rain intensity')
    return parser.parse_args()


def load_image_list(list_file):
    """Load list of image IDs from file"""
    with open(list_file, 'r', encoding='utf-8', errors='ignore') as f:
        return [line.strip() for line in f if line.strip()]


def process_batch(image_ids, image_dir, output_dir, style_path, args, models=None):
    """
    Process a batch of images
    
    Args:
        image_ids: List of image IDs to process
        image_dir: Input directory
        output_dir: Output directory
        style_path: Path to style image
        args: Arguments
        models: Dict containing pre-loaded models
    """
    start_idx = args.start
    end_idx = min(args.end, len(image_ids))
    
    batch_ids = image_ids[start_idx:end_idx]
    
    print(f"📊 Processing samples {start_idx} to {end_idx-1} ({len(batch_ids)} samples)")
    
    # Extract models
    midas_model = models.get('midas_model') if models else None
    midas_transform = models.get('midas_transform') if models else None
    midas_device = models.get('midas_device') if models else None
    vgg_model = models.get('vgg_model') if models else None
    device = models.get('device', 'auto') if models else 'auto'
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for idx, image_id in enumerate(batch_ids):
        global_idx = start_idx + idx
        
        # Find input image
        input_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = Path(image_dir) / f"{image_id}{ext}"
            if candidate.exists():
                input_path = candidate
                break
        
        if input_path is None:
            print(f"  ❌ Sample {global_idx}: {image_id} - Image not found")
            error_count += 1
            continue
        
        # Output path
        output_path = Path(output_dir) / f"{image_id}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create temp output file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_out:
                tmp_output_path = tmp_out.name
            
            # Process image with rain pipeline
            process_image(
                input_path=str(input_path),
                output_path=tmp_output_path,
                style_image_path=str(style_path),
                steps=args.steps,
                style_weight=args.style_weight,
                intensity=args.intensity,
                use_fake_depth=False,
                weather='rain',
                midas_model=midas_model,
                midas_transform=midas_transform,
                midas_device=midas_device,
                vgg_model=vgg_model,
                device=device
            )
            
            # Move to final output
            shutil.move(tmp_output_path, output_path)
            
            print(f"  ✓ Sample {global_idx}/{end_idx-1}: {image_id}")
            success_count += 1
            
            # Free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  ❌ Sample {global_idx}: {image_id} - {e}")
            error_count += 1
            
            # Clean up temp file
            if 'tmp_output_path' in locals() and os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
            
            # Free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            continue
    
    return success_count, error_count


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🌧️  SODA VOC RAIN AUGMENTATION WORKER")
    print("=" * 70)
    print(f"Image directory: {args.image_dir}")
    print(f"Image list: {args.image_list}")
    print(f"Output directory: {args.output_dir}")
    print(f"Style: {args.style}")
    print(f"Range: {args.start} to {args.end}")
    print(f"Steps: {args.steps}")
    print(f"Style weight: {args.style_weight}")
    print(f"Intensity: {args.intensity}")
    print("=" * 70)
    
    # Load image list
    print("\n📂 Loading image list...")
    image_ids = load_image_list(args.image_list)
    print(f"   Total images in list: {len(image_ids)}")
    
    # Validate range
    if args.start >= len(image_ids):
        print(f"❌ Start index {args.start} >= total images {len(image_ids)}")
        sys.exit(1)
    
    # ===== PRE-LOAD ALL MODELS =====
    print("\n🔧 Loading models (this happens once for all images)...")
    
    # Load MiDaS model
    print("   Loading MiDaS depth model...")
    midas_model, midas_transform, midas_device = load_midas_model()
    print(f"   ✓ MiDaS loaded on {midas_device}")
    
    # Load VGG style transfer model (rain checkpoint)
    print("   Loading VGG style transfer model (rain)...")
    vgg_dir = PROJECT_ROOT / 'VGG'
    vgg_checkpoint = resolve_vgg_checkpoint('rain', vgg_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vgg_model = load_style_transfer_model(pretrained=vgg_checkpoint)
    vgg_model = vgg_model.to(device).eval()
    print(f"   ✓ VGG loaded on {device}")
    
    models = {
        'midas_model': midas_model,
        'midas_transform': midas_transform,
        'midas_device': midas_device,
        'vgg_model': vgg_model,
        'device': device
    }
    
    # Process batch
    print(f"\n🎯 Processing batch...")
    success, errors = process_batch(
        image_ids=image_ids,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        style_path=args.style,
        args=args,
        models=models
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"Processed: {args.start} to {args.end}")
    print(f"Success: {success}")
    print(f"Errors: {errors}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Exit with error code if all failed
    if success == 0 and errors > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
