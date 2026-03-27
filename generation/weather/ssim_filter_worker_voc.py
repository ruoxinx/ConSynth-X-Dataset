#!/usr/bin/env python3
"""
Worker script để filter ảnh VOC format dựa trên SSIM score
Không yêu cầu GPU - chỉ dùng CPU

Input: Augmented JPEGImages folder
Output: Filtered images (copy/symlink) or removal list
"""

import argparse
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import shutil
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Filter VOC images based on SSIM score')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input style directory containing JPEGImages folder')
    parser.add_argument('--original-dir', type=str, required=True,
                        help='Original JPEGImages directory for comparison')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for filtered images (optional, if not set, removes bad images)')
    parser.add_argument('--ssim-min', type=float, default=0.5,
                        help='Minimum SSIM threshold (default: 0.5)')
    parser.add_argument('--ssim-max', type=float, default=0.95,
                        help='Maximum SSIM threshold (default: 0.95)')
    parser.add_argument('--csv-output', type=str, default=None,
                        help='Output CSV with SSIM scores')
    parser.add_argument('--remove-bad', action='store_true',
                        help='Remove images outside SSIM range')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run - only calculate SSIM without modifying')
    return parser.parse_args()


def calculate_ssim(orig_path, aug_path):
    """Calculate SSIM between two images"""
    try:
        orig_img = Image.open(orig_path).convert('RGB')
        aug_img = Image.open(aug_path).convert('RGB')
        
        orig_arr = np.array(orig_img)
        aug_arr = np.array(aug_img)
        
        # Resize if different shapes
        if orig_arr.shape != aug_arr.shape:
            aug_img_resized = aug_img.resize((orig_arr.shape[1], orig_arr.shape[0]))
            aug_arr = np.array(aug_img_resized)
        
        # Calculate SSIM
        ssim_val = ssim(orig_arr, aug_arr, channel_axis=2)
        return ssim_val
    except Exception as e:
        print(f"Error calculating SSIM for {aug_path.name}: {e}")
        return None


def process_image(args_tuple):
    """Process a single image - for parallel execution"""
    orig_path, aug_path, ssim_min, ssim_max = args_tuple
    
    ssim_val = calculate_ssim(orig_path, aug_path)
    if ssim_val is None:
        return None
    
    in_range = ssim_min <= ssim_val <= ssim_max
    return {
        'filename': aug_path.name,
        'ssim': ssim_val,
        'in_range': in_range,
        'orig_path': str(orig_path),
        'aug_path': str(aug_path)
    }


def main():
    args = parse_args()
    
    print("="*70)
    print("SSIM-based VOC Image Filter")
    print("="*70)
    print(f"Input directory: {args.input_dir}")
    print(f"Original directory: {args.original_dir}")
    print(f"SSIM range: [{args.ssim_min}, {args.ssim_max}]")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    if args.remove_bad:
        print("Mode: Remove images outside SSIM range")
    if args.dry_run:
        print("DRY RUN - no modifications will be made")
    print("="*70)
    
    # Setup paths
    input_path = Path(args.input_dir)
    original_path = Path(args.original_dir)
    
    # Find JPEGImages folder
    if (input_path / 'JPEGImages').exists():
        aug_images_dir = input_path / 'JPEGImages'
    else:
        aug_images_dir = input_path
    
    if not original_path.exists():
        print(f"❌ Original directory not found: {original_path}")
        sys.exit(1)
    
    # Get list of augmented images
    aug_images = sorted(aug_images_dir.glob("*.jpg")) + sorted(aug_images_dir.glob("*.jpeg")) + sorted(aug_images_dir.glob("*.png"))
    
    if not aug_images:
        print(f"❌ No images found in {aug_images_dir}")
        sys.exit(1)
    
    print(f"\n📦 Found {len(aug_images)} augmented images")
    
    # Prepare tasks
    tasks = []
    missing_originals = []
    
    for aug_img in aug_images:
        orig_img = original_path / aug_img.name
        if not orig_img.exists():
            # Try different extensions
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                alt_orig = original_path / (aug_img.stem + ext)
                if alt_orig.exists():
                    orig_img = alt_orig
                    found = True
                    break
            if not found:
                missing_originals.append(aug_img.name)
                continue
        
        tasks.append((orig_img, aug_img, args.ssim_min, args.ssim_max))
    
    if missing_originals:
        print(f"⚠️  {len(missing_originals)} images without matching original (will be skipped)")
    
    print(f"\n🎯 Processing {len(tasks)} images with {args.workers} workers...")
    
    # Process images
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Calculating SSIM"):
            result = future.result()
            if result:
                results.append(result)
    
    # Statistics
    ssim_values = [r['ssim'] for r in results]
    in_range_count = sum(1 for r in results if r['in_range'])
    out_range_count = len(results) - in_range_count
    
    print(f"\n📊 SSIM Statistics:")
    print(f"  • Total processed: {len(ssim_values)}")
    print(f"  • Min: {np.min(ssim_values):.4f}")
    print(f"  • Max: {np.max(ssim_values):.4f}")
    print(f"  • Mean: {np.mean(ssim_values):.4f}")
    print(f"  • Median: {np.median(ssim_values):.4f}")
    print(f"  • Std: {np.std(ssim_values):.4f}")
    
    print(f"\n🎯 Filtering results:")
    print(f"  • In range [{args.ssim_min}, {args.ssim_max}]: {in_range_count} ({100*in_range_count/len(results):.1f}%)")
    print(f"  • Out of range: {out_range_count} ({100*out_range_count/len(results):.1f}%)")
    
    # Output CSV
    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'ssim', 'in_range'])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'filename': r['filename'],
                    'ssim': f"{r['ssim']:.4f}",
                    'in_range': 'YES' if r['in_range'] else 'NO'
                })
        print(f"\n✅ SSIM results saved to {csv_path}")
    
    # Handle output
    if not args.dry_run:
        if args.output_dir:
            # Copy good images to output directory
            output_path = Path(args.output_dir)
            output_images_dir = output_path / 'JPEGImages'
            output_images_dir.mkdir(parents=True, exist_ok=True)
            
            copied = 0
            for r in results:
                if r['in_range']:
                    src = Path(r['aug_path'])
                    dst = output_images_dir / r['filename']
                    shutil.copy2(src, dst)
                    copied += 1
            
            print(f"\n✅ Copied {copied} images to {output_images_dir}")
            
            # Copy annotations if exist
            input_annotations = input_path / 'Annotations'
            if input_annotations.exists():
                output_annotations = output_path / 'Annotations'
                output_annotations.mkdir(parents=True, exist_ok=True)
                
                for r in results:
                    if r['in_range']:
                        xml_name = Path(r['filename']).stem + '.xml'
                        src_xml = input_annotations / xml_name
                        if src_xml.exists():
                            shutil.copy2(src_xml, output_annotations / xml_name)
        
        elif args.remove_bad:
            # Remove images outside range
            removed = 0
            for r in results:
                if not r['in_range']:
                    aug_path = Path(r['aug_path'])
                    if aug_path.exists():
                        aug_path.unlink()
                        removed += 1
                        
                        # Also remove annotation if exists
                        ann_path = input_path / 'Annotations' / (aug_path.stem + '.xml')
                        if ann_path.exists():
                            ann_path.unlink()
            
            print(f"\n✅ Removed {removed} images outside SSIM range")
    else:
        print("\n🔍 DRY RUN - no files modified")
        if args.remove_bad:
            to_remove = [r['filename'] for r in results if not r['in_range']]
            print(f"   Would remove {len(to_remove)} images")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
