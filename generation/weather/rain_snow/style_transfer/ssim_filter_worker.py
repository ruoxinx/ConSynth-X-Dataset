#!/usr/bin/env python3
"""
Worker script để filter samples dựa trên SSIM score
Không yêu cầu GPU - chỉ dùng CPU

Input: Arrow files từ augmented output
Output: Filtered arrow file với samples có SSIM trong range cho phép
"""

import argparse
import sys
import os
from pathlib import Path
import pyarrow as pa
from PIL import Image
import io
import numpy as np
from skimage.metrics import structural_similarity as ssim


def parse_args():
    parser = argparse.ArgumentParser(description='Filter samples based on SSIM score')
    parser.add_argument('--input-dir', type=str, required=True, 
                        help='Input directory containing batch_*.arrow files')
    parser.add_argument('--original-arrow', type=str, required=True,
                        help='Original dataset arrow file for comparison')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output arrow file')
    parser.add_argument('--ssim-min', type=float, default=0.5,
                        help='Minimum SSIM threshold (default: 0.5)')
    parser.add_argument('--ssim-max', type=float, default=0.95,
                        help='Maximum SSIM threshold (default: 0.95)')
    parser.add_argument('--csv-output', type=str, default=None,
                        help='Optional: output CSV with SSIM scores')
    return parser.parse_args()


def load_arrow_file(path):
    """Load dataset from arrow file"""
    try:
        with pa.memory_map(str(path), 'r') as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()
        # Convert to list of dicts
        samples = []
        for i in range(len(table)):
            sample = {}
            for col in table.column_names:
                sample[col] = table[col][i].as_py()
            samples.append(sample)
        return samples
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def to_pil_image(img_data):
    """Convert image data to PIL Image"""
    if isinstance(img_data, Image.Image):
        return img_data
    elif isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data))
    elif isinstance(img_data, dict) and 'bytes' in img_data:
        return Image.open(io.BytesIO(img_data['bytes']))
    else:
        return img_data


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    try:
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Resize if different shapes
        if arr1.shape != arr2.shape:
            img1_resized = Image.fromarray(arr1).resize((arr2.shape[1], arr2.shape[0]))
            arr1 = np.array(img1_resized)
        
        # Ensure uint8
        if arr1.dtype != np.uint8:
            arr1 = np.clip(arr1 * 255, 0, 255).astype(np.uint8) if arr1.max() <= 1 else arr1.astype(np.uint8)
        if arr2.dtype != np.uint8:
            arr2 = np.clip(arr2 * 255, 0, 255).astype(np.uint8) if arr2.max() <= 1 else arr2.astype(np.uint8)
        
        # Calculate SSIM
        if len(arr1.shape) == 3 and arr1.shape[2] == 3:
            ssim_val = ssim(arr1, arr2, channel_axis=2)
        else:
            ssim_val = ssim(arr1, arr2)
        
        return ssim_val
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return None


def image_to_bytes(img, format='JPEG', quality=95):
    """Convert PIL Image to bytes"""
    buffer = io.BytesIO()
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def build_original_index(original_samples):
    """Build index from ref_id/image_id to sample"""
    index = {}
    for idx, sample in enumerate(original_samples):
        image_id = sample.get('image_id', str(idx))
        index[image_id] = sample
    return index


def export_filtered_to_arrow(filtered_samples, output_path):
    """Export filtered samples to arrow file"""
    if len(filtered_samples) == 0:
        print("⚠️  No samples to export")
        return None
    
    # Get column names from first sample
    sample_keys = [k for k in filtered_samples[0]['aug_sample'].keys() 
                   if not k.startswith('_')]
    sample_keys.append('ssim_score')
    
    # Prepare data
    data = {key: [] for key in sample_keys}
    
    for sample_info in filtered_samples:
        aug_sample = sample_info['aug_sample']
        
        for key in sample_keys:
            if key == 'ssim_score':
                data[key].append(sample_info['ssim'])
            else:
                value = aug_sample.get(key)
                if key == 'image':
                    if isinstance(value, Image.Image):
                        value = {'bytes': image_to_bytes(value)}
                    elif isinstance(value, dict) and 'bytes' in value:
                        pass
                    elif isinstance(value, bytes):
                        value = {'bytes': value}
                data[key].append(value)
    
    # Create arrays
    arrays = []
    for key in sample_keys:
        if key == 'image':
            byte_arrays = [v['bytes'] if isinstance(v, dict) and 'bytes' in v else v 
                          for v in data[key]]
            arrays.append(pa.StructArray.from_arrays(
                [pa.array(byte_arrays, type=pa.binary())],
                names=['bytes']
            ))
        elif key == 'ssim_score':
            arrays.append(pa.array(data[key], type=pa.float64()))
        else:
            arrays.append(pa.array(data[key]))
    
    table = pa.table(dict(zip(sample_keys, arrays)))
    
    # Write
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pa.OSFile(str(output_path), 'wb') as f:
        writer = pa.ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()
    
    return output_path


def main():
    args = parse_args()
    
    print("="*70)
    print("SSIM-based Sample Filter")
    print("="*70)
    print(f"Input directory: {args.input_dir}")
    print(f"Original dataset: {args.original_arrow}")
    print(f"SSIM range: [{args.ssim_min}, {args.ssim_max}]")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Load original dataset
    print("\n📂 Loading original dataset...")
    original_samples = load_arrow_file(args.original_arrow)
    if original_samples is None:
        print("❌ Failed to load original dataset")
        sys.exit(1)
    print(f"✅ Loaded {len(original_samples)} original samples")
    
    # Build index for fast lookup
    original_index = build_original_index(original_samples)
    
    # Load all augmented batches
    input_dir = Path(args.input_dir)
    arrow_files = sorted(input_dir.glob("batch_*.arrow"))
    
    if not arrow_files:
        print(f"❌ No batch_*.arrow files found in {input_dir}")
        sys.exit(1)
    
    print(f"\n📦 Found {len(arrow_files)} batch files")
    
    # Process all samples
    all_samples = []
    ssim_results = []
    
    for arrow_file in arrow_files:
        print(f"  Loading {arrow_file.name}...")
        samples = load_arrow_file(arrow_file)
        if samples:
            for sample in samples:
                sample['_batch_file'] = arrow_file.name
                all_samples.append(sample)
    
    print(f"\n✅ Loaded {len(all_samples)} total augmented samples")
    
    # Calculate SSIM for all samples
    print(f"\n🎯 Calculating SSIM scores...")
    filtered_samples = []
    
    for idx, aug_sample in enumerate(all_samples):
        ref_id = aug_sample.get('ref_id', aug_sample.get('image_id'))
        
        if ref_id not in original_index:
            continue
        
        orig_sample = original_index[ref_id]
        
        # Convert images
        orig_img = to_pil_image(orig_sample['image'])
        aug_img = to_pil_image(aug_sample['image'])
        
        # Calculate SSIM
        score = calculate_ssim(orig_img, aug_img)
        
        if score is not None:
            ssim_results.append({
                'ref_id': ref_id,
                'ssim': score,
                'batch_file': aug_sample.get('_batch_file', '')
            })
            
            # Check if within threshold
            if args.ssim_min <= score <= args.ssim_max:
                filtered_samples.append({
                    'ref_id': ref_id,
                    'ssim': score,
                    'aug_sample': aug_sample,
                    'orig_sample': orig_sample
                })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(all_samples)} samples")
    
    # Statistics
    ssim_values = [r['ssim'] for r in ssim_results]
    print(f"\n📊 SSIM Statistics:")
    print(f"  • Total calculated: {len(ssim_values)}")
    print(f"  • Min: {np.min(ssim_values):.4f}")
    print(f"  • Max: {np.max(ssim_values):.4f}")
    print(f"  • Mean: {np.mean(ssim_values):.4f}")
    print(f"  • Median: {np.median(ssim_values):.4f}")
    
    print(f"\n🎯 Filtering results:")
    print(f"  • Samples in range [{args.ssim_min}, {args.ssim_max}]: {len(filtered_samples)}")
    print(f"  • Keep ratio: {100*len(filtered_samples)/len(ssim_results):.1f}%")
    
    # Export filtered samples
    if len(filtered_samples) > 0:
        print(f"\n💾 Exporting filtered samples...")
        output_path = export_filtered_to_arrow(filtered_samples, args.output)
        if output_path:
            file_size = output_path.stat().st_size / (1024*1024)
            print(f"✅ Exported {len(filtered_samples)} samples to {output_path}")
            print(f"   File size: {file_size:.2f} MB")
    else:
        print("⚠️  No samples to export")
    
    # Export CSV if requested
    if args.csv_output:
        import csv
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ref_id', 'ssim', 'batch_file', 'quality'])
            writer.writeheader()
            for r in ssim_results:
                r['quality'] = 'HIGH' if args.ssim_min <= r['ssim'] <= args.ssim_max else 'LOW'
                writer.writerow(r)
        
        print(f"✅ SSIM results saved to {csv_path}")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
