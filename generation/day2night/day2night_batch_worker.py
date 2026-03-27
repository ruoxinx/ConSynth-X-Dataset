#!/usr/bin/env python3
"""
Day2Night Batch Worker
Process a batch of images from Arrow file using img2img-turbo
Model is loaded once and applied to all images in the batch

Output structure:
  output_dir/
    images/       - Night images (JPG)
    annotations/  - Annotations (JSON)
"""
import os
import sys
import argparse
import json
from pathlib import Path
import io

# Add img2img-turbo to path BEFORE importing
BASE_DIR = Path(__file__).parent
IMG2IMG_SRC = BASE_DIR / "img2img-turbo" / "src"
sys.path.insert(0, str(IMG2IMG_SRC))

import pyarrow as pa
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform


def load_arrow_file(path):
    """Load Arrow IPC file"""
    with open(path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    return table


def get_image_from_row(row):
    """Extract PIL Image from Arrow row"""
    image_data = row['image'][0].as_py()
    if isinstance(image_data, dict):
        img_bytes = image_data['bytes']
        return Image.open(io.BytesIO(img_bytes))
    return None


def main():
    parser = argparse.ArgumentParser(description='Day2Night Batch Worker')
    parser.add_argument('--arrow-file', required=True, help='Path to Arrow file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--start-idx', type=int, required=True, help='Start index (inclusive)')
    parser.add_argument('--end-idx', type=int, required=True, help='End index (exclusive)')
    parser.add_argument('--batch-id', type=int, default=0, help='Batch ID for logging')
    args = parser.parse_args()
    
    arrow_path = Path(args.arrow_file)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"DAY2NIGHT BATCH WORKER - Batch {args.batch_id}")
    print("=" * 60)
    print(f"Arrow file: {arrow_path}")
    print(f"Output images: {images_dir}")
    print(f"Output annotations: {annotations_dir}")
    print(f"Range: [{args.start_idx}, {args.end_idx})")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/3] Loading Arrow file...")
    table = load_arrow_file(arrow_path)
    total_samples = len(table)
    print(f"Total samples in dataset: {total_samples}")
    print(f"Columns: {table.column_names}")
    
    # Adjust indices
    start_idx = max(0, args.start_idx)
    end_idx = min(total_samples, args.end_idx)
    batch_size = end_idx - start_idx
    
    if batch_size <= 0:
        print("No samples to process in this range!")
        return
    
    print(f"Processing indices: [{start_idx}, {end_idx}) = {batch_size} samples")
    
    # Load model ONCE for the entire batch
    print("\n[2/3] Loading CycleGAN-Turbo day_to_night model...")
    model = CycleGAN_Turbo(pretrained_name="day_to_night")
    model.eval()
    
    # Try xformers
    try:
        model.unet.enable_xformers_memory_efficient_attention()
        print("Using xformers for memory efficient attention")
    except Exception as e:
        print(f"xformers not available: {e}")
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        model.half()
        print("Using FP16 for faster inference")
    
    # Image transform
    T_val = build_transform("resize_512x512")
    
    # Process all images in batch
    print(f"\n[3/3] Processing {batch_size} images...")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx in tqdm(range(start_idx, end_idx), desc=f"Batch {args.batch_id}"):
        try:
            # Get image_id for filename
            if 'image_id' in table.column_names:
                image_id = str(table['image_id'][idx].as_py())
            else:
                image_id = f"{idx:07d}"
            
            # Check if already processed
            output_image_path = images_dir / f"{image_id}.jpg"
            output_anno_path = annotations_dir / f"{image_id}.json"
            
            if output_image_path.exists():
                skip_count += 1
                continue
            
            # Extract image from table
            row = table.slice(idx, 1)
            input_image = get_image_from_row(row)
            
            if input_image is None:
                error_count += 1
                continue
            
            input_image = input_image.convert('RGB')
            original_size = (input_image.width, input_image.height)
            
            # Run inference
            with torch.no_grad():
                input_img = T_val(input_image)
                x_t = transforms.ToTensor()(input_img)
                x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0)
                
                if device == "cuda":
                    x_t = x_t.cuda().half()
                
                output = model(x_t, direction=None, caption=None)
            
            # Convert output to PIL
            output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
            output_pil = output_pil.resize(original_size, Image.LANCZOS)
            
            # Save output image
            output_pil.save(output_image_path, 'JPEG', quality=95)
            
            # Collect and save annotations
            annotation = {"image_id": image_id, "ref_idx": idx}
            for col in table.column_names:
                if col != 'image':
                    annotation[col] = table[col][idx].as_py()
            
            with open(output_anno_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            error_count += 1
            continue
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"BATCH {args.batch_id} COMPLETED!")
    print(f"{'=' * 60}")
    print(f"Successful: {success_count}")
    print(f"Skipped (exists): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Output images: {images_dir}")
    print(f"Output annotations: {annotations_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
