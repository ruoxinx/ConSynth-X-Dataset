#!/usr/bin/env python3
"""
Day2Night SODA Worker
Process a batch of images from SODA folder using img2img-turbo
Model is loaded once and applied to all images in the batch

Output structure:
  output_dir/
    images/       - Night images (JPG)
"""
import os
import sys
import argparse
from pathlib import Path

# Add img2img-turbo to path BEFORE importing
BASE_DIR = Path(__file__).parent
IMG2IMG_SRC = BASE_DIR / "img2img-turbo" / "src"
sys.path.insert(0, str(IMG2IMG_SRC))

from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform


def get_image_list(images_dir):
    """Get sorted list of image files"""
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = []
    for f in Path(images_dir).iterdir():
        if f.suffix in extensions:
            images.append(f)
    return sorted(images, key=lambda x: x.stem)


def main():
    parser = argparse.ArgumentParser(description='Day2Night SODA Worker')
    parser.add_argument('--images-dir', required=True, help='Path to SODA images folder')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--start-idx', type=int, required=True, help='Start index (inclusive)')
    parser.add_argument('--end-idx', type=int, required=True, help='End index (exclusive)')
    parser.add_argument('--batch-id', type=int, default=0, help='Batch ID for logging')
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"DAY2NIGHT SODA WORKER - Batch {args.batch_id}")
    print("=" * 60)
    print(f"Input images: {images_dir}")
    print(f"Output images: {output_images_dir}")
    print(f"Range: [{args.start_idx}, {args.end_idx})")
    print("=" * 60)
    
    # Get image list
    print("\n[1/3] Loading image list...")
    all_images = get_image_list(images_dir)
    total_images = len(all_images)
    print(f"Total images in folder: {total_images}")
    
    # Adjust indices
    start_idx = max(0, args.start_idx)
    end_idx = min(total_images, args.end_idx)
    batch_size = end_idx - start_idx
    
    if batch_size <= 0:
        print("No images to process in this range!")
        return
    
    # Get batch images
    batch_images = all_images[start_idx:end_idx]
    print(f"Processing indices: [{start_idx}, {end_idx}) = {batch_size} images")
    
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
    
    for img_path in tqdm(batch_images, desc=f"Batch {args.batch_id}"):
        try:
            # Output filename same as input
            output_path = output_images_dir / img_path.name
            
            # Check if already processed
            if output_path.exists():
                skip_count += 1
                continue
            
            # Load image
            input_image = Image.open(img_path).convert('RGB')
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
            output_pil.save(output_path, 'JPEG', quality=95)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            error_count += 1
            continue
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"BATCH {args.batch_id} COMPLETED!")
    print(f"{'=' * 60}")
    print(f"Successful: {success_count}")
    print(f"Skipped (exists): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Output images: {output_images_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
