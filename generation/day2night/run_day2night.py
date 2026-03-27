"""
Script to apply img2img-turbo day_to_night model on construction site images
"""
import os
import sys
import random
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

# Paths
BASE_DIR = Path(__file__).parent
ARROW_FILE = BASE_DIR / "LouisChen15___construction_site/construction_site-test.arrow"
OUTPUT_DIR = BASE_DIR / "output_day2night"
TEMP_INPUT_DIR = BASE_DIR / "temp_input_images"

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
    print("="*60)
    print("IMG2IMG-TURBO: Day to Night Translation")
    print("="*60)
    
    # Load original data
    print("\n[1/4] Loading arrow file...")
    table = load_arrow_file(ARROW_FILE)
    print(f"Loaded {len(table)} samples")
    
    # Sample 100 images
    random.seed(42)
    sample_indices = random.sample(range(len(table)), min(100, len(table)))
    print(f"\n[2/4] Extracting {len(sample_indices)} sample images...")
    
    # Create directories
    TEMP_INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Extract and save images
    image_paths = []
    for idx in tqdm(sample_indices, desc="Extracting images"):
        row = table.slice(idx, 1)
        img = get_image_from_row(row)
        if img is not None:
            image_id = table['image_id'][idx].as_py()
            img_path = TEMP_INPUT_DIR / f"{image_id}.jpg"
            # Convert to RGB and save
            img.convert('RGB').save(img_path, 'JPEG', quality=95)
            image_paths.append(img_path)
    
    print(f"Extracted {len(image_paths)} images to {TEMP_INPUT_DIR}")
    
    # Initialize model
    print("\n[3/4] Loading CycleGAN-Turbo day_to_night model...")
    model = CycleGAN_Turbo(pretrained_name="day_to_night")
    model.eval()
    
    # Try xformers but don't fail if not available
    try:
        model.unet.enable_xformers_memory_efficient_attention()
        print("Using xformers for memory efficient attention")
    except Exception as e:
        print(f"xformers not available, using standard attention: {e}")
    
    # Check if GPU available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        model.half()  # Use FP16 for faster inference
    
    # Image transform
    T_val = build_transform("resize_512x512")
    
    # Run inference
    print(f"\n[4/4] Running day_to_night translation on {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths, desc="Translating"):
        try:
            input_image = Image.open(img_path).convert('RGB')
            original_size = (input_image.width, input_image.height)
            
            with torch.no_grad():
                input_img = T_val(input_image)
                x_t = transforms.ToTensor()(input_img)
                x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0)
                
                if device == "cuda":
                    x_t = x_t.cuda().half()
                
                output = model(x_t, direction=None, caption=None)
            
            output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
            output_pil = output_pil.resize(original_size, Image.LANCZOS)
            
            # Save output
            output_path = OUTPUT_DIR / f"{img_path.stem}_night.jpg"
            output_pil.save(output_path, 'JPEG', quality=95)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Count results
    output_files = list(OUTPUT_DIR.glob("*.jpg"))
    print(f"\n{'='*60}")
    print(f"COMPLETED!")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total images translated: {len(output_files)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
