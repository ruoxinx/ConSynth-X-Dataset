#!/usr/bin/env python3
"""
Worker script để augment dataset với RAIN effect và output .arrow file
Giữ nguyên tất cả labels, chỉ thay đổi image và thêm ref_id

Load models once, reuse for all images - much faster than loading per image!
"""

import argparse
import sys
import os
from pathlib import Path
import pyarrow as pa
from PIL import Image
import io
import tempfile
import torch
import gc

# Add Weather_Effect_Generator to path
sys.path.insert(0, str(Path(__file__).parent / 'Weather_Effect_Generator'))

from rain_pipeline import (
    process_image, 
    load_midas_model,
    resolve_vgg_checkpoint,
    PROJECT_ROOT
)
from lib.style_transfer_utils import load_style_transfer_model


def parse_args():
    parser = argparse.ArgumentParser(description='Augment dataset batch with RAIN and output .arrow')
    parser.add_argument('--input', type=str, required=True, help='Input .arrow file')
    parser.add_argument('--output', type=str, required=True, help='Output .arrow file')
    parser.add_argument('--style', type=str, required=True, help='Style image path')
    parser.add_argument('--start', type=int, required=True, help='Start index (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='End index (exclusive)')
    parser.add_argument('--steps', type=int, default=10, help='Style transfer steps')
    parser.add_argument('--style-weight', type=float, default=10000, help='Style weight')
    parser.add_argument('--intensity', type=str, default='light', 
                        choices=['light', 'medium', 'heavy', 'extreme', 'quiet_night'], 
                        help='Rain intensity')
    return parser.parse_args()


def load_dataset(path):
    """Load dataset từ Arrow file"""
    with open(path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    return table


def image_to_bytes(img, format='JPEG', quality=95):
    """Convert PIL Image to bytes"""
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def process_batch(table, start_idx, end_idx, style_path, args, models=None):
    """
    Process một batch samples và trả về augmented data
    
    Args:
        table: PyArrow table
        start_idx: Start index
        end_idx: End index  
        style_path: Path to style image
        args: Arguments
        models: Dict containing pre-loaded models:
            - 'midas_model': MiDaS model
            - 'midas_transform': MiDaS transform
            - 'midas_device': MiDaS device
            - 'vgg_model': VGG model
            - 'device': Device string
    """
    # Validate indices
    total_samples = len(table)
    start_idx = max(0, start_idx)
    end_idx = min(total_samples, end_idx)
    
    print(f"📊 Processing samples {start_idx} to {end_idx-1} ({end_idx - start_idx} samples)")
    
    # Extract models if provided
    midas_model = models.get('midas_model') if models else None
    midas_transform = models.get('midas_transform') if models else None
    midas_device = models.get('midas_device') if models else None
    vgg_model = models.get('vgg_model') if models else None
    device = models.get('device', 'auto') if models else 'auto'
    
    # Columns to keep (all original columns)
    original_columns = table.column_names
    
    # Prepare new data
    new_data = {col: [] for col in original_columns}
    new_data['ref_id'] = []  # New column for reference to original
    
    # Process each sample
    for idx in range(start_idx, end_idx):
        try:
            row = table.slice(idx, 1)
            
            # Get original image
            image_data = row['image'][0].as_py()
            image_id = row['image_id'][0].as_py()
            
            # Extract image bytes
            if isinstance(image_data, dict):
                img_bytes = image_data['bytes']
                img_path = image_data.get('path', f'{image_id}.jpg')
                original_img = Image.open(io.BytesIO(img_bytes))
            else:
                print(f"  ⚠️ Sample {idx}: Unknown image format")
                continue
            
            # Create temp files for processing
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_in:
                original_img.save(tmp_in.name, 'JPEG', quality=95)
                tmp_input_path = tmp_in.name
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_out:
                tmp_output_path = tmp_out.name
            
            try:
                # Process image with rain pipeline (using pre-loaded models)
                process_image(
                    input_path=tmp_input_path,
                    output_path=tmp_output_path,
                    style_image_path=str(style_path),
                    steps=args.steps,
                    style_weight=args.style_weight,
                    intensity=args.intensity,
                    use_fake_depth=False,  # Always use real depth when models are loaded
                    weather='rain',  # Use rain VGG checkpoint
                    midas_model=midas_model,
                    midas_transform=midas_transform,
                    midas_device=midas_device,
                    vgg_model=vgg_model,
                    device=device
                )
                
                # Read augmented image
                augmented_img = Image.open(tmp_output_path)
                augmented_bytes = image_to_bytes(augmented_img)
                del augmented_img  # Free immediately
                
                # Build new row
                for col in original_columns:
                    if col == 'image':
                        # Replace with augmented image
                        new_image_data = {
                            'bytes': augmented_bytes,
                            'path': img_path
                        }
                        new_data[col].append(new_image_data)
                    else:
                        # Keep original value
                        new_data[col].append(row[col][0].as_py())
                
                # Add ref_id (reference to original)
                new_data['ref_id'].append(image_id)
                
                print(f"  ✓ Sample {idx}/{end_idx-1}: {image_id}")
                
                # ===== Free memory after each successful image =====
                del original_img, img_bytes, image_data, row
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            finally:
                # Cleanup temp files
                if os.path.exists(tmp_input_path):
                    os.remove(tmp_input_path)
                if os.path.exists(tmp_output_path):
                    os.remove(tmp_output_path)
                    
        except Exception as e:
            print(f"  ❌ Sample {idx}: Error - {e}")
            import traceback
            traceback.print_exc()
            
            # ===== CRITICAL: Free GPU memory after error =====
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            continue
    
    return new_data, original_columns


def create_arrow_table(data, original_columns, original_schema):
    """Create PyArrow table from processed data"""
    # Build schema with ref_id added
    fields = list(original_schema)
    fields.append(pa.field('ref_id', pa.string()))
    new_schema = pa.schema(fields)
    
    # Convert to PyArrow arrays
    arrays = []
    for col in original_columns:
        if col == 'image':
            # Create struct array for image
            bytes_array = pa.array([d['bytes'] for d in data[col]], type=pa.binary())
            path_array = pa.array([d['path'] for d in data[col]], type=pa.string())
            struct_array = pa.StructArray.from_arrays(
                [bytes_array, path_array],
                names=['bytes', 'path']
            )
            arrays.append(struct_array)
        else:
            # Use original type from schema
            field_type = original_schema.field(col).type
            arrays.append(pa.array(data[col], type=field_type))
    
    # Add ref_id column
    arrays.append(pa.array(data['ref_id'], type=pa.string()))
    
    return pa.table(arrays, schema=new_schema)


def save_arrow(table, output_path):
    """Save table to Arrow IPC file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pa.ipc.new_stream(str(output_path), table.schema) as writer:
        writer.write_table(table)
    
    print(f"📁 Saved: {output_path}")


def main():
    args = parse_args()
    
    # Detect device
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print("=" * 70)
    print("🌧️  ARROW AUGMENTATION WORKER (RAIN)")
    print("=" * 70)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Style:       {args.style}")
    print(f"Range:       {args.start} - {args.end}")
    print(f"Steps:       {args.steps}")
    print(f"StyleWeight: {args.style_weight}")
    print(f"Intensity:   {args.intensity}")
    print(f"Device:      {DEVICE}")
    print("=" * 70)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    table = load_dataset(args.input)
    print(f"✓ Loaded {len(table)} samples")
    
    # ========== Load models ONCE ==========
    print("\n🔧 Loading models (one time)...")
    models = {'device': DEVICE}
    
    # Load MiDaS model
    print("  Loading MiDaS model...")
    midas_model, midas_transform, midas_device = load_midas_model('DPT_Large', DEVICE)
    models['midas_model'] = midas_model
    models['midas_transform'] = midas_transform
    models['midas_device'] = midas_device
    print("  ✓ MiDaS loaded!")
    
    # Load VGG model (RAIN checkpoint)
    print("  Loading VGG model (rain)...")
    vgg_dir = PROJECT_ROOT / 'VGG'
    vgg_ckpt = resolve_vgg_checkpoint('rain', vgg_dir)
    vgg_model = load_style_transfer_model(pretrained=vgg_ckpt)
    vgg_model = vgg_model.to(DEVICE).eval()
    models['vgg_model'] = vgg_model
    print("  ✓ VGG loaded!")
    
    print("✓ All models loaded and ready!")
    
    # Process batch
    print("\n🔄 Processing batch...")
    new_data, original_columns = process_batch(
        table, args.start, args.end, args.style, args, models=models
    )
    
    if len(new_data['ref_id']) == 0:
        print("❌ No samples processed successfully")
        sys.exit(1)
    
    # Create new table
    print(f"\n📊 Creating Arrow table ({len(new_data['ref_id'])} samples)...")
    new_table = create_arrow_table(new_data, original_columns, table.schema)
    
    # Save
    save_arrow(new_table, args.output)
    
    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
