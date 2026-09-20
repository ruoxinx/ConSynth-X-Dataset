#!/usr/bin/env python3
"""
FLUX Outpainting Pipeline Worker

Reads samples directly from arrow file, applies random Gaussian scale outpainting,
and saves: outpainted image, transferred annotation, and meta.

Output structure:
  output_dir/
    images/        - Outpainted images
    annotations/   - Transferred annotations (xyxy format)
    meta/          - Full metadata for debugging
"""

import os
import sys
import json
import random
import gc
from pathlib import Path
from PIL import Image, ImageDraw
import io
import argparse
from typing import Dict, List, Tuple, Optional


# ============================================================
# Scale Generation
# ============================================================

def generate_gaussian_scale(
    mean: float = 0.25,
    std: float = None,
    min_scale: float = 0.2,
    max_scale: float = 0.4
) -> float:
    """Generate random scale using truncated Gaussian distribution."""
    if std is None:
        std = (max_scale - min_scale) / 4
    
    scale = random.gauss(mean, std)
    return max(min_scale, min(max_scale, scale))


# ============================================================
# Bbox Transfer
# ============================================================

def transfer_bbox(
    bbox: List[float],
    orig_w: int,
    orig_h: int,
    paste_x: int,
    paste_y: int,
    new_w: int,
    new_h: int,
    resize_factor: float = 1.0
) -> List[float]:
    """
    Transfer bbox from original to outpainted coordinates.
    
    Args:
        bbox: Original bbox [x1, y1, x2, y2] in normalized coords (0-1)
        orig_w, orig_h: Original image size
        paste_x, paste_y: Position where image is pasted on canvas
        new_w, new_h: New canvas size
        resize_factor: Input resize factor applied before outpainting
    
    Returns:
        Transferred bbox in normalized coords
    """
    x1, y1, x2, y2 = bbox
    
    # Convert to pixel coords (after resize)
    x1_px = x1 * orig_w * resize_factor
    y1_px = y1 * orig_h * resize_factor
    x2_px = x2 * orig_w * resize_factor
    y2_px = y2 * orig_h * resize_factor
    
    # Add paste offset
    x1_new = (x1_px + paste_x) / new_w
    y1_new = (y1_px + paste_y) / new_h
    x2_new = (x2_px + paste_x) / new_w
    y2_new = (y2_px + paste_y) / new_h
    
    return [round(x1_new, 6), round(y1_new, 6), round(x2_new, 6), round(y2_new, 6)]


def transfer_all_annotations(
    annotations: Dict,
    orig_size: Tuple[int, int],
    paste_pos: Tuple[int, int],
    canvas_size: Tuple[int, int],
    resize_factor: float
) -> Dict:
    """Transfer all annotations to outpainted coordinates."""
    
    orig_w, orig_h = orig_size
    paste_x, paste_y = paste_pos
    new_w, new_h = canvas_size
    
    transferred = {}
    
    # Object detection fields
    bbox_fields = ['excavator', 'rebar', 'worker_with_white_hard_hat']
    for field in bbox_fields:
        if field in annotations and annotations[field]:
            transferred[field] = [
                transfer_bbox(bbox, orig_w, orig_h, paste_x, paste_y, new_w, new_h, resize_factor)
                for bbox in annotations[field]
                if isinstance(bbox, list) and len(bbox) == 4
            ]
        else:
            transferred[field] = []
    
    # Rule violation fields
    rule_fields = ['rule_1_violation', 'rule_2_violation', 'rule_3_violation', 'rule_4_violation']
    for field in rule_fields:
        if field in annotations and annotations[field] and isinstance(annotations[field], dict):
            violation = annotations[field].copy()
            if 'bounding_box' in violation and violation['bounding_box']:
                violation['bounding_box'] = [
                    transfer_bbox(bbox, orig_w, orig_h, paste_x, paste_y, new_w, new_h, resize_factor)
                    for bbox in violation['bounding_box']
                    if isinstance(bbox, list) and len(bbox) == 4
                ]
            transferred[field] = violation
        else:
            transferred[field] = annotations.get(field)
    
    # Copy non-bbox fields
    non_bbox_fields = ['image_id', 'image_caption', 'illumination', 'camera_distance', 'view', 'quality_of_info']
    for field in non_bbox_fields:
        if field in annotations:
            transferred[field] = annotations[field]
    
    return transferred


# ============================================================
# Outpainting
# ============================================================

def create_outpainting_canvas(
    image: Image.Image,
    scale_factor: float
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """
    Create canvas and mask for outpainting.
    
    Returns:
        canvas: Image with original pasted in center
        mask: White (inpaint) everywhere except original image (black)
        position: (paste_x, paste_y, orig_width, orig_height)
    """
    orig_w, orig_h = image.size
    
    expand_w = int(orig_w * scale_factor)
    expand_h = int(orig_h * scale_factor)
    
    new_w = orig_w + 2 * expand_w
    new_h = orig_h + 2 * expand_h
    
    # Create canvas with gray background
    canvas = Image.new('RGB', (new_w, new_h), (128, 128, 128))
    paste_x, paste_y = expand_w, expand_h
    canvas.paste(image, (paste_x, paste_y))
    
    # Create mask
    mask = Image.new('L', (new_w, new_h), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([paste_x, paste_y, paste_x + orig_w, paste_y + orig_h], fill=0)
    
    return canvas, mask, (paste_x, paste_y, orig_w, orig_h)


def setup_flux_pipeline():
    """Setup FLUX pipeline with memory optimizations."""
    import torch
    from diffusers import FluxFillPipeline
    print("Loading FLUX.1-Fill-dev model...")
    
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    )
    
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    print("FLUX model ready!")
    return pipe


def run_outpainting(
    pipe,
    canvas: Image.Image,
    mask: Image.Image,
    prompt: str,
    num_steps: int = 28,
    guidance_scale: float = 30.0,
    max_size: int = 728
) -> Image.Image:
    """Run FLUX outpainting."""
    
    orig_canvas_size = canvas.size
    
    # Resize if needed for memory
    if max(canvas.size) > max_size:
        ratio = max_size / max(canvas.size)
        new_size = (int(canvas.size[0] * ratio), int(canvas.size[1] * ratio))
        canvas_resized = canvas.resize(new_size, Image.Resampling.LANCZOS)
        mask_resized = mask.resize(new_size, Image.Resampling.NEAREST)
    else:
        canvas_resized = canvas
        mask_resized = mask
    
    # Generate
    result = pipe(
        prompt=prompt,
        image=canvas_resized,
        mask_image=mask_resized,
        height=canvas_resized.height,
        width=canvas_resized.width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        max_sequence_length=512,
    ).images[0]
    
    # Resize back if needed
    if result.size != orig_canvas_size:
        result = result.resize(orig_canvas_size, Image.Resampling.LANCZOS)
    
    return result


# ============================================================
# Main Processing
# ============================================================

def process_arrow_batch(
    arrow_file: str,
    output_dir: str,
    start_idx: int = 0,
    end_idx: int = None,
    resize_input: float = 0.5,
    scale_mean: float = 0.25,
    scale_min: float = 0.2,
    scale_max: float = 0.4,
    num_steps: int = 28,
    guidance_scale: float = 30.0,
    prompt: str = "Extend the image edges seamlessly. Continue only the existing ground texture, dirt, concrete, and sky. Match lighting, colors, and perspective. Do not add any new objects.",
    seed: int = 42
):
    import pyarrow as pa
    import torch

    """
    Process samples from arrow file.
    
    Outputs to:
        output_dir/images/{image_id}.jpg
        output_dir/annotations/{image_id}.json
        output_dir/meta/{image_id}.json
    """
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    meta_dir = output_path / "meta"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Load arrow file
    print(f"Loading arrow file: {arrow_file}")
    with open(arrow_file, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    
    total = table.num_rows
    columns = table.column_names
    print(f"Total samples: {total}")
    print(f"Columns: {columns}")
    
    if end_idx is None:
        end_idx = total
    end_idx = min(end_idx, total)
    
    print(f"Processing samples {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
    
    # Setup FLUX
    pipe = setup_flux_pipeline()
    
    processed = 0
    skipped = 0
    
    for i in range(start_idx, end_idx):
        # Get image_id
        image_id = str(table.column('image_id')[i].as_py()) if 'image_id' in columns else f"{i:07d}"
        
        # Check if already processed
        output_image = images_dir / f"{image_id}.jpg"
        if output_image.exists():
            print(f"[{i}] {image_id}: Already exists, skipping")
            skipped += 1
            continue
        
        print(f"\n[{i}] Processing: {image_id}")
        
        try:
            # Load image
            image_data = table.column('image')[i].as_py()
            if isinstance(image_data, dict) and 'bytes' in image_data:
                img = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                print(f"  Unknown image format, skipping")
                skipped += 1
                continue
            
            orig_size = img.size
            print(f"  Original size: {orig_size}")
            
            # Get original annotations
            original_annotations = {}
            for col in columns:
                if col != 'image':
                    original_annotations[col] = table.column(col)[i].as_py()
            
            # Resize input
            if resize_input and resize_input != 1.0:
                new_size = (int(img.width * resize_input), int(img.height * resize_input))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"  Resized to: {img.size}")
            
            resized_size = img.size
            
            # Generate random scale
            scale_factor = generate_gaussian_scale(scale_mean, None, scale_min, scale_max)
            print(f"  Scale factor: {scale_factor:.4f}")
            
            # Create canvas and mask
            canvas, mask, position = create_outpainting_canvas(img, scale_factor)
            paste_x, paste_y, paste_w, paste_h = position
            canvas_size = canvas.size
            print(f"  Canvas size: {canvas_size}")
            
            # Run outpainting
            print(f"  Generating...")
            result = run_outpainting(
                pipe=pipe,
                canvas=canvas,
                mask=mask,
                prompt=prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale
            )
            
            # Transfer annotations
            transferred_annotations = transfer_all_annotations(
                annotations=original_annotations,
                orig_size=orig_size,
                paste_pos=(paste_x, paste_y),
                canvas_size=canvas_size,
                resize_factor=resize_input if resize_input else 1.0
            )
            
            # Save outputs
            
            # 1. Save image
            result.save(images_dir / f"{image_id}.jpg", quality=95)
            
            # 2. Save transferred annotation
            annotation_output = {
                "image_id": image_id,
                "bbox_format": "xyxy",
                **transferred_annotations
            }
            with open(annotations_dir / f"{image_id}.json", 'w') as f:
                json.dump(annotation_output, f, indent=2)
            
            # 3. Save meta (for debugging/recovery)
            meta_output = {
                "image_id": image_id,
                "original_size": list(orig_size),
                "resized_size": list(resized_size),
                "canvas_size": list(canvas_size),
                "scale_factor": round(scale_factor, 6),
                "resize_input": resize_input,
                "paste_position": {
                    "x": paste_x,
                    "y": paste_y,
                    "width": paste_w,
                    "height": paste_h
                },
                "prompt": prompt,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "original_annotations": original_annotations,
                "transferred_annotations": transferred_annotations
            }
            with open(meta_dir / f"{image_id}.json", 'w') as f:
                json.dump(meta_output, f, indent=2, default=str)
            
            print(f"  Saved: {image_id}.jpg, annotation, meta")
            processed += 1
            
            # Cleanup
            del result, canvas, mask
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Completed!")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="FLUX Outpainting Pipeline Worker")
    
    parser.add_argument("--arrow", "-a", required=True, help="Input arrow file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--num-samples", "-n", type=int, default=None,
                        help="Number of samples to process (alternative to --end)")
    
    parser.add_argument("--resize-input", type=float, default=0.5,
                        help="Resize input before outpainting (default: 0.5)")
    
    # Scale parameters
    parser.add_argument("--scale-mean", type=float, default=0.25,
                        help="Mean scale factor (default: 0.25)")
    parser.add_argument("--scale-min", type=float, default=0.2,
                        help="Minimum scale factor (default: 0.2)")
    parser.add_argument("--scale-max", type=float, default=0.4,
                        help="Maximum scale factor (default: 0.4)")
    
    # Generation parameters
    parser.add_argument("--num-steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=30.0, help="Guidance scale")
    parser.add_argument("--prompt", type=str,
                        default="Extend the image edges seamlessly. Continue only the existing ground texture, dirt, concrete, and sky. Match lighting, colors, and perspective. Do not add any new objects.",
                        help="Generation prompt")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    # Handle --num-samples
    end_idx = args.end
    if args.num_samples is not None:
        end_idx = args.start + args.num_samples
    
    process_arrow_batch(
        arrow_file=args.arrow,
        output_dir=args.output,
        start_idx=args.start,
        end_idx=end_idx,
        resize_input=args.resize_input,
        scale_mean=args.scale_mean,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        prompt=args.prompt,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
