#!/usr/bin/env python3
"""
FLUX Outpainting Pipeline Worker for VOC Format (SODA Dataset)

Reads images from JPEGImages/ and annotations from Annotations/ (Pascal VOC XML),
applies FLUX outpainting, and saves outpainted images with transferred annotations.

Output structure:
  output_dir/
    JPEGImages/    - Outpainted images
    Annotations/   - Transferred annotations (VOC XML with updated pixel coords)
    meta/          - Full metadata for debugging
"""

from __future__ import annotations

import os
import sys
import json
import random
import gc
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from xml.dom import minidom


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
# VOC XML Parsing & Writing
# ============================================================

def parse_voc_xml(xml_path: str) -> Dict:
    """Parse Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotation = {
        "folder": root.findtext("folder", ""),
        "filename": root.findtext("filename", ""),
        "path": root.findtext("path", ""),
        "size": {
            "width": int(root.find("size/width").text),
            "height": int(root.find("size/height").text),
            "depth": int(root.find("size/depth").text),
        },
        "segmented": root.findtext("segmented", "0"),
        "objects": [],
    }

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        annotation["objects"].append({
            "name": obj.findtext("name", ""),
            "pose": obj.findtext("pose", "Unspecified"),
            "truncated": int(obj.findtext("truncated", "0")),
            "difficult": int(obj.findtext("difficult", "0")),
            "bndbox": {
                "xmin": int(float(bndbox.findtext("xmin"))),
                "ymin": int(float(bndbox.findtext("ymin"))),
                "xmax": int(float(bndbox.findtext("xmax"))),
                "ymax": int(float(bndbox.findtext("ymax"))),
            },
        })

    return annotation


def write_voc_xml(annotation: Dict, output_path: str):
    """Write Pascal VOC XML annotation file."""
    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = annotation.get("folder", "")
    ET.SubElement(root, "filename").text = annotation.get("filename", "")
    ET.SubElement(root, "path").text = annotation.get("path", "")

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(annotation["size"]["width"])
    ET.SubElement(size, "height").text = str(annotation["size"]["height"])
    ET.SubElement(size, "depth").text = str(annotation["size"]["depth"])

    ET.SubElement(root, "segmented").text = str(annotation.get("segmented", "0"))

    for obj in annotation.get("objects", []):
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "name").text = obj["name"]
        ET.SubElement(obj_elem, "pose").text = obj.get("pose", "Unspecified")
        ET.SubElement(obj_elem, "truncated").text = str(obj.get("truncated", 0))
        ET.SubElement(obj_elem, "difficult").text = str(obj.get("difficult", 0))

        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["bndbox"]["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["bndbox"]["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["bndbox"]["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["bndbox"]["ymax"])

    xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="\t")
    # Remove extra XML declaration from minidom
    lines = xml_str.split("\n")
    lines = [l for l in lines if not l.startswith("<?xml")]
    with open(output_path, "w") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write("\n".join(lines))


# ============================================================
# Bbox Transfer (Pixel Coordinates)
# ============================================================

def transfer_bbox_pixel(
    bbox: Dict,
    paste_x: int,
    paste_y: int,
    resize_factor: float,
    canvas_w: int,
    canvas_h: int
) -> Dict:
    """
    Transfer bbox from original pixel coords to outpainted canvas coords.

    After resize by resize_factor, the image is pasted at (paste_x, paste_y).
    New pixel coords = original_pixel * resize_factor + paste_offset.
    Clamp to canvas bounds.
    """
    xmin = int(round(bbox["xmin"] * resize_factor)) + paste_x
    ymin = int(round(bbox["ymin"] * resize_factor)) + paste_y
    xmax = int(round(bbox["xmax"] * resize_factor)) + paste_x
    ymax = int(round(bbox["ymax"] * resize_factor)) + paste_y

    # Clamp to canvas
    xmin = max(0, min(xmin, canvas_w))
    ymin = max(0, min(ymin, canvas_h))
    xmax = max(0, min(xmax, canvas_w))
    ymax = max(0, min(ymax, canvas_h))

    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


# ============================================================
# Outpainting (reuse from original)
# ============================================================

def create_outpainting_canvas(
    image: Image.Image,
    scale_factor: float
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    from PIL import Image, ImageDraw
    """Create canvas and mask for outpainting."""
    orig_w, orig_h = image.size

    expand_w = int(orig_w * scale_factor)
    expand_h = int(orig_h * scale_factor)

    new_w = orig_w + 2 * expand_w
    new_h = orig_h + 2 * expand_h

    canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
    paste_x, paste_y = expand_w, expand_h
    canvas.paste(image, (paste_x, paste_y))

    mask = Image.new("L", (new_w, new_h), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([paste_x, paste_y, paste_x + orig_w, paste_y + orig_h], fill=0)

    return canvas, mask, (paste_x, paste_y, orig_w, orig_h)


def setup_flux_pipeline():
    import torch
    from diffusers import FluxFillPipeline
    """Setup FLUX pipeline with memory optimizations."""
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
    num_steps: int = 35,
    guidance_scale: float = 30.0,
    max_size: int = 1024
) -> Image.Image:
    """Run FLUX outpainting."""
    from PIL import Image
    orig_canvas_size = canvas.size

    if max(canvas.size) > max_size:
        ratio = max_size / max(canvas.size)
        new_size = (int(canvas.size[0] * ratio), int(canvas.size[1] * ratio))
        canvas_resized = canvas.resize(new_size, Image.Resampling.LANCZOS)
        mask_resized = mask.resize(new_size, Image.Resampling.NEAREST)
    else:
        canvas_resized = canvas
        mask_resized = mask

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

    if result.size != orig_canvas_size:
        result = result.resize(orig_canvas_size, Image.Resampling.LANCZOS)

    return result


# ============================================================
# Main Processing
# ============================================================

def process_voc_batch(
    images_dir: str,
    annotations_dir: str,
    image_list: List[str],
    output_dir: str,
    start_idx: int = 0,
    end_idx: int = None,
    resize_input: float = 0.7,
    scale_mean: float = 0.3,
    scale_min: float = 0.25,
    scale_max: float = 0.4,
    num_steps: int = 35,
    guidance_scale: float = 30.0,
    prompt: str = "Extend the image edges seamlessly. Continue only the existing ground texture, dirt, concrete, and sky. Match lighting, colors, and perspective. Do not add any new objects.",
    seed: int = 42,
    max_size: int = 1024
):
    """Process VOC format images with outpainting."""
    import torch
    from PIL import Image

    random.seed(seed)

    output_path = Path(output_dir)
    out_images_dir = output_path / "JPEGImages"
    out_annotations_dir = output_path / "Annotations"
    meta_dir = output_path / "meta"

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_annotations_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    total = len(image_list)
    if end_idx is None:
        end_idx = total
    end_idx = min(end_idx, total)

    print(f"Total images in list: {total}")
    print(f"Processing: {start_idx} to {end_idx} ({end_idx - start_idx} images)")

    pipe = setup_flux_pipeline()

    processed = 0
    skipped = 0

    for i in range(start_idx, end_idx):
        image_id = image_list[i]
        image_file = Path(images_dir) / f"{image_id}.jpg"
        xml_file = Path(annotations_dir) / f"{image_id}.xml"

        out_image_file = out_images_dir / f"{image_id}.jpg"
        if out_image_file.exists():
            print(f"[{i}] {image_id}: Already exists, skipping")
            skipped += 1
            continue

        if not image_file.exists():
            print(f"[{i}] {image_id}: Image not found, skipping")
            skipped += 1
            continue

        print(f"\n[{i}] Processing: {image_id}")

        try:
            img = Image.open(str(image_file)).convert("RGB")
            orig_w, orig_h = img.size
            print(f"  Original size: {orig_w}x{orig_h}")

            # Parse annotation if exists
            has_annotation = xml_file.exists()
            if has_annotation:
                voc_annotation = parse_voc_xml(str(xml_file))
            else:
                voc_annotation = {
                    "folder": "",
                    "filename": f"{image_id}.jpg",
                    "size": {"width": orig_w, "height": orig_h, "depth": 3},
                    "segmented": "0",
                    "objects": [],
                }
                print(f"  No annotation file, proceeding without objects")

            # Resize input
            actual_resize = resize_input if resize_input and resize_input != 1.0 else 1.0
            if actual_resize != 1.0:
                new_size = (int(orig_w * actual_resize), int(orig_h * actual_resize))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"  Resized to: {img.size}")

            resized_w, resized_h = img.size

            # Generate random scale
            scale_factor = generate_gaussian_scale(scale_mean, None, scale_min, scale_max)
            print(f"  Scale factor: {scale_factor:.4f}")

            # Create canvas and mask
            canvas, mask, position = create_outpainting_canvas(img, scale_factor)
            paste_x, paste_y, paste_w, paste_h = position
            canvas_w, canvas_h = canvas.size
            print(f"  Canvas size: {canvas_w}x{canvas_h}")

            # Run outpainting
            print(f"  Generating (steps={num_steps}, max_size={max_size})...")
            result = run_outpainting(
                pipe=pipe,
                canvas=canvas,
                mask=mask,
                prompt=prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                max_size=max_size,
            )

            # Transfer annotations
            transferred_objects = []
            for obj in voc_annotation.get("objects", []):
                new_bbox = transfer_bbox_pixel(
                    bbox=obj["bndbox"],
                    paste_x=paste_x,
                    paste_y=paste_y,
                    resize_factor=actual_resize,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                )
                # Skip degenerate boxes
                if new_bbox["xmax"] > new_bbox["xmin"] and new_bbox["ymax"] > new_bbox["ymin"]:
                    transferred_objects.append({
                        "name": obj["name"],
                        "pose": obj.get("pose", "Unspecified"),
                        "truncated": obj.get("truncated", 0),
                        "difficult": obj.get("difficult", 0),
                        "bndbox": new_bbox,
                    })

            # Build output annotation
            out_annotation = {
                "folder": "JPEGImages",
                "filename": f"{image_id}.jpg",
                "path": str(out_image_file),
                "size": {"width": canvas_w, "height": canvas_h, "depth": 3},
                "segmented": voc_annotation.get("segmented", "0"),
                "objects": transferred_objects,
            }

            # Save outputs
            result.save(str(out_image_file), quality=95)
            write_voc_xml(out_annotation, str(out_annotations_dir / f"{image_id}.xml"))

            # Save meta
            meta = {
                "image_id": image_id,
                "original_size": [orig_w, orig_h],
                "resized_size": [resized_w, resized_h],
                "canvas_size": [canvas_w, canvas_h],
                "scale_factor": round(scale_factor, 6),
                "resize_input": actual_resize,
                "paste_position": {"x": paste_x, "y": paste_y, "width": paste_w, "height": paste_h},
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "max_size": max_size,
                "prompt": prompt,
                "num_objects_original": len(voc_annotation.get("objects", [])),
                "num_objects_transferred": len(transferred_objects),
            }
            with open(meta_dir / f"{image_id}.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  Saved: {image_id}.jpg ({len(transferred_objects)} objects)")
            processed += 1

            del result, canvas, mask
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
            continue

    print(f"\n{'='*60}")
    print(f"Completed!")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="FLUX Outpainting Worker for VOC Format")

    parser.add_argument("--images-dir", required=True, help="Path to JPEGImages/")
    parser.add_argument("--annotations-dir", required=True, help="Path to Annotations/")
    parser.add_argument("--image-list", required=True, help="Text file with image IDs (one per line)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument("--resize-input", type=float, default=0.7)
    parser.add_argument("--scale-mean", type=float, default=0.3)
    parser.add_argument("--scale-min", type=float, default=0.25)
    parser.add_argument("--scale-max", type=float, default=0.4)

    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--max-size", type=int, default=1024,
                        help="Max canvas size for generation (default: 1024 for A100)")
    parser.add_argument("--prompt", type=str,
                        default="Extend the image edges seamlessly. Continue only the existing ground texture, dirt, concrete, and sky. Match lighting, colors, and perspective. Do not add any new objects.")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load image list
    with open(args.image_list, "r", encoding="latin-1") as f:
        image_list = [line.strip() for line in f if line.strip()]

    process_voc_batch(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        image_list=image_list,
        output_dir=args.output,
        start_idx=args.start,
        end_idx=args.end,
        resize_input=args.resize_input,
        scale_mean=args.scale_mean,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        prompt=args.prompt,
        seed=args.seed,
        max_size=args.max_size,
    )


if __name__ == "__main__":
    main()
