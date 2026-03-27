#!/usr/bin/env python3
"""
Export all arrow files from output/construction_site_train/filtered to train folder
with structure: train/<style_name>/images/ and train/<style_name>/annotations/
"""

import os
import sys
import pyarrow as pa
from pathlib import Path
from PIL import Image
import io
import json
from tqdm import tqdm


def extract_arrow_to_dataset(arrow_file: str, output_dir: str, style_name: str):
    """
    Extract images and annotations from arrow file to dataset format.
    
    Output structure:
        output_dir/
            images/
                0000001.jpg
                ...
            annotations/
                0000001.json
                ...
    """
    print(f"\n{'='*60}")
    print(f"Processing: {arrow_file}")
    print(f"Style: {style_name}")
    print(f"Output: {output_dir}")
    
    # Load arrow file
    with open(arrow_file, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    
    total = table.num_rows
    print(f"Total samples: {total}")
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    columns = table.column_names
    print(f"Columns: {columns}")
    
    extracted = 0
    
    for i in tqdm(range(total), desc=f"Extracting {style_name}"):
        # Get image_id
        if 'image_id' in columns:
            image_id = table.column('image_id')[i].as_py()
        else:
            image_id = f"{i:07d}"
        
        # Get image
        if 'image' in columns:
            image_data = table.column('image')[i].as_py()
            
            # Handle different image formats in arrow
            if isinstance(image_data, dict):
                if 'bytes' in image_data:
                    img_bytes = image_data['bytes']
                    image = Image.open(io.BytesIO(img_bytes))
                elif 'path' in image_data:
                    image = Image.open(image_data['path'])
                else:
                    print(f"  [{i}] Unknown image format: {image_data.keys()}")
                    continue
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                print(f"  [{i}] Unknown image type: {type(image_data)}")
                continue
            
            # Save image
            image_path = images_dir / f"{image_id}.jpg"
            image.convert('RGB').save(image_path, quality=95)
            
            # Build annotation
            annotation = {
                "image_id": str(image_id),
                "ref_id": str(image_id),
                "style": style_name,
            }
            
            # Get image size
            annotation["image_size"] = {
                "width": image.width,
                "height": image.height
            }
            
            # Extract objects (bounding boxes)
            if 'objects' in columns:
                objects_data = table.column('objects')[i].as_py()
                annotation["objects"] = objects_data if objects_data else {}
            
            # Extract SSIM score if available
            if 'ssim_score' in columns:
                ssim = table.column('ssim_score')[i].as_py()
                annotation["ssim_score"] = ssim
            
            # Extract violations if available
            if 'violations' in columns:
                violations_data = table.column('violations')[i].as_py()
                annotation["violations"] = violations_data if violations_data else {
                    "rule_1": None,
                    "rule_2": None,
                    "rule_3": None,
                    "rule_4": None
                }
            else:
                annotation["violations"] = {
                    "rule_1": None,
                    "rule_2": None,
                    "rule_3": None,
                    "rule_4": None
                }
            
            # Extract metadata if available
            if 'metadata' in columns:
                metadata = table.column('metadata')[i].as_py()
                annotation["metadata"] = metadata if metadata else {}
            elif 'illumination' in columns or 'camera_distance' in columns or 'view' in columns:
                annotation["metadata"] = {}
                for meta_col in ['illumination', 'camera_distance', 'view']:
                    if meta_col in columns:
                        annotation["metadata"][meta_col] = table.column(meta_col)[i].as_py()
            
            # Save annotation
            anno_path = annotations_dir / f"{image_id}.json"
            with open(anno_path, 'w') as f:
                json.dump(annotation, f, indent=2, default=str)
            
            extracted += 1
    
    print(f"✅ Extracted {extracted} images to {output_dir}")
    return extracted


def main():
    # Source directory with arrow files
    source_dir = Path("output/construction_site_train/filtered")
    
    # Output base directory
    output_base = Path("train")
    
    # Find all arrow files
    arrow_files = list(source_dir.glob("*.arrow"))
    
    if not arrow_files:
        print(f"No arrow files found in {source_dir}")
        sys.exit(1)
    
    print(f"Found {len(arrow_files)} arrow files:")
    for f in arrow_files:
        print(f"  - {f.name}")
    
    total_extracted = 0
    
    for arrow_file in arrow_files:
        # Extract style name from filename
        # Example: filtered_style_rain_0_ssim_0.6_0.95.arrow -> style_rain_0
        name = arrow_file.stem  # filtered_style_rain_0_ssim_0.6_0.95
        
        # Parse style name
        parts = name.split('_')
        if 'style' in parts:
            style_idx = parts.index('style')
            # Get style_<type>_<num>
            style_parts = []
            for p in parts[style_idx:]:
                if p.startswith('ssim'):
                    break
                style_parts.append(p)
            style_name = '_'.join(style_parts)
        else:
            style_name = name.replace('filtered_', '').split('_ssim')[0]
        
        # Output directory for this style
        output_dir = output_base / style_name
        
        # Extract
        count = extract_arrow_to_dataset(
            str(arrow_file),
            str(output_dir),
            style_name
        )
        total_extracted += count
    
    print(f"\n{'='*60}")
    print(f"🎉 Total extracted: {total_extracted} images")
    print(f"Output directory: {output_base}")
    print(f"\nStructure:")
    for style_dir in sorted(output_base.iterdir()):
        if style_dir.is_dir():
            img_count = len(list((style_dir / "images").glob("*.jpg")))
            print(f"  {style_dir.name}/")
            print(f"    images/      ({img_count} files)")
            print(f"    annotations/ ({img_count} files)")


if __name__ == "__main__":
    main()
