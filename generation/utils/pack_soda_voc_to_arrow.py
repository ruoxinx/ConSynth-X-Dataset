#!/usr/bin/env python3
"""
Pack SODA VOC augmented JPGs + original XML annotations into Arrow format.
Weather augmentation preserves geometry, so original annotations apply directly.

Validates:
  - Every image has a matching XML annotation
  - Image dimensions match annotation <size> (warns if augmented size differs)
  - Bounding boxes are within image bounds

Usage:
    python pack_soda_voc_to_arrow.py \
        --image-dir experiments/ablation_style_transfer/soda_voc/rain_snow/style_transfer/rain_0/JPEGImages \
        --annotation-dir /path/to/VOC2007/Annotations \
        --output experiments/ablation_style_transfer/soda_voc/rain_snow/style_transfer/rain_0.arrow \
        --weather rain --style 0
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import io
import pyarrow as pa


def parse_voc_xml(xml_path):
    """Parse VOC XML annotation, return dict with size and objects."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        truncated = int(obj.find('truncated').text) if obj.find('truncated') is not None else 0
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax],
            'difficult': difficult,
            'truncated': truncated,
        })

    return {'width': w, 'height': h, 'objects': objects}


def image_to_bytes(img_path):
    with open(img_path, 'rb') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--annotation-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--weather', type=str, default='')
    parser.add_argument('--style', type=str, default='')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    annotation_dir = Path(args.annotation_dir)
    output_path = Path(args.output)

    images = sorted(image_dir.glob('*.jpg'))
    if not images:
        images = sorted(image_dir.glob('*.jpeg')) + sorted(image_dir.glob('*.png'))

    print(f"Images: {len(images)} in {image_dir}")
    print(f"Annotations: {annotation_dir}")
    print(f"Output: {output_path}")

    # Validate all images have annotations
    missing = []
    for img in images:
        xml_path = annotation_dir / f'{img.stem}.xml'
        if not xml_path.exists():
            missing.append(img.stem)
    if missing:
        print(f"ERROR: {len(missing)} images missing annotations: {missing[:5]}...")
        return

    # Build Arrow table
    records = {
        'image': [],
        'image_id': [],
        'filename': [],
        'width': [],
        'height': [],
        'objects_name': [],
        'objects_bbox': [],
        'objects_difficult': [],
        'objects_truncated': [],
        'ref_id': [],
    }
    if args.weather:
        records['weather'] = []
    if args.style:
        records['style'] = []

    errors = 0
    for i, img_path in enumerate(images):
        xml_path = annotation_dir / f'{img_path.stem}.xml'
        ann = parse_voc_xml(xml_path)

        # Read augmented image to verify dimensions
        aug_img = Image.open(img_path)
        aug_w, aug_h = aug_img.size

        # Warn if dimensions differ (style transfer may resize)
        if aug_w != ann['width'] or aug_h != ann['height']:
            print(f"  WARN: {img_path.stem} size mismatch: "
                  f"aug={aug_w}x{aug_h} vs xml={ann['width']}x{ann['height']}")

        # Validate bboxes against actual image size
        valid_objects = []
        for obj in ann['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']
            # Clamp to image bounds (use augmented size)
            xmin = max(0, min(xmin, aug_w))
            ymin = max(0, min(ymin, aug_h))
            xmax = max(0, min(xmax, aug_w))
            ymax = max(0, min(ymax, aug_h))
            if xmax > xmin and ymax > ymin:
                obj['bbox'] = [xmin, ymin, xmax, ymax]
                valid_objects.append(obj)

        img_bytes = image_to_bytes(img_path)

        records['image'].append({'bytes': img_bytes, 'path': img_path.name})
        records['image_id'].append(img_path.stem)
        records['filename'].append(img_path.name)
        records['width'].append(aug_w)
        records['height'].append(aug_h)
        records['objects_name'].append([o['name'] for o in valid_objects])
        records['objects_bbox'].append([o['bbox'] for o in valid_objects])
        records['objects_difficult'].append([o['difficult'] for o in valid_objects])
        records['objects_truncated'].append([o['truncated'] for o in valid_objects])
        records['ref_id'].append(img_path.stem)
        if args.weather:
            records['weather'].append(args.weather)
        if args.style:
            records['style'].append(args.style)

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(images)}] processed")

    # Write Arrow
    table = pa.table(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(output_path), 'wb') as f:
        writer = pa.ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()

    print(f"\nArrow saved: {output_path}")
    print(f"  Total: {len(table)} images")
    print(f"  Schema: {table.column_names}")
    if errors:
        print(f"  Errors: {errors}")


if __name__ == '__main__':
    main()
