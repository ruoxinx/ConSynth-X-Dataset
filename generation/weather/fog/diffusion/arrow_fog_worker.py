#!/usr/bin/env python3
"""
Worker script: apply fog augmentation to a batch from Arrow file.
Preserves all original annotations. Adds ref_id, fog_label, fog_visibility columns.

Usage:
    python arrow_fog_worker.py --input data.arrow --output fog_heavy_0-300.arrow \
        --start 0 --end 300 --label heavy
"""

import argparse
import sys
import os
import io
import gc
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pyarrow as pa
import pyarrow.ipc as ipc

# Setup path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fog_gen import generate_fog

# Haze zones: visibility ranges per label
HAZE_ZONES = {
    'heavy':  (300, 500),
    'medium': (500, 750),
    'light':  (750, 1000),
}
FOG_COLOR_RANGE = (210, 245)


def parse_args():
    parser = argparse.ArgumentParser(description='Fog augmentation Arrow worker')
    parser.add_argument('--input', required=True, help='Input .arrow file')
    parser.add_argument('--output', required=True, help='Output .arrow file')
    parser.add_argument('--start', type=int, required=True, help='Start index (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='End index (exclusive)')
    parser.add_argument('--label', required=True, choices=['heavy', 'medium', 'light'],
                        help='Fog intensity label (determines visibility range)')
    parser.add_argument('--depth-model', default='depth-anything-small',
                        choices=['depth-anything-small', 'depth-anything-base', 'depth-anything-large'])
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_depth_anything(model_size='Small', device='cpu'):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_id = f'depth-anything/Depth-Anything-V2-{model_size}-hf'
    print(f'  Loading Depth Anything V2 ({model_size})...')
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval().to(device)
    return model, processor


def estimate_depth(pil_img, model, processor, device):
    inputs = processor(images=pil_img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    h, w = pil_img.size[1], pil_img.size[0]
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1), size=(h, w),
        mode='bicubic', align_corners=False,
    ).squeeze()

    disp = prediction.detach().cpu().numpy()
    p2, p98 = np.percentile(disp, [2, 98])
    disp_clipped = np.clip(disp, p2, p98)
    NEAR_M, FAR_M = 2.0, 200.0
    disp_norm = (disp_clipped - p2) / (p98 - p2 + 1e-8)
    depth = NEAR_M + (1.0 - disp_norm) * (FAR_M - NEAR_M)
    return depth.astype(np.float32)


def apply_fog_with_label(img_array, depth_map, label):
    vis_min, vis_max = HAZE_ZONES[label]
    visibility = np.random.uniform(vis_min, vis_max)
    fog_color = np.random.randint(*FOG_COLOR_RANGE)
    foggy = generate_fog(img_array, depth_map, visibility=visibility, fog_color=fog_color)
    return foggy, visibility, fog_color


def image_to_bytes(img, fmt='JPEG', quality=95):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print('FOG AUGMENTATION ARROW WORKER')
    print('=' * 70)
    print(f'Input:    {args.input}')
    print(f'Output:   {args.output}')
    print(f'Range:    {args.start} - {args.end}')
    print(f'Label:    {args.label} (vis {HAZE_ZONES[args.label][0]}-{HAZE_ZONES[args.label][1]}m)')
    print(f'Device:   {DEVICE}')
    print('=' * 70)

    # Load dataset
    print('\nLoading dataset...')
    with open(args.input, 'rb') as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    total = len(table)
    start = max(0, args.start)
    end = min(total, args.end)
    print(f'Total: {total}, processing {start}-{end-1} ({end-start} samples)')

    # Load depth model
    print('\nLoading depth model...')
    size_map = {'depth-anything-small': 'Small', 'depth-anything-base': 'Base', 'depth-anything-large': 'Large'}
    da_model, da_processor = load_depth_anything(size_map[args.depth_model], DEVICE)
    print('Depth model ready!')

    # Prepare output data
    original_columns = table.column_names
    new_data = {col: [] for col in original_columns}
    new_data['ref_id'] = []
    new_data['fog_label'] = []
    new_data['fog_visibility'] = []

    # Process batch
    print(f'\nProcessing {end-start} images...')
    for idx in range(start, end):
        try:
            image_data = table['image'][idx].as_py()
            image_id = table['image_id'][idx].as_py()

            img_bytes = image_data['bytes']
            img_path = image_data.get('path', f'{image_id}.jpg')
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Depth estimation
            depth_map = estimate_depth(pil_img, da_model, da_processor, DEVICE)

            # Apply fog
            img_array = np.array(pil_img)
            foggy, visibility, fog_color = apply_fog_with_label(img_array, depth_map, args.label)
            foggy_img = Image.fromarray(foggy)
            foggy_bytes = image_to_bytes(foggy_img)

            # Store all original columns
            for col in original_columns:
                if col == 'image':
                    new_data[col].append({'bytes': foggy_bytes, 'path': img_path})
                else:
                    new_data[col].append(table[col][idx].as_py())

            new_data['ref_id'].append(image_id)
            new_data['fog_label'].append(args.label)
            new_data['fog_visibility'].append(round(visibility, 1))

            if (idx - start + 1) % 50 == 0 or idx == end - 1:
                print(f'  [{idx-start+1}/{end-start}] {image_id} | {args.label} | vis={visibility:.0f}m')

            del pil_img, depth_map, img_array, foggy, foggy_img, foggy_bytes, img_bytes
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  ERROR sample {idx} ({image_id}): {e}')
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            continue

    # Build Arrow table
    n_processed = len(new_data['ref_id'])
    print(f'\nBuilding Arrow table ({n_processed} samples)...')

    # Build schema: original + new columns
    fields = list(table.schema)
    fields.append(pa.field('ref_id', pa.string()))
    fields.append(pa.field('fog_label', pa.string()))
    fields.append(pa.field('fog_visibility', pa.float32()))
    new_schema = pa.schema(fields)

    arrays = []
    for col in original_columns:
        if col == 'image':
            bytes_arr = pa.array([d['bytes'] for d in new_data[col]], type=pa.binary())
            path_arr = pa.array([d['path'] for d in new_data[col]], type=pa.string())
            arrays.append(pa.StructArray.from_arrays([bytes_arr, path_arr], names=['bytes', 'path']))
        else:
            arrays.append(pa.array(new_data[col], type=table.schema.field(col).type))

    arrays.append(pa.array(new_data['ref_id'], type=pa.string()))
    arrays.append(pa.array(new_data['fog_label'], type=pa.string()))
    arrays.append(pa.array(new_data['fog_visibility'], type=pa.float32()))

    new_table = pa.table(arrays, schema=new_schema)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_stream(str(out_path), new_table.schema) as writer:
        writer.write_table(new_table)

    print(f'Saved: {out_path} ({n_processed} samples)')
    print('\nDONE')


if __name__ == '__main__':
    main()
