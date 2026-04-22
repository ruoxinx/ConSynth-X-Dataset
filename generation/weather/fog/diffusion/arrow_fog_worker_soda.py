#!/usr/bin/env python3
"""
Fog worker for SODA VOC Arrow schema: [image_id, image (binary), annotation (str), meta (str)].
Outputs Arrow with same schema + [ref_id, fog_label, fog_visibility].
"""

import argparse
import sys
import io
import gc
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pyarrow as pa
import pyarrow.ipc as ipc

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fog_gen import generate_fog

HAZE_ZONES = {'heavy': (300, 500), 'medium': (500, 750), 'light': (750, 1000)}
FOG_COLOR_RANGE = (210, 245)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--label', required=True, choices=['heavy', 'medium', 'light'])
    p.add_argument('--depth-model', default='depth-anything-small',
                   choices=['depth-anything-small', 'depth-anything-base', 'depth-anything-large'])
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_depth_anything(size, device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_id = f'depth-anything/Depth-Anything-V2-{size}-hf'
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).eval().to(device)
    return model, processor


def estimate_depth(pil_img, model, processor, device):
    inputs = processor(images=pil_img, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inputs).predicted_depth
    h, w = pil_img.size[1], pil_img.size[0]
    pred = torch.nn.functional.interpolate(out.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False).squeeze()
    disp = pred.detach().cpu().numpy()
    p2, p98 = np.percentile(disp, [2, 98])
    disp_clipped = np.clip(disp, p2, p98)
    NEAR_M, FAR_M = 2.0, 200.0
    disp_norm = (disp_clipped - p2) / (p98 - p2 + 1e-8)
    depth = NEAR_M + (1.0 - disp_norm) * (FAR_M - NEAR_M)
    return depth.astype(np.float32)


def apply_fog(img_arr, depth, label):
    vmin, vmax = HAZE_ZONES[label]
    vis = np.random.uniform(vmin, vmax)
    col = np.random.randint(*FOG_COLOR_RANGE)
    foggy = generate_fog(img_arr, depth, visibility=vis, fog_color=col)
    return foggy, vis, col


def img_to_bytes(img, quality=95):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Input:  {args.input}')
    print(f'Output: {args.output}')
    print(f'Range:  {args.start}-{args.end}  Label: {args.label}  Device: {device}')

    with open(args.input, 'rb') as f:
        table = ipc.open_stream(f).read_all()
    total = len(table)
    s, e = max(0, args.start), min(total, args.end)
    print(f'Total in arrow: {total}, processing {s}-{e-1}')

    size_map = {'depth-anything-small': 'Small', 'depth-anything-base': 'Base', 'depth-anything-large': 'Large'}
    model, proc = load_depth_anything(size_map[args.depth_model], device)
    print('Depth model ready')

    orig_cols = table.column_names
    out_data = {c: [] for c in orig_cols}
    out_data['ref_id'] = []
    out_data['fog_label'] = []
    out_data['fog_visibility'] = []

    for idx in range(s, e):
        try:
            img_bytes = table['image'][idx].as_py()
            image_id = table['image_id'][idx].as_py()
            pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            depth = estimate_depth(pil, model, proc, device)
            arr = np.array(pil)
            foggy, vis, col = apply_fog(arr, depth, args.label)
            foggy_pil = Image.fromarray(foggy)
            new_bytes = img_to_bytes(foggy_pil)

            for c in orig_cols:
                if c == 'image':
                    out_data[c].append(new_bytes)
                else:
                    out_data[c].append(table[c][idx].as_py())
            out_data['ref_id'].append(image_id)
            out_data['fog_label'].append(args.label)
            out_data['fog_visibility'].append(round(float(vis), 1))

            if (idx - s + 1) % 50 == 0 or idx == e - 1:
                print(f'  [{idx-s+1}/{e-s}] {image_id} vis={vis:.0f}m')

            del pil, depth, arr, foggy, foggy_pil, new_bytes
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as ex:
            print(f'  ERROR idx={idx}: {ex}')
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            continue

    n = len(out_data['ref_id'])
    print(f'Building Arrow table, {n} samples...')

    fields = list(table.schema) + [
        pa.field('ref_id', pa.string()),
        pa.field('fog_label', pa.string()),
        pa.field('fog_visibility', pa.float32()),
    ]
    schema = pa.schema(fields)

    arrays = []
    for c in orig_cols:
        arrays.append(pa.array(out_data[c], type=table.schema.field(c).type))
    arrays.append(pa.array(out_data['ref_id'], type=pa.string()))
    arrays.append(pa.array(out_data['fog_label'], type=pa.string()))
    arrays.append(pa.array(out_data['fog_visibility'], type=pa.float32()))

    new_table = pa.table(arrays, schema=schema)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_stream(str(out), new_table.schema) as w:
        w.write_table(new_table)
    print(f'Saved: {out}  rows={n}')


if __name__ == '__main__':
    main()
