#!/usr/bin/env python3
"""
Pack SODA-KTSH augmented JPGs + original captions into Arrow format.
Matches augmented images (by filename) back to original captions from HDF5/JSON.

Usage:
    python pack_soda_ktsh_to_arrow.py \
        --image-dir augmentation_data/soda_ktsh/rain_snow/diffusion/rain \
        --original-image-dir /path/to/soda-ktsh/images \
        --caption-dir /path/to/soda-ktsh/captiondata \
        --output augmentation_data/soda_ktsh/rain_snow/diffusion/rain.arrow \
        --weather rain
"""

import argparse
import json
from pathlib import Path
from PIL import Image
import pyarrow as pa


def load_wordmap(caption_dir):
    with open(caption_dir / 'WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json') as f:
        wordmap = json.load(f)
    return {v: k for k, v in wordmap.items()}


def load_captions(caption_dir, split, rev_map):
    """Load and decode captions for a split. Returns list of decoded caption strings."""
    with open(caption_dir / f'{split}_CAPTIONS_flickr8k_5_cap_per_img_5_min_word_freq.json') as f:
        raw = json.load(f)
    with open(caption_dir / f'{split}_CAPLENS_flickr8k_5_cap_per_img_5_min_word_freq.json') as f:
        caplens = json.load(f)

    decoded = []
    for cap in raw:
        tokens = [rev_map.get(t, '') for t in cap if t != 0]
        # Remove <start> and <end> tokens
        text = ' '.join(t for t in tokens if t not in ('<start>', '<end>', '<pad>', ''))
        decoded.append(text)
    return decoded


def build_image_to_captions(original_image_dir, caption_dir):
    """Build mapping: image filename stem -> list of 5 caption strings."""
    rev_map = load_wordmap(caption_dir)

    # Original images sorted = same order as HDF5
    orig_images = sorted(Path(original_image_dir).glob('*.jpg'))
    print(f"Original images: {len(orig_images)}")

    # Load all splits and concatenate
    all_captions = []
    all_images = []
    for split in ['TRAIN', 'VAL', 'TEST']:
        caps = load_captions(caption_dir, split, rev_map)
        n_images = len(caps) // 5
        all_captions.extend(caps)
        all_images.append(n_images)
        print(f"  {split}: {n_images} images, {len(caps)} captions")

    total_images = sum(all_images)
    print(f"  Total from captions: {total_images} images")

    # Map: image stem -> 5 captions
    # HDF5 order = sorted glob order of original images
    mapping = {}
    for i, img in enumerate(orig_images[:total_images]):
        captions = all_captions[i*5 : (i+1)*5]
        mapping[img.stem] = captions

    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--original-image-dir', type=str, required=True)
    parser.add_argument('--caption-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--weather', type=str, default='')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    caption_dir = Path(args.caption_dir)
    output_path = Path(args.output)

    aug_images = sorted(image_dir.glob('*.jpg'))
    print(f"Augmented images: {len(aug_images)} in {image_dir}")

    # Build caption mapping
    print("Loading captions...")
    img2cap = build_image_to_captions(args.original_image_dir, caption_dir)
    print(f"Caption mapping: {len(img2cap)} images")

    # Match augmented images to captions
    records = {
        'image': [],
        'image_id': [],
        'filename': [],
        'captions': [],
        'ref_id': [],
    }
    if args.weather:
        records['weather'] = []

    matched = 0
    missing = 0
    for i, img_path in enumerate(aug_images):
        stem = img_path.stem
        captions = img2cap.get(stem)
        if captions is None:
            missing += 1
            if missing <= 5:
                print(f"  WARN: no captions for {stem}")
            continue

        with open(img_path, 'rb') as f:
            img_bytes = f.read()

        records['image'].append({'bytes': img_bytes, 'path': img_path.name})
        records['image_id'].append(stem)
        records['filename'].append(img_path.name)
        records['captions'].append(captions)
        records['ref_id'].append(stem)
        if args.weather:
            records['weather'].append(args.weather)
        matched += 1

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(aug_images)}] processed")

    # Write Arrow
    table = pa.table(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(output_path), 'wb') as f:
        writer = pa.ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()

    print(f"\nArrow saved: {output_path}")
    print(f"  Matched: {matched}, Missing captions: {missing}")
    print(f"  Schema: {table.column_names}")


if __name__ == '__main__':
    main()
