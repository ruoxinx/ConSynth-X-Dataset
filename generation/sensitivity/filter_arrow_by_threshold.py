#!/usr/bin/env python3
"""
Filter existing Arrow batch files using specified SSIM/LPIPS thresholds.
Reads pre-computed metric CSVs to decide KEEP/DROP, then builds filtered Arrow output.

No GPU required — reads Arrow + CSV only.

Usage:
  python filter_arrow_by_threshold.py --config current
  python filter_arrow_by_threshold.py --ssim-rain 0.50 --ssim-snow 0.50 --lpips 0.40 --tag moderate
"""

import argparse
import os
import sys
from pathlib import Path
import pyarrow as pa
import pandas as pd
import json

BASE = Path(os.environ.get('CONSYNTH_REPO_ROOT', Path(__file__).resolve().parents[2]))
AUG  = BASE / 'augmentation_data' / 'construction_site' / 'rain_snow'

# Pre-defined threshold combos (same as ssim_lpips_sweep.py)
CONFIGS = {
    'loose':    {'ssim_rain': 0.40, 'ssim_snow': 0.40, 'lpips': 0.50},
    'moderate': {'ssim_rain': 0.50, 'ssim_snow': 0.50, 'lpips': 0.40},
    'current':  {'ssim_rain': 0.60, 'ssim_snow': 0.50, 'lpips': 0.35},
    'strict':   {'ssim_rain': 0.65, 'ssim_snow': 0.60, 'lpips': 0.30},
    'tight':    {'ssim_rain': 0.70, 'ssim_snow': 0.65, 'lpips': 0.25},
    'no_lpips': {'ssim_rain': 0.60, 'ssim_snow': 0.50, 'lpips': 1.00},
}


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def save_arrow(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), 'wb') as f:
        writer = pa.ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()


def load_meta_csvs(meta_dir):
    """Load all meta_*.csv in a directory into one DataFrame."""
    parts = []
    for csv in sorted(Path(meta_dir).glob('meta_*.csv')):
        parts.append(pd.read_csv(csv))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def filter_diffusion(weather, split, ssim_lower, ssim_upper, lpips_thresh, output_dir):
    """
    Filter diffusion Arrow batches for one weather type.
    Reads meta CSVs to determine which image_ids to keep,
    then selects matching rows from Arrow batch files.
    """
    base_dir = AUG / 'diffusion' / split / weather

    # Collect meta from direct and sharded dirs
    all_meta = []
    arrow_dirs = []

    # Direct meta files
    direct_meta = load_meta_csvs(base_dir)
    if len(direct_meta) > 0:
        all_meta.append(direct_meta)
        arrow_dirs.append(base_dir)

    # Sharded (s0, s1, ...)
    for shard_dir in sorted(base_dir.glob('s*')):
        if shard_dir.is_dir():
            shard_meta = load_meta_csvs(shard_dir)
            if len(shard_meta) > 0:
                all_meta.append(shard_meta)
                arrow_dirs.append(shard_dir)

    if not all_meta:
        print(f'    No meta CSVs found for {weather}/{split}')
        return 0

    df_meta = pd.concat(all_meta, ignore_index=True)

    # Apply thresholds
    ssim_mask = (df_meta['ssim'] >= ssim_lower) & (df_meta['ssim'] <= ssim_upper)
    has_lpips = 'lpips' in df_meta.columns and (df_meta['lpips'] != 0).any()
    if has_lpips and lpips_thresh < 1.0:
        lpips_mask = df_meta['lpips'] < lpips_thresh
        mask = ssim_mask & lpips_mask
    else:
        mask = ssim_mask

    keep_ids = set(df_meta.loc[mask, 'image_id'].astype(str).values)
    total = len(df_meta)
    kept = len(keep_ids)
    print(f'    {weather}/{split}: {kept}/{total} kept ({kept/total*100:.1f}%)')

    # Now filter Arrow files
    kept_tables = []
    for d in arrow_dirs:
        for arrow_file in sorted(d.glob('batch_*.arrow')):
            table = load_arrow(arrow_file)
            # Find matching rows
            indices = []
            for i in range(len(table)):
                img_id = str(table.column('image_id')[i].as_py())
                if img_id in keep_ids:
                    indices.append(i)
            if indices:
                kept_tables.append(table.take(indices))

    if not kept_tables:
        print(f'    No Arrow rows matched for {weather}/{split}')
        return 0

    merged = pa.concat_tables(kept_tables)
    out_path = output_dir / f'{weather}_{split}.arrow'
    save_arrow(merged, out_path)
    print(f'    Saved: {out_path} ({len(merged)} rows)')
    return len(merged)


def filter_style_transfer(style, ssim_lower, ssim_upper, output_dir):
    """
    Filter style-transfer data by re-applying SSIM threshold.
    The original filtered Arrow files used fixed thresholds.
    We re-filter from the SSIM CSV + raw batch Arrow files.
    """
    csv_path = AUG / 'style_transfer' / 'train' / f'ssim_results_style_{style}.csv'
    if not csv_path.exists():
        print(f'    CSV not found: {csv_path}')
        return 0

    df = pd.read_csv(csv_path)
    mask = (df['ssim'] >= ssim_lower) & (df['ssim'] <= ssim_upper)
    kept = mask.sum()
    total = len(df)
    print(f'    ST {style}: {kept}/{total} kept ({kept/total*100:.1f}%)')

    # For style transfer, we use the existing filtered Arrow files as source
    # (they contain all generated images regardless of current threshold)
    # We need the unfiltered batch files to re-filter properly
    # Check if raw batches exist
    raw_dir = AUG / 'style_transfer' / 'train'
    batch_files = sorted(raw_dir.glob(f'*{style}*.arrow'))

    if not batch_files:
        # Fallback: use the existing filtered file and sub-filter
        existing = raw_dir / f'filtered_style_{style}_ssim_0.5_0.95.arrow'
        if not existing.exists():
            existing = raw_dir / f'filtered_style_{style}_ssim_0.6_0.95.arrow'
        if not existing.exists():
            print(f'    No Arrow source found for ST {style}')
            return 0

        table = load_arrow(existing)
        # Re-filter by ref_id matching
        keep_refs = set(df.loc[mask, 'ref_id'].astype(str).values)
        indices = []
        ref_col = 'ref_id' if 'ref_id' in table.column_names else 'image_id'
        for i in range(len(table)):
            ref = str(table.column(ref_col)[i].as_py())
            if ref in keep_refs:
                indices.append(i)
        if not indices:
            return 0
        filtered = table.take(indices)
    else:
        # Direct filter from batch files
        keep_refs = set(df.loc[mask, 'ref_id'].astype(str).values)
        parts = []
        for bf in batch_files:
            table = load_arrow(bf)
            ref_col = 'ref_id' if 'ref_id' in table.column_names else 'image_id'
            indices = [i for i in range(len(table))
                       if str(table.column(ref_col)[i].as_py()) in keep_refs]
            if indices:
                parts.append(table.take(indices))
        if not parts:
            return 0
        filtered = pa.concat_tables(parts)

    out_path = output_dir / f'st_{style}_train.arrow'
    save_arrow(filtered, out_path)
    print(f'    Saved: {out_path} ({len(filtered)} rows)')
    return len(filtered)


def main():
    parser = argparse.ArgumentParser(description='Filter Arrow data by SSIM/LPIPS threshold')
    parser.add_argument('--config', type=str, default=None,
                        choices=list(CONFIGS.keys()),
                        help='Pre-defined threshold config')
    parser.add_argument('--ssim-rain', type=float, default=None)
    parser.add_argument('--ssim-snow', type=float, default=None)
    parser.add_argument('--lpips', type=float, default=None)
    parser.add_argument('--tag', type=str, default=None, help='Custom tag for output dir')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'both'])
    parser.add_argument('--method', type=str, default='both',
                        choices=['diffusion', 'style_transfer', 'both'])
    args = parser.parse_args()

    if args.config:
        cfg = CONFIGS[args.config]
        tag = args.config
    elif args.ssim_rain is not None:
        cfg = {
            'ssim_rain': args.ssim_rain,
            'ssim_snow': args.ssim_snow or args.ssim_rain,
            'lpips': args.lpips or 1.0,
        }
        tag = args.tag or f'ssim{args.ssim_rain}_lpips{cfg["lpips"]}'
    else:
        parser.error('Provide --config or --ssim-rain')

    output_dir = AUG / 'filtered' / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'Filtering with config: {tag}')
    print(f'  SSIM rain: [{cfg["ssim_rain"]}, 0.95]')
    print(f'  SSIM snow: [{cfg["ssim_snow"]}, 0.95]')
    print(f'  LPIPS:     < {cfg["lpips"]}' + (' (disabled)' if cfg['lpips'] >= 1.0 else ''))
    print(f'  Output:    {output_dir}')
    print('=' * 60)

    summary = {'config': tag, **cfg}
    splits = ['train', 'test'] if args.split == 'both' else [args.split]

    # Diffusion
    if args.method in ('diffusion', 'both'):
        print('\n── Diffusion (IP2P) ──')
        for split in splits:
            for weather in ['rain', 'snow']:
                ssim_lo = cfg['ssim_rain'] if weather == 'rain' else cfg['ssim_snow']
                n = filter_diffusion(weather, split, ssim_lo, 0.95, cfg['lpips'], output_dir)
                summary[f'diff_{weather}_{split}'] = n

    # Style transfer
    if args.method in ('style_transfer', 'both'):
        print('\n── Style Transfer ──')
        for style in ['rain_0', 'rain_1', 'rain_2', 'snow_0', 'snow_1', 'snow_2']:
            weather = 'rain' if 'rain' in style else 'snow'
            ssim_lo = cfg['ssim_rain'] if weather == 'rain' else cfg['ssim_snow']
            n = filter_style_transfer(style, ssim_lo, 0.95, output_dir)
            summary[f'st_{style}'] = n

    # Save summary
    summary_path = output_dir / 'filter_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {summary_path}')


if __name__ == '__main__':
    main()
