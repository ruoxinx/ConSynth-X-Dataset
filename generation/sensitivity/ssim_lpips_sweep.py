#!/usr/bin/env python3
"""
Sensitivity analysis: sweep SSIM and LPIPS thresholds on existing metric CSVs.
No GPU required — pure data analysis on pre-computed scores.

Outputs:
  - CSV summary table (threshold × pass_rate × mean_ssim)
  - Figures: pass rate curves, distribution plots
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(os.environ.get('CONSYNTH_REPO_ROOT', Path(__file__).resolve().parents[2]))
AUG  = BASE / 'augmentation_data' / 'construction_site' / 'rain_snow'

# Style Transfer CSVs (train)
ST_DIR = AUG / 'style_transfer' / 'train'

# Diffusion CSVs (test — complete 3004 images, both rain+snow)
DIFF_TEST_DIR = AUG / 'diffusion' / 'test'

# Diffusion CSVs (train)
DIFF_TRAIN_DIR = AUG / 'diffusion' / 'train'

OUT_DIR = BASE / 'generation' / 'sensitivity' / 'results'


# ── Load helpers ───────────────────────────────────────────────────────
def load_style_transfer_csvs():
    """Load all style-transfer SSIM CSV files."""
    frames = {}
    for csv in sorted(ST_DIR.glob('ssim_results_style_*.csv')):
        style = csv.stem.replace('ssim_results_style_', '')
        df = pd.read_csv(csv)
        df['style'] = style
        df['weather'] = 'rain' if 'rain' in style else 'snow'
        df['method'] = 'style_transfer'
        frames[style] = df
    return frames


def load_diffusion_csvs(base_dir):
    """Load all diffusion meta CSVs for rain and snow."""
    frames = {}
    for weather in ['rain', 'snow']:
        parts = []
        weather_dir = base_dir / weather
        # Direct meta files
        for csv in sorted(weather_dir.glob('meta_*.csv')):
            parts.append(pd.read_csv(csv))
        # Sharded (s0, s1, ...)
        for shard_dir in sorted(weather_dir.glob('s*')):
            if shard_dir.is_dir():
                for csv in sorted(shard_dir.glob('meta_*.csv')):
                    parts.append(pd.read_csv(csv))
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df['weather'] = weather
            df['method'] = 'diffusion'
            frames[weather] = df
    return frames


# ── SSIM sweep ─────────────────────────────────────────────────────────
def ssim_sweep(df, ssim_upper=0.95,
               ssim_lowers=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)):
    """Sweep SSIM lower threshold. Returns list of dicts."""
    rows = []
    n = len(df)
    for lo in ssim_lowers:
        mask = (df['ssim'] >= lo) & (df['ssim'] <= ssim_upper)
        kept = mask.sum()
        rows.append({
            'ssim_lower': lo,
            'ssim_upper': ssim_upper,
            'total': n,
            'kept': int(kept),
            'dropped': int(n - kept),
            'pass_rate': kept / n * 100,
            'kept_ssim_mean': float(df.loc[mask, 'ssim'].mean()) if kept > 0 else np.nan,
            'kept_ssim_std':  float(df.loc[mask, 'ssim'].std())  if kept > 0 else np.nan,
        })
    return rows


# ── LPIPS sweep ────────────────────────────────────────────────────────
def lpips_sweep(df, ssim_lower=0.60, ssim_upper=0.95,
                lpips_thresholds=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0)):
    """Sweep LPIPS threshold (with fixed SSIM range). 1.0 = no LPIPS filter."""
    rows = []
    n = len(df)
    has_lpips = 'lpips' in df.columns and df['lpips'].notna().any() and (df['lpips'] != 0).any()
    for lp in lpips_thresholds:
        ssim_mask = (df['ssim'] >= ssim_lower) & (df['ssim'] <= ssim_upper)
        if has_lpips and lp < 1.0:
            lpips_mask = df['lpips'] < lp
            mask = ssim_mask & lpips_mask
        else:
            mask = ssim_mask
        kept = mask.sum()
        row = {
            'lpips_threshold': lp,
            'ssim_lower': ssim_lower,
            'ssim_upper': ssim_upper,
            'total': n,
            'kept': int(kept),
            'dropped': int(n - kept),
            'pass_rate': kept / n * 100,
            'kept_ssim_mean': float(df.loc[mask, 'ssim'].mean()) if kept > 0 else np.nan,
        }
        if has_lpips and kept > 0:
            row['kept_lpips_mean'] = float(df.loc[mask, 'lpips'].mean())
            row['kept_lpips_std']  = float(df.loc[mask, 'lpips'].std())
        rows.append(row)
    return rows


# ── Plotting ───────────────────────────────────────────────────────────
def plot_ssim_sweep(results_by_group, out_path):
    """Plot pass rate vs SSIM lower threshold per group."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, weather in zip(axes, ['rain', 'snow']):
        for label, df_sweep in results_by_group.items():
            if weather not in label:
                continue
            ax.plot(df_sweep['ssim_lower'], df_sweep['pass_rate'],
                    'o-', label=label, markersize=5)
        ax.set_xlabel('SSIM Lower Threshold')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title(f'{weather.title()}: Pass Rate vs SSIM Lower Threshold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    plt.suptitle('SSIM Threshold Sensitivity Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out_path}')


def plot_lpips_sweep(results_by_group, out_path):
    """Plot pass rate vs LPIPS threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, df_sweep in results_by_group.items():
        ax.plot(df_sweep['lpips_threshold'], df_sweep['pass_rate'],
                'o-', label=label, markersize=5)

    ax.set_xlabel('LPIPS Threshold')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('LPIPS Threshold Sensitivity (with SSIM pre-filter)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out_path}')


def plot_kept_count(results_by_group, out_path):
    """Plot absolute image count kept at each threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, weather in zip(axes, ['rain', 'snow']):
        for label, df_sweep in results_by_group.items():
            if weather not in label:
                continue
            ax.plot(df_sweep['ssim_lower'], df_sweep['kept'],
                    's-', label=label, markersize=5)
        ax.set_xlabel('SSIM Lower Threshold')
        ax.set_ylabel('Images Kept')
        ax.set_title(f'{weather.title()}: Images Kept vs SSIM Threshold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('Data Quantity vs Quality Tradeoff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out_path}')


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='SSIM/LPIPS threshold sensitivity analysis')
    parser.add_argument('--split', choices=['train', 'test', 'both'], default='both',
                        help='Which split to analyze')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('SSIM / LPIPS Threshold Sensitivity Analysis')
    print('=' * 60)

    # ── Load data ──────────────────────────────────────────────────
    print('\nLoading Style Transfer CSVs ...')
    st_frames = load_style_transfer_csvs()
    for k, df in st_frames.items():
        print(f'  {k}: {len(df)} rows')

    diff_frames = {}
    if args.split in ('test', 'both'):
        print('\nLoading Diffusion CSVs (test) ...')
        diff_frames_test = load_diffusion_csvs(DIFF_TEST_DIR)
        for k, df in diff_frames_test.items():
            key = f'ip2p_{k}_test'
            diff_frames[key] = df
            print(f'  {key}: {len(df)} rows')

    if args.split in ('train', 'both'):
        print('\nLoading Diffusion CSVs (train) ...')
        diff_frames_train = load_diffusion_csvs(DIFF_TRAIN_DIR)
        for k, df in diff_frames_train.items():
            key = f'ip2p_{k}_train'
            diff_frames[key] = df
            print(f'  {key}: {len(df)} rows')

    # ── SSIM sweep ─────────────────────────────────────────────────
    print('\n' + '─' * 40)
    print('SSIM Lower Threshold Sweep')
    print('─' * 40)

    ssim_results = {}

    # Style transfer
    for style, df in st_frames.items():
        rows = ssim_sweep(df)
        key = f'ST_{style}'
        ssim_results[key] = pd.DataFrame(rows)
        print(f'\n  {key}:')
        print(ssim_results[key][['ssim_lower', 'kept', 'pass_rate']].to_string(index=False))

    # Diffusion
    for key, df in diff_frames.items():
        rows = ssim_sweep(df)
        ssim_results[key] = pd.DataFrame(rows)
        print(f'\n  {key}:')
        print(ssim_results[key][['ssim_lower', 'kept', 'pass_rate']].to_string(index=False))

    # Combine and save
    all_ssim = []
    for label, df_sweep in ssim_results.items():
        df_sweep = df_sweep.copy()
        df_sweep['group'] = label
        all_ssim.append(df_sweep)
    df_ssim_all = pd.concat(all_ssim, ignore_index=True)
    csv_path = OUT_DIR / 'ssim_sweep_results.csv'
    df_ssim_all.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # Plot
    plot_ssim_sweep(ssim_results, OUT_DIR / 'ssim_sweep_pass_rate.png')
    plot_kept_count(ssim_results, OUT_DIR / 'ssim_sweep_kept_count.png')

    # ── LPIPS sweep ────────────────────────────────────────────────
    print('\n' + '─' * 40)
    print('LPIPS Threshold Sweep')
    print('─' * 40)

    lpips_results = {}

    for key, df in diff_frames.items():
        has_lpips = 'lpips' in df.columns and (df['lpips'] != 0).any()
        if not has_lpips:
            print(f'\n  {key}: No LPIPS data (snow uses SSIM only) — running with SSIM-only')
        weather = 'rain' if 'rain' in key else 'snow'
        ssim_lo = 0.60 if weather == 'rain' else 0.50
        rows = lpips_sweep(df, ssim_lower=ssim_lo)
        lpips_results[key] = pd.DataFrame(rows)
        cols = ['lpips_threshold', 'kept', 'pass_rate']
        if 'kept_lpips_mean' in lpips_results[key].columns:
            cols.append('kept_lpips_mean')
        print(f'\n  {key} (SSIM>={ssim_lo}):')
        print(lpips_results[key][cols].to_string(index=False))

    # Combine and save
    all_lpips = []
    for label, df_sweep in lpips_results.items():
        df_sweep = df_sweep.copy()
        df_sweep['group'] = label
        all_lpips.append(df_sweep)
    df_lpips_all = pd.concat(all_lpips, ignore_index=True)
    csv_path = OUT_DIR / 'lpips_sweep_results.csv'
    df_lpips_all.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    plot_lpips_sweep(lpips_results, OUT_DIR / 'lpips_sweep_pass_rate.png')

    # ── Summary recommendation table ──────────────────────────────
    print('\n' + '=' * 60)
    print('SUMMARY — Threshold Combinations to Evaluate Downstream')
    print('=' * 60)

    combos = [
        {'label': 'loose',    'ssim_rain': 0.40, 'ssim_snow': 0.40, 'lpips': 0.50},
        {'label': 'moderate', 'ssim_rain': 0.50, 'ssim_snow': 0.50, 'lpips': 0.40},
        {'label': 'current',  'ssim_rain': 0.60, 'ssim_snow': 0.50, 'lpips': 0.35},
        {'label': 'strict',   'ssim_rain': 0.65, 'ssim_snow': 0.60, 'lpips': 0.30},
        {'label': 'tight',    'ssim_rain': 0.70, 'ssim_snow': 0.65, 'lpips': 0.25},
        {'label': 'no_lpips', 'ssim_rain': 0.60, 'ssim_snow': 0.50, 'lpips': 1.00},
    ]

    combo_rows = []
    for c in combos:
        row = {'config': c['label']}
        for key, df in diff_frames.items():
            weather = 'rain' if 'rain' in key else 'snow'
            ssim_lo = c[f'ssim_{weather}']
            ssim_mask = (df['ssim'] >= ssim_lo) & (df['ssim'] <= 0.95)
            has_lpips = 'lpips' in df.columns and (df['lpips'] != 0).any()
            if has_lpips and c['lpips'] < 1.0:
                mask = ssim_mask & (df['lpips'] < c['lpips'])
            else:
                mask = ssim_mask
            row[f'{key}_kept'] = int(mask.sum())
            row[f'{key}_pct'] = mask.sum() / len(df) * 100
        combo_rows.append(row)

    df_combos = pd.DataFrame(combo_rows)
    print(df_combos.to_string(index=False, float_format='%.1f'))

    csv_path = OUT_DIR / 'threshold_combinations.csv'
    df_combos.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    print('\n' + '=' * 60)
    print('Next step: use filter_arrow_by_threshold.py to create filtered')
    print('Arrow datasets, then train YOLOv8 per threshold combo.')
    print('=' * 60)


if __name__ == '__main__':
    main()
