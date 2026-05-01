#!/usr/bin/env python3
"""
Submit SLURM jobs for Order B weather-night pipeline (full construction-site test set).

Generates job .sh files under jobs/weather_night/ and submits them.
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow as pa


REPO = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')
ORIG_ARROW = Path('/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data_arrow/construction_site_test.arrow')
WORKER = REPO / 'generation/weather/rain_snow/diffusion/weather_night_batch_worker.py'
JOBS_DIR = REPO / 'generation/weather/rain_snow/diffusion/jobs/weather_night'
LOGS_DIR = REPO / 'generation/weather/rain_snow/diffusion/logs/weather_night'
OUT_BASE = REPO / 'augmentation_data/construction_site/night_weather'


JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=wn_{weather}_{start}-{end}_{ts}
#SBATCH --partition=nextgen
#SBATCH --account=pgs0407
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --output={logs_dir}/wn_{weather}_{start}-{end}_{ts}_%j.out
#SBATCH --error={logs_dir}/wn_{weather}_{start}-{end}_{ts}_%j.err

module load cuda/12.8.1
module load miniconda3/24.1.2-py310
eval "$(conda shell.bash hook)"
conda activate /users/PGS0407/binben14/.conda/envs/VLM

cd {repo}
export PYTHONUNBUFFERED=1

echo "Weather-night Order B {weather} batch {start}-{end} at $(date)"
nvidia-smi

python {worker} \\
    --orig-input {orig} \\
    --output-dir {output_dir} \\
    --start {start} --end {end} \\
    --weather {weather} --seed 42 \\
    {filter_flag}

echo "Done at $(date)"
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=500)
    p.add_argument('--weather', choices=['rain', 'snow', 'all'], default='all')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--filter', action='store_true',
                   help='Enable SSIM/LPIPS filter (default: --no-filter, keep all). '
                        'Order B SSIM-vs-original lies far below Order A thresholds; '
                        'filter post-hoc instead.')
    return p.parse_args()


def count_arrow(path):
    with open(path, 'rb') as f:
        return len(pa.ipc.open_stream(f).read_all())


def main():
    args = parse_args()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    total = count_arrow(ORIG_ARROW)
    ts = datetime.now().strftime('%m%d_%H%M')
    weathers = ['rain', 'snow'] if args.weather == 'all' else [args.weather]

    print(f'Total images: {total}')
    print(f'Batch size: {args.batch_size}')
    print(f'Weathers: {weathers}')
    print(f'Output base: {OUT_BASE}')
    print(f'Timestamp: {ts}')
    print('-' * 60)

    submitted = []
    for w in weathers:
        out_dir = OUT_BASE / f'{w}_night'
        out_dir.mkdir(parents=True, exist_ok=True)

        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            sh = JOB_TEMPLATE.format(
                weather=w, start=start, end=end, ts=ts,
                logs_dir=LOGS_DIR, repo=REPO, worker=WORKER,
                orig=ORIG_ARROW, output_dir=out_dir,
                filter_flag='' if args.filter else '--no-filter',
            )
            sh_path = JOBS_DIR / f'wn_{w}_{start}-{end}_{ts}.sh'
            sh_path.write_text(sh)
            sh_path.chmod(0o755)

            if args.dry_run:
                print(f'  [DRY] {sh_path.name}')
                continue
            r = subprocess.run(['sbatch', str(sh_path)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f'  FAIL {sh_path.name}: {r.stderr.strip()}')
            else:
                jid = r.stdout.strip().split()[-1]
                print(f'  {jid}  {sh_path.name}')
                submitted.append(jid)

    print('-' * 60)
    print(f'Total submitted: {len(submitted)}')
    if submitted:
        manifest = JOBS_DIR / f'jobs_{ts}.txt'
        manifest.write_text('\n'.join(submitted) + '\n')
        print(f'Manifest: {manifest}')


if __name__ == '__main__':
    main()
