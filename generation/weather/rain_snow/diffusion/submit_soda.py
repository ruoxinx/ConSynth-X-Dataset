#!/usr/bin/env python3
"""
Submit SLURM jobs for diffusion augmentation (IP2P + physics + filter) — SODA JPG folder.

Usage:
    python submit_soda.py                    # submit all
    python submit_soda.py --weather rain      # rain only
    python submit_soda.py --dry-run           # preview
"""

import argparse
import os
import subprocess
import math
from pathlib import Path
from datetime import datetime


def count_images(path):
    exts = {'.jpg', '.jpeg', '.png'}
    return len([f for f in Path(path).iterdir() if f.suffix.lower() in exts])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str,
                        default=os.environ.get('CONSYNTH_SODA_KTSH_IMG',
                                               str(Path(os.environ.get('CONSYNTH_DATA_ROOT', Path.home() / 'consynth_data')) / 'SODA' / 'data' / 'soda-ktsh' / 'images')))
    parser.add_argument('--output-dir', type=str,
                        default='augmentation_data/soda_voc/rain_snow/diffusion')
    parser.add_argument('--weather', type=str, default='all', choices=['rain', 'snow', 'all'])
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--partition', type=str, default='nextgen')
    parser.add_argument('--account', type=str, default='pgs0407')
    parser.add_argument('--time', type=str, default='06:00:00')
    parser.add_argument('--mem', type=str, default='32G')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = Path.cwd()

    total = count_images(args.input_dir)
    start = args.start
    end = args.end if args.end else total
    end = min(end, total)
    num_samples = end - start
    num_batches = math.ceil(num_samples / args.batch_size)

    weather_types = ['rain', 'snow'] if args.weather == 'all' else [args.weather]

    jobs_dir = work_dir / 'generation' / 'weather' / 'rain_snow' / 'diffusion' / 'jobs' / 'soda'
    logs_dir = work_dir / 'generation' / 'weather' / 'rain_snow' / 'diffusion' / 'logs' / 'soda'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    worker_script = 'generation/weather/rain_snow/diffusion/batch_worker_soda.py'

    print("=" * 60)
    print("DIFFUSION BATCH SUBMISSION — SODA")
    print("=" * 60)
    print(f"Input:      {args.input_dir} ({total} images)")
    print(f"Range:      [{start}, {end}) = {num_samples} images")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches:    {num_batches} per weather type")
    print(f"Weather:    {weather_types}")
    print(f"Output:     {args.output_dir}/{{rain,snow}}/")
    print("=" * 60)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    all_jobs = []

    for weather in weather_types:
        output_dir = work_dir / args.output_dir / weather
        print(f"\n--- {weather.upper()} ---")

        for batch_idx in range(num_batches):
            b_start = start + batch_idx * args.batch_size
            b_end = min(b_start + args.batch_size, end)
            job_name = f"dif_{weather}_soda_{b_start}-{b_end}_{timestamp}"

            cmd = (f"python {worker_script} "
                   f"--input-dir '{args.input_dir}' "
                   f"--output-dir '{output_dir}' "
                   f"--start {b_start} --end {b_end} "
                   f"--weather {weather} --seed {args.seed}")

            script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
#SBATCH --time={args.time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem={args.mem}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

module load cuda/11.8.0
module load miniconda3/24.1.2-py310
eval "$(conda shell.bash hook)"
conda activate "${CONSYNTH_CONDA_ENV:-VLM}"

cd {work_dir}
echo "Starting {weather} SODA batch {b_start}-{b_end} at $(date)"
nvidia-smi
{cmd}
echo "Done at $(date)"
"""
            script_path = jobs_dir / f'{job_name}.sh'
            with open(script_path, 'w') as f:
                f.write(script)

            if args.dry_run:
                print(f"  [DRY-RUN] {b_start}-{b_end}")
                all_jobs.append({'job_id': 'DRY', 'weather': weather, 'batch': f'{b_start}-{b_end}'})
            else:
                result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    print(f"  Batch {b_start}-{b_end}: Job {job_id}")
                    all_jobs.append({'job_id': job_id, 'weather': weather, 'batch': f'{b_start}-{b_end}'})
                else:
                    print(f"  FAILED {b_start}-{b_end}: {result.stderr}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total jobs: {len(all_jobs)}")
    for weather in weather_types:
        print(f"  {args.output_dir}/{weather}/")

    # Save job list
    job_list = jobs_dir / f'jobs_{timestamp}.txt'
    with open(job_list, 'w') as f:
        f.write(f"# Diffusion SODA batch jobs - {timestamp}\n")
        for j in all_jobs:
            f.write(f"{j['job_id']}\t{j['weather']}\t{j['batch']}\n")
    print(f"Job list: {job_list}")

    if not args.dry_run:
        ids = [j['job_id'] for j in all_jobs if j['job_id'] != 'DRY']
        if ids:
            print(f"\nMonitor: squeue -u $USER | grep dif_")


if __name__ == '__main__':
    main()
