#!/usr/bin/env python3
"""
Submit SLURM jobs for night weather augmentation (IP2P + physics) — TEST set.

Pipeline: Night images → IP2P (rain/snow) → Physics particles → SSIM/LPIPS filter → Arrow

Usage:
    python submit_night_weather_test.py                    # submit all (rain + snow)
    python submit_night_weather_test.py --weather rain     # rain only
    python submit_night_weather_test.py --dry-run          # preview
"""

import argparse
import os
import subprocess
import math
from pathlib import Path
from datetime import datetime
import pyarrow as pa


def count_arrow(path):
    with open(path, 'rb') as f:
        return len(pa.ipc.open_stream(f).read_all())


def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit night weather augmentation jobs (test set)')
    parser.add_argument('--night-input', type=str,
                        default='augmentation_data_arrow/night.arrow',
                        help='Path to night arrow (relative to work_dir)')
    parser.add_argument('--orig-input', type=str,
                        default='augmentation_data_arrow/construction_site_test.arrow',
                        help='Path to original arrow (relative to work_dir)')
    parser.add_argument('--output-dir', type=str,
                        default='augmentation_data/construction_site/night_weather',
                        help='Output directory (relative to work_dir)')
    parser.add_argument('--weather', type=str, default='all',
                        choices=['rain', 'snow', 'all'])
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

    # Resolve paths from environment
    work_dir = Path(os.environ.get("CONSYNTH_DATA_ROOT", Path.home() / "consynth_data"))
    consynth_dir = Path(os.environ.get("CONSYNTH_REPO_ROOT", Path(__file__).resolve().parents[4]))

    night_file = work_dir / args.night_input
    orig_file = work_dir / args.orig_input

    total = count_arrow(night_file)
    start = args.start
    end = args.end if args.end else total
    end = min(end, total)
    num_samples = end - start
    num_batches = math.ceil(num_samples / args.batch_size)

    weather_types = ['rain', 'snow'] if args.weather == 'all' else [args.weather]

    jobs_dir = consynth_dir / 'generation' / 'weather' / 'rain_snow' / 'diffusion' / 'jobs' / 'night_weather'
    logs_dir = consynth_dir / 'generation' / 'weather' / 'rain_snow' / 'diffusion' / 'logs' / 'night_weather'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    worker_script = consynth_dir / 'generation' / 'weather' / 'rain_snow' / 'diffusion' / 'night_weather_batch_worker.py'

    print("=" * 60)
    print("NIGHT WEATHER BATCH SUBMISSION")
    print("Pipeline: Night → IP2P (rain/snow) → Physics → Filter")
    print("=" * 60)
    print(f"Night input:  {night_file} ({total} samples)")
    print(f"Orig input:   {orig_file}")
    print(f"Range:        [{start}, {end}) = {num_samples} samples")
    print(f"Batch size:   {args.batch_size}")
    print(f"Batches:      {num_batches} per weather type")
    print(f"Weather:      {weather_types}")
    print(f"Output:       {args.output_dir}/{{rain_night,snow_night}}/")
    print("=" * 60)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    all_jobs = []

    for weather in weather_types:
        output_dir = work_dir / args.output_dir / f'{weather}_night'
        print(f"\n--- {weather.upper()} NIGHT ---")

        for batch_idx in range(num_batches):
            b_start = start + batch_idx * args.batch_size
            b_end = min(b_start + args.batch_size, end)
            job_name = f"nw_{weather}_{b_start}-{b_end}_{timestamp}"

            cmd = (f"python {worker_script} "
                   f"--night-input {night_file} "
                   f"--orig-input {orig_file} "
                   f"--output-dir {output_dir} "
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

echo "Starting {weather} night batch {b_start}-{b_end} at $(date)"
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
                result = subprocess.run(['sbatch', str(script_path)],
                                        capture_output=True, text=True)
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
        out = work_dir / args.output_dir / f'{weather}_night'
        print(f"  {out}/")

    # Save job list
    job_list = jobs_dir / f'jobs_{timestamp}.txt'
    with open(job_list, 'w') as f:
        f.write(f"# Night weather batch jobs - {timestamp}\n")
        for j in all_jobs:
            f.write(f"{j['job_id']}\t{j['weather']}\t{j['batch']}\n")
    print(f"Job list: {job_list}")

    if not args.dry_run:
        ids = [j['job_id'] for j in all_jobs if j['job_id'] != 'DRY']
        if ids:
            print(f"\nMonitor: squeue -u $USER | grep nw_")


if __name__ == '__main__':
    main()
