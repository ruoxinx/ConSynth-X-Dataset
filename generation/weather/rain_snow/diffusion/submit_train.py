#!/usr/bin/env python3
"""
Submit SLURM jobs for diffusion augmentation (IP2P + physics + filter) — TRAIN set (multi-shard).

Usage:
    python submit_train.py                    # submit all
    python submit_train.py --weather rain      # rain only
    python submit_train.py --dry-run           # preview
"""

import argparse
import subprocess
import math
from pathlib import Path
from datetime import datetime
import pyarrow as pa


def count_arrow(path):
    with open(path, 'rb') as f:
        return len(pa.ipc.open_stream(f).read_all())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str,
                        default='LouisChen15___construction_site')
    parser.add_argument('--output-dir', type=str,
                        default='augmentation_data/construction_site/rain_snow_v4_train')
    parser.add_argument('--weather', type=str, default='all', choices=['rain', 'snow', 'all'])
    parser.add_argument('--batch-size', type=int, default=500)
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
    input_dir = work_dir / args.input_dir

    # Find all train shards
    train_shards = sorted(input_dir.glob('construction_site-train-*.arrow'))
    if not train_shards:
        print(f"ERROR: No train shards found in {input_dir}")
        return

    shard_info = []
    total_samples = 0
    for shard in train_shards:
        n = count_arrow(shard)
        shard_info.append((shard, n))
        total_samples += n

    weather_types = ['rain', 'snow'] if args.weather == 'all' else [args.weather]

    jobs_dir = work_dir / 'generation' / 'weather' / 'diffusion' / 'jobs' / 'train'
    logs_dir = work_dir / 'generation' / 'weather' / 'diffusion' / 'logs' / 'train'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    worker_script = 'generation/weather/diffusion/batch_worker.py'

    print("=" * 60)
    print("V4 BATCH SUBMISSION — TRAIN SET")
    print("=" * 60)
    print(f"Shards:     {len(train_shards)}")
    for shard, n in shard_info:
        print(f"  {shard.name}: {n} samples")
    print(f"Total:      {total_samples} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Weather:    {weather_types}")
    print(f"Output:     {args.output_dir}/{{rain,snow}}/")
    print("=" * 60)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    all_jobs = []

    for weather in weather_types:
        output_dir = work_dir / args.output_dir / weather
        print(f"\n--- {weather.upper()} ---")

        for shard_idx, (shard_path, shard_len) in enumerate(shard_info):
            num_batches = math.ceil(shard_len / args.batch_size)
            shard_tag = f"s{shard_idx}"

            for batch_idx in range(num_batches):
                b_start = batch_idx * args.batch_size
                b_end = min(b_start + args.batch_size, shard_len)
                job_name = f"v4t_{weather}_{shard_tag}_{b_start}-{b_end}_{timestamp}"

                cmd = (f"python {worker_script} "
                       f"--input {shard_path} "
                       f"--output-dir {output_dir}/{shard_tag} "
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
conda activate /users/PGS0407/binben14/.conda/envs/VLM

cd {work_dir}
echo "Starting {weather} shard {shard_idx} batch {b_start}-{b_end} at $(date)"
nvidia-smi
{cmd}
echo "Done at $(date)"
"""
                script_path = jobs_dir / f'{job_name}.sh'
                with open(script_path, 'w') as f:
                    f.write(script)

                if args.dry_run:
                    print(f"  [DRY-RUN] shard {shard_idx} {b_start}-{b_end}")
                    all_jobs.append({'job_id': 'DRY', 'weather': weather,
                                     'shard': shard_idx, 'batch': f'{b_start}-{b_end}'})
                else:
                    result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
                    if result.returncode == 0:
                        job_id = result.stdout.strip().split()[-1]
                        print(f"  Shard {shard_idx} batch {b_start}-{b_end}: Job {job_id}")
                        all_jobs.append({'job_id': job_id, 'weather': weather,
                                         'shard': shard_idx, 'batch': f'{b_start}-{b_end}'})
                    else:
                        print(f"  FAILED shard {shard_idx} {b_start}-{b_end}: {result.stderr}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total jobs: {len(all_jobs)}")
    for weather in weather_types:
        wjobs = [j for j in all_jobs if j['weather'] == weather]
        print(f"  {weather}: {len(wjobs)} jobs → {args.output_dir}/{weather}/")

    # Save job list
    job_list = jobs_dir / f'jobs_{timestamp}.txt'
    with open(job_list, 'w') as f:
        f.write(f"# V4 train batch jobs - {timestamp}\n")
        for j in all_jobs:
            f.write(f"{j['job_id']}\t{j['weather']}\tshard{j['shard']}\t{j['batch']}\n")
    print(f"Job list: {job_list}")

    if not args.dry_run:
        ids = [j['job_id'] for j in all_jobs if j['job_id'] != 'DRY']
        if ids:
            print(f"\nMonitor: squeue -u $USER | grep v4t_")


if __name__ == '__main__':
    main()
