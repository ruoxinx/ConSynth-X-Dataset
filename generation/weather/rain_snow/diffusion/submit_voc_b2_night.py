#!/usr/bin/env python3
"""
Submit SLURM jobs for B2 voc night-from-weather pipeline.

Generates one job per weather covering [0, N) of the pre-sliced 1000-row inputs.
Inputs must already exist at augmentation_data/soda_voc/_b2_inputs/.
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')
INPUTS = REPO / 'augmentation_data/soda_voc/_b2_inputs'
WORKER = REPO / 'generation/weather/rain_snow/diffusion/voc_b2_night_worker.py'
JOBS_DIR = REPO / 'generation/weather/rain_snow/diffusion/jobs/voc_b2_night'
LOGS_DIR = REPO / 'generation/weather/rain_snow/diffusion/logs/voc_b2_night'
OUT_BASE = REPO / 'augmentation_data/soda_voc/_b2_outputs'

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=vocb2_{weather}_{start}-{end}_{ts}
#SBATCH --partition=nextgen
#SBATCH --account=pgs0407
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --output={logs_dir}/vocb2_{weather}_{start}-{end}_{ts}_%j.out
#SBATCH --error={logs_dir}/vocb2_{weather}_{start}-{end}_{ts}_%j.err

module load cuda/12.8.1
module load miniconda3/24.1.2-py310
eval "$(conda shell.bash hook)"
conda activate /users/PGS0407/binben14/.conda/envs/VLM

cd {repo}
export PYTHONUNBUFFERED=1

echo "VOC B2 night-from-weather {weather} batch {start}-{end} at $(date)"
nvidia-smi

python {worker} \\
    --weather-input {weather_input} \\
    --orig-input {orig_input} \\
    --output-dir {output_dir} \\
    --start {start} --end {end} \\
    --weather {weather} --seed 42

echo "Done at $(date)"
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weather', choices=['rain', 'snow', 'all'], default='all')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=1000)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    weathers = ['rain', 'snow'] if args.weather == 'all' else [args.weather]
    ts = datetime.now().strftime('%m%d_%H%M')

    submitted = []
    for w in weathers:
        weather_input = INPUTS / f'{w}_1000.arrow'
        orig_input = INPUTS / 'clear_1000.arrow'
        output_dir = OUT_BASE / w

        if not weather_input.exists():
            raise FileNotFoundError(f'Missing input: {weather_input}')
        if not orig_input.exists():
            raise FileNotFoundError(f'Missing input: {orig_input}')

        job_text = JOB_TEMPLATE.format(
            weather=w, start=args.start, end=args.end, ts=ts,
            logs_dir=LOGS_DIR, repo=REPO, worker=WORKER,
            weather_input=weather_input, orig_input=orig_input,
            output_dir=output_dir,
        )
        job_path = JOBS_DIR / f'vocb2_{w}_{args.start}-{args.end}_{ts}.sh'
        with open(job_path, 'w') as f:
            f.write(job_text)
        job_path.chmod(0o755)

        if args.dry_run:
            print(f'[DRY] would submit: {job_path}')
        else:
            r = subprocess.run(['sbatch', str(job_path)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f'FAILED to submit {job_path}: {r.stderr}')
            else:
                jobid = r.stdout.strip().split()[-1]
                submitted.append((w, jobid, job_path))
                print(f'Submitted {w} job {jobid}: {job_path}')

    if submitted:
        print(f'\nSubmitted {len(submitted)} jobs:')
        for w, jid, p in submitted:
            print(f'  {w}: {jid}  ({p.name})')


if __name__ == '__main__':
    main()
