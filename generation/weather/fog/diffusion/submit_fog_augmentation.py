#!/usr/bin/env python3
"""
Submit batch fog augmentation jobs to SLURM.

Splits 3004 images into 3 intensity zones (~1001 each),
each zone into batches of ~300 images → ~12 jobs total.
Output: Arrow files preserving all annotations + fog metadata.

Usage:
    python submit_fog_augmentation.py                  # Submit all
    python submit_fog_augmentation.py --dry-run         # Preview without submitting
    python submit_fog_augmentation.py --label heavy     # Only heavy jobs
"""

import argparse
import subprocess
import math
from pathlib import Path
import pyarrow.ipc as ipc


def count_arrow_samples(path):
    with open(path, 'rb') as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    return len(table)


def main():
    parser = argparse.ArgumentParser(description='Submit fog augmentation batch jobs')
    parser.add_argument('--batch-size', type=int, default=300, help='Samples per batch')
    parser.add_argument('--label', default=None, choices=['heavy', 'medium', 'light'],
                        help='Only submit jobs for this label')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--partition', default='nextgen')
    parser.add_argument('--time', default='02:00:00')
    parser.add_argument('--mem', default='32G')
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.resolve()
    REPO_ROOT = PROJECT_ROOT.parents[2]
    ARROW_FILE = REPO_ROOT / 'LouisChen15___construction_site' / 'construction_site-test.arrow'
    OUTPUT_DIR = REPO_ROOT / 'augmentation_data' / 'construction_site' / 'fog'

    total = count_arrow_samples(str(ARROW_FILE))
    print(f'Total samples: {total}')

    # Split into 3 equal zones
    zone_size = total // 3
    labels = ['heavy', 'medium', 'light']
    zones = []
    for i, label in enumerate(labels):
        z_start = i * zone_size
        z_end = (i + 1) * zone_size if i < 2 else total
        zones.append((label, z_start, z_end))

    # Filter by label if specified
    if args.label:
        zones = [(l, s, e) for l, s, e in zones if l == args.label]

    # Create batch jobs
    jobs_dir = PROJECT_ROOT / 'jobs'
    logs_dir = PROJECT_ROOT / 'logs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nZone assignment:')
    for label, z_start, z_end in zones:
        n_batches = math.ceil((z_end - z_start) / args.batch_size)
        print(f'  {label:8s}: images {z_start}-{z_end-1} ({z_end-z_start} images, {n_batches} batches)')

    print()

    job_count = 0
    for label, z_start, z_end in zones:
        n_images = z_end - z_start
        n_batches = math.ceil(n_images / args.batch_size)

        for batch_idx in range(n_batches):
            b_start = z_start + batch_idx * args.batch_size
            b_end = min(z_start + (batch_idx + 1) * args.batch_size, z_end)

            out_file = OUTPUT_DIR / label / f'fog_{label}_{b_start}-{b_end}.arrow'
            job_name = f'fog_{label}_{b_start}'

            script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={args.partition}
#SBATCH --account=pgs0407
#SBATCH --time={args.time}
#SBATCH --gpus-per-node=1
#SBATCH --mem={args.mem}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

source ~/.bashrc
conda activate VLM

cd {PROJECT_ROOT}

python3 arrow_fog_worker.py \\
  --input {ARROW_FILE} \\
  --output {out_file} \\
  --start {b_start} \\
  --end {b_end} \\
  --label {label} \\
  --depth-model depth-anything-small \\
  --seed {42 + b_start}

echo "Job completed at $(date)"
"""
            script_path = jobs_dir / f'{job_name}.sh'
            with open(script_path, 'w') as f:
                f.write(script)

            if args.dry_run:
                print(f'  [DRY-RUN] {job_name}: {b_start}-{b_end} ({b_end-b_start} imgs) → {out_file.name}')
            else:
                result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    print(f'  {job_name}: {b_start}-{b_end} ({b_end-b_start} imgs) → {job_id}')
                else:
                    print(f'  ERROR {job_name}: {result.stderr.strip()}')

            job_count += 1

    print(f'\n{"[DRY-RUN] " if args.dry_run else ""}Submitted {job_count} jobs')
    print(f'Output dir: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
