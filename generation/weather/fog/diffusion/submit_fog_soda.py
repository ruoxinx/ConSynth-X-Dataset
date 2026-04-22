#!/usr/bin/env python3
"""
Submit SODA VOC fog augmentation jobs to SLURM.
Splits the input Arrow into 3 intensity zones (heavy/medium/light), each zone into batches.
"""

import argparse
import subprocess
import math
from pathlib import Path
import pyarrow.ipc as ipc


def count_rows(path):
    with open(path, 'rb') as f:
        return len(ipc.open_stream(f).read_all())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input Arrow file (original SODA VOC)')
    p.add_argument('--output-dir', required=True, help='Base output dir (e.g., augmentation_data/soda_voc/fog)')
    p.add_argument('--batch-size', type=int, default=300)
    p.add_argument('--label', default=None, choices=['heavy', 'medium', 'light'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--partition', default='nextgen')
    p.add_argument('--time', default='02:00:00')
    p.add_argument('--mem', default='32G')
    args = p.parse_args()

    PROJECT_ROOT = Path(__file__).parent.resolve()
    ARROW = Path(args.input).resolve()
    OUT = Path(args.output_dir).resolve()

    total = count_rows(str(ARROW))
    print(f'Total samples: {total}')

    zone = total // 3
    labels = ['heavy', 'medium', 'light']
    zones = [(lab, i * zone, (i + 1) * zone if i < 2 else total) for i, lab in enumerate(labels)]
    if args.label:
        zones = [z for z in zones if z[0] == args.label]

    jobs_dir = PROJECT_ROOT / 'jobs'
    logs_dir = PROJECT_ROOT / 'logs'
    jobs_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    print('Zones:')
    for lab, s, e in zones:
        nb = math.ceil((e - s) / args.batch_size)
        print(f'  {lab}: {s}-{e-1} ({e-s} imgs, {nb} batches)')

    job_count = 0
    for lab, zs, ze in zones:
        nb = math.ceil((ze - zs) / args.batch_size)
        for i in range(nb):
            bs = zs + i * args.batch_size
            be = min(zs + (i + 1) * args.batch_size, ze)
            out_file = OUT / lab / f'fog_{lab}_{bs}-{be}.arrow'
            job_name = f'fog_soda_{lab}_{bs}'

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

python3 arrow_fog_worker_soda.py \\
  --input {ARROW} \\
  --output {out_file} \\
  --start {bs} \\
  --end {be} \\
  --label {lab} \\
  --depth-model depth-anything-small \\
  --seed {42 + bs}

echo "Done at $(date)"
"""
            sp = jobs_dir / f'{job_name}.sh'
            with open(sp, 'w') as f:
                f.write(script)

            if args.dry_run:
                print(f'  [DRY] {job_name}: {bs}-{be} → {out_file.name}')
            else:
                r = subprocess.run(['sbatch', str(sp)], capture_output=True, text=True)
                if r.returncode == 0:
                    print(f'  {job_name}: {bs}-{be} → {r.stdout.strip().split()[-1]}')
                else:
                    print(f'  ERROR {job_name}: {r.stderr.strip()}')
            job_count += 1

    print(f'\n{"[DRY-RUN] " if args.dry_run else ""}Total jobs: {job_count}')


if __name__ == '__main__':
    main()
