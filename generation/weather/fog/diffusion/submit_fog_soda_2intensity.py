#!/usr/bin/env python3
"""
Submit SODA VOC fog augmentation for 2 intensities, 1000 images per intensity.

Default: heavy + light (bracket the visibility range: 300-500m vs 750-1000m).
Each intensity processes a disjoint slice of the input Arrow so the two
outputs cover 2000 distinct images.

Usage:
    python submit_fog_soda_2intensity.py --input <arrow> --output-dir <dir>
    python submit_fog_soda_2intensity.py ... --intensities heavy medium
    python submit_fog_soda_2intensity.py ... --n-per-intensity 1000 --dry-run
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
    p.add_argument('--output-dir', required=True, help='Base output dir')
    p.add_argument('--intensities', nargs=2, default=['heavy', 'light'],
                   choices=['heavy', 'medium', 'light'],
                   help='Exactly 2 intensities to generate')
    p.add_argument('--n-per-intensity', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=300)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--partition', default='nextgen')
    p.add_argument('--time', default='02:00:00')
    p.add_argument('--mem', default='32G')
    args = p.parse_args()

    if args.intensities[0] == args.intensities[1]:
        raise SystemExit('Error: --intensities must be two distinct labels')

    PROJECT_ROOT = Path(__file__).parent.resolve()
    ARROW = Path(args.input).resolve()
    OUT = Path(args.output_dir).resolve()

    total = count_rows(str(ARROW))
    print(f'Total samples in input: {total}')

    n = args.n_per_intensity
    need = n * 2
    if need > total:
        raise SystemExit(f'Error: need {need} images but input has only {total}')

    # Disjoint slices: intensity[0] -> [0, n), intensity[1] -> [n, 2n)
    zones = [
        (args.intensities[0], 0, n),
        (args.intensities[1], n, 2 * n),
    ]

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
                print(f'  [DRY] {job_name}: {bs}-{be} -> {out_file.name}')
            else:
                r = subprocess.run(['sbatch', str(sp)], capture_output=True, text=True)
                if r.returncode == 0:
                    print(f'  {job_name}: {bs}-{be} -> {r.stdout.strip().split()[-1]}')
                else:
                    print(f'  ERROR {job_name}: {r.stderr.strip()}')
            job_count += 1

    print(f'\n{"[DRY-RUN] " if args.dry_run else ""}Total jobs: {job_count}')


if __name__ == '__main__':
    main()
