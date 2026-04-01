#!/usr/bin/env python3
"""
Submit SLURM job to generate fog samples with different depth models.
Usage:
    python submit_fog_samples.py                        # Depth Anything V2 medium only
    python submit_fog_samples.py --depth-model midas    # MiDaS only
    python submit_fog_samples.py --compare              # Both models side by side
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
REPO_ROOT = PROJECT_ROOT.parents[2]

ARROW_FILE = REPO_ROOT / 'LouisChen15___construction_site' / 'construction_site-test.arrow'

SLURM_CONFIG = {
    'partition': 'nextgen',
    'account': 'pgs0407',
    'time': '00:30:00',
    'gpu': '1',
    'mem': '32G',
}


def build_fog_cmd(depth_model, intensity, out_dir):
    return (
        f'python3 {PROJECT_ROOT}/fog_pipeline.py'
        f' --arrow {ARROW_FILE}'
        f' --num-samples 5'
        f' --intensity {intensity}'
        f' --depth-model {depth_model}'
        f' --output {out_dir}'
        f' --seed 42'
    )


def submit(job_name, commands):
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = PROJECT_ROOT / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)

    all_cmds = ' && '.join(commands)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --account={SLURM_CONFIG['account']}
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --gpus-per-node={SLURM_CONFIG['gpu']}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err

source ~/.bashrc
conda activate VLM

cd {PROJECT_ROOT}

echo "Device: $(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"

{all_cmds}

echo "Job completed at $(date)"
"""

    script_path = jobs_dir / f'{job_name}.sh'
    with open(script_path, 'w') as f:
        f.write(script)

    print(f'SLURM script: {script_path}')
    result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f'Error: {result.stderr.strip()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-model', default='depth-anything-small',
                        choices=['midas', 'depth-anything-small', 'depth-anything-base',
                                 'depth-anything-large'])
    parser.add_argument('--intensity', default='medium',
                        choices=['haze', 'light', 'medium', 'heavy', 'extreme'])
    parser.add_argument('--all-intensities', action='store_true',
                        help='Run all 4 intensities')
    parser.add_argument('--compare', action='store_true',
                        help='Run both MiDaS and Depth Anything V2 for comparison')
    args = parser.parse_args()

    intensities = ['light', 'medium', 'heavy', 'extreme'] if args.all_intensities else [args.intensity]

    if args.compare:
        models = ['midas', args.depth_model]
    else:
        models = [args.depth_model]

    commands = []
    for model in models:
        for intensity in intensities:
            tag = model.replace('depth-anything-', 'da2_')
            out_dir = PROJECT_ROOT / 'output_samples' / f'{tag}_{intensity}'
            cmd = build_fog_cmd(model, intensity, out_dir)
            commands.append(f'echo "\\n===== {model} / {intensity} =====" && {cmd}')

    job_name = 'fog_' + '_'.join(models).replace('depth-anything-', 'da2_')
    submit(job_name, commands)


if __name__ == '__main__':
    main()
