#!/usr/bin/env python3
"""
Submit 6 SLURM jobs for SSIM/LPIPS sensitivity analysis.
Each job runs run_sensitivity.py with one threshold config.
"""

import os
from pathlib import Path

BASE = Path(os.environ.get('CONSYNTH_REPO_ROOT', Path(__file__).resolve().parents[2]))
SENS = BASE / 'generation' / 'sensitivity'
JOBS = SENS / 'jobs'
LOGS = SENS / 'logs'

CONFIGS = ['baseline', 'loose', 'moderate', 'current', 'strict', 'tight', 'no_lpips']
EPOCHS = 20  # Reduced for faster iteration; increase to 50 for paper

TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=sens_{cfg}
#SBATCH --output={logs}/sens_{cfg}_%j.out
#SBATCH --error={logs}/sens_{cfg}_%j.err
#SBATCH --partition=nextgen
#SBATCH --account=pgs0407
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4

module load cuda/12.8.1
module load miniconda3/24.1.2-py310
source activate VLM

echo "Config: {cfg} | Node: $HOSTNAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

python {script} --config {cfg} --epochs {epochs} --val-samples 1000

date
echo "Done: {cfg}"
"""

def main():
    JOBS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    script = SENS / 'run_sensitivity.py'

    for cfg in CONFIGS:
        sh = JOBS / f'sens_{cfg}.sh'
        sh.write_text(TEMPLATE.format(
            cfg=cfg,
            logs=LOGS,
            script=script,
            epochs=EPOCHS,
        ))
        ret = os.system(f'sbatch {sh}')
        status = 'submitted' if ret == 0 else 'FAILED'
        print(f'  [{status}] {cfg} → {sh}')

    print(f'\nLogs: {LOGS}/')
    print(f'Results: {SENS / "detection_results"}/')


if __name__ == '__main__':
    main()
