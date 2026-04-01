#!/usr/bin/env python3
"""
Submit a SLURM job to test InstructPix2Pix dust augmentation on A100.

Runs all 4 presets on a few test images for visual comparison.
"""

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
WEATHER_ROOT = PROJECT_ROOT.parents[1]
TEST_IMAGES_DIR = WEATHER_ROOT.parent / 'SODA' / 'data' / 'soda-ktsh' / 'images'
OUTPUT_DIR = PROJECT_ROOT / 'generated_pipeline' / 'instruct_pix2pix'

# Pick a few test images
TEST_IMAGES = ['Ktsh0000.jpg', 'Ktsh0001.jpg', 'Ktsh0002.jpg']

CONDA_ENV = '/users/PGS0407/binben14/.conda/envs/VLM'
SCRIPT_PATH = PROJECT_ROOT / 'instruct_pix2pix_dust.py'

PRESETS = ['light', 'medium', 'heavy', 'sandstorm']

# Build the shell commands
cmds = []
for img_name in TEST_IMAGES:
    input_path = TEST_IMAGES_DIR / img_name
    stem = Path(img_name).stem
    for preset in PRESETS:
        output_path = OUTPUT_DIR / f'{stem}-{preset}.jpg'
        cmds.append(
            f'echo "Processing {img_name} with preset={preset}..." && '
            f'python {SCRIPT_PATH} '
            f'--input {input_path} '
            f'--output {output_path} '
            f'--preset {preset} '
            f'--seed 42'
        )

job_script = f"""#!/bin/bash
#SBATCH --job-name=dust_pix2pix_test
#SBATCH --output={PROJECT_ROOT / 'logs' / 'dust_pix2pix_test_%j.out'}
#SBATCH --error={PROJECT_ROOT / 'logs' / 'dust_pix2pix_test_%j.err'}
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --account=pgs0407

echo "=== InstructPix2Pix Dust Test ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

eval "$(/users/PGS0407/binben14/miniconda3/bin/conda shell.bash hook)"
conda activate {CONDA_ENV}

mkdir -p {OUTPUT_DIR}

{chr(10).join(cmds)}

echo ""
echo "=== All done! ==="
echo "Results in: {OUTPUT_DIR}"
ls -la {OUTPUT_DIR}/
"""

# Write job script
jobs_dir = PROJECT_ROOT / 'jobs'
jobs_dir.mkdir(exist_ok=True)
logs_dir = PROJECT_ROOT / 'logs'
logs_dir.mkdir(exist_ok=True)

job_path = jobs_dir / 'test_instruct_pix2pix.sh'
with open(job_path, 'w') as f:
    f.write(job_script)
os.chmod(str(job_path), 0o755)

print(f'Job script: {job_path}')
print(f'Output dir: {OUTPUT_DIR}')
print(f'Test images: {len(TEST_IMAGES)} x {len(PRESETS)} presets = {len(TEST_IMAGES)*len(PRESETS)} outputs')
print()

# Submit
result = subprocess.run(['sbatch', str(job_path)], capture_output=True, text=True)
if result.returncode == 0:
    print(f'Submitted: {result.stdout.strip()}')
else:
    print(f'Submit failed: {result.stderr}')
    print(f'Run manually: sbatch {job_path}')
