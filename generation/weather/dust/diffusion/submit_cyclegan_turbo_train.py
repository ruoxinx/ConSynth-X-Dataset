#!/usr/bin/env python3
"""
Submit CycleGAN-Turbo training job for clear-to-dust translation.

Data: 600 clean SODA images (train_A) + 600 WEAPD sandstorm images (train_B)
Model: stabilityai/sd-turbo with LoRA adapters
Training: ~25,000 steps, ~4-8 hours on A100
"""

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
IMG2IMG_TURBO = PROJECT_ROOT.parents[2] / 'day2night' / 'img2img-turbo'
DATA_DIR = PROJECT_ROOT / 'data' / 'clear2dust'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'cyclegan_turbo_clear2dust'
LOGS_DIR = PROJECT_ROOT / 'logs'
JOBS_DIR = PROJECT_ROOT / 'jobs'
CONDA_ENV = '/users/PGS0407/binben14/.conda/envs/VLM'

# Training hyperparameters
MAX_TRAIN_STEPS = 25000
BATCH_SIZE = 1
LEARNING_RATE = "1e-5"
VALIDATION_STEPS = 500
CHECKPOINTING_STEPS = 2500
LAMBDA_GAN = 0.5
LAMBDA_CYCLE = 1.0
LAMBDA_IDT = 1.0

LOGS_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

job_script = f"""#!/bin/bash
#SBATCH --job-name=clear2dust_train
#SBATCH --output={LOGS_DIR}/clear2dust_train_%j.out
#SBATCH --error={LOGS_DIR}/clear2dust_train_%j.err
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --account=pgs0407

echo "=== CycleGAN-Turbo: Clear-to-Dust Training ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"
echo ""

eval "$(/users/PGS0407/binben14/miniconda3/bin/conda shell.bash hook)"
conda activate {CONDA_ENV}

# Install missing dependencies
pip install wandb vision-aided-loss clean-fid lpips --quiet 2>/dev/null
# Fix awq/qwen3 conflict: uninstall awq (not needed for LoRA training)
pip uninstall autoawq autoawq-kernels -y --quiet 2>/dev/null
# Restore peft to required version
pip install peft==0.18.1 --quiet 2>/dev/null
export WANDB_MODE=offline

cd {IMG2IMG_TURBO}

export NCCL_P2P_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --main_process_port 29501 --mixed_precision fp16 src/train_cyclegan_turbo.py \\
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \\
    --output_dir="{OUTPUT_DIR}" \\
    --dataset_folder "{DATA_DIR}" \\
    --train_img_prep "resize_286_randomcrop_256x256_hflip" \\
    --val_img_prep "resize_256" \\
    --learning_rate="{LEARNING_RATE}" \\
    --max_train_steps={MAX_TRAIN_STEPS} \\
    --train_batch_size={BATCH_SIZE} \\
    --gradient_accumulation_steps=1 \\
    --tracker_project_name "clear2dust_cyclegan_turbo" \\
    --report_to "wandb" \\
    --validation_steps {VALIDATION_STEPS} \\
    --checkpointing_steps {CHECKPOINTING_STEPS} \\
    --lambda_gan {LAMBDA_GAN} \\
    --lambda_idt {LAMBDA_IDT} \\
    --lambda_cycle {LAMBDA_CYCLE} \\
    --lora_rank_unet 128 \\
    --lora_rank_vae 4 \\
    --gradient_checkpointing \\
    --allow_tf32 \\
    --seed 42

echo ""
echo "End: $(date)"
echo "=== Training Complete ==="
echo "Checkpoints: {OUTPUT_DIR}/checkpoints/"
ls -la {OUTPUT_DIR}/checkpoints/ 2>/dev/null
"""

job_path = JOBS_DIR / 'train_clear2dust.sh'
with open(job_path, 'w') as f:
    f.write(job_script)
os.chmod(str(job_path), 0o755)

print(f'Job script: {job_path}')
print(f'Data dir:   {DATA_DIR}')
print(f'  train_A:  {len(list(DATA_DIR.glob("train_A/*")))} images (clean)')
print(f'  train_B:  {len(list(DATA_DIR.glob("train_B/*")))} images (dusty)')
print(f'  test_A:   {len(list(DATA_DIR.glob("test_A/*")))} images')
print(f'  test_B:   {len(list(DATA_DIR.glob("test_B/*")))} images')
print(f'Output:     {OUTPUT_DIR}')
print(f'Steps:      {MAX_TRAIN_STEPS}')
print()

result = subprocess.run(['sbatch', str(job_path)], capture_output=True, text=True)
if result.returncode == 0:
    print(f'Submitted: {result.stdout.strip()}')
else:
    print(f'Submit failed: {result.stderr}')
    print(f'Run manually: sbatch {job_path}')
