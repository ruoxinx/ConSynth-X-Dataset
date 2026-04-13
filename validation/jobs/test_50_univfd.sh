#!/bin/bash
#SBATCH --job-name=real_t50u
#SBATCH --account=pgs0407
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=validation/logs/realism_t50u_%j.out
#SBATCH --error=validation/logs/realism_t50u_%j.err

# Quick test: 50 samples, UnivFD mode, ALL conditions incl. diffusion

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

module load cuda/12.8.1
module load miniconda3/24.1.2-py310
conda activate VLM

cd /users/PGS0407/binben14/VietHuy/ConSynth-X

python -c "import clip" 2>/dev/null || pip install git+https://github.com/openai/CLIP.git --quiet

mkdir -p validation/logs validation/results

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python validation/run_realism_validation.py \
    --weights validation/weights/fc_weights.pth \
    --mode univfd \
    --max-samples 50 \
    --batch-size 32 \
    --output-dir validation/results/test50_univfd

echo ""
echo "End: $(date)"
