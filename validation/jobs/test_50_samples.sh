#!/bin/bash
#SBATCH --job-name=realism_test50
#SBATCH --account=pgs0407
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=validation/logs/realism_test50_%j.out
#SBATCH --error=validation/logs/realism_test50_%j.err

# Quick test: 50 samples per condition, CLIP zero-shot mode

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

module load cuda/12.8.1
module load miniconda3/24.1.2-py310

conda activate VLM

cd /users/PGS0407/binben14/VietHuy/ConSynth-X

# Check/install clip
python -c "import clip" 2>/dev/null || {
    echo "Installing CLIP..."
    pip install git+https://github.com/openai/CLIP.git --quiet
}

mkdir -p validation/logs validation/results

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run on a representative subset of conditions
python validation/run_realism_validation.py \
    --mode clip_zeroshot \
    --max-samples 50 \
    --batch-size 32 \
    --conditions original weather_style_rain_0 weather_style_snow_0 night small \
    --output-dir validation/results/test50

echo ""
echo "End: $(date)"
echo "Results: validation/results/test50/"
