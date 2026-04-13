#!/bin/bash
#SBATCH --job-name=realism_full
#SBATCH --account=pgs0407
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=validation/logs/realism_full_%j.out
#SBATCH --error=validation/logs/realism_full_%j.err

# Full realism validation with UnivFD on ALL conditions (no sample limit)

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

# Full run: UnivFD mode, all conditions, no sample limit
python validation/run_realism_validation.py \
    --weights validation/weights/fc_weights.pth \
    --mode univfd \
    --batch-size 64 \
    --output-dir validation/results/full_univfd

echo ""
echo "End: $(date)"
echo "Results: validation/results/full_univfd/"
