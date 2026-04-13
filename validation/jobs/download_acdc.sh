#!/bin/bash
#SBATCH --job-name=dl_acdc
#SBATCH --account=pgs0407
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --output=validation/logs/download_acdc_%j.out
#SBATCH --error=validation/logs/download_acdc_%j.err

# Download ACDC dataset from HuggingFace and organize per weather condition.
# No GPU needed — CPU-only job.

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"

module load miniconda3/24.1.2-py310
conda activate VLM

cd /users/PGS0407/binben14/VietHuy/ConSynth-X

pip install datasets --quiet 2>/dev/null

python validation/download_acdc.py \
    --output-dir validation/reference_data/acdc \
    --resize 640

echo ""
echo "End: $(date)"
echo "Output: validation/reference_data/acdc/"
