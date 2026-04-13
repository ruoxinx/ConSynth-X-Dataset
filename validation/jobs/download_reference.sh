#!/bin/bash
#SBATCH --job-name=dl_ref
#SBATCH --account=pgs0407
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --output=validation/logs/download_ref_%j.out
#SBATCH --error=validation/logs/download_ref_%j.err

# Download weather reference images from:
#   1. WeatherNet-05 (fast, ~2GB, 18K images, 5 classes)
#   2. BDD100K streaming (for night class, ~1K per condition)

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"

module load miniconda3/24.1.2-py310
conda activate VLM

cd /users/PGS0407/binben14/VietHuy/ConSynth-X
mkdir -p validation/logs

SOURCE="${SOURCE:-both}"

python validation/download_reference.py \
    --source "${SOURCE}" \
    --resize 640 \
    --max-per-condition 1000

echo ""
echo "End: $(date)"
