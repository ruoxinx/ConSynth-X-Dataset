#!/bin/bash
#SBATCH --job-name=fid_kid
#SBATCH --account=pgs0407
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=validation/logs/fid_kid_%j.out
#SBATCH --error=validation/logs/fid_kid_%j.err

# Compute FID/KID: ConSynth-X augmented conditions vs ACDC real weather.
# Requires: ACDC downloaded first (run download_acdc.sh)
#
# Usage:
#   sbatch validation/jobs/run_fid_kid.sh          # full run
#   MAX_SAMPLES=200 sbatch validation/jobs/run_fid_kid.sh  # quick test

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

module load cuda/12.8.1
module load miniconda3/24.1.2-py310

conda activate VLM

cd /users/PGS0407/binben14/VietHuy/ConSynth-X

pip install scipy --quiet 2>/dev/null

mkdir -p validation/logs validation/results/fid_kid

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

MAX_SAMPLES="${MAX_SAMPLES:-}"

CMD="python validation/compute_fid_kid.py --batch-size 64"

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
    echo "Max samples per condition: ${MAX_SAMPLES}"
fi

echo "Command: ${CMD}"
echo ""

${CMD}

echo ""
echo "End: $(date)"
echo "Results: validation/results/fid_kid/"
