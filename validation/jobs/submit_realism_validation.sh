#!/bin/bash
#SBATCH --job-name=realism_val
#SBATCH --account=pgs0407
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=validation/logs/realism_val_%j.out
#SBATCH --error=validation/logs/realism_val_%j.err

# Realism validation: Run AI-generated image detector on ConSynth-X data.
# Estimates: ~30 min for full dataset (CLIP inference on A100), ~2 min for 50 samples.
#
# Usage:
#   # Full run
#   sbatch validation/jobs/submit_realism_validation.sh
#
#   # Quick test (override via environment)
#   MAX_SAMPLES=50 sbatch validation/jobs/submit_realism_validation.sh

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

# ── Environment ─────────────────────────────────────────
module load cuda/12.8.1
module load miniconda3/24.1.2-py310

conda activate VLM

cd /users/PGS0407/binben14/VietHuy/ConSynth-X

# Create logs dir
mkdir -p validation/logs

# ── Configuration ───────────────────────────────────────
WEIGHTS="validation/weights/fc_weights.pth"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # empty = all images

# Build command
CMD="python validation/run_realism_validation.py"

if [ -f "${WEIGHTS}" ]; then
    CMD="${CMD} --weights ${WEIGHTS}"
    echo "Mode: UnivFD (pretrained weights)"
else
    CMD="${CMD} --mode clip_zeroshot"
    echo "Mode: CLIP zero-shot (no UnivFD weights found)"
    echo "  For better results, run: bash validation/setup_univfd.sh"
fi

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
    echo "Max samples: ${MAX_SAMPLES}"
else
    echo "Max samples: all"
fi

CMD="${CMD} --batch-size 64"

echo ""
echo "Command: ${CMD}"
echo ""

# ── Run ─────────────────────────────────────────────────
${CMD}

echo ""
echo "End: $(date)"
echo "Results: validation/results/"
