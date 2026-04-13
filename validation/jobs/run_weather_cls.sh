#!/bin/bash
#SBATCH --job-name=weather_cls
#SBATCH --account=pgs0407
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=validation/logs/weather_cls_%j.out
#SBATCH --error=validation/logs/weather_cls_%j.err

# Weather classification validation: does a weather classifier recognize
# our augmented weather conditions as the correct weather type?
#
# Usage:
#   sbatch validation/jobs/run_weather_cls.sh                    # full run
#   MAX_SAMPLES=100 sbatch validation/jobs/run_weather_cls.sh    # quick test

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

module load cuda/12.8.1
module load miniconda3/24.1.2-py310
conda activate VLM

cd /users/PGS0407/binben14/VietHuy/ConSynth-X
mkdir -p validation/logs validation/results/weather_cls

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

MAX_SAMPLES="${MAX_SAMPLES:-}"
CMD="python validation/weather_classifier.py --batch-size 64"

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
    echo "Max samples: ${MAX_SAMPLES}"
fi

echo "Command: ${CMD}"
echo ""

${CMD}

echo ""
echo "End: $(date)"
