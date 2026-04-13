#!/bin/bash
#SBATCH --job-name=cross_cond
#SBATCH --account=pgs0407
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PGS0407/binben14/VietHuy/ConSynth-X/experiments/logs/cross_cond_%j.out
#SBATCH --error=/users/PGS0407/binben14/VietHuy/ConSynth-X/experiments/logs/cross_cond_%j.err

module load cuda/12.8.1 miniconda3/24.1.2-py310
source activate VLM

export PYTHONUNBUFFERED=1

cd /users/PGS0407/binben14/VietHuy/ConSynth-X

echo "=== Cross-Condition Detection Evaluation ==="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Phase 1: Export test sets (~10 min)
python experiments/cross_condition_eval.py --phase export

# Phase 2: Train 2 new configs, 20 epochs each (~60 min total)
python experiments/cross_condition_eval.py --phase train --epochs 20

# Phase 3: Evaluate all 4 models on 6 test sets (~20 min)
python experiments/cross_condition_eval.py --phase eval

echo "Done: $(date)"
