#!/usr/bin/env python3
"""
Submit 3 YOLO training jobs for SODA dataset in parallel on A100.

Configs:
  1. original
  2. original_weather_night
  3. original_weather_night_small
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

WORKSPACE = Path("/users/PGS0407/binben14/VietHuy/ConstructionSite")
SCRIPT = Path(__file__).resolve().parent / "train_yolo_soda.py"
OUTPUT_DIR = WORKSPACE / "validation_data/downstream_detection/SODA"
VLM_BIN = "/users/PGS0407/binben14/.conda/envs/VLM/bin"

CONFIGS = [
    "original",
    "original_weather_night",
    "original_weather_night_small",
]


def create_job_script(config, epochs, batch_size, imgsz, model_size, time_limit, memory):
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    job_name = f"soda_yolo_{config}_{timestamp}"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=PGS0407
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={memory}
#SBATCH --gpus-per-node=1
#SBATCH --partition=batch
#SBATCH --constraint=a100
#SBATCH --gpu_cmode=shared
#SBATCH --output={WORKSPACE}/logs/{job_name}_%j.out
#SBATCH --error={WORKSPACE}/logs/{job_name}_%j.err

echo "=========================================="
echo "SODA YOLO Training: {config}"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi

export PATH={VLM_BIN}:$PATH
cd {WORKSPACE}

python {SCRIPT} \\
    --config {config} \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --imgsz {imgsz} \\
    --model-size {model_size} \\
    --output-dir {OUTPUT_DIR}

echo ""
echo "=========================================="
echo "Job complete: $(date)"
echo "=========================================="
"""
    return job_name, script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model-size", type=str, default="s")
    parser.add_argument("--time", default="08:00:00")
    parser.add_argument("--memory", default="80G")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--configs", nargs="+", default=CONFIGS,
                        help="Which configs to submit")
    args = parser.parse_args()

    (WORKSPACE / "logs").mkdir(exist_ok=True)
    (WORKSPACE / "jobs").mkdir(exist_ok=True)

    submitted = []
    for config in args.configs:
        job_name, script = create_job_script(
            config, args.epochs, args.batch_size, args.imgsz,
            args.model_size, args.time, args.memory,
        )
        script_path = WORKSPACE / "jobs" / f"{job_name}.sh"
        script_path.write_text(script)

        if args.dry_run:
            print(f"[DRY RUN] {config}: {script_path}")
            submitted.append(("dry", config))
        else:
            r = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            if r.returncode == 0:
                jid = r.stdout.strip().split()[-1]
                print(f"Submitted {config}: Job ID {jid}")
                submitted.append((jid, config))
            else:
                print(f"ERROR {config}: {r.stderr}")

    print(f"\n{'='*60}")
    print(f"Submitted {len(submitted)} jobs")
    for jid, cfg in submitted:
        print(f"  {jid:>10s}  {cfg}")
    print(f"\nMonitor: squeue -u $USER")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
