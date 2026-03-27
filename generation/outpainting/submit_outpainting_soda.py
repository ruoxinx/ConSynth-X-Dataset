#!/usr/bin/env python3
"""
Submit FLUX Outpainting Pipeline for SODA Dataset (VOC Format)

Reads images from VOC JPEGImages/ and Annotations/, submits SLURM jobs
on A100 GPUs for high-quality outpainting.

Usage:
  # Small test: 5 jobs x 100 images = 500 samples
  python outpainting/submit_outpainting_soda.py --num-samples 500 --batch-size 100 --max-jobs 5

  # Full dataset
  python outpainting/submit_outpainting_soda.py --batch-size 200 --max-jobs 50
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json


def load_image_list(image_list_file: str) -> list:
    """Load image IDs from text file."""
    with open(image_list_file, "r", encoding="latin-1") as f:
        return [line.strip() for line in f if line.strip()]


def create_job_script(
    job_name: str,
    images_dir: str,
    annotations_dir: str,
    image_list_file: str,
    output_dir: str,
    start_idx: int,
    end_idx: int,
    resize_input: float,
    scale_mean: float,
    scale_min: float,
    scale_max: float,
    num_steps: int,
    guidance_scale: float,
    max_size: int,
    prompt: str,
    seed: int,
    workspace: Path,
    time_limit: str = "06:00:00",
    account: str = "PGS0407",
    memory: str = "80G",
    cpus: int = 8,
) -> str:
    """Create SLURM job script for A100 GPU."""

    escaped_prompt = prompt.replace('"', '\\"')

    script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --gpus-per-node=1
#SBATCH --partition=batch
#SBATCH --constraint=a100
#SBATCH --gpu_cmode=shared
#SBATCH --output={workspace}/logs/{job_name}_%j.out
#SBATCH --error={workspace}/logs/{job_name}_%j.err

echo "=========================================="
echo "FLUX Outpainting - SODA Dataset"
echo "Job: {job_name}"
echo "Samples: {start_idx} to {end_idx}"
echo "Output: {output_dir}"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi

export PATH=/users/PGS0407/binben14/.conda/envs/VLM/bin:$PATH
cd {workspace}

python outpainting/flux_pipeline_worker_voc.py \\
    --images-dir "{images_dir}" \\
    --annotations-dir "{annotations_dir}" \\
    --image-list "{image_list_file}" \\
    --output "{output_dir}" \\
    --start {start_idx} \\
    --end {end_idx} \\
    --resize-input {resize_input} \\
    --scale-mean {scale_mean} \\
    --scale-min {scale_min} \\
    --scale-max {scale_max} \\
    --num-steps {num_steps} \\
    --guidance-scale {guidance_scale} \\
    --max-size {max_size} \\
    --seed {seed} \\
    --prompt "{escaped_prompt}"

echo ""
echo "=========================================="
echo "Job complete: $(date)"
echo "=========================================="
'''
    return script


def main():
    parser = argparse.ArgumentParser(description="Submit FLUX Outpainting for SODA (VOC)")

    # Paths
    parser.add_argument("--voc-root",
                        default="/users/PGS0407/binben14/VietHuy/construction-site/SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007",
                        help="VOC2007 root directory")
    parser.add_argument("--image-list", default=None,
                        help="Image list file (default: ImageSets/Main/trainval.txt)")
    parser.add_argument("--output-dir",
                        default="/users/PGS0407/binben14/VietHuy/construction-site/augmentation_data/SODA/small",
                        help="Output directory")

    # Sample selection
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--num-samples", "-n", type=int, default=None)

    # Batch options
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-jobs", type=int, default=5)

    # Scale parameters
    parser.add_argument("--resize-input", type=float, default=0.7,
                        help="Resize input (0.7 for higher quality on A100)")
    parser.add_argument("--scale-mean", type=float, default=0.3)
    parser.add_argument("--scale-min", type=float, default=0.25)
    parser.add_argument("--scale-max", type=float, default=0.4)

    # Generation parameters (higher quality for A100)
    parser.add_argument("--num-steps", type=int, default=35,
                        help="Inference steps (35 for higher quality)")
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--max-size", type=int, default=1024,
                        help="Max canvas size for generation (1024 for A100)")
    parser.add_argument("--prompt", type=str,
                        default="Extend the image edges seamlessly. Continue only the existing ground texture, dirt, concrete, and sky. Match lighting, colors, and perspective. Do not add any new objects.")
    parser.add_argument("--seed", type=int, default=42)

    # SLURM options
    parser.add_argument("--account", default="PGS0407")
    parser.add_argument("--time", default="06:00:00")
    parser.add_argument("--memory", default="80G")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images that already have output in output-dir/JPEGImages/")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    # Setup paths
    workspace = Path(__file__).parent.parent.resolve()
    voc_root = Path(args.voc_root)
    images_dir = voc_root / "JPEGImages"
    annotations_dir = voc_root / "Annotations"

    # Image list file
    if args.image_list:
        image_list_file = Path(args.image_list)
    else:
        image_list_file = voc_root / "ImageSets" / "Main" / "trainval.txt"

    if not image_list_file.exists():
        print(f"ERROR: Image list not found: {image_list_file}")
        sys.exit(1)

    # Load and count images
    image_ids = load_image_list(str(image_list_file))
    total_images = len(image_ids)
    print(f"VOC root: {voc_root}")
    print(f"Image list: {image_list_file}")
    print(f"Total images in list: {total_images}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-filter: skip images that already have output
    if args.skip_existing:
        out_images_dir = output_dir / "JPEGImages"
        existing = set()
        if out_images_dir.exists():
            existing = {p.stem for p in out_images_dir.iterdir() if p.suffix == ".jpg"}
        original_count = len(image_ids)
        image_ids = [img_id for img_id in image_ids if img_id not in existing]
        print(f"Skip-existing: {original_count - len(image_ids)} already processed, {len(image_ids)} remaining")

        # Write filtered list to a temp file for workers
        filtered_list_file = output_dir / "filtered_image_list.txt"
        with open(filtered_list_file, "w") as f:
            for img_id in image_ids:
                f.write(img_id + "\n")
        image_list_file = filtered_list_file
        total_images = len(image_ids)

    # Determine range
    start_idx = args.start
    if args.num_samples is not None:
        end_idx = start_idx + args.num_samples
    elif args.end is not None:
        end_idx = args.end
    else:
        end_idx = total_images

    end_idx = min(end_idx, total_images)
    samples_to_process = end_idx - start_idx

    print(f"Processing: samples {start_idx} to {end_idx} ({samples_to_process} images)")
    print(f"Output: {output_dir}")

    # Create directories
    (workspace / "logs").mkdir(exist_ok=True)
    (workspace / "jobs").mkdir(exist_ok=True)

    # Calculate batches
    if samples_to_process <= args.batch_size:
        num_batches = 1
        batch_size = samples_to_process
    else:
        num_batches = min(args.max_jobs, (samples_to_process + args.batch_size - 1) // args.batch_size)
        batch_size = (samples_to_process + num_batches - 1) // num_batches

    print(f"Splitting into {num_batches} job(s) of ~{batch_size} images")
    print(f"GPU: A100 (partition=batch, constraint=a100)")
    print(f"Quality: num_steps={args.num_steps}, max_size={args.max_size}, resize_input={args.resize_input}")

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    submitted_jobs = []

    for batch_idx in range(num_batches):
        batch_start = start_idx + batch_idx * batch_size
        batch_end = min(batch_start + batch_size, end_idx)

        if batch_start >= batch_end:
            break

        job_name = f"soda_out_{batch_idx:03d}_{timestamp}"

        script = create_job_script(
            job_name=job_name,
            images_dir=str(images_dir),
            annotations_dir=str(annotations_dir),
            image_list_file=str(image_list_file),
            output_dir=str(output_dir),
            start_idx=batch_start,
            end_idx=batch_end,
            resize_input=args.resize_input,
            scale_mean=args.scale_mean,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            max_size=args.max_size,
            prompt=args.prompt,
            seed=args.seed + batch_idx,
            workspace=workspace,
            time_limit=args.time,
            account=args.account,
            memory=args.memory,
            cpus=args.cpus,
        )

        script_path = workspace / "jobs" / f"{job_name}.sh"
        with open(script_path, "w") as f:
            f.write(script)

        print(f"\nBatch {batch_idx}: images {batch_start}-{batch_end}")

        if args.dry_run:
            print(f"  [DRY RUN] Would submit: {script_path}")
            submitted_jobs.append(f"dry_{batch_idx}")
        else:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"  Submitted: Job ID {job_id}")
                submitted_jobs.append(job_id)
            else:
                print(f"  ERROR: {result.stderr}")

    # Summary
    print(f"\n{'='*60}")
    print("Pipeline Summary")
    print(f"{'='*60}")
    print(f"  VOC root:       {voc_root}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Samples:        {start_idx} to {end_idx} ({samples_to_process} total)")
    print(f"  Batch jobs:     {len(submitted_jobs)}")
    print(f"  GPU:            A100 (batch partition)")
    print(f"  Scale:          Gaussian(mean={args.scale_mean}, range=[{args.scale_min}, {args.scale_max}])")
    print(f"  Resize input:   {args.resize_input}")
    print(f"  Num steps:      {args.num_steps}")
    print(f"  Max size:       {args.max_size}")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    JPEGImages/   - Outpainted images")
    print(f"    Annotations/  - VOC XML annotations (pixel coords transferred)")
    print(f"    meta/         - Metadata for debugging")
    print(f"\nMonitor: squeue -u $USER")
    print(f"{'='*60}")

    # Save pipeline info
    info = {
        "voc_root": str(voc_root),
        "image_list": str(image_list_file),
        "output_dir": str(output_dir),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "samples_to_process": samples_to_process,
        "num_batches": len(submitted_jobs),
        "submitted_jobs": submitted_jobs,
        "parameters": {
            "resize_input": args.resize_input,
            "scale_mean": args.scale_mean,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "max_size": args.max_size,
            "prompt": args.prompt,
            "seed": args.seed,
        },
        "gpu": "A100",
        "timestamp": timestamp,
        "annotation_format": "VOC XML (pixel coords)",
    }

    info_file = output_dir / "pipeline_info.json"
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nPipeline info: {info_file}")


if __name__ == "__main__":
    main()
