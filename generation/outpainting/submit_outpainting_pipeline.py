#!/usr/bin/env python3
"""
Submit FLUX Outpainting Pipeline Jobs

Usage:
  # Test with 10 samples
  python submit_outpainting_pipeline.py --arrow test.arrow --num-samples 10
  
  # Full batch processing
  python submit_outpainting_pipeline.py --arrow test.arrow --batch-size 100 --max-jobs 20
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json
import pyarrow as pa


def count_arrow_samples(arrow_file: str) -> int:
    """Count samples in arrow file."""
    with open(arrow_file, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        return reader.read_all().num_rows


def create_job_script(
    job_name: str,
    arrow_file: str,
    output_dir: str,
    start_idx: int,
    end_idx: int,
    resize_input: float,
    scale_mean: float,
    scale_min: float,
    scale_max: float,
    num_steps: int,
    guidance_scale: float,
    prompt: str,
    seed: int,
    workspace: Path,
    time_limit: str = "04:00:00",
    account: str = "PGS0407",
    memory: str = "64G",
    cpus: int = 8
) -> str:
    """Create SLURM job script."""
    
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
#SBATCH --partition=gpu-exp
#SBATCH --constraint=v100-32g
#SBATCH --gpu_cmode=shared
#SBATCH --output={workspace}/logs/{job_name}_%j.out
#SBATCH --error={workspace}/logs/{job_name}_%j.err

echo "=========================================="
echo "FLUX Outpainting Pipeline"
echo "Job: {job_name}"
echo "Samples: {start_idx} to {end_idx}"
echo "Output: {output_dir}"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi

source ~/.bashrc
conda activate VLM
cd {workspace}

python outpainting/flux_pipeline_worker.py \\
    --arrow "{arrow_file}" \\
    --output "{output_dir}" \\
    --start {start_idx} \\
    --end {end_idx} \\
    --resize-input {resize_input} \\
    --scale-mean {scale_mean} \\
    --scale-min {scale_min} \\
    --scale-max {scale_max} \\
    --num-steps {num_steps} \\
    --guidance-scale {guidance_scale} \\
    --seed {seed} \\
    --prompt "{escaped_prompt}"

echo ""
echo "=========================================="
echo "Job complete: $(date)"
echo "=========================================="
'''
    return script


def main():
    parser = argparse.ArgumentParser(description="Submit FLUX Outpainting Pipeline")
    
    # Input/Output
    parser.add_argument("--arrow", "-a", required=True, help="Input arrow file")
    parser.add_argument("--output-base", default="outpainting_last_output",
                        help="Base output directory (default: outpainting_last_output)")
    
    # Sample selection
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--num-samples", "-n", type=int, default=None,
                        help="Number of samples (for testing)")
    
    # Batch options
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Samples per job (default: 100)")
    parser.add_argument("--max-jobs", type=int, default=10,
                        help="Maximum parallel jobs (default: 10)")
    
    # Scale parameters
    parser.add_argument("--resize-input", type=float, default=0.5,
                        help="Resize input (default: 0.5)")
    parser.add_argument("--scale-mean", type=float, default=0.3,
                        help="Mean scale (default: 0.3)")
    parser.add_argument("--scale-min", type=float, default=0.25,
                        help="Min scale (default: 0.25)")
    parser.add_argument("--scale-max", type=float, default=0.4,
                        help="Max scale (default: 0.35)")
    
    # Generation parameters
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--prompt", type=str,
                        default="Extend the image edges seamlessly. Continue only the existing ground texture, dirt, concrete, and sky. Match lighting, colors, and perspective. Do not add any new objects.")
    parser.add_argument("--seed", type=int, default=42)
    
    # SLURM options
    parser.add_argument("--account", default="PGS0407")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--memory", default="64G", help="Memory per node (64G for FLUX)")
    parser.add_argument("--cpus", type=int, default=8, help="CPUs per task")
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    
    # Setup paths
    workspace = Path(__file__).parent.parent.resolve()
    
    arrow_file = Path(args.arrow)
    if not arrow_file.is_absolute():
        arrow_file = workspace / arrow_file
    
    # Create output directory: outpainting_last_output/{input_name}/
    input_name = arrow_file.stem  # e.g., "construction_site-test"
    output_dir = workspace / args.output_base / input_name
    
    # Create directories
    (workspace / "logs").mkdir(exist_ok=True)
    (workspace / "jobs").mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count samples
    total_samples = count_arrow_samples(str(arrow_file))
    print(f"Input: {arrow_file}")
    print(f"Total samples in arrow: {total_samples}")
    
    # Determine range
    start_idx = args.start
    if args.num_samples is not None:
        end_idx = start_idx + args.num_samples
    elif args.end is not None:
        end_idx = args.end
    else:
        end_idx = total_samples
    
    end_idx = min(end_idx, total_samples)
    samples_to_process = end_idx - start_idx
    
    print(f"Processing: samples {start_idx} to {end_idx} ({samples_to_process} samples)")
    print(f"Output: {output_dir}")
    
    # Calculate batches
    if samples_to_process <= args.batch_size:
        # Single job
        num_batches = 1
        batch_size = samples_to_process
    else:
        num_batches = min(args.max_jobs, (samples_to_process + args.batch_size - 1) // args.batch_size)
        batch_size = (samples_to_process + num_batches - 1) // num_batches
    
    print(f"Splitting into {num_batches} batch(es) of ~{batch_size} samples")
    
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    submitted_jobs = []
    
    # Submit batch jobs
    for batch_idx in range(num_batches):
        batch_start = start_idx + batch_idx * batch_size
        batch_end = min(batch_start + batch_size, end_idx)
        
        if batch_start >= batch_end:
            break
        
        job_name = f"flux_out_{batch_idx:03d}_{timestamp}"
        
        script = create_job_script(
            job_name=job_name,
            arrow_file=str(arrow_file),
            output_dir=str(output_dir),
            start_idx=batch_start,
            end_idx=batch_end,
            resize_input=args.resize_input,
            scale_mean=args.scale_mean,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            prompt=args.prompt,
            seed=args.seed + batch_idx,  # Different seed per batch
            workspace=workspace,
            time_limit=args.time,
            account=args.account,
            memory=args.memory,
            cpus=args.cpus
        )
        
        script_path = workspace / "jobs" / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        print(f"\nBatch {batch_idx}: samples {batch_start}-{batch_end}")
        
        if args.dry_run:
            print(f"  [DRY RUN] Would submit: {script_path}")
            submitted_jobs.append(f"dry_{batch_idx}")
        else:
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True
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
    print(f"  Input arrow:    {arrow_file}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Samples:        {start_idx} to {end_idx} ({samples_to_process} total)")
    print(f"  Batch jobs:     {len(submitted_jobs)}")
    print(f"  Scale:          Gaussian(mean={args.scale_mean}, range=[{args.scale_min}, {args.scale_max}])")
    print(f"  Resize input:   {args.resize_input}")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    images/       - Outpainted images")
    print(f"    annotations/  - Transferred annotations (xyxy)")
    print(f"    meta/         - Full metadata for debugging")
    print(f"\nMonitor: squeue -u $USER")
    print(f"{'='*60}")
    
    # Save pipeline info
    info = {
        "input_arrow": str(arrow_file),
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
            "prompt": args.prompt,
            "seed": args.seed
        },
        "timestamp": timestamp,
        "bbox_format": "xyxy"
    }
    
    info_file = output_dir / "pipeline_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nPipeline info: {info_file}")


if __name__ == "__main__":
    main()



# python outpainting/submit_outpainting_pipeline.py     --arrow LouisChen15___construction_site/construction_site-test.arrow     --num-samples 10     --scale-mean 0.5 --scale-min 0.4 --scale-max 0.6