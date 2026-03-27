#!/usr/bin/env python3
"""
Submit Day2Night Batch Jobs to SLURM
Splits dataset into batches and submits each batch as a separate job
Each job loads model once and processes all images in its batch

Output structure:
  day2night_output/{arrow_name}/
    images/       - Night images (JPG)
    annotations/  - Annotations (JSON)
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow as pa

WORK_DIR = Path(__file__).parent.resolve()  # day2night/
PROJECT_DIR = WORK_DIR.parent  # construction-site/

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --time={time}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --gres=gpu:{gpu_type}:1

# ============================================
# Day2Night Batch Job
# Batch ID: {batch_id}
# Range: [{start_idx}, {end_idx})
# ============================================

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Batch: {batch_id} ({start_idx} - {end_idx})"
echo "============================================"

# Load environment
module load cuda/11.8.0
module load miniconda3/24.1.2-py310
eval "$(/apps/spack/0.21/pitzer/linux-rhel9-skylake/miniconda3/gcc/11.4.1/24.1.2-py310-ghbxrie/bin/conda shell.bash hook)"
conda activate /users/PGS0407/binben14/.conda/envs/VLM

cd {work_dir}

echo "Python: $(which python)"
echo "CUDA available:"
nvidia-smi
echo ""
echo "Running Day2Night batch worker..."
echo ""

python day2night_batch_worker.py \\
    --arrow-file "{arrow_file}" \\
    --output-dir "{output_dir}" \\
    --start-idx {start_idx} \\
    --end-idx {end_idx} \\
    --batch-id {batch_id}

echo ""
echo "============================================"
echo "Job completed at: $(date)"
echo "Exit code: $?"
echo "============================================"
"""


def get_dataset_size(arrow_path):
    """Get total number of samples in Arrow file"""
    with open(arrow_path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    return len(table)


def main():
    parser = argparse.ArgumentParser(
        description='Submit Day2Night batch jobs to SLURM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--arrow-file', '-i', 
                       default='../LouisChen15___construction_site/construction_site-test.arrow',
                       help='Path to Arrow file (default: construction_site-test.arrow)')
    parser.add_argument('--output-dir', '-o',
                       default=None,
                       help='Output directory (default: day2night_output/{arrow_name})')
    
    # Batch settings
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                       help='Number of images per batch (default: 100)')
    parser.add_argument('--start-batch', type=int, default=0,
                       help='Start from this batch number (default: 0)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to submit (default: all)')
    
    # SLURM settings
    parser.add_argument('--account', '-A', default='pgs0407',
                       help='SLURM account (default: pgs0407)')
    parser.add_argument('--time', default='4:00:00',
                       help='Job time limit per batch (default: 4:00:00)')
    parser.add_argument('--memory', default='32GB',
                       help='Memory per job (default: 32GB)')
    parser.add_argument('--cpus', type=int, default=4,
                       help='CPUs per job (default: 4)')
    parser.add_argument('--gpu-type', default='v100',
                       help='GPU type (default: v100)')
    
    # Control
    parser.add_argument('--dry-run', action='store_true',
                       help='Show jobs without submitting')
    
    args = parser.parse_args()
    
    # Setup paths
    arrow_path = Path(args.arrow_file)
    if not arrow_path.is_absolute():
        arrow_path = WORK_DIR / arrow_path
    
    # Auto-generate output dir from arrow filename
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        arrow_name = arrow_path.stem  # e.g., "construction_site-test"
        output_dir = PROJECT_DIR / 'day2night_output' / arrow_name
    
    log_dir = WORK_DIR / 'logs'
    job_dir = WORK_DIR / 'jobs'
    
    log_dir.mkdir(exist_ok=True)
    job_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate input
    if not arrow_path.exists():
        print(f"❌ Arrow file not found: {arrow_path}")
        sys.exit(1)
    
    # Get dataset size
    print("📂 Loading dataset info...")
    total_samples = get_dataset_size(arrow_path)
    print(f"   Total samples: {total_samples}")
    
    # Calculate batches
    batch_size = args.batch_size
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    # Apply limits
    start_batch = args.start_batch
    if args.max_batches:
        end_batch = min(start_batch + args.max_batches, total_batches)
    else:
        end_batch = total_batches
    
    num_batches = end_batch - start_batch
    
    print(f"\n{'=' * 70}")
    print(f"🌙  DAY2NIGHT BATCH SUBMISSION")
    print(f"{'=' * 70}")
    print(f"Arrow file:   {arrow_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Output format: images/ + annotations/ folders")
    print(f"Total samples: {total_samples}")
    print(f"Batch size:   {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"Submitting:   batches {start_batch} to {end_batch - 1} ({num_batches} batches)")
    print(f"GPU type:     {args.gpu_type}")
    print(f"Time limit:   {args.time}")
    print(f"{'=' * 70}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - showing job details without submitting\n")
    
    # Submit batches
    jobs = []
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    
    for batch_id in range(start_batch, end_batch):
        start_idx = batch_id * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        
        job_name = f"d2n_b{batch_id}_{timestamp}"
        
        job_content = JOB_TEMPLATE.format(
            job_name=job_name,
            account=args.account,
            log_dir=str(log_dir),
            time=args.time,
            cpus=args.cpus,
            memory=args.memory,
            gpu_type=args.gpu_type,
            work_dir=str(WORK_DIR),
            arrow_file=str(arrow_path),
            output_dir=str(output_dir),
            batch_id=batch_id,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        
        job_file = job_dir / f"{job_name}.sh"
        
        if args.dry_run:
            print(f"  Batch {batch_id}: [{start_idx}, {end_idx}) - {end_idx - start_idx} images")
            continue
        
        # Write and submit
        with open(job_file, 'w') as f:
            f.write(job_content)
        os.chmod(job_file, 0o755)
        
        result = subprocess.run(['sbatch', str(job_file)], capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"  ✅ Batch {batch_id}: [{start_idx}, {end_idx}) → Job {job_id}")
            jobs.append({
                'batch_id': batch_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'job_id': job_id,
                'job_file': str(job_file)
            })
        else:
            print(f"  ❌ Batch {batch_id}: Failed - {result.stderr}")
    
    if args.dry_run:
        print(f"\n💡 Remove --dry-run to submit {num_batches} jobs")
        return
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"📋 SUMMARY")
    print(f"{'=' * 70}")
    print(f"Jobs submitted: {len(jobs)}")
    print(f"Output directory: {output_dir}")
    
    # Save job list
    job_list_path = output_dir / f'day2night_jobs_{timestamp}.txt'
    with open(job_list_path, 'w') as f:
        f.write(f"# Day2Night batch jobs - {len(jobs)} jobs\n")
        f.write(f"# Submitted: {datetime.now().isoformat()}\n")
        f.write(f"# Arrow file: {arrow_path}\n")
        f.write(f"# Batch size: {batch_size}\n\n")
        f.write("# job_id\tbatch_id\tstart_idx\tend_idx\n")
        for job in jobs:
            f.write(f"{job['job_id']}\t{job['batch_id']}\t{job['start_idx']}\t{job['end_idx']}\n")
    print(f"\n📄 Job list saved: {job_list_path}")
    
    # Check command
    if jobs:
        job_ids = ','.join([j['job_id'] for j in jobs])
        print(f"\n📋 Check all jobs:")
        print(f"   squeue -j {job_ids}")


if __name__ == '__main__':
    main()


# Usage:
# python submit_day2night_batch.py                          # Submit all batches
# python submit_day2night_batch.py --batch-size 50          # 50 images per batch
# python submit_day2night_batch.py --start-batch 5          # Start from batch 5
# python submit_day2night_batch.py --max-batches 3          # Only submit 3 batches
# python submit_day2night_batch.py --dry-run                # Preview without submitting
