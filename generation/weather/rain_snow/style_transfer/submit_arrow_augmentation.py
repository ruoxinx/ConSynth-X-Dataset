#!/usr/bin/env python3
"""
Submit batch augmentation jobs to SLURM
Output format: ./output/{dataset_name}/{style_name}/batch_{start}-{end}.arrow
"""

import argparse
import subprocess
import math
from pathlib import Path
from datetime import datetime
import pyarrow as pa


def count_arrow_samples(arrow_file):
    """Count total samples in arrow file"""
    with open(arrow_file, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    return len(table)


def parse_args():
    parser = argparse.ArgumentParser(description='Submit batch augmentation jobs')
    
    # Required
    parser.add_argument('--dataset', type=str, default='construction_site_train',
                        help='Dataset name (default: construction_site_test)')
    
    # Batch config
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of samples per batch (default: 100)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start sample index (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='End sample index (default: all samples)')
    
    # Style config
    parser.add_argument('--snow-style-dir', type=str, default='generation/weather/snow_style',
                        help='Directory containing snow style images')
    parser.add_argument('--rain-style-dir', type=str, default='generation/weather/rain_style',
                        help='Directory containing rain style images')
    parser.add_argument('--pipeline', type=str, default='all',
                        choices=['snow', 'rain', 'all'],
                        help='Which pipeline to run (default: all)')
    
    # Processing config  
    parser.add_argument('--steps', type=int, default=10, help='Style transfer steps')
    parser.add_argument('--style-weight', type=float, default=10000, help='Style weight')
    parser.add_argument('--intensity', type=str, default='light',
                        choices=['light', 'medium', 'heavy', 'extreme', 'quiet_night'], 
                        help='Snow intensity')
    
    # Output config
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Base output directory (default: output)')
    
    # SLURM config
    parser.add_argument('--partition', type=str, default='gpu', help='SLURM partition')
    parser.add_argument('--account', type=str, default='pgs0407', help='SLURM account')
    parser.add_argument('--time', type=str, default='04:00:00', help='Job time limit')
    parser.add_argument('--gpu', type=str, default='v100:1', help='GPU config')
    parser.add_argument('--mem', type=str, default='32G', help='Memory')
    
    # Misc
    parser.add_argument('--dry-run', action='store_true', help='Print jobs without submitting')
    
    return parser.parse_args()


def get_styles(style_dir):
    """Get list of style files from directory"""
    style_path = Path(style_dir)
    if not style_path.exists():
        return []
    styles = sorted(style_path.glob('*.jpg')) + sorted(style_path.glob('*.png'))
    return styles


def get_pipeline_type(style_name):
    """Determine pipeline type from style name"""
    if 'rain' in style_name.lower():
        return 'rain'
    elif 'snow' in style_name.lower():
        return 'snow'
    else:
        return 'snow'  # default to snow


def get_worker_script(pipeline_type):
    """Get appropriate worker script for pipeline"""
    if pipeline_type == 'rain':
        return 'generation/weather/arrow_augmentation_worker_rain.py'
    else:
        return 'generation/weather/arrow_augmentation_worker.py'


def create_slurm_script(job_name, worker_cmd, args, log_dir):
    """Create SLURM job script"""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
#SBATCH --time={args.time}
#SBATCH --gres=gpu:{args.gpu}
#SBATCH --mem={args.mem}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err

# Load conda
source ~/.bashrc
conda activate VLM

# Change to working directory
cd {Path.cwd()}

# Run worker
{worker_cmd}

echo "Job completed at $(date)"
"""
    return script


def submit_job(script_content, job_name, jobs_dir, dry_run=False):
    """Submit job to SLURM"""
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = jobs_dir / f'{job_name}.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    if dry_run:
        print(f"  [DRY-RUN] Would submit: {script_path}")
        return 'DRY-RUN'
    
    result = subprocess.run(['sbatch', str(script_path)], capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract job ID
        job_id = result.stdout.strip().split()[-1]
        return job_id
    else:
        print(f"  ❌ Failed to submit: {result.stderr}")
        return None


def main():
    args = parse_args()
    
    # Setup paths
    work_dir = Path.cwd()
    
    # Handle input file path based on dataset name
    dataset_dir = work_dir / 'LouisChen15___construction_site'
    if args.dataset == 'construction_site_night':
        input_file = "/users/PGS0407/binben14/VietHuy/ConstructionSite/output/construction_site_test/style_snow_0/batch_0-100.arrow"
    elif args.dataset == 'construction_site_test':
        input_file = dataset_dir / 'construction_site-test.arrow'
    elif args.dataset == 'construction_site_train':
        # Train has multiple shards
        input_file = dataset_dir / 'construction_site-train-00000-of-00002.arrow'
    elif args.dataset == 'construction_site_train_2':
        input_file = dataset_dir / 'construction_site-train-00001-of-00002.arrow'
    else:
        input_file = dataset_dir / f'{args.dataset}.arrow'
    
    output_base = work_dir / args.output_dir / args.dataset
    jobs_dir = work_dir / 'jobs' / 'arrow_aug'
    log_dir = work_dir / 'logs' / 'arrow_aug'
    
    # Ensure dirs exist
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get styles based on pipeline option
    all_styles = []
    
    if args.pipeline in ['snow', 'all']:
        snow_styles = get_styles(work_dir / args.snow_style_dir)
        all_styles.extend([(s, 'snow') for s in snow_styles])
        print(f"Snow styles: {[s.stem for s in snow_styles]}")
    
    if args.pipeline in ['rain', 'all']:
        rain_styles = get_styles(work_dir / args.rain_style_dir)
        all_styles.extend([(s, 'rain') for s in rain_styles])
        print(f"Rain styles: {[s.stem for s in rain_styles]}")
    
    if not all_styles:
        print(f"❌ No styles found!")
        return
    
    # Auto-detect total samples from arrow file
    print(f"📂 Loading dataset info from {input_file}...")
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    total_samples = count_arrow_samples(input_file)
    print(f"   Total samples in file: {total_samples}")
    
    # Calculate batches
    start_idx = args.start
    end_idx = args.end if args.end else total_samples
    end_idx = min(end_idx, total_samples)  # Ensure we don't exceed file size
    num_samples = end_idx - start_idx
    num_batches = math.ceil(num_samples / args.batch_size)
    
    print("=" * 70)
    print("🚀 ARROW AUGMENTATION BATCH SUBMISSION")
    print("=" * 70)
    print(f"Dataset:      {args.dataset}")
    print(f"Input:        {input_file}")
    print(f"Output:       {output_base}/")
    print(f"Samples:      {start_idx} to {end_idx} ({num_samples} total)")
    print(f"Batch size:   {args.batch_size}")
    print(f"Num batches:  {num_batches}")
    print(f"Pipeline:     {args.pipeline}")
    print(f"Total styles: {len(all_styles)}")
    print(f"Steps:        {args.steps}")
    print(f"StyleWeight:  {args.style_weight}")
    print(f"Intensity:    {args.intensity}")
    print("=" * 70)
    
    if args.dry_run:
        print("⚠️  DRY-RUN MODE - No jobs will be submitted")
    
    all_jobs = []
    timestamp = datetime.now().strftime('%m%d_%H%M')
    
    for style_path, pipeline_type in all_styles:
        style_name = style_path.stem
        style_output_dir = output_base / style_name
        worker_script = get_worker_script(pipeline_type)
        
        print(f"\n📁 Style: {style_name} ({pipeline_type} pipeline)")
        print("-" * 50)
        
        for batch_idx in range(num_batches):
            batch_start = start_idx + batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, end_idx)
            
            # Output file name
            output_file = style_output_dir / f'batch_{batch_start}-{batch_end}.arrow'
            
            # Job name
            job_name = f'aug_{style_name}_{batch_start}-{batch_end}_{timestamp}'
            
            # Build worker command
            worker_cmd = [
                'python', worker_script,
                '--input', str(input_file),
                '--output', str(output_file),
                '--style', str(style_path),
                '--start', str(batch_start),
                '--end', str(batch_end),
                '--steps', str(args.steps),
                '--style-weight', str(args.style_weight),
                '--intensity', args.intensity
            ]
            
            worker_cmd_str = ' '.join(worker_cmd)
            
            # Create and submit job
            script = create_slurm_script(job_name, worker_cmd_str, args, log_dir)
            job_id = submit_job(script, job_name, jobs_dir, args.dry_run)
            
            if job_id:
                print(f"  ✓ Batch {batch_start}-{batch_end}: Job {job_id}")
                all_jobs.append({
                    'job_id': job_id,
                    'style': style_name,
                    'batch': f'{batch_start}-{batch_end}',
                    'output': str(output_file)
                })
            else:
                print(f"  ❌ Batch {batch_start}-{batch_end}: Failed")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"Total jobs submitted: {len(all_jobs)}")
    print(f"\nOutput structure:")
    for style_path, pipeline_type in all_styles:
        style_name = style_path.stem
        print(f"  {output_base.relative_to(work_dir)}/{style_name}/ ({pipeline_type})")
        for batch_idx in range(min(3, num_batches)):
            batch_start = start_idx + batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, end_idx)
            print(f"    └── batch_{batch_start}-{batch_end}.arrow")
        if num_batches > 3:
            print(f"    └── ... ({num_batches - 3} more batches)")
    
    # Save job list
    job_list_path = jobs_dir / f'jobs_{timestamp}.txt'
    with open(job_list_path, 'w') as f:
        f.write(f"# Arrow augmentation jobs - {timestamp}\n")
        f.write(f"# Total: {len(all_jobs)} jobs\n")
        f.write(f"# Config: steps={args.steps}, style_weight={args.style_weight}, intensity={args.intensity}, depth=MiDaS\n\n")
        for job in all_jobs:
            f.write(f"{job['job_id']}\t{job['style']}\t{job['batch']}\t{job['output']}\n")
    print(f"\n📄 Job list: {job_list_path}")
    
    if not args.dry_run and all_jobs:
        job_ids = ','.join([j['job_id'] for j in all_jobs if j['job_id'] != 'DRY-RUN'])
        print(f"\n📋 Check all jobs:")
        print(f"   squeue -j {job_ids}")


if __name__ == '__main__':
    main()
