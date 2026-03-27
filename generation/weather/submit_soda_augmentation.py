#!/usr/bin/env python3
"""
Submit batch weather augmentation jobs for SODA VOC dataset
Supports both rain and snow augmentation

Output structure:
./output/soda_voc/{split}/{style_name}/JPEGImages/
"""

import argparse
import subprocess
import math
from pathlib import Path
from datetime import datetime


def count_image_list(list_file):
    """Count total images in list file"""
    with open(list_file, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for line in f if line.strip())


def parse_args():
    parser = argparse.ArgumentParser(description='Submit SODA VOC augmentation jobs')
    
    # Required
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test', 'trainval', 'val'],
                        help='Dataset split to process (default: train)')
    
    # VOC paths
    parser.add_argument('--voc-root', type=str, 
                        default='SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007',
                        help='VOC dataset root directory')
    
    # Batch config
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of images per batch (default: 100)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start image index (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='End image index (default: all images)')
    
    # Style config
    parser.add_argument('--snow-style-dir', type=str, default='weather_aug/snow_style',
                        help='Directory containing snow style images')
    parser.add_argument('--rain-style-dir', type=str, default='weather_aug/rain_style',
                        help='Directory containing rain style images')
    parser.add_argument('--pipeline', type=str, default='all',
                        choices=['snow', 'rain', 'all'],
                        help='Which pipeline to run (default: all)')
    
    # Processing config  
    parser.add_argument('--steps', type=int, default=10, help='Style transfer steps')
    parser.add_argument('--style-weight', type=float, default=10000, help='Style weight')
    parser.add_argument('--intensity', type=str, default='light',
                        choices=['light', 'medium', 'heavy', 'extreme', 'quiet_night'], 
                        help='Weather intensity')
    
    # Output config
    parser.add_argument('--output-dir', type=str, default='output/soda_voc',
                        help='Base output directory (default: output/soda_voc)')
    
    # SLURM config
    parser.add_argument('--partition', type=str, default='gpu', help='SLURM partition')
    parser.add_argument('--account', type=str, default='pgs0407', help='SLURM account')
    parser.add_argument('--time', type=str, default='04:00:00', help='Job time limit')
    parser.add_argument('--gpu', type=str, default='v100:1', help='GPU config')
    parser.add_argument('--mem', type=str, default='32G', help='Memory')
    
    # Misc
    parser.add_argument('--dry-run', action='store_true', help='Print jobs without submitting')
    parser.add_argument('--conda-env', type=str, default='VLM', help='Conda environment')
    
    return parser.parse_args()


def get_styles(style_dir):
    """Get list of style files from directory"""
    style_path = Path(style_dir)
    if not style_path.exists():
        return []
    styles = sorted(style_path.glob('*.jpg')) + sorted(style_path.glob('*.png'))
    return styles


def get_worker_script(pipeline_type):
    """Get appropriate worker script for pipeline"""
    if pipeline_type == 'rain':
        return 'generation/weather/soda_augmentation_worker_rain.py'
    else:
        return 'generation/weather/soda_augmentation_worker_snow.py'


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
conda activate {args.conda_env}

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
        job_id = result.stdout.strip().split()[-1]
        return job_id
    else:
        print(f"  ❌ Failed to submit: {result.stderr}")
        return None


def main():
    args = parse_args()
    
    # Setup paths
    work_dir = Path.cwd()
    voc_root = work_dir / args.voc_root
    
    # Validate VOC structure
    images_dir = voc_root / 'JPEGImages'
    image_list_file = voc_root / 'ImageSets' / 'Main' / f'{args.split}.txt'
    
    if not images_dir.exists():
        print(f"❌ JPEGImages directory not found: {images_dir}")
        return
    
    if not image_list_file.exists():
        print(f"❌ Image list not found: {image_list_file}")
        return
    
    # Output paths
    output_base = work_dir / args.output_dir / args.split
    jobs_dir = work_dir / 'jobs' / 'soda_aug'
    log_dir = work_dir / 'logs' / 'soda_aug'
    
    # Ensure dirs exist
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get styles
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
    
    # Count images
    print(f"📂 Loading dataset info from {image_list_file}...")
    total_images = count_image_list(image_list_file)
    print(f"   Total images in {args.split}: {total_images}")
    
    # Calculate batches
    start_idx = args.start
    end_idx = args.end if args.end else total_images
    end_idx = min(end_idx, total_images)
    num_images = end_idx - start_idx
    num_batches = math.ceil(num_images / args.batch_size)
    
    print("=" * 70)
    print("🚀 SODA VOC WEATHER AUGMENTATION BATCH SUBMISSION")
    print("=" * 70)
    print(f"VOC Root:     {voc_root}")
    print(f"Split:        {args.split}")
    print(f"Images:       {images_dir}")
    print(f"Image list:   {image_list_file}")
    print(f"Output:       {output_base}/")
    print(f"Samples:      {start_idx} to {end_idx} ({num_images} total)")
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
        style_output_dir = output_base / f"style_{style_name}" / 'JPEGImages'
        worker_script = get_worker_script(pipeline_type)
        
        print(f"\n📁 Style: {style_name} ({pipeline_type} pipeline)")
        print("-" * 50)
        
        for batch_idx in range(num_batches):
            batch_start = start_idx + batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, end_idx)
            
            # Job name
            job_name = f'soda_{args.split}_{style_name}_{batch_start}-{batch_end}_{timestamp}'
            
            # Build worker command (quote paths to handle spaces)
            worker_cmd_str = (
                f'python {worker_script} '
                f'--image-dir "{images_dir}" '
                f'--image-list "{image_list_file}" '
                f'--output-dir "{style_output_dir}" '
                f'--style "{style_path}" '
                f'--start {batch_start} '
                f'--end {batch_end} '
                f'--steps {args.steps} '
                f'--style-weight {args.style_weight} '
                f'--intensity {args.intensity}'
            )
            
            # Create and submit job
            script = create_slurm_script(job_name, worker_cmd_str, args, log_dir)
            job_id = submit_job(script, job_name, jobs_dir, args.dry_run)
            
            if job_id:
                print(f"  ✓ Batch {batch_start}-{batch_end}: Job {job_id}")
                all_jobs.append({
                    'job_id': job_id,
                    'style': style_name,
                    'batch': f'{batch_start}-{batch_end}',
                    'output': str(style_output_dir)
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
        print(f"  {output_base.relative_to(work_dir)}/style_{style_name}/JPEGImages/ ({pipeline_type})")
    
    # Save job list
    job_list_path = jobs_dir / f'jobs_soda_{args.split}_{timestamp}.txt'
    with open(job_list_path, 'w') as f:
        f.write(f"# SODA VOC augmentation jobs - {timestamp}\n")
        f.write(f"# Split: {args.split}\n")
        f.write(f"# Total: {len(all_jobs)} jobs\n")
        f.write(f"# Config: steps={args.steps}, style_weight={args.style_weight}, intensity={args.intensity}\n\n")
        for job in all_jobs:
            f.write(f"{job['job_id']}\t{job['style']}\t{job['batch']}\t{job['output']}\n")
    print(f"\n📄 Job list: {job_list_path}")
    
    if not args.dry_run and all_jobs:
        job_ids = ','.join([j['job_id'] for j in all_jobs if j['job_id'] != 'DRY-RUN'])
        print(f"\n📋 Check all jobs:")
        print(f"   squeue -j {job_ids}")


if __name__ == '__main__':
    main()
