#!/usr/bin/env python3
"""
Submit SSIM filtering jobs for SODA VOC dataset (no GPU required)
Xử lý VOC format (JPEGImages + Annotations)

SSIM thresholds:
- Rain styles: 0.6 - 0.95
- Snow styles: 0.5 - 0.95
"""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# SSIM thresholds by style type
SSIM_THRESHOLDS = {
    'rain': {'min': 0.6, 'max': 0.95},
    'snow': {'min': 0.5, 'max': 0.95}
}


def parse_args():
    parser = argparse.ArgumentParser(description='Submit SSIM filter jobs for SODA VOC dataset')
    
    # Input/Output
    parser.add_argument('--input-root', type=str, default='output/soda_voc/train',
                        help='Root directory containing style_* folders')
    parser.add_argument('--original-dir', type=str, 
                        default='SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007/JPEGImages',
                        help='Directory containing original JPEGImages')
    parser.add_argument('--output-root', type=str, default=None,
                        help='Output directory for filtered images (optional)')
    
    # Style selection
    parser.add_argument('--styles', type=str, default=None,
                        help='Comma-separated style names (default: all style_* dirs)')
    
    # Filter mode
    parser.add_argument('--remove-bad', action='store_true', default=True,
                        help='Remove images outside SSIM range (default: True)')
    parser.add_argument('--copy-good', action='store_true',
                        help='Copy good images to output dir instead of removing bad')
    
    # SLURM config
    parser.add_argument('--partition', type=str, default='cpu', help='SLURM partition (no GPU)')
    parser.add_argument('--account', type=str, default='pgs0407', help='SLURM account')
    parser.add_argument('--time', type=str, default='04:00:00', help='Job time limit')
    parser.add_argument('--mem', type=str, default='32G', help='Memory')
    parser.add_argument('--cpus', type=int, default=8, help='Number of CPUs')
    
    # Misc
    parser.add_argument('--dry-run', action='store_true', help='Print jobs without submitting')
    parser.add_argument('--conda-env', type=str, default='VLM', help='Conda environment name')
    
    return parser.parse_args()


def get_style_type(style_name):
    """Determine style type (rain or snow) from style name"""
    style_lower = style_name.lower()
    if 'rain' in style_lower:
        return 'rain'
    elif 'snow' in style_lower:
        return 'snow'
    else:
        # Default to snow thresholds for unknown styles
        return 'snow'


def get_styles(input_root, styles_arg):
    """Get list of style directories to process"""
    input_path = Path(input_root)
    
    if styles_arg:
        # Use specified styles
        style_names = [s.strip() for s in styles_arg.split(',')]
        styles = []
        for name in style_names:
            style_dir = input_path / name
            if style_dir.exists():
                styles.append(style_dir)
            else:
                print(f"⚠️  Style directory not found: {style_dir}")
    else:
        # Use all style_* directories
        styles = sorted(input_path.glob('style_*'))
    
    return styles


def create_slurm_script(job_name, worker_cmd, args, log_dir):
    """Create SLURM job script (no GPU)"""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err

# Load conda
source ~/.bashrc
conda activate {args.conda_env}

# Run worker
cd {Path.cwd()}
{worker_cmd}

echo "Job completed at $(date)"
"""
    return script


def main():
    args = parse_args()
    
    print("="*70)
    print("SSIM Filter Job Submission (SODA VOC Dataset)")
    print("="*70)
    print(f"Input root: {args.input_root}")
    print(f"Original directory: {args.original_dir}")
    print(f"Partition: {args.partition} (no GPU)")
    print(f"Mode: {'Copy good images' if args.copy_good else 'Remove bad images'}")
    print("="*70)
    
    # Verify original directory exists
    original_dir = Path(args.original_dir)
    if not original_dir.exists():
        print(f"❌ Original directory not found: {original_dir}")
        return
    
    # Count original images
    orig_images = list(original_dir.glob("*.jpg")) + list(original_dir.glob("*.jpeg"))
    print(f"\n📁 Found {len(orig_images)} original images")
    
    # Get styles to process
    styles = get_styles(args.input_root, args.styles)
    
    if not styles:
        print("❌ No style directories found")
        return
    
    print(f"\n📁 Found {len(styles)} style directories:")
    for s in styles:
        style_type = get_style_type(s.name)
        thresh = SSIM_THRESHOLDS[style_type]
        # Count images
        img_dir = s / 'JPEGImages' if (s / 'JPEGImages').exists() else s
        num_imgs = len(list(img_dir.glob("*.jpg")))
        print(f"  • {s.name} ({style_type}): {num_imgs} images, SSIM [{thresh['min']}, {thresh['max']}]")
    
    # Create directories
    jobs_dir = Path('jobs/ssim_filter_soda_voc')
    logs_dir = Path('logs/ssim_filter_soda_voc')
    
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output_root:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    
    # Submit jobs
    submitted_jobs = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n🚀 {'DRY RUN - ' if args.dry_run else ''}Submitting jobs...")
    
    for style_dir in styles:
        style_name = style_dir.name
        style_type = get_style_type(style_name)
        thresh = SSIM_THRESHOLDS[style_type]
        
        # Job name
        job_name = f"ssim_soda_{style_name}"
        
        # CSV output
        csv_file = Path(args.input_root) / 'filtered' / f"ssim_results_{style_name}.csv"
        
        # Worker command
        worker_cmd = (
            f"python generation/weather/ssim_filter_worker_voc.py "
            f"--input-dir {style_dir} "
            f"--original-dir '{original_dir}' "
            f"--ssim-min {thresh['min']} "
            f"--ssim-max {thresh['max']} "
            f"--workers {args.cpus} "
            f"--csv-output {csv_file} "
        )
        
        if args.copy_good and args.output_root:
            output_style_dir = Path(args.output_root) / style_name
            worker_cmd += f"--output-dir {output_style_dir} "
        else:
            worker_cmd += "--remove-bad "
        
        # Create SLURM script
        script_content = create_slurm_script(job_name, worker_cmd, args, logs_dir)
        script_path = jobs_dir / f"{job_name}_{timestamp}.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"\n📝 {style_name}:")
        print(f"   SSIM range: [{thresh['min']}, {thresh['max']}]")
        print(f"   Script: {script_path}")
        
        if not args.dry_run:
            # Submit job
            result = subprocess.run(['sbatch', str(script_path)], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs.append(job_id)
                print(f"   ✅ Submitted: Job ID {job_id}")
            else:
                print(f"   ❌ Failed to submit: {result.stderr}")
        else:
            print(f"   🔍 DRY RUN - would submit job")
    
    # Summary
    print("\n" + "="*70)
    if args.dry_run:
        print(f"DRY RUN complete. {len(styles)} jobs would be submitted.")
        print("Run without --dry-run to actually submit jobs.")
    else:
        print(f"Submitted {len(submitted_jobs)} jobs")
        if submitted_jobs:
            print(f"Job IDs: {', '.join(submitted_jobs)}")
            print(f"\nMonitor with: squeue -j {','.join(submitted_jobs)}")
            print(f"Or: squeue -u $USER")
    print("="*70)


if __name__ == '__main__':
    main()
