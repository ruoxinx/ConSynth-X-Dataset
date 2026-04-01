#!/usr/bin/env python3
"""
Submit SSIM filtering jobs for TRAIN dataset (no GPU required)
Modified version to handle multiple original arrow files.

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
    parser = argparse.ArgumentParser(description='Submit SSIM filter jobs for train dataset')
    
    # Input/Output
    parser.add_argument('--input-root', type=str, default='output/construction_site_train',
                        help='Root directory containing style_* folders')
    parser.add_argument('--original-arrow-dir', type=str, 
                        default='LouisChen15___construction_site',
                        help='Directory containing original arrow files')
    parser.add_argument('--original-arrow-pattern', type=str, 
                        default='construction_site-train-*.arrow',
                        help='Glob pattern for original dataset arrow files')
    parser.add_argument('--output-root', type=str, default='output/construction_site_train/filtered',
                        help='Output directory for filtered arrow files')
    
    # Style selection
    parser.add_argument('--styles', type=str, default=None,
                        help='Comma-separated style names (default: all style_* dirs)')
    
    # SLURM config
    parser.add_argument('--partition', type=str, default='cpu', help='SLURM partition (no GPU)')
    parser.add_argument('--account', type=str, default='pgs0407', help='SLURM account')
    parser.add_argument('--time', type=str, default='04:00:00', help='Job time limit')
    parser.add_argument('--mem', type=str, default='32G', help='Memory')
    parser.add_argument('--cpus', type=int, default=4, help='Number of CPUs')
    
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
    print("SSIM Filter Job Submission (Train Dataset)")
    print("="*70)
    print(f"Input root: {args.input_root}")
    print(f"Original arrow dir: {args.original_arrow_dir}")
    print(f"Original arrow pattern: {args.original_arrow_pattern}")
    print(f"Output root: {args.output_root}")
    print(f"Partition: {args.partition} (no GPU)")
    print("="*70)
    
    # Verify original arrow files exist
    original_arrow_dir = Path(args.original_arrow_dir)
    original_arrow_files = sorted(original_arrow_dir.glob(args.original_arrow_pattern))
    
    if not original_arrow_files:
        print(f"❌ No original arrow files found matching: {original_arrow_dir}/{args.original_arrow_pattern}")
        return
    
    print(f"\n📁 Found {len(original_arrow_files)} original arrow files:")
    for f in original_arrow_files:
        print(f"  • {f}")
    
    # Get styles to process
    styles = get_styles(args.input_root, args.styles)
    
    if not styles:
        print("❌ No style directories found")
        return
    
    print(f"\n📁 Found {len(styles)} style directories:")
    for s in styles:
        style_type = get_style_type(s.name)
        thresh = SSIM_THRESHOLDS[style_type]
        print(f"  • {s.name} ({style_type}): SSIM [{thresh['min']}, {thresh['max']}]")
    
    # Create directories
    jobs_dir = Path('jobs/ssim_filter_train')
    logs_dir = Path('logs/ssim_filter_train')
    output_root = Path(args.output_root)
    
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Submit jobs
    submitted_jobs = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n🚀 {'DRY RUN - ' if args.dry_run else ''}Submitting jobs...")
    
    for style_dir in styles:
        style_name = style_dir.name
        style_type = get_style_type(style_name)
        thresh = SSIM_THRESHOLDS[style_type]
        
        # Output file
        output_file = output_root / f"filtered_{style_name}_ssim_{thresh['min']}_{thresh['max']}.arrow"
        csv_file = output_root / f"ssim_results_{style_name}.csv"
        
        # Job name
        job_name = f"ssim_train_{style_name}"
        
        # Worker command - use new worker that supports multiple arrow files
        worker_cmd = (
            f"python generation/weather/ssim_filter_worker_multi.py "
            f"--input-dir {style_dir} "
            f"--original-arrow-dir {original_arrow_dir} "
            f"--original-arrow-pattern '{args.original_arrow_pattern}' "
            f"--output {output_file} "
            f"--ssim-min {thresh['min']} "
            f"--ssim-max {thresh['max']} "
            f"--csv-output {csv_file}"
        )
        
        # Create SLURM script
        script_content = create_slurm_script(job_name, worker_cmd, args, logs_dir)
        script_path = jobs_dir / f"{job_name}_{timestamp}.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"\n📝 {style_name}:")
        print(f"   SSIM range: [{thresh['min']}, {thresh['max']}]")
        print(f"   Output: {output_file}")
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
    print("="*70)


if __name__ == '__main__':
    main()
