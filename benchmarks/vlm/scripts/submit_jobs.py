#!/usr/bin/env python3
"""
SLURM job submission script for Benchmark Runner.

Usage:
    # Submit all tasks for a model + condition
    python scripts/submit_jobs.py --model qwen2.5-vl-7b --condition original --task vqa_safety

    # Submit all conditions for a model
    python scripts/submit_jobs.py --model internvl2.5-8b --task description --all-conditions

    # All models × all conditions × all tasks
    python scripts/submit_jobs.py --all

    # Just generate scripts without submitting
    python scripts/submit_jobs.py --model qwen2.5-vl-7b --condition original --task vqa_safety --no-submit

    # Submit evaluation jobs for existing predictions
    python scripts/submit_jobs.py --evaluate                    # eval both tasks
    python scripts/submit_jobs.py --evaluate --task description # eval description only
    python scripts/submit_jobs.py --evaluate --task vqa_safety  # eval vqa only
    python scripts/submit_jobs.py --evaluate --no-submit        # generate scripts only
"""
import argparse
import glob
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(PROJECT_ROOT, 'jobs')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

ALL_MODELS = [
    'qwen2.5-vl-7b',
    'qwen2.5-vl-32b',
    'internvl2.5-8b',
    'internvl2.5-26b',
    'llama-3.2-11b-vision',
    'gpt-4o',
    'gpt-4o-mini',
    'llava-v1.5-7b',
    'qwen3-vl-8b',
    'phi-4-multimodal',
]

# API models (no GPU needed)
API_MODELS = {'gpt-4o', 'gpt-4o-mini'}

ALL_CONDITIONS = ['original', 'weather', 'night', 'small']

ALL_TASKS = ['description', 'vqa_safety']

# Conda environment paths
CONDA_ENV_DEFAULT = '/users/PGS0407/binben14/.conda/envs/VLM'        # transformers 4.46.3 (InternVL, LLaVA, etc.)
CONDA_ENV_NEW = '/users/PGS0407/binben14/.conda/envs/vlm-new'        # transformers 5.x (Qwen3.5, future models)

# Models that need the newer env. All others use CONDA_ENV_DEFAULT.
MODEL_CONDA_ENV = {
    'qwen3-vl-8b': CONDA_ENV_NEW,
    'gemma-3-27b-it': CONDA_ENV_NEW,
}

# GPU requirements per model
GPU_CONFIG = {
    'qwen2.5-vl-7b': {'gpus': 1, 'mem': '32G', 'time': '4:00:00'},
    'qwen2.5-vl-32b': {'gpus': 2, 'mem': '48G', 'time': '10:00:00'},
    'qwen2.5-vl-72b': {'gpus': 4, 'mem': '80G', 'time': '24:00:00'},
    'internvl2.5-8b': {'gpus': 1, 'mem': '32G', 'time': '4:00:00'},
    'internvl2.5-26b': {'gpus': 3, 'mem': '64G', 'time': '10:00:00'},
    'llama-3.2-11b-vision': {'gpus': 1, 'mem': '32G', 'time': '6:00:00'},
    'gpt-4o': {'gpus': 0, 'mem': '8G', 'time': '6:00:00'},
    'gpt-4o-mini': {'gpus': 0, 'mem': '8G', 'time': '4:00:00'},
    'llava-v1.5-7b': {'gpus': 1, 'mem': '32G', 'time': '4:00:00'},
    'gemma-3-27b-it': {'gpus': 3, 'mem': '32G', 'time': '10:00:00'},
    'qwen3-vl-8b': {'gpus': 1, 'mem': '32G', 'time': '4:00:00'},
    'phi-4-multimodal': {'gpus': 1, 'mem': '32G', 'time': '4:00:00'},
}

SLURM_TEMPLATE_GPU = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=pgs0407
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:{gpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={log_dir}/{model}/{task}/{job_name}_%j.out
#SBATCH --error={log_dir}/{model}/{task}/{job_name}_%j.err

module load cuda/12.8.1
module load miniconda3/24.1.2-py310
conda activate {conda_env}

export LD_LIBRARY_PATH={conda_env}/lib:$LD_LIBRARY_PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME=/users/PGS0407/binben14/.cache/huggingface

cd {project_root}

python run.py \\
    --model {model} \\
    --task {task} \\
    --condition {condition} \\
    --resume

echo "Job completed: {job_name}"
"""

SLURM_TEMPLATE_API = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=pgs0407
#SBATCH --partition=batch
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={log_dir}/{model}/{task}/{job_name}_%j.out
#SBATCH --error={log_dir}/{model}/{task}/{job_name}_%j.err

module load miniconda3/24.1.2-py310
conda activate {conda_env}

cd {project_root}

python run.py \\
    --model {model} \\
    --task {task} \\
    --condition {condition} \\
    --resume

echo "Job completed: {job_name}"
"""


def generate_job(model, task, condition, no_submit=False):
    """Generate and optionally submit a SLURM job."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOGS_DIR, model, task), exist_ok=True)

    job_name = f'{model}_{task}_{condition}'
    gpu_cfg = GPU_CONFIG.get(model, {'gpus': 1, 'mem': '32G', 'time': '6:00:00'})
    is_api = model in API_MODELS
    template = SLURM_TEMPLATE_API if is_api else SLURM_TEMPLATE_GPU

    script_path = os.path.join(JOBS_DIR, f'{job_name}.sh')
    conda_env = MODEL_CONDA_ENV.get(model, CONDA_ENV_DEFAULT)

    script = template.format(
        job_name=job_name,
        gpus=gpu_cfg['gpus'],
        mem=gpu_cfg['mem'],
        time=gpu_cfg['time'],
        log_dir=LOGS_DIR,
        project_root=PROJECT_ROOT,
        model=model,
        task=task,
        condition=condition,
        script_path=script_path,
        conda_env=conda_env,
    )

    with open(script_path, 'w') as f:
        f.write(script)

    print(f'Created: {script_path}')

    if not no_submit:
        result = subprocess.run(['sbatch', script_path],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f'  Submitted: {result.stdout.strip()}')
        else:
            print(f'  ERROR: {result.stderr.strip()}')

    return script_path


# ── Evaluation job templates ──────────────────────────────────────────

SLURM_EVAL_GPU = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=pgs0407
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output={log_dir}/eval/{job_name}_%j.out
#SBATCH --error={log_dir}/eval/{job_name}_%j.err

module load cuda/12.8.1
module load miniconda3/24.1.2-py310
conda activate {conda_env}

export LD_LIBRARY_PATH={conda_env}/lib:$LD_LIBRARY_PATH
export HF_HOME=/users/PGS0407/binben14/.cache/huggingface

cd {project_root}

python {script}

echo "Eval completed: {job_name}"
"""

SLURM_EVAL_CPU = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=pgs0407
#SBATCH --partition=batch
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output={log_dir}/eval/{job_name}_%j.out
#SBATCH --error={log_dir}/eval/{job_name}_%j.err

module load miniconda3/24.1.2-py310
conda activate {conda_env}

cd {project_root}

python {script}

echo "Eval completed: {job_name}"
"""


def submit_eval_jobs(tasks=None, no_submit=False):
    """Submit evaluation SLURM jobs for existing predictions."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOGS_DIR, 'eval'), exist_ok=True)

    if tasks is None:
        tasks = ALL_TASKS

    count = 0

    if 'description' in tasks:
        # Description eval needs GPU (BERTScore + CLIPScore) — 1 job for all files
        desc_preds = glob.glob(os.path.join(PROJECT_ROOT, 'output', 'description', '*.json'))
        # Filter out checkpoint files with < 100 samples
        desc_preds = [f for f in desc_preds if 'checkpoint' not in os.path.basename(f)]
        if desc_preds:
            job_name = 'eval_description_all'
            script_path = os.path.join(JOBS_DIR, f'{job_name}.sh')
            script = SLURM_EVAL_GPU.format(
                job_name=job_name,
                log_dir=LOGS_DIR,
                project_root=PROJECT_ROOT,
                conda_env=CONDA_ENV_DEFAULT,
                script='scripts/evaluate_description_gpu.py',
            )
            with open(script_path, 'w') as f:
                f.write(script)
            print(f'Created: {script_path}  ({len(desc_preds)} prediction files)')
            if not no_submit:
                result = subprocess.run(['sbatch', script_path],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    print(f'  Submitted: {result.stdout.strip()}')
                else:
                    print(f'  ERROR: {result.stderr.strip()}')
            count += 1
        else:
            print('No description prediction files found, skipping.')

    if 'vqa_safety' in tasks:
        # VQA eval is CPU-only — 1 job for all files
        vqa_preds = glob.glob(os.path.join(PROJECT_ROOT, 'output', 'vqa_safety', '*.json'))
        if vqa_preds:
            job_name = 'eval_vqa_safety_all'
            script_path = os.path.join(JOBS_DIR, f'{job_name}.sh')
            script = SLURM_EVAL_CPU.format(
                job_name=job_name,
                log_dir=LOGS_DIR,
                project_root=PROJECT_ROOT,
                conda_env=CONDA_ENV_DEFAULT,
                script='scripts/evaluate_vqa_batch.py',
            )
            with open(script_path, 'w') as f:
                f.write(script)
            print(f'Created: {script_path}  ({len(vqa_preds)} prediction files)')
            if not no_submit:
                result = subprocess.run(['sbatch', script_path],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    print(f'  Submitted: {result.stdout.strip()}')
                else:
                    print(f'  ERROR: {result.stderr.strip()}')
            count += 1
        else:
            print('No VQA prediction files found, skipping.')

    print(f'\nTotal eval jobs: {count}')


def main():
    parser = argparse.ArgumentParser(description='Submit benchmark SLURM jobs')
    parser.add_argument('--model', choices=ALL_MODELS)
    parser.add_argument('--task', choices=ALL_TASKS)
    parser.add_argument('--condition', choices=ALL_CONDITIONS)
    parser.add_argument('--all-conditions', action='store_true')
    parser.add_argument('--all', action='store_true',
                        help='Submit all models x conditions x tasks')
    parser.add_argument('--evaluate', action='store_true',
                        help='Submit evaluation jobs for existing predictions')
    parser.add_argument('--no-submit', action='store_true',
                        help='Generate scripts only, do not submit')
    args = parser.parse_args()

    # Evaluation mode
    if args.evaluate:
        eval_tasks = [args.task] if args.task else ALL_TASKS
        submit_eval_jobs(tasks=eval_tasks, no_submit=args.no_submit)
        return

    if args.all:
        models = ALL_MODELS
        tasks = ALL_TASKS
        conditions = ALL_CONDITIONS
    else:
        models = [args.model] if args.model else ALL_MODELS
        tasks = [args.task] if args.task else ALL_TASKS
        conditions = ALL_CONDITIONS if args.all_conditions else (
            [args.condition] if args.condition else ALL_CONDITIONS)

    count = 0
    for model in models:
        for task in tasks:
            for condition in conditions:
                generate_job(model, task, condition, args.no_submit)
                count += 1

    print(f'\nTotal jobs: {count}')


if __name__ == '__main__':
    main()
