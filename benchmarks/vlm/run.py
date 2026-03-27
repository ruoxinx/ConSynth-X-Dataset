#!/usr/bin/env python3
"""
Main entry point for Benchmark Runner.

Usage:
    # Run inference
    python run.py --model qwen2.5-vl-7b --task vqa_safety --condition original
    python run.py --model internvl2.5-8b --task description --condition small

    # Run evaluation only (on existing results)
    python run.py --evaluate --task vqa_safety --condition original \
        --predictions output/vqa_safety/Qwen2.5-VL-7B-Instruct_vqa_safety_original.json

    # List available models
    python run.py --list-models
"""
import argparse
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Data paths
LMUDATA = os.path.expanduser('~/LMUData')
AUG_ROOT = '/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data'
ANNOTATION_ROOT = '/users/PGS0407/binben14/VietHuy/ConstructionSite-10k-Implementation/Annotations'

# TSV mapping: (task, condition) → tsv path
TSV_MAP = {
    ('description', 'original'): os.path.join(LMUDATA, 'CSAug_original_500.tsv'),
    ('description', 'weather'): os.path.join(LMUDATA, 'CSAug_weather_500.tsv'),
    ('description', 'night'): os.path.join(LMUDATA, 'CSAug_night_500.tsv'),
    ('description', 'small'): os.path.join(LMUDATA, 'CSAug_small_500.tsv'),
    ('vqa_safety', 'original'): os.path.join(LMUDATA, 'CSAug_original_500.tsv'),
    ('vqa_safety', 'weather'): os.path.join(LMUDATA, 'CSAug_weather_500.tsv'),
    ('vqa_safety', 'night'): os.path.join(LMUDATA, 'CSAug_night_500.tsv'),
    ('vqa_safety', 'small'): os.path.join(LMUDATA, 'CSAug_small_500.tsv'),
    ('vqa_rule1', 'original'): os.path.join(LMUDATA, 'CSAug_Rule1_original_500.tsv'),
    ('vqa_rule1', 'weather'): os.path.join(LMUDATA, 'CSAug_Rule1_weather_500.tsv'),
    ('vqa_rule1', 'night'): os.path.join(LMUDATA, 'CSAug_Rule1_night_500.tsv'),
    ('vqa_rule1', 'small'): os.path.join(LMUDATA, 'CSAug_Rule1_small_500.tsv'),
}

# Few-shot images always from original condition
FEWSHOT_DIR = os.path.join(AUG_ROOT, 'construction_site-test', 'images')


def run_inference(args):
    """Run model inference on a dataset."""
    from models import get_model, list_models
    from tasks import get_task
    from runners import InferenceRunner

    if args.list_models:
        print('Available models:')
        for name in list_models():
            print(f'  - {name}')
        return

    # Get TSV path
    tsv_key = (args.task, args.condition)
    tsv_path = TSV_MAP.get(tsv_key)
    if not tsv_path or not os.path.exists(tsv_path):
        print(f'ERROR: TSV not found for {tsv_key}: {tsv_path}')
        sys.exit(1)

    # Create output dir
    output_dir = os.path.join(PROJECT_ROOT, 'output', args.task)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize task
    task_kwargs = {}
    if args.task == 'vqa_safety':
        task_kwargs['fewshot_image_dir'] = FEWSHOT_DIR
    if args.task == 'object_detection':
        task_kwargs['target_object'] = args.object_type or 'excavator'

    task = get_task(args.task, **task_kwargs)

    # Initialize and load model
    print(f'Loading model: {args.model}')
    model = get_model(args.model)
    model.load()

    # Run inference
    runner = InferenceRunner(
        model=model,
        task=task,
        condition=args.condition,
        output_dir=output_dir,
        tsv_path=tsv_path,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    results = runner.run()

    # Cleanup
    model.unload()
    print(f'Inference complete. Results: {len(results)} images')


def run_evaluation(args):
    """Run evaluation on existing prediction files."""
    from evaluations import DescriptionEvaluator, VQAEvaluator, DetectionEvaluator

    if not args.predictions:
        print('ERROR: --predictions required for evaluation')
        sys.exit(1)

    if args.task == 'description':
        ref_path = os.path.join(ANNOTATION_ROOT, 'random1_references.json')
        evaluator = DescriptionEvaluator(args.predictions, ref_path)
        results = evaluator.compute_all()

    elif args.task == 'vqa_safety':
        ref_path = os.path.join(ANNOTATION_ROOT, 'random1_vqa_references.json')
        bbox_refs = {
            f'rule{i}': os.path.join(ANNOTATION_ROOT, f'vqa_rule{i}_bbox_references.json')
            for i in range(1, 5)
        }
        evaluator = VQAEvaluator(args.predictions, ref_path, bbox_refs)
        results = evaluator.evaluate_all()

    else:
        print(f'Evaluation not yet implemented for task: {args.task}')
        sys.exit(1)

    # Print results
    print('\n' + '=' * 60)
    print(f'EVALUATION RESULTS: {args.task} / {args.condition}')
    print('=' * 60)
    print(json.dumps(results, indent=2))

    # Save results
    if args.eval_output:
        with open(args.eval_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved: {args.eval_output}')


def main():
    parser = argparse.ArgumentParser(description='Benchmark Runner for VLMs')
    parser.add_argument('--model', type=str, help='Model name from registry')
    parser.add_argument('--task', type=str,
                        choices=['description', 'vqa_safety', 'vqa_rule1', 'object_detection'],
                        help='Task to run')
    parser.add_argument('--condition', type=str,
                        choices=['original', 'weather', 'night', 'small'],
                        help='Dataset condition/augmentation')
    parser.add_argument('--object-type', type=str,
                        choices=['excavator', 'workers', 'rebars'],
                        help='Object type for detection task')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Do not resume from checkpoint')
    parser.add_argument('--checkpoint-every', type=int, default=1,
                        help='Checkpoint frequency (default: every image)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')

    # Evaluation mode
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation on existing predictions')
    parser.add_argument('--predictions', type=str,
                        help='Path to predictions JSON for evaluation')
    parser.add_argument('--eval-output', type=str,
                        help='Path to save evaluation results')

    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    if args.list_models:
        run_inference(args)
    elif args.evaluate:
        run_evaluation(args)
    else:
        if not args.model or not args.task or not args.condition:
            parser.error('--model, --task, and --condition are required for inference')
        run_inference(args)


if __name__ == '__main__':
    main()
