#!/usr/bin/env python3
"""
Batch evaluation script for VQA safety task.
Evaluates all prediction files in output/vqa_safety/ against condition-specific references.

Usage:
    python scripts/evaluate_vqa_batch.py
"""
import json
import os
import sys
import glob
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / 'output' / 'vqa_safety'
VQA_REF_DIR = PROJECT_ROOT / 'output' / 'vqa_references'


def parse_filename(fname):
    """Parse model name and condition from prediction filename.
    Format: ModelName_vqa_safety_condition.json or _checkpoint.json
    """
    base = fname.replace('.json', '').replace('_checkpoint', '')
    match = re.match(r'^(.+)_vqa_safety_(.+)$', base)
    if match:
        return match.group(1), match.group(2)
    return None, None


def main():
    from evaluations.vqa_eval import VQAEvaluator

    pred_files = sorted(glob.glob(str(OUTPUT_DIR / '*.json')))
    print(f'Found {len(pred_files)} prediction files')

    if not pred_files:
        print('No prediction files found. Exiting.')
        return

    all_results = []

    for pred_file in pred_files:
        fname = os.path.basename(pred_file)
        model_name, condition = parse_filename(fname)

        if not condition:
            print(f'  SKIPPED (cannot parse): {fname}')
            continue

        # Load condition-specific references
        ref_path = VQA_REF_DIR / f'{condition}_references.json'
        if not ref_path.exists():
            print(f'  SKIPPED (no reference for {condition}): {fname}')
            continue

        bbox_refs = {}
        for i in range(1, 5):
            bbox_path = VQA_REF_DIR / f'{condition}_rule{i}_bbox.json'
            if bbox_path.exists():
                bbox_refs[f'rule{i}'] = str(bbox_path)

        print(f'\n{"="*60}')
        print(f'Evaluating: {fname}')
        print(f'  Model: {model_name}, Condition: {condition}')
        print(f'{"="*60}')

        # Quick stats
        with open(pred_file) as f:
            predictions = json.load(f)
        n_total = len(predictions)
        n_errors = sum(1 for v in predictions.values()
                       if isinstance(v, dict) and v.get('error'))
        print(f'  Samples: {n_total}, Errors: {n_errors}')

        if n_errors == n_total:
            print('  SKIPPED: all samples have errors')
            continue

        try:
            evaluator = VQAEvaluator(pred_file, str(ref_path), bbox_refs)
            results = evaluator.evaluate_all()
        except Exception as e:
            print(f'  ERROR: {e}')
            continue

        clf = results.get('classification', {})
        overall = clf.get('overall', {})
        print(f'  Classification (micro): P={overall.get("precision", 0):.4f}  '
              f'R={overall.get("recall", 0):.4f}  F1={overall.get("f1", 0):.4f}')

        for r in '1234':
            rule = clf.get(f'rule_{r}', {})
            print(f'    Rule {r}: P={rule.get("precision", 0):.4f}  '
                  f'R={rule.get("recall", 0):.4f}  F1={rule.get("f1", 0):.4f}  '
                  f'(TP={rule.get("tp", 0)} FP={rule.get("fp", 0)} FN={rule.get("fn", 0)})')

        bbox_iou = results.get('bbox_iou', {})
        for rule_id, iou_data in bbox_iou.items():
            if isinstance(iou_data, dict):
                print(f'  BBox IoU {rule_id}: {json.dumps(iou_data)}')

        result_entry = {
            'model': model_name,
            'condition': condition,
            'file': fname,
            'n_samples': n_total,
            'n_errors': n_errors,
            **results,
        }
        all_results.append(result_entry)

    # Save all results
    results_path = PROJECT_ROOT / 'output' / 'vqa_eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n{"="*60}')
    print(f'Saved results: {results_path}')
    print(f'Evaluated {len(all_results)} files total.')

    # Summary table
    print(f'\n{"Model":<30} {"Cond":<10} {"P":>8} {"R":>8} {"F1":>8}')
    print('-' * 70)
    for r in all_results:
        clf = r.get('classification', {}).get('overall', {})
        print(f'{r["model"]:<30} {r["condition"]:<10} '
              f'{clf.get("precision", 0):>8.4f} {clf.get("recall", 0):>8.4f} '
              f'{clf.get("f1", 0):>8.4f}')


if __name__ == '__main__':
    main()
