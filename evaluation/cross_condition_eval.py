"""Cross-condition evaluation harness.

- Load model predictions
- Load split and condition labels
- Run per-condition and combined-condition evaluations
"""
import json


def run_cross_condition_eval(predictions_path: str, splits_path: str, conditions_path: str):
    # TODO: implement loading and evaluation pipeline using robustness_metrics
    with open(predictions_path) as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} predictions")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python cross_condition_eval.py <preds.json> <splits.json> <condition_labels.json>")
    else:
        run_cross_condition_eval(sys.argv[1], sys.argv[2], sys.argv[3])
