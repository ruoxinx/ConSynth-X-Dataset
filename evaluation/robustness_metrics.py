"""Robustness metrics for per-condition evaluation.

Functions:
- compute_per_condition_ap(preds, gts, condition_mask)
- compute_overall_metrics(preds, gts)
"""
from typing import List, Dict


def compute_per_condition_ap(preds: List[Dict], gts: List[Dict], conditions: List[str]):
    """Placeholder: compute AP per condition.

    preds/gts are lists of predictions and ground-truths aligned by image index.
    conditions is list of condition keys to evaluate.
    """
    results = {c: None for c in conditions}
    # TODO: implement per-condition filtering and AP computation (COCO/PASCAL metrics)
    return results


def compute_overall_metrics(preds: List[Dict], gts: List[Dict]):
    """Compute overall detection/segmentation metrics."""
    return {"mAP": None}
