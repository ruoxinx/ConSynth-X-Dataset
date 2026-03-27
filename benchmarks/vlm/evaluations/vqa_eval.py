"""
Evaluation for safety violation VQA.
- Multi-label classification (P/R/F1 per rule)
- Bounding box IoU for correctly detected violations
Ported from ConstructionSite-10k-Implementation.
"""
import json
import re
import numpy as np
from .detection_eval import DetectionEvaluator


class VQAEvaluator:
    """Evaluates VQA safety violation detection."""

    RULE_KEYWORDS = {
        '1': ['ppe', 'hard hat', 'helmet', 'vest', 'clothing', 'shoes'],
        '2': ['harness', 'safety harness', 'height'],
        '3': ['edge', 'guardrail', 'fence', 'protection', 'underground'],
        '4': ['excavator', 'blind spot', 'radius', 'operator'],
    }

    def __init__(self, predictions_path, references_path,
                 bbox_references=None):
        """
        Args:
            predictions_path: path to inference JSON
            references_path: path to VQA reference JSON
            bbox_references: dict {rule_id: path_to_bbox_json}
        """
        with open(predictions_path) as f:
            self.predictions = json.load(f)
        with open(references_path) as f:
            self.references = json.load(f)
        self.bbox_references = bbox_references or {}
        self.det_evaluator = DetectionEvaluator()

    def _extract_rules(self, pred_data):
        """Extract predicted rule IDs from model output."""
        if isinstance(pred_data, dict):
            parsed = pred_data.get('parsed', pred_data)
            if isinstance(parsed, dict):
                if parsed.get('parse_error'):
                    return set()
                if '0' in parsed:
                    return set()  # No violations
                return {k for k in parsed.keys() if k in {'1', '2', '3', '4'}}
        return set()

    def _extract_ref_rules(self, ref_data):
        """Extract ground truth rule IDs."""
        rules = set()
        for key in ref_data:
            # Handle formats: "rule_1_violation", "rule_1", or plain "1"/"2"/"3"/"4"
            if key in {'1', '2', '3', '4'} and ref_data[key] is not None:
                rules.add(key)
            else:
                match = re.search(r'rule[_\s]*(\d)', key)
                if match and ref_data[key] is not None:
                    rules.add(match.group(1))
        return rules

    def multilabel_classification(self):
        """
        Multi-label classification metrics per rule.
        Returns P/R/F1 for each rule and overall.
        """
        rule_tp = {r: 0 for r in '1234'}
        rule_fp = {r: 0 for r in '1234'}
        rule_fn = {r: 0 for r in '1234'}

        for img_id, ref_data in self.references.items():
            pred_data = self.predictions.get(img_id)
            if pred_data is None:
                pred_rules = set()
            else:
                pred_rules = self._extract_rules(pred_data)

            ref_rules = self._extract_ref_rules(ref_data)

            for r in '1234':
                pred_has = r in pred_rules
                ref_has = r in ref_rules
                if pred_has and ref_has:
                    rule_tp[r] += 1
                elif pred_has and not ref_has:
                    rule_fp[r] += 1
                elif not pred_has and ref_has:
                    rule_fn[r] += 1

        results = {}
        total_tp = total_fp = total_fn = 0

        for r in '1234':
            tp, fp, fn = rule_tp[r], rule_fp[r], rule_fn[r]
            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)

            results[f'rule_{r}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn,
            }

        # Overall (micro)
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results['overall'] = {'precision': p, 'recall': r, 'f1': f1}

        return results

    def bounding_box_iou(self):
        """Compute IoU for correctly detected violations, per rule."""
        results = {}

        for rule_id, bbox_ref_path in self.bbox_references.items():
            with open(bbox_ref_path) as f:
                bbox_refs = json.load(f)

            pred_bboxes = {}
            ref_bboxes = {}

            for img_id, ref_bbox_data in bbox_refs.items():
                # Get normalized bboxes from reference
                if isinstance(ref_bbox_data, dict):
                    ref_boxes = ref_bbox_data.get('normalized_scale',
                                                   ref_bbox_data.get('bounding_box', []))
                elif isinstance(ref_bbox_data, list):
                    ref_boxes = ref_bbox_data
                else:
                    continue

                if not ref_boxes:
                    continue

                # Check if model correctly detected this rule
                pred_data = self.predictions.get(img_id)
                if pred_data is None:
                    continue

                parsed = pred_data.get('parsed', pred_data)
                if not isinstance(parsed, dict):
                    continue

                rule_num = rule_id.replace('rule', '')
                if rule_num not in parsed:
                    continue

                # Extract predicted bbox
                pred_rule = parsed[rule_num]
                if isinstance(pred_rule, dict):
                    pb = pred_rule.get('bounding_box', [])
                    if pb and isinstance(pb[0], (int, float)):
                        pb = [pb]  # Single bbox
                    pred_bboxes[img_id] = pb
                else:
                    pred_bboxes[img_id] = []

                if isinstance(ref_boxes[0], (int, float)):
                    ref_boxes = [ref_boxes]
                ref_bboxes[img_id] = ref_boxes

            if ref_bboxes:
                iou_result = self.det_evaluator.evaluate(pred_bboxes, ref_bboxes)
                results[rule_id] = iou_result

        return results

    def evaluate_all(self):
        """Run all evaluations."""
        return {
            'classification': self.multilabel_classification(),
            'bbox_iou': self.bounding_box_iou(),
        }
