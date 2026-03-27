"""
Evaluation for object detection via IoU (Intersection over Union).
Ported from ConstructionSite-10k-Implementation/Evaluations/object_detection/detection_evaluator.py
"""
import json
import numpy as np


class DetectionEvaluator:
    """IoU-based evaluation for bounding box detection."""

    def __init__(self, grid_size=100):
        self.grid_size = grid_size

    def bbox_to_mask(self, bboxes, grid_size=None):
        """Convert list of [x_min, y_min, x_max, y_max] (0-1) to binary mask."""
        gs = grid_size or self.grid_size
        mask = np.zeros((gs, gs), dtype=bool)
        for bbox in bboxes:
            if len(bbox) != 4:
                continue
            x_min, y_min, x_max, y_max = bbox
            # Clamp to [0, 1]
            x_min = max(0, min(1, x_min))
            y_min = max(0, min(1, y_min))
            x_max = max(0, min(1, x_max))
            y_max = max(0, min(1, y_max))
            # Convert to grid coords
            c_min = int(x_min * gs)
            r_min = int(y_min * gs)
            c_max = int(x_max * gs)
            r_max = int(y_max * gs)
            mask[r_min:r_max, c_min:c_max] = True
        return mask

    def compute_iou(self, pred_bboxes, ref_bboxes):
        """Compute IoU between predicted and reference bounding boxes."""
        pred_mask = self.bbox_to_mask(pred_bboxes)
        ref_mask = self.bbox_to_mask(ref_bboxes)

        intersection = np.logical_and(pred_mask, ref_mask).sum()
        union = np.logical_or(pred_mask, ref_mask).sum()

        if union == 0:
            return 0.0
        return intersection / union

    def evaluate(self, predictions, references, positive_only=True):
        """
        Compute macro and micro IoU across all images.

        Args:
            predictions: dict {image_id: list of bboxes}
            references: dict {image_id: list of bboxes}
            positive_only: skip images with no reference objects

        Returns:
            dict with macro_iou, micro_iou
        """
        ious = []
        total_intersection = 0
        total_union = 0

        for img_id, ref_bboxes in references.items():
            if positive_only and not ref_bboxes:
                continue

            pred_bboxes = predictions.get(img_id, [])
            if isinstance(pred_bboxes, dict) and 'parsed' in pred_bboxes:
                pred_bboxes = pred_bboxes['parsed']

            pred_mask = self.bbox_to_mask(pred_bboxes)
            ref_mask = self.bbox_to_mask(ref_bboxes)

            inter = np.logical_and(pred_mask, ref_mask).sum()
            union = np.logical_or(pred_mask, ref_mask).sum()

            total_intersection += inter
            total_union += union

            iou = inter / union if union > 0 else 0.0
            ious.append(iou)

        macro_iou = np.mean(ious) if ious else 0.0
        micro_iou = total_intersection / total_union if total_union > 0 else 0.0

        return {
            'macro_iou': float(macro_iou),
            'micro_iou': float(micro_iou),
            'num_images': len(ious),
        }
