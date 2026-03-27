"""
Downstream Detection Module for Data Augmentation Validation

This module provides tools to evaluate data augmentation quality
through downstream object detection tasks.

Models:
- Faster R-CNN with ResNet50-FPN backbone
- Pre-trained on COCO, fine-tuned on construction site data

Training Configurations:
1. original: Only original data
2. original_weather: Original + weather augmentation
3. original_weather_small: Original + weather + small object augmentation

Classes:
- excavator
- rebar
- worker_with_white_hard_hat
"""

from .dataset import (
    ConstructionSiteDataset,
    collate_fn,
    get_transform,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_IDX
)

__all__ = [
    'ConstructionSiteDataset',
    'collate_fn',
    'get_transform',
    'NUM_CLASSES',
    'CLASS_NAMES',
    'CLASS_TO_IDX'
]
