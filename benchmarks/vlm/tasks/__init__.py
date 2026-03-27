from .base_task import BaseTask
from .description import DescriptionTask
from .vqa_safety import VQASafetyTask
from .vqa_rule1 import VQARule1Task
from .object_detection import ObjectDetectionTask

TASK_REGISTRY = {
    'description': DescriptionTask,
    'vqa_safety': VQASafetyTask,
    'vqa_rule1': VQARule1Task,
    'object_detection': ObjectDetectionTask,
}


def get_task(name, **kwargs):
    if name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Task '{name}' not found. Available: {available}")
    return TASK_REGISTRY[name](**kwargs)
