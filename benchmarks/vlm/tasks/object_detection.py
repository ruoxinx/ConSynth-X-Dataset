"""
Task: Object detection (excavator, workers, rebars).
"""
import json
import re
from .base_task import BaseTask
from .prompts import SYSTEM_PROMPT_INSPECTOR, OBJECT_DETECTION_PROMPTS


class ObjectDetectionTask(BaseTask):

    task_name = 'object_detection'

    def __init__(self, target_object='excavator', **kwargs):
        super().__init__(**kwargs)
        self.target_object = target_object

    def get_system_prompt(self):
        return SYSTEM_PROMPT_INSPECTOR

    def get_user_prompt(self, sample):
        return OBJECT_DETECTION_PROMPTS[self.target_object]

    def parse_response(self, response):
        """Parse bounding box response."""
        text = response.strip()
        text = text.replace('```json', '').replace('```', '').strip()

        if 'none' in text.lower() or 'None' in text:
            return []

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return [result]
        except json.JSONDecodeError:
            # Try to extract bbox patterns
            bbox_pattern = r'\[[\d.,\s]+\]'
            matches = re.findall(bbox_pattern, text)
            bboxes = []
            for m in matches:
                try:
                    bbox = json.loads(m)
                    if len(bbox) == 4:
                        bboxes.append(bbox)
                except json.JSONDecodeError:
                    continue
            return bboxes

    def uses_fewshot(self):
        return False
