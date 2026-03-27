"""
Task: Safety violation VQA (4 rules, 5-shot with bounding boxes).
Follows author methodology exactly.
"""
import os
import json
import re
from .base_task import BaseTask
from .prompts import (
    SYSTEM_PROMPT_INSPECTOR, FEW_SHOT_INTRO, VQA_SAFETY_PROMPT,
    FEWSHOT_ANSWERS, FEWSHOT_IDS,
)


class VQASafetyTask(BaseTask):

    task_name = 'vqa_safety'

    def __init__(self, fewshot_image_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.fewshot_image_dir = fewshot_image_dir

    def get_system_prompt(self):
        return SYSTEM_PROMPT_INSPECTOR

    def get_user_prompt(self, sample):
        return VQA_SAFETY_PROMPT

    def get_few_shot_intro(self):
        return FEW_SHOT_INTRO

    def get_few_shot_examples(self):
        """Return 5-shot examples as (image_path, prompt, response)."""
        if not self.fewshot_image_dir:
            return []
        examples = []
        for fs_id in FEWSHOT_IDS:
            img_path = os.path.join(self.fewshot_image_dir, f'{fs_id}.jpg')
            examples.append((img_path, '', FEWSHOT_ANSWERS[fs_id]))
        return examples

    def uses_fewshot(self):
        return True

    def parse_response(self, response):
        """Parse VQA response into structured dict."""
        text = response.strip()
        # Clean common artifacts
        text = text.replace('```json', '').replace('```', '').strip()

        try:
            result = json.loads(text)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {'raw': text, 'parse_error': True}
