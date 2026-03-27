"""
Task: Rule 1 binary VQA (Yes/No PPE compliance).
"""
from .base_task import BaseTask
from .prompts import SYSTEM_PROMPT_INSPECTOR, VQA_RULE1_PROMPT


class VQARule1Task(BaseTask):

    task_name = 'vqa_rule1'

    def get_system_prompt(self):
        return SYSTEM_PROMPT_INSPECTOR

    def get_user_prompt(self, sample):
        return VQA_RULE1_PROMPT

    def parse_response(self, response):
        """Parse Yes/No response."""
        text = response.strip().lower()
        if 'yes' in text:
            return 'Yes'
        elif 'no' in text:
            return 'No'
        return text

    def uses_fewshot(self):
        return False
