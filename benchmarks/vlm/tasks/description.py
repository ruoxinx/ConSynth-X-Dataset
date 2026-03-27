"""
Task: Image captioning / description generation.
"""
from .base_task import BaseTask
from .prompts import SYSTEM_PROMPT_INSPECTOR, DESCRIPTION_PROMPT_DETAILED


class DescriptionTask(BaseTask):

    task_name = 'description'

    def get_system_prompt(self):
        return SYSTEM_PROMPT_INSPECTOR

    def get_user_prompt(self, sample):
        return DESCRIPTION_PROMPT_DETAILED

    def parse_response(self, response):
        # Caption is the raw text
        return response.strip()

    def uses_fewshot(self):
        return False
