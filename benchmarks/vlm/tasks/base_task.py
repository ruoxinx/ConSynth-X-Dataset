"""
Base class for benchmark tasks.
"""
import abc
import os
import json
import pandas as pd
from PIL import Image


class BaseTask(abc.ABC):
    """Base class for all benchmark tasks."""

    def __init__(self, config=None):
        self.config = config or {}

    @property
    @abc.abstractmethod
    def task_name(self):
        """Unique task identifier."""
        pass

    @abc.abstractmethod
    def get_system_prompt(self):
        """Return the system prompt for this task."""
        pass

    @abc.abstractmethod
    def get_user_prompt(self, sample):
        """Return the user prompt for a specific sample."""
        pass

    @abc.abstractmethod
    def parse_response(self, response):
        """Parse model response into structured output."""
        pass

    def get_few_shot_examples(self):
        """Return list of (image_path, prompt, expected_response) tuples."""
        return []

    def get_few_shot_intro(self):
        """Return intro text for few-shot prompting."""
        return None

    def load_samples_from_tsv(self, tsv_path):
        """Load samples from TSV file (LMUData format)."""
        df = pd.read_csv(tsv_path, sep='\t')
        samples = []
        for _, row in df.iterrows():
            sample = {
                'index': row['index'],
                'image_path': row['image_path'],
                'question': row.get('question', ''),
                'image_id': os.path.splitext(os.path.basename(row['image_path']))[0],
            }
            if 'answer' in row:
                sample['answer'] = row['answer']
            samples.append(sample)
        return samples

    def load_references(self, ref_path):
        """Load reference/ground-truth JSON."""
        with open(ref_path) as f:
            return json.load(f)
