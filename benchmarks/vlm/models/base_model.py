"""
Base class for all VLM model wrappers.
Each HuggingFace model needs a specific wrapper because they have
different image preprocessing, prompt formatting, and output parsing.
"""
import abc
from PIL import Image


class BaseVLM(abc.ABC):
    """Base class for Vision-Language Model wrappers."""

    def __init__(self, model_id, device="cuda", dtype="bfloat16", **kwargs):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

    @abc.abstractmethod
    def load(self):
        """Load model and processor from HuggingFace."""
        pass

    @abc.abstractmethod
    def generate(self, image, prompt, system_prompt=None, max_new_tokens=1024,
                 temperature=0.2, **kwargs):
        """
        Generate response for a single image + prompt.

        Args:
            image: PIL.Image or str (path)
            prompt: str - the user prompt
            system_prompt: optional system instruction
            max_new_tokens: max tokens to generate
            temperature: sampling temperature

        Returns:
            str: generated text
        """
        pass

    @abc.abstractmethod
    def generate_with_fewshot(self, few_shot_examples, image, prompt,
                               system_prompt=None, max_new_tokens=1024,
                               temperature=0.2, **kwargs):
        """
        Generate response with few-shot examples.

        Args:
            few_shot_examples: list of (PIL.Image, str_prompt, str_response)
            image: PIL.Image - the query image
            prompt: str
            system_prompt: optional
            max_new_tokens: int
            temperature: float

        Returns:
            str: generated text
        """
        pass

    def load_image(self, image_path):
        """Load image from path."""
        return Image.open(image_path).convert("RGB")

    def unload(self):
        """Free GPU memory."""
        import torch
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
