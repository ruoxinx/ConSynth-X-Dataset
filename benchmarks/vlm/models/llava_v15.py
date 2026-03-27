"""
LLaVA v1.5 model wrapper.
Supports: liuhaotian/llava-v1.5-7b
"""
import torch
from PIL import Image
from .base_model import BaseVLM
from .registry import register_model


class LLaVA15Base(BaseVLM):
    """Wrapper for LLaVA v1.5 models using LlavaForConditionalGeneration."""

    def load(self):
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            low_cpu_mem_usage=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, image, prompt, system_prompt=None, max_new_tokens=1024,
                 temperature=0.2, **kwargs):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        else:
            full_prompt = ""
        full_prompt += f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            text=full_prompt, images=image, return_tensors='pt'
        ).to(self.model.device)

        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, repetition_penalty=1.2)
        response = self.processor.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        return response.strip()

    def generate_with_fewshot(self, few_shot_examples, image, prompt,
                               system_prompt=None, max_new_tokens=1024,
                               temperature=0.2, **kwargs):
        # LLaVA v1.5 only supports single-image input — use zero-shot.
        return self.generate(
            image=image, prompt=prompt, system_prompt=system_prompt,
            max_new_tokens=max_new_tokens, temperature=temperature)


@register_model('llava-v1.5-7b')
class LLaVA15_7B(LLaVA15Base):
    def __init__(self, **kwargs):
        super().__init__(
            model_id='llava-hf/llava-1.5-7b-hf', **kwargs)
