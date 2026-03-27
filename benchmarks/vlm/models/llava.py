"""
LLaVA / Llama Vision model wrappers.
Supports: Llama-3.2-11B-Vision-Instruct, LLaVA-OneVision, etc.
"""
import torch
from PIL import Image
from .base_model import BaseVLM
from .registry import register_model


class LlamaVisionBase(BaseVLM):
    """Wrapper for Llama 3.2 Vision models."""

    def load(self):
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, image, prompt, system_prompt=None, max_new_tokens=1024,
                 temperature=0.2, **kwargs):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        content = []
        if system_prompt:
            content.append({'type': 'text', 'text': system_prompt + '\n\n'})
        content.append({'type': 'image'})
        content.append({'type': 'text', 'text': prompt})

        messages = [{'role': 'user', 'content': content}]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)
        inputs = self.processor(
            image, input_text, return_tensors='pt').to(self.model.device)

        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=1.0)
        response = self.processor.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        return response

    def generate_with_fewshot(self, few_shot_examples, image, prompt,
                               system_prompt=None, max_new_tokens=1024,
                               temperature=0.2, **kwargs):
        images = []
        content_parts = []

        if system_prompt:
            content_parts.append({'type': 'text', 'text': system_prompt + '\n\n'})

        if kwargs.get('fewshot_intro'):
            content_parts.append({'type': 'text', 'text': kwargs['fewshot_intro'] + '\n\n'})

        for fs_img, fs_prompt, fs_response in few_shot_examples:
            if isinstance(fs_img, str):
                fs_img = Image.open(fs_img).convert('RGB')
            images.append(fs_img)
            content_parts.append({'type': 'image'})
            content_parts.append({
                'type': 'text',
                'text': f'\nExample:\n\n{fs_response}\n\n'
            })

        # Target
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        images.append(image)
        content_parts.append({'type': 'text', 'text': f'\n{prompt}'})
        content_parts.append({'type': 'image'})

        messages = [{'role': 'user', 'content': content_parts}]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)
        inputs = self.processor(
            images, input_text, return_tensors='pt').to(self.model.device)

        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=1.0)
        response = self.processor.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        return response


@register_model('llama-3.2-11b-vision')
class Llama32Vision11B(LlamaVisionBase):
    def __init__(self, **kwargs):
        super().__init__(
            model_id='meta-llama/Llama-3.2-11B-Vision-Instruct', **kwargs)
