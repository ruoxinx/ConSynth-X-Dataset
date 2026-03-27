"""
Gemma 3 Vision model wrapper.
Supports: google/gemma-3-27b-it
"""
import torch
from PIL import Image
from .base_model import BaseVLM
from .registry import register_model


class Gemma3Base(BaseVLM):
    """Wrapper for Gemma 3 vision-language models."""

    def load(self):
        from transformers import Gemma3ForConditionalGeneration, AutoProcessor

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, image, prompt, system_prompt=None, max_new_tokens=1024,
                 temperature=0.2, **kwargs):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        messages = []
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': [{'type': 'text', 'text': system_prompt}]
            })
        messages.append({
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt},
            ]
        })

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt'
        ).to(self.model.device)

        input_len = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=1.0)
        response = self.processor.decode(
            output[0][input_len:], skip_special_tokens=True)
        return response.strip()

    def generate_with_fewshot(self, few_shot_examples, image, prompt,
                               system_prompt=None, max_new_tokens=1024,
                               temperature=0.2, **kwargs):
        messages = []
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': [{'type': 'text', 'text': system_prompt}]
            })

        # Build user content with few-shot examples and target
        user_content = []

        if kwargs.get('fewshot_intro'):
            user_content.append({'type': 'text', 'text': kwargs['fewshot_intro'] + '\n\n'})

        for fs_img, fs_prompt, fs_response in few_shot_examples:
            if isinstance(fs_img, str):
                fs_img = Image.open(fs_img).convert('RGB')
            user_content.append({'type': 'image', 'image': fs_img})
            user_content.append({'type': 'text', 'text': f'\nExample:\n\n{fs_response}\n\n'})

        # Target image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        user_content.append({'type': 'text', 'text': f'\n{prompt}'})
        user_content.append({'type': 'image', 'image': image})

        messages.append({'role': 'user', 'content': user_content})

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt'
        ).to(self.model.device)

        input_len = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=1.0)
        response = self.processor.decode(
            output[0][input_len:], skip_special_tokens=True)
        return response.strip()


@register_model('gemma-3-27b-it')
class Gemma3_27B(Gemma3Base):
    def __init__(self, **kwargs):
        super().__init__(
            model_id='google/gemma-3-27b-it', **kwargs)
