"""
Qwen3-VL model wrapper.
Supports: Qwen/Qwen3-VL-8B-Instruct
"""
import torch
from .base_model import BaseVLM
from .registry import register_model


class Qwen3VLBase(BaseVLM):
    """Base wrapper for Qwen3-VL family."""

    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            import flash_attn  # noqa: F401
            attn_impl = 'flash_attention_2'
        except ImportError:
            attn_impl = 'sdpa'
        print(f'[Qwen3-VL] attention: {attn_impl}')

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
            attn_implementation=attn_impl,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )

    def generate(self, image, prompt, system_prompt=None, max_new_tokens=1024,
                 temperature=0.2, **kwargs):
        from qwen_vl_utils import process_vision_info

        if isinstance(image, str):
            image = f'file://{image}'
        else:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            image.save(tmp.name)
            image = f'file://{tmp.name}'

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image', 'image': image},
            ],
        })

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors='pt'
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output_ids[0, inputs['input_ids'].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)

    def generate_with_fewshot(self, few_shot_examples, image, prompt,
                               system_prompt=None, max_new_tokens=1024,
                               temperature=0.2, **kwargs):
        from qwen_vl_utils import process_vision_info

        content_parts = []
        if kwargs.get('fewshot_intro'):
            content_parts.append({'type': 'text', 'text': kwargs['fewshot_intro'] + '\n\n'})

        for fs_img, fs_prompt, fs_response in few_shot_examples:
            if isinstance(fs_img, str):
                img_ref = f'file://{fs_img}'
            else:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                fs_img.save(tmp.name)
                img_ref = f'file://{tmp.name}'

            content_parts.append({
                'type': 'image', 'image': img_ref,
                'resized_height': 280, 'resized_width': 420,
            })
            content_parts.append({
                'type': 'text',
                'text': f'\nExample:\n\n{fs_response}\n\n'
            })

        content_parts.append({'type': 'text', 'text': f'\n{prompt}'})
        if isinstance(image, str):
            img_ref = f'file://{image}'
        else:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            image.save(tmp.name)
            img_ref = f'file://{tmp.name}'
        content_parts.append({'type': 'image', 'image': img_ref})

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': content_parts})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors='pt'
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output_ids[0, inputs['input_ids'].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)


@register_model('qwen3-vl-8b')
class Qwen3VL8B(Qwen3VLBase):
    def __init__(self, **kwargs):
        super().__init__(model_id='Qwen/Qwen3-VL-8B-Instruct', **kwargs)
