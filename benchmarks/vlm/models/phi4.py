"""
Phi-4 Multimodal model wrapper.
Supports: microsoft/Phi-4-multimodal-instruct (~5.6B, vision+audio+text)
Requires: trust_remote_code=True, transformers 4.46.x (VLM env)

Phi-4 config defaults to flash_attention_2 which may not be installed.
We override with _attn_implementation='sdpa' (PyTorch native scaled dot-product
attention) which is universally available and performs comparably.
"""
import sys
import torch
from PIL import Image
from .base_model import BaseVLM
from .registry import register_model


def _block_broken_awq():
    """Block awq import if it would crash on missing transformers.models.qwen3.

    The autoawq package in the VLM env references transformers.models.qwen3
    which doesn't exist in transformers 4.46.x.  PEFT's LoRA dispatcher
    tries to import awq unconditionally, causing a crash during Phi-4 init
    (which uses LoRA adapters for vision).  Blocking the import makes
    peft.import_utils.is_auto_awq_available() return False so the
    dispatcher is skipped entirely.
    """
    try:
        import awq  # noqa: F401 — succeeds only if qwen3 module exists
    except (ImportError, ModuleNotFoundError):
        # Mark awq as failed so importlib.util.find_spec returns None
        sys.modules['awq'] = None


def _patch_num_logits_to_keep(model):
    """Fix num_logits_to_keep=None crash on transformers 4.46.x.

    Phi-4's forward() declares num_logits_to_keep: int = 0, but transformers
    4.46.x doesn't know about this parameter. During generate(), the value
    arrives as None via **kwargs, causing ``-None`` to crash at line 2137:
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
    """
    original_forward = model.forward

    def _patched_forward(*args, **kwargs):
        if kwargs.get('num_logits_to_keep') is None:
            kwargs['num_logits_to_keep'] = 0
        return original_forward(*args, **kwargs)

    model.forward = _patched_forward


class Phi4MultimodalBase(BaseVLM):
    """Wrapper for Microsoft Phi-4-multimodal-instruct."""

    def load(self):
        _block_broken_awq()
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            _attn_implementation='sdpa',
        ).to('cuda').eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        _patch_num_logits_to_keep(self.model)

    def _build_inputs(self, images, messages, max_new_tokens=1024,
                      temperature=0.2):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors='pt',
        ).to(self.model.device)
        return inputs

    def generate(self, image, prompt, system_prompt=None,
                 max_new_tokens=1024, temperature=0.2, **kwargs):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({
            'role': 'user',
            'content': '<|image_1|>\n' + prompt,
        })

        inputs = self._build_inputs([image], messages,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature)
        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=1.0,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        response = self.processor.tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        )
        return response.strip()

    def generate_with_fewshot(self, few_shot_examples, image, prompt,
                               system_prompt=None, max_new_tokens=1024,
                               temperature=0.2, **kwargs):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        fewshot_intro = kwargs.get('fewshot_intro', '')
        all_images = []
        img_idx = 1

        for fs_img, _fs_prompt, fs_response in few_shot_examples:
            if isinstance(fs_img, str):
                fs_img = Image.open(fs_img).convert('RGB')
            all_images.append(fs_img)
            user_text = ''
            if img_idx == 1 and fewshot_intro:
                user_text += fewshot_intro + '\n\n'
            user_text += f'<|image_{img_idx}|>\n{prompt}'
            messages.append({'role': 'user', 'content': user_text})
            messages.append({'role': 'assistant', 'content': fs_response})
            img_idx += 1

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        all_images.append(image)
        messages.append({
            'role': 'user',
            'content': f'<|image_{img_idx}|>\n{prompt}',
        })

        inputs = self._build_inputs(all_images, messages,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature)
        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=1.0,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        response = self.processor.tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        )
        return response.strip()


@register_model('phi-4-multimodal')
class Phi4Multimodal(Phi4MultimodalBase):
    def __init__(self, **kwargs):
        super().__init__(
            model_id='microsoft/Phi-4-multimodal-instruct', **kwargs)
