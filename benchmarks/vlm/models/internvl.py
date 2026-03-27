"""
InternVL2.5 model wrapper.
Supports: InternVL2_5-8B, InternVL2_5-26B, etc.
"""
import torch
from PIL import Image
from .base_model import BaseVLM
from .registry import register_model


class InternVLBase(BaseVLM):
    """Wrapper for InternVL2.5 models."""

    def load(self):
        from transformers import AutoModel, AutoTokenizer
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True)

    def _get_device(self):
        """Get the device of the first model parameter (respects device_map)."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device('cuda')

    def _prepare_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        device = self._get_device()
        return self.transform(image).unsqueeze(0).to(torch.bfloat16).to(device)

    def generate(self, image, prompt, system_prompt=None, max_new_tokens=1024,
                 temperature=0.2, **kwargs):
        pixel_values = self._prepare_image(image)
        full_prompt = '<image>\n'
        if system_prompt:
            full_prompt = system_prompt + '\n\n' + full_prompt
        full_prompt += prompt

        gen_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(
            self.tokenizer, pixel_values, full_prompt, gen_config,
            history=None, return_history=False)
        return response

    def generate_with_fewshot(self, few_shot_examples, image, prompt,
                               system_prompt=None, max_new_tokens=1024,
                               temperature=0.2, **kwargs):
        images = []
        prompt_parts = []

        if kwargs.get('fewshot_intro'):
            prompt_parts.append(kwargs['fewshot_intro'] + '\n\n')

        for fs_img, fs_prompt, fs_response in few_shot_examples:
            pixel_values = self._prepare_image(fs_img)
            images.append(pixel_values)
            prompt_parts.append(f'<image>\nExample:\n\n{fs_response}\n\n')

        # Target image
        pixel_values = self._prepare_image(image)
        images.append(pixel_values)
        prompt_parts.append(f'\n{prompt}\n<image>')

        all_pixels = torch.cat(images, dim=0)
        full_prompt = ''.join(prompt_parts)

        gen_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(
            self.tokenizer, all_pixels, full_prompt, gen_config,
            history=None, return_history=False)
        return response

    def unload(self):
        if hasattr(self, 'transform'):
            del self.transform
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        super().unload()


@register_model('internvl2.5-8b')
class InternVL25_8B(InternVLBase):
    def __init__(self, **kwargs):
        super().__init__(model_id='OpenGVLab/InternVL2_5-8B', **kwargs)


@register_model('internvl2.5-26b')
class InternVL25_26B(InternVLBase):
    def __init__(self, **kwargs):
        super().__init__(model_id='OpenGVLab/InternVL2_5-26B-MPO', **kwargs)
