"""
OpenAI API model wrapper (GPT-4o, GPT-4o-mini, etc.).
No GPU required — runs via API calls.
"""
import base64
import os
import time

from dotenv import load_dotenv

from .base_model import BaseVLM
from .registry import register_model


class OpenAIBase(BaseVLM):
    """Wrapper for OpenAI vision API models."""

    def __init__(self, model_id, **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self.client = None

    def load(self):
        from openai import OpenAI

        # Load .env from project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(project_root, '.env'))

        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key or api_key == 'your-openai-api-key-here':
            raise ValueError(
                "OPENAI_API_KEY not set. Add your key to "
                f"{os.path.join(project_root, '.env')}"
            )
        self.client = OpenAI(api_key=api_key)
        print(f'[OpenAI] Loaded client for {self.model_id}')

    def _encode_image(self, image):
        """Encode image to base64 data URL."""
        if isinstance(image, str):
            with open(image, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
        else:
            # PIL Image
            import io
            buf = io.BytesIO()
            image.save(buf, format='JPEG')
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    def _call_api(self, messages, max_tokens=1024, temperature=0.2,
                  max_retries=5):
        """Call OpenAI API with retry and exponential backoff."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = min(2 ** (attempt + 1), 60)
                    print(f"  Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

    def generate(self, image, prompt, system_prompt=None, max_new_tokens=1024,
                 temperature=0.2, **kwargs):
        data_url = self._encode_image(image)

        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        return self._call_api(messages, max_tokens=max_new_tokens,
                              temperature=temperature)

    def generate_with_fewshot(self, few_shot_examples, image, prompt,
                               system_prompt=None, max_new_tokens=1024,
                               temperature=0.2, **kwargs):
        user_content = []

        # Few-shot intro
        if kwargs.get('fewshot_intro'):
            user_content.append({"type": "text", "text": kwargs['fewshot_intro'] + "\n\n"})

        # Few-shot examples: (image_path, prompt_text, response_text)
        for fs_img, fs_prompt, fs_response in few_shot_examples:
            data_url = self._encode_image(fs_img)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "low"},
            })
            user_content.append({
                "type": "text",
                "text": f"\nExample:\n\n{fs_response}\n\n",
            })

        # Target prompt + image
        data_url = self._encode_image(image)
        user_content.append({"type": "text", "text": f"\n{prompt}"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data_url, "detail": "high"},
        })

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        return self._call_api(messages, max_tokens=max_new_tokens,
                              temperature=temperature)

    def unload(self):
        self.client = None


@register_model('gpt-4o')
class GPT4o(OpenAIBase):
    def __init__(self, **kwargs):
        super().__init__(model_id='gpt-4o', **kwargs)


@register_model('gpt-4o-mini')
class GPT4oMini(OpenAIBase):
    def __init__(self, **kwargs):
        super().__init__(model_id='gpt-4o-mini', **kwargs)
