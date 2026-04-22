#!/usr/bin/env python3
"""
IP2P v3: natural prompts + subtle physics overlay
- NO forced puddles (let IP2P handle wet ground naturally)
- Rain streaks: lighter, varied density per region, multiple angles
- Snow: lighter flakes, let IP2P do accumulation
"""

import sys, gc, random
from pathlib import Path
import numpy as np
import pyarrow as pa
from PIL import Image
import io, cv2, torch

SCRIPT_DIR = Path(__file__).parent
_GEN_ROOT = next(p for p in SCRIPT_DIR.resolve().parents if (p / '_paths.py').exists())
sys.path.insert(0, str(_GEN_ROOT))
from _paths import DATA_ROOT as BASE_DIR
OUT_DIR = SCRIPT_DIR / 'output_samples' / 'v4_stronger_rain_20'


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()

def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    return Image.open(io.BytesIO(row['bytes'])).convert('RGB')


def add_rain_fog(image: np.ndarray, strength=0.25) -> np.ndarray:
    """Add atmospheric haze from rain — reduces contrast, adds grey tone."""
    h, w = image.shape[:2]
    result = image.astype(np.float64)
    fog_color = np.array([170, 175, 185], dtype=np.float64)
    # Stronger at top (sky), lighter at bottom
    gradient = np.linspace(strength * 1.3, strength * 0.5, h).reshape(h, 1, 1)
    gradient = np.broadcast_to(gradient, (h, w, 3))
    result = result * (1 - gradient) + fog_color * gradient
    return np.clip(result, 0, 255).astype(np.uint8)


def add_natural_rain(image: np.ndarray, intensity: str = 'heavy') -> np.ndarray:
    """Rain streaks at 3 depth layers + atmospheric fog.

    intensity:
      'heavy' (default, unchanged) — current production behavior
      'light' — ~1/3 the streak density, thinner fog haze
      'heavy_fog' — heavy streaks + stronger fog haze + mild blur (reduced visibility)
    """
    h, w = image.shape[:2]

    blur_after = False
    if intensity == 'light':
        fog_strength = random.uniform(0.05, 0.12)
        layers = [
            {'n': random.randint(500, 900),  'len': (8, 18),  'thick': 1,
             'alpha': 0.15, 'y_range': (0, h)},
            {'n': random.randint(300, 600),  'len': (14, 25), 'thick': 1,
             'alpha': 0.20, 'y_range': (0, h)},
            {'n': random.randint(80, 200),   'len': (20, 35), 'thick': 1,
             'alpha': 0.25, 'y_range': (0, h)},
        ]
    elif intensity == 'heavy_fog':
        fog_strength = random.uniform(0.30, 0.45)
        layers = [
            {'n': random.randint(3000, 4500), 'len': (12, 26), 'thick': 1,
             'alpha': 0.35, 'y_range': (0, h)},
            {'n': random.randint(2200, 3600), 'len': (22, 42), 'thick': random.choice([1, 2]),
             'alpha': 0.48, 'y_range': (0, h)},
            {'n': random.randint(900, 1600),  'len': (30, 60), 'thick': random.choice([2, 2, 3]),
             'alpha': 0.55, 'y_range': (0, h)},
        ]
        blur_after = True
    else:
        fog_strength = random.uniform(0.15, 0.30)
        layers = [
            {'n': random.randint(1500, 2500), 'len': (10, 22), 'thick': 1,
             'alpha': 0.25, 'y_range': (0, h)},
            {'n': random.randint(1000, 2000), 'len': (18, 35), 'thick': 1,
             'alpha': 0.35, 'y_range': (0, h)},
            {'n': random.randint(300, 700),   'len': (25, 50), 'thick': random.choice([1, 2]),
             'alpha': 0.40, 'y_range': (0, h)},
        ]

    result = add_rain_fog(image, strength=fog_strength)
    result = result.astype(np.float64)

    base_angle = random.choice([-1, 1]) * random.randint(75, 87)
    wind_var = random.uniform(-5, 5)

    for layer in layers:
        canvas = np.zeros((h, w), dtype=np.float64)
        angle = np.radians(base_angle + random.uniform(-3, 3) + wind_var)

        y_lo, y_hi = layer['y_range']
        for _ in range(layer['n']):
            y = random.randint(y_lo, y_hi - 1)
            x = random.randint(0, w - 1)
            length = random.randint(*layer['len'])
            dx = int(length * np.cos(angle))
            dy = int(length * np.sin(angle))
            cv2.line(canvas, (x, y), (x+dx, y+dy), 255, layer['thick'], cv2.LINE_AA)

        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        a = (canvas / 255.0) * layer['alpha']
        a = np.expand_dims(a, axis=2)
        color = np.array([200 + random.randint(0,20),
                          205 + random.randint(0,15),
                          220 + random.randint(0,15)], dtype=np.float64)
        result = result * (1 - a) + color * a

    result = np.clip(result, 0, 255).astype(np.uint8)
    if blur_after:
        result = cv2.GaussianBlur(result, (5, 5), sigmaX=1.3)
    return result


def add_natural_snow(image: np.ndarray) -> np.ndarray:
    """Subtle falling snowflakes — soft, varied sizes."""
    h, w = image.shape[:2]
    result = image.astype(np.float64)

    layers = [
        {'n': random.randint(1500, 3000), 'r': (1, 2), 'alpha': 0.35, 'blur': 3},
        {'n': random.randint(500, 1000), 'r': (2, 4), 'alpha': 0.50, 'blur': 5},
        {'n': random.randint(100, 300), 'r': (4, 6), 'alpha': 0.65, 'blur': 5},
    ]

    for layer in layers:
        canvas = np.zeros((h, w), dtype=np.float64)
        for _ in range(layer['n']):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            r = random.randint(*layer['r'])
            cv2.circle(canvas, (x, y), r, 255, -1, cv2.LINE_AA)

        k = layer['blur']
        if k % 2 == 0: k += 1
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

        a = (canvas / 255.0) * layer['alpha']
        a = np.expand_dims(a, axis=2)
        color = np.array([240, 245, 255], dtype=np.float64)
        # Screen blend
        result = result + color * a * (1 - result / 255.0)

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_arrow(BASE_DIR / 'LouisChen15___construction_site' / 'construction_site-test.arrow')

    random.seed(77)
    indices = random.sample(range(len(table)), 20)
    indices.sort()

    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    rain_prompt = "a rainy day with dark overcast sky, rain falling, grey clouds"
    snow_prompt = "a cold winter day with snow, frost on surfaces, grey sky, snow on the ground"

    for i, idx in enumerate(indices):
        image_id = table.column('image_id')[idx].as_py()
        img_pil = get_image(table, idx)
        h, w = img_pil.height, img_pil.width

        max_dim = 768
        scale = min(max_dim / max(h, w), 1.0)
        nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
        img_resized = img_pil.resize((nw, nh), Image.LANCZOS)

        print(f"[{i+1}/20] id={image_id} ({w}x{h})")
        img_pil.save(OUT_DIR / f"{i+1:02d}_{image_id}_original.jpg", quality=95)

        # Rain: IP2P (igs=1.5) + fog + streaks
        g = torch.Generator("cuda").manual_seed(42 + idx)
        result = pipe(rain_prompt, image=img_resized,
                      num_inference_steps=30, image_guidance_scale=1.5,
                      guidance_scale=10.0, generator=g).images[0]
        rain_np = np.array(result.resize((w, h), Image.LANCZOS))
        rain_np = add_natural_rain(rain_np)
        Image.fromarray(rain_np).save(OUT_DIR / f"{i+1:02d}_{image_id}_rain.jpg", quality=95)

        # Snow: IP2P (igs=1.5) + subtle flakes only
        g = torch.Generator("cuda").manual_seed(42 + idx)
        result = pipe(snow_prompt, image=img_resized,
                      num_inference_steps=30, image_guidance_scale=1.5,
                      guidance_scale=8.0, generator=g).images[0]
        snow_np = np.array(result.resize((w, h), Image.LANCZOS))
        snow_np = add_natural_snow(snow_np)
        Image.fromarray(snow_np).save(OUT_DIR / f"{i+1:02d}_{image_id}_snow.jpg", quality=95)

        print(f"  Saved")
        del result, rain_np, snow_np; gc.collect(); torch.cuda.empty_cache()

    print(f"\nDone! {OUT_DIR}")


if __name__ == '__main__':
    main()
