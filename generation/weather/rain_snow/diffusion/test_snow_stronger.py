#!/usr/bin/env python3
"""
Test IP2P snow with stronger guidance to produce more visible snow effects.

Current: guidance_scale=8.0, image_guidance_scale=1.5 → too subtle
Test:    guidance_scale=12-15, image_guidance_scale=1.2 → stronger snow

Generates 20 images at 3 intensity levels for comparison.
"""

import os
import sys
import io
import random
from pathlib import Path

_REPO = Path(os.environ.get("CONSYNTH_REPO_ROOT", Path(__file__).resolve().parents[4]))
_DATA = Path(os.environ.get("CONSYNTH_DATA_ROOT", Path.home() / "consynth_data"))

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from physics import add_natural_snow

OUTPUT_DIR = _REPO / "validation" / "results" / "snow_intensity_test"
ARROW_PATH = _DATA / "augmentation_data_arrow" / "construction_site_test.arrow"

# Test configs: (name, guidance_scale, image_guidance_scale, prompt)
CONFIGS = {
    "current": {
        "guidance_scale": 8.0,
        "image_guidance_scale": 1.5,
        "prompt": "a cold winter day with snow, frost on surfaces, grey sky, snow on the ground",
    },
    "strong": {
        "guidance_scale": 12.0,
        "image_guidance_scale": 1.2,
        "prompt": "a cold winter day with heavy snow, thick snow covering the ground and surfaces, grey overcast sky, snowfall",
    },
    "extreme": {
        "guidance_scale": 15.0,
        "image_guidance_scale": 1.0,
        "prompt": "a blizzard with heavy snowfall, everything covered in thick white snow, grey sky, snow accumulation on all surfaces",
    },
}

N_SAMPLES = 20
SEED = 42


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(io.BytesIO(row['bytes'])).convert('RGB')
    return Image.open(io.BytesIO(row)).convert('RGB')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    table = load_arrow(ARROW_PATH)

    # Sample 20 indices deterministically
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(table), size=N_SAMPLES, replace=False)

    # Save originals
    orig_dir = OUTPUT_DIR / "original"
    orig_dir.mkdir(exist_ok=True)
    for i, idx in enumerate(indices):
        img = get_image(table, int(idx))
        img.save(orig_dir / f"{i:03d}.jpg")

    print(f"Saved {N_SAMPLES} originals")

    # Load IP2P
    print("Loading InstructPix2Pix...")
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Generate for each config
    for config_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  guidance_scale={cfg['guidance_scale']}, image_guidance_scale={cfg['image_guidance_scale']}")
        print(f"  prompt: {cfg['prompt'][:80]}...")
        print(f"{'='*60}")

        out_dir = OUTPUT_DIR / config_name
        out_dir.mkdir(exist_ok=True)

        for i, idx in enumerate(indices):
            orig_pil = get_image(table, int(idx))
            h, w = orig_pil.height, orig_pil.width

            max_dim = 768
            scale = min(max_dim / max(h, w), 1.0)
            nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
            img_resized = orig_pil.resize((nw, nh), Image.LANCZOS)

            random.seed(SEED + int(idx))
            g = torch.Generator("cuda").manual_seed(SEED + int(idx))

            result = pipe(
                cfg["prompt"],
                image=img_resized,
                num_inference_steps=30,
                image_guidance_scale=cfg["image_guidance_scale"],
                guidance_scale=cfg["guidance_scale"],
                generator=g,
            ).images[0]

            aug_np = np.array(result.resize((w, h), Image.LANCZOS))
            aug_np = add_natural_snow(aug_np)
            aug_pil = Image.fromarray(aug_np)
            aug_pil.save(out_dir / f"{i:03d}.jpg")

            print(f"  [{i+1}/{N_SAMPLES}] done")

    print(f"\nAll done! Results at: {OUTPUT_DIR}")
    print("Compare: original/ vs current/ vs strong/ vs extreme/")


if __name__ == "__main__":
    main()
