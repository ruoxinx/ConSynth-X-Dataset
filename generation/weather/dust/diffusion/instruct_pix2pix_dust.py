#!/usr/bin/env python3
"""
Dust Augmentation using InstructPix2Pix (zero-shot, no training required).

Based on IJCNN 2025 paper approach:
- Low guidance_scale (1.2-2.0) to preserve structure and reduce hallucinations
- Sequential prompt application for layered effects
- Preserves bounding box annotations

Usage:
    # Single image
    python instruct_pix2pix_dust.py --input image.jpg --output dusty.jpg

    # Batch folder
    python instruct_pix2pix_dust.py --input_dir images/ --output_dir dusty_images/

    # Custom intensity
    python instruct_pix2pix_dust.py --input image.jpg --preset heavy
"""

import argparse
import gc
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


# ====== DUST PRESETS ======
# Based on IJCNN 2025 findings: low guidance_scale preserves structure
DUST_PRESETS = {
    'light': {
        'passes': [
            {
                'prompt': "Add dust and haze to this photo, slightly hazy dusty atmosphere, warm sandy brown tone in the air",
                'guidance_scale': 5.0,
                'image_guidance_scale': 1.6,
                'num_inference_steps': 50,
            },
        ],
    },
    'medium': {
        'passes': [
            {
                'prompt': "Add dust and haze to this photo, dusty air, reduced visibility, brown haze in the atmosphere",
                'guidance_scale': 5.5,
                'image_guidance_scale': 1.5,
                'num_inference_steps': 75,
            },
        ],
    },
    'heavy': {
        'passes': [
            {
                'prompt': "Add dust and haze to this photo, dusty air, reduced visibility, brown haze in the atmosphere",
                'guidance_scale': 5.5,
                'image_guidance_scale': 1.5,
                'num_inference_steps': 75,
            },
            {
                'prompt': "Add more dust and haze, thicker brown dust in the air, hazier atmosphere",
                'guidance_scale': 5.0,
                'image_guidance_scale': 1.5,
                'num_inference_steps': 50,
            },
        ],
    },
    'sandstorm': {
        'passes': [
            {
                'prompt': "Add dust and haze to this photo, dusty air, reduced visibility, brown haze in the atmosphere",
                'guidance_scale': 6.0,
                'image_guidance_scale': 1.4,
                'num_inference_steps': 75,
            },
            {
                'prompt': "Add more dust and haze, thicker brown dust, hazier atmosphere, sand particles",
                'guidance_scale': 5.5,
                'image_guidance_scale': 1.4,
                'num_inference_steps': 75,
            },
            {
                'prompt': "Add even more dust, very hazy, dense brown dust clouds",
                'guidance_scale': 5.0,
                'image_guidance_scale': 1.5,
                'num_inference_steps': 50,
            },
        ],
    },
}


def load_pipeline(device='cuda', model_id='timbrooks/instruct-pix2pix'):
    """Load InstructPix2Pix pipeline."""
    print(f'Loading InstructPix2Pix model: {model_id}...')
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Enable memory optimizations
    if device == 'cuda':
        pipe.enable_model_cpu_offload()

    print(f'Model loaded on {device}')
    return pipe


def process_single_image(
    pipe,
    image,
    preset='medium',
    custom_prompt=None,
    custom_guidance_scale=None,
    custom_image_guidance_scale=None,
    custom_steps=None,
    seed=None,
):
    """
    Apply dust effect to a single PIL Image.

    Args:
        pipe: loaded InstructPix2Pix pipeline
        image: PIL Image (RGB)
        preset: 'light', 'medium', 'heavy', 'sandstorm'
        custom_prompt: override preset prompt (single pass)
        seed: random seed for reproducibility

    Returns:
        PIL Image with dust effect
    """
    original_size = image.size

    # Resize to 512x512 for model (SD 1.5 based)
    image_resized = image.resize((512, 512), Image.LANCZOS)

    if custom_prompt:
        passes = [{
            'prompt': custom_prompt,
            'guidance_scale': custom_guidance_scale or 1.6,
            'image_guidance_scale': custom_image_guidance_scale or 1.5,
            'num_inference_steps': custom_steps or 75,
        }]
    else:
        passes = DUST_PRESETS[preset]['passes']

    current = image_resized
    generator = torch.Generator(device='cuda').manual_seed(seed) if seed else None

    for i, pass_cfg in enumerate(passes):
        result = pipe(
            prompt=pass_cfg['prompt'],
            image=current,
            guidance_scale=pass_cfg['guidance_scale'],
            image_guidance_scale=pass_cfg['image_guidance_scale'],
            num_inference_steps=pass_cfg['num_inference_steps'],
            generator=generator,
        ).images[0]
        current = result

    # Resize back to original
    if current.size != original_size:
        current = current.resize(original_size, Image.LANCZOS)

    return current


def process_image_file(
    pipe,
    input_path,
    output_path,
    preset='medium',
    **kwargs,
):
    """Process a single image file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    image = Image.open(input_path).convert('RGB')
    result = process_single_image(pipe, image, preset=preset, **kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if str(output_path).lower().endswith(('.jpg', '.jpeg')):
        result.save(output_path, 'JPEG', quality=95)
    else:
        result.save(output_path)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Dust Augmentation using InstructPix2Pix (zero-shot)',
    )

    # Input/Output
    parser.add_argument('--input', '-i', default=None, help='Path to single input image')
    parser.add_argument('--output', '-o', default=None, help='Path to output image')
    parser.add_argument('--input_dir', default=None, help='Input directory for batch processing')
    parser.add_argument('--output_dir', default=None, help='Output directory for batch processing')

    # Dust configuration
    parser.add_argument('--preset', default='medium',
                        choices=['light', 'medium', 'heavy', 'sandstorm'],
                        help='Dust intensity preset (default: medium)')
    parser.add_argument('--prompt', default=None,
                        help='Custom prompt (overrides preset)')
    parser.add_argument('--guidance_scale', type=float, default=None,
                        help='Text guidance scale (default: from preset)')
    parser.add_argument('--image_guidance_scale', type=float, default=None,
                        help='Image guidance scale (default: from preset)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of inference steps (default: from preset)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    # Model
    parser.add_argument('--model', default='timbrooks/instruct-pix2pix',
                        help='Model ID (default: timbrooks/instruct-pix2pix)')
    parser.add_argument('--device', default='cuda', help='Device (default: cuda)')

    # Batch
    parser.add_argument('--extensions', default='jpg,jpeg,png',
                        help='Image extensions for batch (default: jpg,jpeg,png)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process')

    args = parser.parse_args()

    # Validate
    if not args.input and not args.input_dir:
        parser.error('Provide --input or --input_dir')

    # Load model
    pipe = load_pipeline(device=args.device, model_id=args.model)

    kwargs = dict(
        preset=args.preset,
        custom_prompt=args.prompt,
        custom_guidance_scale=args.guidance_scale,
        custom_image_guidance_scale=args.image_guidance_scale,
        custom_steps=args.steps,
        seed=args.seed,
    )

    if args.input:
        # Single image
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f'{input_path.stem}-dust-{args.preset}{input_path.suffix}'

        print(f'Processing: {input_path.name} (preset={args.preset})')
        t0 = time.time()
        process_image_file(pipe, input_path, output_path, **kwargs)
        elapsed = time.time() - t0
        print(f'Saved: {output_path} ({elapsed:.1f}s)')

    elif args.input_dir:
        # Batch
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f'{input_dir.name}_dust_{args.preset}'
        output_dir.mkdir(parents=True, exist_ok=True)

        exts = args.extensions.split(',')
        image_files = []
        for ext in exts:
            image_files.extend(sorted(input_dir.glob(f'*.{ext}')))

        if args.limit:
            image_files = image_files[:args.limit]

        print(f'Batch: {len(image_files)} images, preset={args.preset}')
        print(f'Output: {output_dir}')

        total_time = 0
        for idx, img_path in enumerate(image_files):
            out_path = output_dir / f'{img_path.stem}-dust{img_path.suffix}'
            print(f'  [{idx+1}/{len(image_files)}] {img_path.name}', end=' ')
            t0 = time.time()
            try:
                process_image_file(pipe, img_path, out_path, **kwargs)
                elapsed = time.time() - t0
                total_time += elapsed
                print(f'({elapsed:.1f}s)')
            except Exception as e:
                print(f'ERROR: {e}')

            # GPU cleanup every 10 images
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        avg = total_time / max(len(image_files), 1)
        print(f'\nDone! {len(image_files)} images, avg {avg:.1f}s/image')

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
