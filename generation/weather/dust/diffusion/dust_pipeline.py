#!/usr/bin/env python3
"""
Dust Effect Pipeline - Combines Style Transfer + Dust Particles
Pipeline: Input Image -> Style Transfer (Dust tone) -> Weather Effect (Dust haze + particles) -> Output

Usage:
    python dust_pipeline.py --input image.jpg --style dust_style.jpg
    python dust_pipeline.py --input image.jpg --style dust_style.jpg --intensity heavy --steps 100
"""

import sys
import os
import gc
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Setup path (Weather_Effect_Generator is vendored at generation/weather/libs/Weather_Effect_Generator)
PROJECT_ROOT = Path(__file__).parent.resolve()
WEATHER_GEN_ROOT = PROJECT_ROOT.parents[1] / 'libs' / 'Weather_Effect_Generator'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WEATHER_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(WEATHER_GEN_ROOT))

from lib.style_transfer_utils import (
    tensor2pil,
    load_style_transfer_model,
    run_style_transfer,
    style_content_image_loader,
)
from Dust_Effect_Generator import DustEffectGenerator


# ====== DEPTH ESTIMATION (MiDaS) ======
def load_midas_model(model_name='DPT_Large', device='cpu'):
    """Load MiDaS model once for reuse."""
    dev = torch.device(device)
    print(f'  Loading MiDaS model ({model_name})...')
    model = torch.hub.load('intel-isl/MiDaS', model_name, trust_repo=True)
    model.eval()
    model.to(dev)
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    transform = midas_transforms.dpt_transform if model_name in ['DPT_Large', 'DPT_Hybrid'] else midas_transforms.small_transform
    return model, transform, dev


def estimate_depth_with_model(image_path, midas_model, midas_transform, device):
    """Generate depth map using pre-loaded MiDaS model."""
    img = np.array(Image.open(image_path).convert('RGB'))
    input_batch = midas_transform(img).to(device)
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
    disp = prediction.detach().cpu().numpy()
    disp[disp < 0] = 0
    disp = disp + 1e-3
    baseline, focal = 0.54, 721.09
    depth = baseline * focal / disp
    return depth.astype(np.float32)


def estimate_depth_midas(image_path, model_name='DPT_Large', device='cpu'):
    """Generate depth map using MiDaS model (loads model each time - legacy)."""
    model, transform, dev = load_midas_model(model_name, device)
    return estimate_depth_with_model(image_path, model, transform, dev)


def fake_depth_map(h, w):
    """Generate fake depth map (gradient from near to far)."""
    y = np.linspace(1.0, 80.0, h, dtype=np.float32)[:, None]
    return np.repeat(y, w, axis=1)


# ====== VGG CHECKPOINT RESOLUTION ======
def resolve_vgg_checkpoint(mode, vgg_dir):
    """Find or extract VGG checkpoint for weather mode."""
    if mode == 'imagenet':
        return None
    pth_candidates = sorted(vgg_dir.glob(f'*{mode}*.pth'))
    if pth_candidates:
        print(f'  Found checkpoint: {pth_candidates[0].name}')
        return str(pth_candidates[0])
    zip_candidates = sorted(vgg_dir.glob(f'*{mode}*.zip'))
    if zip_candidates:
        zip_path = zip_candidates[0]
        pth_path = vgg_dir / f'{zip_path.stem}.pth'
        print(f'  Extracting: {zip_path.name} -> {pth_path.name}...')
        state_dict = torch.load(str(zip_path), map_location='cpu', weights_only=False)
        torch.save(state_dict, str(pth_path))
        return str(pth_path)
    print(f'  No checkpoint for "{mode}", using ImageNet VGG19')
    return None


# ====== INTENSITY CONFIG FOR DUST ======
INTENSITY_CONFIG = {
    'none': {
        'visibility_ratio': (1.0, 1.0),
        'darkness': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        'dust_height': (15, 20),
        'particle_density': (0.0, 0.0),
    },
    'light': {
        'visibility_ratio': (0.7, 0.95),
        'darkness': {0: 1.0, 1: 0.97, 2: 0.94, 3: 0.91},
        'dust_height': (15, 20),
        'particle_density': (0.00005, 0.0002),
    },
    'medium': {
        'visibility_ratio': (0.5, 0.8),
        'darkness': {0: 1.0, 1: 0.95, 2: 0.9, 3: 0.85},
        'dust_height': (12, 18),
        'particle_density': (0.0001, 0.0004),
    },
    'heavy': {
        'visibility_ratio': (0.3, 0.5),
        'darkness': {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7},
        'dust_height': (10, 16),
        'particle_density': (0.0003, 0.0006),
    },
    'extreme': {
        'visibility_ratio': (0.15, 0.3),
        'darkness': {0: 1.0, 1: 0.8, 2: 0.65, 3: 0.5},
        'dust_height': (8, 14),
        'particle_density': (0.0005, 0.001),
    },
    'sandstorm': {
        'visibility_ratio': (0.05, 0.15),
        'darkness': {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3},
        'dust_height': (5, 10),
        'particle_density': (0.001, 0.003),
    },
}


# ====== PROCESS IMAGE FUNCTION (for batch processing) ======
def process_image(
    input_path,
    output_path,
    style_image_path,
    steps=50,
    style_weight=100000,
    content_weight=2,
    intensity='medium',
    use_fake_depth=False,
    weather='dust',
    max_size=None,
    midas_model=None,
    midas_transform=None,
    midas_device=None,
    vgg_model=None,
    device='auto'
):
    """
    Process a single image with dust pipeline.
    Can accept pre-loaded models for batch processing efficiency.

    Returns:
        PIL Image of result
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    style_path = Path(style_image_path)

    if device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        DEVICE = device

    current_image = Image.open(input_path).convert('RGB')
    original_size = current_image.size

    # ========== STEP 1: STYLE TRANSFER ==========
    if style_path.exists():
        if vgg_model is None:
            vgg_dir = WEATHER_GEN_ROOT / 'VGG'
            vgg_ckpt = resolve_vgg_checkpoint(weather, vgg_dir)
            vgg_model = load_style_transfer_model(pretrained=vgg_ckpt)
            vgg_model = vgg_model.to(DEVICE).eval()

        content_img, style_img = style_content_image_loader(input_path, style_path, max_size=max_size)
        content_img = content_img.to(DEVICE, torch.float)
        style_img = style_img.to(DEVICE, torch.float)
        input_img = content_img.clone()

        output = run_style_transfer(
            cnn=vgg_model,
            content_img=content_img,
            style_img=style_img,
            input_img=input_img,
            num_steps=steps,
            style_weight=style_weight,
            content_weight=content_weight,
            device=DEVICE,
        )

        current_image = tensor2pil(output[0].detach().cpu())

        if max_size is not None and current_image.size != original_size:
            current_image = current_image.resize(original_size, Image.LANCZOS)

        del content_img, style_img, input_img, output
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # ========== STEP 2: DUST EFFECT ==========
    unique_id = f"{input_path.stem}_{os.getpid()}_{id(current_image)}"
    temp_dir = PROJECT_ROOT / 'generated_pipeline' / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_image_path = temp_dir / f'temp_style_{unique_id}.png'
    current_image.save(temp_image_path, 'PNG')

    img_array = np.array(current_image)
    h, w = img_array.shape[:2]

    if use_fake_depth:
        depth_map = fake_depth_map(h, w)
    else:
        if midas_model is not None and midas_transform is not None:
            depth_map = estimate_depth_with_model(temp_image_path, midas_model, midas_transform, midas_device or DEVICE)
        else:
            depth_map = estimate_depth_midas(temp_image_path, 'DPT_Large', DEVICE)

    temp_depth_path = temp_dir / f'temp_depth_{unique_id}.npy'
    np.save(temp_depth_path, depth_map)

    # Configure dust intensity
    cfg = INTENSITY_CONFIG[intensity]
    d_max = float(depth_map.max())
    ratio_min, ratio_max = cfg['visibility_ratio']
    vis_min = int(d_max * ratio_min)
    vis_max = int(d_max * ratio_max)
    dust_height = np.random.uniform(cfg['dust_height'][0], cfg['dust_height'][1])

    dust_gen = DustEffectGenerator()
    dust_gen._weather2visibility = (max(vis_min, 150), max(vis_max, 300))
    dust_gen._illumination2darkness = cfg['darkness']
    dust_gen._dust_height_range = cfg['dust_height']
    dust_gen._particle_density_ratio = cfg['particle_density']

    dust_result = dust_gen.genEffect(str(temp_image_path), str(temp_depth_path))
    current_image = Image.fromarray(dust_result)

    # Free memory
    del dust_gen, dust_result, depth_map, img_array
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Cleanup temp files
    if temp_image_path.exists():
        temp_image_path.unlink()
    temp_depth_path.unlink(missing_ok=True)
    try:
        temp_dir.rmdir()
    except:
        pass

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if str(output_path).lower().endswith(('.jpg', '.jpeg')):
        current_image.save(output_path, 'JPEG', quality=95)
    else:
        current_image.save(output_path)

    return current_image


def main():
    parser = argparse.ArgumentParser(
        description='Dust Effect Pipeline: Style Transfer + Dust Haze & Particles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python dust_pipeline.py --input image.jpg --style dust_style.jpg

  # Heavy construction dust
  python dust_pipeline.py --input image.jpg --style dust_style.jpg --intensity heavy --steps 100

  # Skip style transfer (only add dust effect)
  python dust_pipeline.py --input image.jpg --skip-style-transfer --intensity medium

  # Sandstorm effect
  python dust_pipeline.py --input image.jpg --style dust_style.jpg --intensity sandstorm
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Path to input image')
    parser.add_argument('--style', '-s', default=None, help='Path to style image (dust reference)')
    parser.add_argument('--output', '-o', default=None, help='Output file path')

    parser.add_argument('--weather', '-w', default='imagenet',
                        choices=['snow', 'rain', 'fog', 'dust', 'imagenet'],
                        help='VGG checkpoint mode (default: imagenet)')
    parser.add_argument('--steps', type=int, default=50, help='Style transfer steps (default: 50)')
    parser.add_argument('--style-weight', type=float, default=100000, help='Style loss weight (default: 100000)')
    parser.add_argument('--content-weight', type=float, default=2, help='Content loss weight (default: 2)')
    parser.add_argument('--max-size', type=int, default=None, help='Max image dimension for style transfer')

    parser.add_argument('--intensity', default='medium',
                        choices=['none', 'light', 'medium', 'heavy', 'extreme', 'sandstorm'],
                        help='Dust intensity (default: medium)')
    parser.add_argument('--midas-model', default='DPT_Large',
                        choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                        help='MiDaS model for depth estimation')
    parser.add_argument('--use-fake-depth', action='store_true',
                        help='Use fake depth map instead of MiDaS')

    parser.add_argument('--skip-style-transfer', action='store_true',
                        help='Skip style transfer, only add dust effect')
    parser.add_argument('--skip-dust-particles', action='store_true',
                        help='Skip dust effect, only do style transfer')
    parser.add_argument('--save-intermediate', action='store_true',
                        help='Save intermediate results')

    parser.add_argument('--device', default='auto', help='Device: auto/cpu/cuda/mps')

    args = parser.parse_args()

    if args.device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        DEVICE = args.device

    print('=' * 70)
    print('DUST EFFECT PIPELINE')
    print('    Style Transfer + Dust Haze & Particles')
    print('=' * 70)
    print(f'Device: {DEVICE}')
    print(f'PyTorch: {torch.__version__}')
    print()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input image not found: {input_path}')

    output_dir = PROJECT_ROOT / 'generated_pipeline'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f'{input_path.stem}-dust-pipeline.jpg'

    print(f'Input: {input_path.name}')
    current_image = Image.open(input_path).convert('RGB')

    # ========== STEP 1: STYLE TRANSFER ==========
    if not args.skip_style_transfer:
        print()
        print('-' * 50)
        print('STEP 1: Style Transfer (Dust tone)')
        print('-' * 50)

        if args.style is None:
            style_dir = PROJECT_ROOT / 'dust_style'
            style_candidates = list(style_dir.glob('*dust*.jpg')) if style_dir.exists() else []
            if not style_candidates:
                style_dir = PROJECT_ROOT.parent
                style_candidates = list(style_dir.glob('*dust*.jpg')) if style_dir.exists() else []
            if style_candidates:
                style_path = style_candidates[0]
                print(f'  Auto-found style image: {style_path.name}')
            else:
                print('  No style image found. Skipping style transfer.')
                args.skip_style_transfer = True

        if not args.skip_style_transfer:
            style_path = Path(args.style) if args.style else style_path
            if not style_path.exists():
                raise FileNotFoundError(f'Style image not found: {style_path}')

            print(f'  Style: {style_path.name}')
            print(f'  Steps: {args.steps} | Style weight: {args.style_weight} | Content weight: {args.content_weight}')

            vgg_dir = WEATHER_GEN_ROOT / 'VGG'
            vgg_ckpt = resolve_vgg_checkpoint(args.weather, vgg_dir)

            print(f'  Loading VGG19 model...')
            cnn = load_style_transfer_model(pretrained=vgg_ckpt)
            cnn = cnn.to(DEVICE).eval()

            original_size = current_image.size
            content_img, style_img = style_content_image_loader(input_path, style_path, max_size=args.max_size)
            content_img = content_img.to(DEVICE, torch.float)
            style_img = style_img.to(DEVICE, torch.float)
            input_img = content_img.clone()
            print(f'  Processing at: {content_img.shape[2]}x{content_img.shape[3]} (original: {original_size[0]}x{original_size[1]})')

            print(f'  Running style transfer ({args.steps} steps)...')
            output = run_style_transfer(
                cnn=cnn,
                content_img=content_img,
                style_img=style_img,
                input_img=input_img,
                num_steps=args.steps,
                style_weight=args.style_weight,
                content_weight=args.content_weight,
                device=DEVICE,
            )

            current_image = tensor2pil(output[0].detach().cpu())
            print('  Style transfer complete!')

            if args.max_size is not None and current_image.size != original_size:
                print(f'  Upscaling to original size: {original_size[0]}x{original_size[1]}')
                current_image = current_image.resize(original_size, Image.LANCZOS)

            if args.save_intermediate:
                intermediate_path = output_dir / f'{input_path.stem}-style-transfer.jpg'
                current_image.save(intermediate_path)
                print(f'  Intermediate saved: {intermediate_path.name}')

            del cnn, content_img, style_img, input_img, output
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

    # ========== STEP 2: DUST EFFECT ==========
    if not args.skip_dust_particles:
        print()
        print('-' * 50)
        print('STEP 2: Dust Effect (Atmospheric Scattering + Particles)')
        print('-' * 50)

        unique_id = f"{input_path.stem}_{os.getpid()}"
        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / f'temp_style_{unique_id}.png'
        current_image.save(temp_image_path, 'PNG')

        img_array = np.array(current_image)
        h, w = img_array.shape[:2]

        print(f'  Generating depth map...')
        if args.use_fake_depth:
            print(f'    Using fake depth (gradient)')
            depth_map = fake_depth_map(h, w)
        else:
            print(f'    Using MiDaS ({args.midas_model})')
            depth_map = estimate_depth_midas(temp_image_path, args.midas_model, DEVICE)

        temp_depth_path = temp_dir / f'temp_depth_{unique_id}.npy'
        np.save(temp_depth_path, depth_map)

        print(f'  Intensity: {args.intensity}')
        cfg = INTENSITY_CONFIG[args.intensity]
        d_max = float(depth_map.max())
        ratio_min, ratio_max = cfg['visibility_ratio']
        vis_min = int(d_max * ratio_min)
        vis_max = int(d_max * ratio_max)
        print(f'    Depth max: {d_max:.1f} | Visibility: ({vis_min}, {vis_max})')

        dust_gen = DustEffectGenerator()
        dust_gen._weather2visibility = (max(vis_min, 150), max(vis_max, 300))
        dust_gen._illumination2darkness = cfg['darkness']
        dust_gen._dust_height_range = cfg['dust_height']
        dust_gen._particle_density_ratio = cfg['particle_density']

        print(f'  Applying dust effect...')
        dust_result = dust_gen.genEffect(str(temp_image_path), str(temp_depth_path))
        current_image = Image.fromarray(dust_result)
        print('  Dust effect applied!')

        if temp_image_path.exists():
            temp_image_path.unlink()
        temp_depth_path.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except:
            pass

    # ========== SAVE OUTPUT ==========
    print()
    print('-' * 50)
    print('OUTPUT')
    print('-' * 50)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if str(output_path).lower().endswith(('.jpg', '.jpeg')):
        current_image.save(output_path, 'JPEG', quality=95)
    else:
        current_image.save(output_path)
    print(f'Final result saved: {output_path}')

    print()
    print('=' * 70)
    print('Pipeline complete!')
    print('=' * 70)


if __name__ == '__main__':
    main()
