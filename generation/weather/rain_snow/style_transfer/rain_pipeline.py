#!/usr/bin/env python3
"""
Rain Effect Pipeline - Combines Style Transfer + Rain Particles
Pipeline: Input Image → Style Transfer (Rain tone) → Weather Effect (Rain particles) → Output

Usage:
    python rain_pipeline.py --input image.jpg --style rain_style.jpg
    python rain_pipeline.py --input image.jpg --style rain_style.jpg --intensity heavy --steps 100
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Setup path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.style_transfer_utils import (
    tensor2pil,
    load_style_transfer_model,
    run_style_transfer,
    style_content_image_loader,
)
from Rain_Effect_Generator import RainEffectGenerator


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


def estimate_depth_with_model(image_path: Path, midas_model, midas_transform, device):
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
    
    # Convert disparity to depth
    baseline, focal = 0.54, 721.09
    depth = baseline * focal / disp
    return depth.astype(np.float32)


def estimate_depth_midas(image_path: Path, model_name='DPT_Large', device='cpu'):
    """Generate depth map using MiDaS model (loads model each time - legacy)."""
    model, transform, dev = load_midas_model(model_name, device)
    return estimate_depth_with_model(image_path, model, transform, dev)


def fake_depth_map(h: int, w: int):
    """Generate fake depth map (gradient from near to far)."""
    y = np.linspace(1.0, 80.0, h, dtype=np.float32)[:, None]
    return np.repeat(y, w, axis=1)


# ====== VGG CHECKPOINT RESOLUTION ======
def resolve_vgg_checkpoint(mode: str, vgg_dir: Path):
    """Find or extract VGG checkpoint for weather mode."""
    if mode == 'imagenet':
        return None
    
    pth_candidates = sorted(vgg_dir.glob(f'*{mode}*.pth'))
    if pth_candidates:
        print(f'  ✓ Found checkpoint: {pth_candidates[0].name}')
        return str(pth_candidates[0])
    
    zip_candidates = sorted(vgg_dir.glob(f'*{mode}*.zip'))
    if zip_candidates:
        zip_path = zip_candidates[0]
        pth_path = vgg_dir / f'{zip_path.stem}.pth'
        print(f'  📦 Extracting: {zip_path.name} → {pth_path.name}...')
        state_dict = torch.load(str(zip_path), map_location='cpu', weights_only=False)
        torch.save(state_dict, str(pth_path))
        return str(pth_path)
    
    print(f'  ⚠ No checkpoint for "{mode}", using ImageNet VGG19')
    return None


# ====== INTENSITY CONFIG FOR RAIN ======
INTENSITY_CONFIG = {
    'none': {
        'visibility_ratio': (1.0, 1.0),  # No particles visible
        'darkness': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},  # No darkness
    },
    'light': {
        'visibility_ratio': (0.6, 0.9),
        'darkness': {0: 1.0, 1: 0.98, 2: 0.95, 3: 0.92},
    },
    'medium': {
        'visibility_ratio': (0.4, 0.6),
        'darkness': {0: 1.0, 1: 0.95, 2: 0.85, 3: 0.8},
    },
    'heavy': {
        'visibility_ratio': (0.2, 0.4),
        'darkness': {0: 1.0, 1: 0.85, 2: 0.75, 3: 0.65},
    },
    'extreme': {
        'visibility_ratio': (0.1, 0.2),
        'darkness': {0: 1.0, 1: 0.7, 2: 0.55, 3: 0.4},
    },
    'quiet_night': {
        'visibility_ratio': (0.005, 0.05),
        'darkness': {0: 1.0, 1: 0.45, 2: 0.25, 3: 0.1},
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
    weather='rain',
    max_size=None,
    midas_model=None,
    midas_transform=None,
    midas_device=None,
    vgg_model=None,
    device='auto'
):
    """
    Process a single image with rain pipeline.
    Can accept pre-loaded models for batch processing efficiency.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output
        style_image_path: Path to style image
        steps: Style transfer steps
        style_weight: Style loss weight
        content_weight: Content loss weight
        intensity: Rain intensity (light/medium/heavy/extreme/quiet_night)
        use_fake_depth: Use fake depth instead of MiDaS
        weather: VGG checkpoint mode
        max_size: Max image dimension (None = keep original)
        midas_model: Pre-loaded MiDaS model (for batch efficiency)
        midas_transform: Pre-loaded MiDaS transform
        midas_device: MiDaS device
        vgg_model: Pre-loaded VGG model (for batch efficiency)
        device: Device to use (auto/cpu/cuda/mps)
    
    Returns:
        PIL Image of result
    """
    import os
    import gc
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    style_path = Path(style_image_path)
    
    # Detect device
    if device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        DEVICE = device
    
    # Load input image
    current_image = Image.open(input_path).convert('RGB')
    original_size = current_image.size
    
    # ========== STEP 1: STYLE TRANSFER ==========
    if style_path.exists():
        # Load VGG model if not provided
        if vgg_model is None:
            vgg_dir = PROJECT_ROOT / 'VGG'
            vgg_ckpt = resolve_vgg_checkpoint(weather, vgg_dir)
            vgg_model = load_style_transfer_model(pretrained=vgg_ckpt)
            vgg_model = vgg_model.to(DEVICE).eval()
        
        # Load images for style transfer
        content_img, style_img = style_content_image_loader(input_path, style_path, max_size=max_size)
        content_img = content_img.to(DEVICE, torch.float)
        style_img = style_img.to(DEVICE, torch.float)
        input_img = content_img.clone()
        
        # Run style transfer
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
        
        # Upscale back to original size if needed
        if max_size is not None and current_image.size != original_size:
            current_image = current_image.resize(original_size, Image.LANCZOS)
        
        # Clean up after style transfer
        del content_img, style_img, input_img, output
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    # ========== STEP 2: RAIN PARTICLES ==========
    # Create unique temp files
    unique_id = f"{input_path.stem}_{os.getpid()}_{id(current_image)}"
    temp_dir = PROJECT_ROOT / 'generated_pipeline' / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_image_path = temp_dir / f'temp_style_{unique_id}.png'
    current_image.save(temp_image_path, 'PNG')
    
    # Get image dimensions
    img_array = np.array(current_image)
    h, w = img_array.shape[:2]
    
    # Generate depth map
    if use_fake_depth:
        depth_map = fake_depth_map(h, w)
    else:
        # Use pre-loaded MiDaS model if provided
        if midas_model is not None and midas_transform is not None:
            depth_map = estimate_depth_with_model(temp_image_path, midas_model, midas_transform, midas_device or DEVICE)
        else:
            depth_map = estimate_depth_midas(temp_image_path, 'DPT_Large', DEVICE)
    
    # Save depth temporarily
    temp_depth_path = temp_dir / f'temp_depth_{unique_id}.npy'
    np.save(temp_depth_path, depth_map)
    
    # Configure rain intensity
    cfg = INTENSITY_CONFIG[intensity]
    d_max = float(depth_map.max())
    ratio_min, ratio_max = cfg['visibility_ratio']
    vis_min = int(d_max * ratio_min)
    vis_max = int(d_max * ratio_max)
    
    # Create rain generator
    rain_gen = RainEffectGenerator()
    rain_gen._weather2visibility = (vis_min, vis_max)
    rain_gen._illumination2darkness = cfg['darkness']
    
    # Apply rain effect
    rain_result = rain_gen.genEffect(str(temp_image_path), str(temp_depth_path))
    current_image = Image.fromarray(rain_result)
    
    # ===== CRITICAL: Free ALL memory =====
    del rain_gen, rain_result, depth_map, img_array
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
        description='Rain Effect Pipeline: Style Transfer + Rain Particles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python rain_pipeline.py --input image.jpg --style rain_style.jpg
  
  # With custom parameters
  python rain_pipeline.py --input image.jpg --style rain_style.jpg \\
    --steps 100 --style-weight 200000 --intensity heavy
  
  # Skip style transfer (only add rain particles)
  python rain_pipeline.py --input image.jpg --skip-style-transfer --intensity medium
  
  # Skip rain particles (only style transfer)  
  python rain_pipeline.py --input image.jpg --style rain_style.jpg --skip-rain-particles
        """
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True, help='Path to input image')
    parser.add_argument('--style', '-s', default=None, help='Path to style image (rain reference)')
    parser.add_argument('--output', '-o', default=None, help='Output file path (auto-generated if not set)')
    
    # Style Transfer parameters
    parser.add_argument('--weather', '-w', default='rain', choices=['snow', 'rain', 'fog', 'imagenet'],
                       help='Weather mode for VGG checkpoint (default: rain)')
    parser.add_argument('--steps', type=int, default=50, help='Style transfer optimization steps (default: 50)')
    parser.add_argument('--style-weight', type=float, default=100000, help='Style loss weight (default: 100000)')
    parser.add_argument('--content-weight', type=float, default=2, help='Content loss weight (default: 2)')
    parser.add_argument('--max-size', type=int, default=None,
                       help='Max image dimension for style transfer. None=keep original (default: None). Use 640/800 for faster processing.')
    
    # Rain Particles parameters
    parser.add_argument('--intensity', default='medium', 
                       choices=['none', 'light', 'medium', 'heavy', 'extreme', 'quiet_night'],
                       help='Rain particle intensity (default: medium). Use "none" to skip weather particles.')
    parser.add_argument('--midas-model', default='DPT_Large', 
                       choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                       help='MiDaS model for depth estimation (default: DPT_Large)')
    parser.add_argument('--use-fake-depth', action='store_true',
                       help='Use fake depth map instead of MiDaS (faster but less realistic)')
    
    # Pipeline control
    parser.add_argument('--skip-style-transfer', action='store_true',
                       help='Skip style transfer, only add rain particles')
    parser.add_argument('--skip-rain-particles', action='store_true',
                       help='Skip rain particles, only do style transfer')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate results (style transfer output)')
    
    # Device
    parser.add_argument('--device', default='auto', help='Device: auto/cpu/cuda/mps')
    
    args = parser.parse_args()
    
    # Detect device
    if args.device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        DEVICE = args.device
    
    print('=' * 70)
    print('🌧️  RAIN EFFECT PIPELINE')
    print('    Style Transfer + Rain Particles')
    print('=' * 70)
    print(f'Device: {DEVICE}')
    print(f'PyTorch: {torch.__version__}')
    print()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input image not found: {input_path}')
    
    # Setup output
    output_dir = PROJECT_ROOT / 'generated_pipeline'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f'{input_path.stem}-rain-pipeline.jpg'
    
    # Load input image
    print(f'📷 Input: {input_path.name}')
    current_image = Image.open(input_path).convert('RGB')
    
    # ========================================
    # STEP 1: STYLE TRANSFER (Rain tone)
    # ========================================
    if not args.skip_style_transfer:
        print()
        print('─' * 50)
        print('STEP 1: Style Transfer (Rain tone)')
        print('─' * 50)
        
        if args.style is None:
            # Try to find style image
            style_dir = PROJECT_ROOT.parent / 'rain_style'
            style_candidates = list(style_dir.glob('*rain*.jpg')) if style_dir.exists() else []
            if style_candidates:
                style_path = style_candidates[0]
                print(f'  ✓ Auto-found style image: {style_path.name}')
            else:
                raise FileNotFoundError('No style image provided. Use --style <path> or place rain style in rain_style/ folder')
        else:
            style_path = Path(args.style)
            if not style_path.exists():
                raise FileNotFoundError(f'Style image not found: {style_path}')
        
        print(f'  Style: {style_path.name}')
        print(f'  Steps: {args.steps} | Style weight: {args.style_weight} | Content weight: {args.content_weight}')
        
        # Load VGG model
        vgg_dir = PROJECT_ROOT / 'VGG'
        vgg_ckpt = resolve_vgg_checkpoint(args.weather, vgg_dir)
        
        print(f'  Loading VGG19 model...')
        cnn = load_style_transfer_model(pretrained=vgg_ckpt)
        cnn = cnn.to(DEVICE).eval()
        
        # Load images for style transfer (keep original resolution by default)
        original_size = current_image.size  # Save for later upscaling if needed
        content_img, style_img = style_content_image_loader(input_path, style_path, max_size=args.max_size)
        content_img = content_img.to(DEVICE, torch.float)
        style_img = style_img.to(DEVICE, torch.float)
        input_img = content_img.clone()
        print(f'  Processing at: {content_img.shape[2]}x{content_img.shape[3]} (original: {original_size[0]}x{original_size[1]})')
        
        # Run style transfer
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
        print('  ✓ Style transfer complete!')
        
        # Upscale back to original size if we used max_size
        if args.max_size is not None and current_image.size != original_size:
            print(f'  Upscaling to original size: {original_size[0]}x{original_size[1]}')
            current_image = current_image.resize(original_size, Image.LANCZOS)
        
        # Save intermediate if requested
        if args.save_intermediate:
            intermediate_path = output_dir / f'{input_path.stem}-style-transfer.jpg'
            current_image.save(intermediate_path)
            print(f'  📁 Intermediate saved: {intermediate_path.name}')
        
        # Clean up GPU memory
        del cnn, content_img, style_img, input_img, output
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None
    
    # ========================================
    # STEP 2: RAIN PARTICLES (Weather Effect)
    # ========================================
    if not args.skip_rain_particles:
        print()
        print('─' * 50)
        print('STEP 2: Rain Particles (Weather Effect)')
        print('─' * 50)
        
        # Save current image temporarily for rain generator (use PNG to avoid quality loss)
        # Use unique ID to avoid race condition when running multiple jobs in parallel
        import os
        unique_id = f"{input_path.stem}_{os.getpid()}"
        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / f'temp_style_{unique_id}.png'
        current_image.save(temp_image_path, 'PNG')
        
        # Get image dimensions
        img_array = np.array(current_image)
        h, w = img_array.shape[:2]
        
        # Generate or load depth map
        print(f'  Generating depth map...')
        if args.use_fake_depth:
            print(f'    Using fake depth (gradient)')
            depth_map = fake_depth_map(h, w)
        else:
            print(f'    Using MiDaS ({args.midas_model})')
            depth_map = estimate_depth_midas(temp_image_path, args.midas_model, DEVICE)
        
        # Save depth temporarily (with unique ID)
        temp_depth_path = temp_dir / f'temp_depth_{unique_id}.npy'
        np.save(temp_depth_path, depth_map)
        
        # Configure rain intensity
        print(f'  Intensity: {args.intensity}')
        cfg = INTENSITY_CONFIG[args.intensity]
        d_max = float(depth_map.max())
        ratio_min, ratio_max = cfg['visibility_ratio']
        vis_min = int(d_max * ratio_min)
        vis_max = int(d_max * ratio_max)
        print(f'    Depth max: {d_max:.1f} | Visibility: ({vis_min}, {vis_max})')
        
        # Create rain generator with custom parameters
        rain_gen = RainEffectGenerator()
        rain_gen._weather2visibility = (vis_min, vis_max)
        rain_gen._illumination2darkness = cfg['darkness']
        
        # Apply rain effect
        print(f'  Applying rain particles...')
        rain_result = rain_gen.genEffect(str(temp_image_path), str(temp_depth_path))
        current_image = Image.fromarray(rain_result)
        print('  ✓ Rain particles added!')
        
        # Cleanup temp files
        if temp_image_path.exists():
            temp_image_path.unlink()
        temp_depth_path.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except:
            pass
    
    # ========================================
    # SAVE FINAL OUTPUT
    # ========================================
    print()
    print('─' * 50)
    print('OUTPUT')
    print('─' * 50)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save with high quality
    if str(output_path).lower().endswith('.jpg') or str(output_path).lower().endswith('.jpeg'):
        current_image.save(output_path, 'JPEG', quality=95)
    else:
        current_image.save(output_path)
    print(f'✅ Final result saved: {output_path}')
    
    print()
    print('=' * 70)
    print('🌧️  Pipeline complete!')
    print('=' * 70)


if __name__ == '__main__':
    main()
