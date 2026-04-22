#!/usr/bin/env python3
"""
Neural Style Transfer Script - Apply weather effects to images using VGG19
Usage:
    python style_transfer.py --input <input_image> --style <style_image> [options]
    python style_transfer.py --input cons.jpg --style night_snow.jpg --weather rain --steps 100
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

import torch
import matplotlib.pyplot as plt

# Setup path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.style_transfer_utils import (
    tensor2pil,
    pil2tensor,
    load_style_transfer_model,
    run_style_transfer,
    style_content_image_loader,
)


def resolve_vgg_checkpoint(mode: str, vgg_dir: Path):
    """Find or extract VGG checkpoint for weather mode."""
    if mode == 'imagenet':
        return None
    
    # Try .pth first (already extracted)
    pth_candidates = sorted(vgg_dir.glob(f'*{mode}*.pth'))
    if pth_candidates:
        print(f'✓ Found extracted checkpoint: {pth_candidates[0].name}')
        return str(pth_candidates[0])
    
    # Try .zip → extract and save as .pth
    zip_candidates = sorted(vgg_dir.glob(f'*{mode}*.zip'))
    if zip_candidates:
        zip_path = zip_candidates[0]
        pth_path = vgg_dir / f'{zip_path.stem}.pth'
        
        print(f'📦 Extracting checkpoint: {zip_path.name} → {pth_path.name}...')
        state_dict = torch.load(str(zip_path), map_location='cpu', weights_only=False)
        torch.save(state_dict, str(pth_path))
        print(f'✓ Saved extracted checkpoint ({pth_path.stat().st_size / 1e6:.1f} MB)')
        return str(pth_path)
    
    print(f'⚠ No checkpoint found for mode "{mode}", using ImageNet VGG19')
    return None


def resolve_style_image(style_path, weather_mode: str, project_root: Path):
    """Resolve style image path. If None, try to find in generated_gan/"""
    if style_path is not None:
        p = Path(style_path)
        if not p.exists():
            raise FileNotFoundError(f'Style image not found: {p}')
        return p
    
    # Fallback: look in generated_gan/
    gan_dir = project_root / 'generated_gan'
    if gan_dir.exists():
        mode_map = {'rain': 'rainy', 'snow': 'snowy', 'fog': 'fog'}
        search_key = mode_map.get(weather_mode, weather_mode)
        candidates = [f for f in gan_dir.glob('*.jpg') if search_key in f.stem.lower()]
        if candidates:
            print(f'✓ Auto-picked style image: {candidates[0].name}')
            return candidates[0]
    
    raise FileNotFoundError(
        f'Style image not found for mode "{weather_mode}".\n'
        f'Please provide --style <path_to_style_image>'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Apply neural style transfer (weather effects) to images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with both images specified
  python style_transfer.py --input cons.jpg --style night_snow.jpg --output result.jpg
  
  # With weather mode and parameters
  python style_transfer.py --input cons.jpg --style rain_effect.jpg \
    --weather rain --steps 100 --style-weight 500000 --content-weight 2
  
  # Using ImageNet VGG (no fine-tuned checkpoint needed)
  python style_transfer.py --input image.jpg --style style.jpg --weather imagenet
  
  # Show preview without saving
  python style_transfer.py --input image.jpg --style style.jpg --preview-only
        """
    )
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Path to content image')
    parser.add_argument('--style', default=None, help='Path to style image (if None, auto-search)')
    
    # Optional parameters
    parser.add_argument('--output', default=None, help='Output file path (default: auto-generated)')
    parser.add_argument('--weather', default='rain', 
                       choices=['rain', 'fog', 'snow', 'imagenet'],
                       help='Weather mode for checkpoint selection')
    parser.add_argument('--steps', type=int, default=50, 
                       help='Number of optimization steps (default: 50)')
    parser.add_argument('--style-weight', type=float, default=100000,
                       help='Style loss weight (higher = stronger style effect, default: 100000)')
    parser.add_argument('--content-weight', type=float, default=2,
                       help='Content loss weight (higher = preserve more structure, default: 2)')
    parser.add_argument('--preview', action='store_true', 
                       help='Show preview instead of saving')
    parser.add_argument('--device', default=None,
                       help='Device: auto/cpu/cuda/mps (default: auto-detect)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display result image after processing')
    
    args = parser.parse_args()
    
    # Detect device
    if args.device == 'auto' or args.device is None:
        DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = args.device
    
    print('=' * 60)
    print('🎨 Neural Style Transfer - Weather Effect Generator')
    print('=' * 60)
    print(f'Device: {DEVICE}')
    print(f'PyTorch version: {torch.__version__}')
    print()
    
    # Validate and resolve paths
    content_path = Path(args.input)
    if not content_path.exists():
        raise FileNotFoundError(f'Content image not found: {content_path}')
    
    vgg_dir = PROJECT_ROOT / 'VGG'
    if not vgg_dir.exists():
        print(f'⚠ VGG directory not found at {vgg_dir}')
    
    vgg_ckpt = resolve_vgg_checkpoint(args.weather, vgg_dir)
    style_path = resolve_style_image(args.style, args.weather, PROJECT_ROOT)
    
    print(f'📷 Content image: {content_path}')
    print(f'🎭 Style image: {style_path}')
    print(f'🔧 Weather mode: {args.weather}')
    print(f'⚙️  VGG checkpoint: {Path(vgg_ckpt).name if vgg_ckpt else "ImageNet pretrained"}')
    print(f'📊 Optimization: {args.steps} steps | Style weight: {args.style_weight} | Content weight: {args.content_weight}')
    print()
    
    # Load model
    print('🔄 Loading VGG19 model...')
    cnn = load_style_transfer_model(pretrained=vgg_ckpt)
    cnn = cnn.to(DEVICE).eval()
    print('✓ Model loaded')
    print()
    
    # Load images
    print('🖼️  Loading and processing images...')
    content_img, style_img = style_content_image_loader(content_path, style_path)
    content_img = content_img.to(DEVICE, torch.float)
    style_img = style_img.to(DEVICE, torch.float)
    input_img = content_img.clone()
    
    print(f'  Content tensor: {content_img.shape}')
    print(f'  Style tensor: {style_img.shape}')
    print()
    
    # Run style transfer
    print(f'⏳ Running style transfer ({args.steps} steps)...')
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
    print('✓ Style transfer complete!')
    print()
    
    result_img = tensor2pil(output[0].detach().cpu())
    
    # Save or preview
    if args.preview:
        print('📋 Showing preview (not saving)...')
        if not args.no_display:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(Image.open(content_path).convert('RGB'))
            axes[0].set_title('Content (Original)')
            axes[0].axis('off')
            axes[1].imshow(Image.open(style_path).convert('RGB'))
            axes[1].set_title(f'Style ({args.weather})')
            axes[1].axis('off')
            axes[2].imshow(result_img)
            axes[2].set_title(f'Result (Style Transfer)')
            axes[2].axis('off')
            plt.tight_layout()
            plt.show()
    else:
        # Auto-generate output path if not provided
        if args.output is None:
            output_dir = PROJECT_ROOT / 'generated_vgg'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'{content_path.stem}-{args.weather}-vgg.jpg'
        else:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_img.save(output_path)
        print(f'✅ Result saved: {output_path}')
        
        # Display if not disabled
        if not args.no_display:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(Image.open(content_path).convert('RGB'))
            axes[0].set_title('Content (Original)')
            axes[0].axis('off')
            axes[1].imshow(Image.open(style_path).convert('RGB'))
            axes[1].set_title(f'Style ({args.weather})')
            axes[1].axis('off')
            axes[2].imshow(result_img)
            axes[2].set_title(f'Result (Style Transfer)')
            axes[2].axis('off')
            plt.tight_layout()
            plt.show()
    
    print()
    print('=' * 60)
    print('✨ Done!')
    print('=' * 60)


if __name__ == '__main__':
    main()
