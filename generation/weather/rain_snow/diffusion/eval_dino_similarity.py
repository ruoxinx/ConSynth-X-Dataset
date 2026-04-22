#!/usr/bin/env python3
"""
Compute DINO cosine similarity for all augmentation methods tested.
Uses DINOv2-ViT-S/14 patch features — measures semantic/structural preservation
without penalizing desired color/atmosphere changes (unlike SSIM).

Evaluates all saved images from previous tests:
  - Vanilla IP2P (new prompt)
  - ControlNet IP2P
  - Canny CN + img2img
  - OLD prompt IP2P (from prompt_test)
  - SDXL IP2P (from model_test)
  - FLUX Kontext (from flux_test)

Usage:
  python eval_dino_similarity.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

_REPO = Path(os.environ.get("CONSYNTH_REPO_ROOT", Path(__file__).resolve().parents[4]))
_FIG = _REPO / 'paper' / 'figures'

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

OUT_DIR = _FIG / 'dino_eval'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All test directories with their method names
TEST_DIRS = {
    # ControlNet test (A/B/C)
    'controlnet_test': {
        'dir': _FIG / 'controlnet_test',
        'methods': {
            'A) Vanilla IP2P': '_a.jpg',
            'B) ControlNet IP2P': '_b.jpg',
            'C) Canny CN+img2img': '_c.jpg',
        }
    },
    # Prompt test (old vs new)
    'prompt_test': {
        'dir': _FIG / 'prompt_test',
        'methods': {
            'OLD prompt IP2P': '_old.jpg',
            'NEW prompt IP2P': '_new.jpg',
        }
    },
    # Model test (SD1.5 vs SDXL)
    'model_test': {
        'dir': _FIG / 'model_test',
        'methods': {
            'IP2P SD1.5': '_sd15.jpg',
            'SDXL IP2P': '_sdxl.jpg',
        }
    },
    # FLUX test
    'flux_test': {
        'dir': _FIG / 'flux_test',
        'methods': {
            'IP2P SD1.5 (flux cmp)': '_sd15.jpg',
            'FLUX Kontext': '_flux.jpg',
        }
    },
}


def load_dino_model():
    """Load DINOv2-ViT-S/14, patching xformers issue."""
    print('Loading DINOv2-ViT-S/14...')
    # Monkey-patch xformers memory_efficient_attention to use PyTorch native
    try:
        import xformers.ops
        def _native_attention(q, k, v, attn_bias=None, p=0.0, scale=None):
            """Replace xformers attention with native PyTorch scaled_dot_product_attention."""
            # q,k,v shape: (B, N, H, D) -> need (B, H, N, D) for torch
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
            return out.transpose(1, 2)  # back to (B, N, H, D)
        xformers.ops.memory_efficient_attention = _native_attention
        print('  Patched xformers -> native PyTorch attention')
    except ImportError:
        pass

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval().cuda()
    return model


def get_dino_features(model, img_pil, size=518):
    """Extract patch-level features from DINOv2."""
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(img_pil).unsqueeze(0).cuda()
    with torch.no_grad():
        features = model.forward_features(img_t)
        # Use patch tokens (exclude CLS)
        patch_tokens = features['x_norm_patchtokens']  # [1, N, D]
    return patch_tokens


def dino_cosine_similarity(model, img1_pil, img2_pil):
    """Compute cosine similarity between two images using DINOv2 features."""
    feat1 = get_dino_features(model, img1_pil)  # [1, N, D]
    feat2 = get_dino_features(model, img2_pil)  # [1, N, D]

    # Global: average pool then cosine
    global_sim = F.cosine_similarity(feat1.mean(dim=1), feat2.mean(dim=1)).item()

    # Patch-level: mean cosine across all patches
    patch_sim = F.cosine_similarity(feat1.squeeze(0), feat2.squeeze(0), dim=1).mean().item()

    return global_sim, patch_sim


def compute_ssim_quick(img1_pil, img2_pil):
    """Quick SSIM for comparison."""
    from skimage.metrics import structural_similarity as ssim
    a = np.array(img1_pil.resize((256, 256)))
    b = np.array(img2_pil.resize((256, 256)))
    return ssim(a, b, channel_axis=2)


def main():
    model = load_dino_model()

    all_results = []  # (test_name, method, img_id, dino_global, dino_patch, ssim)

    for test_name, config in TEST_DIRS.items():
        test_dir = config['dir']
        if not test_dir.exists():
            print(f'\n  SKIP {test_name}: directory not found')
            continue

        print(f'\n{"="*60}')
        print(f'Test: {test_name}')
        print(f'{"="*60}')

        # Find original images
        orig_files = sorted(test_dir.glob('*_orig.jpg'))
        if not orig_files:
            print(f'  No _orig.jpg files found')
            continue

        for method_name, suffix in config['methods'].items():
            print(f'\n  Method: {method_name}')

            for orig_path in orig_files:
                img_id = orig_path.stem.replace('_orig', '')
                aug_path = test_dir / f'{img_id}{suffix}'

                if not aug_path.exists():
                    continue

                orig_img = Image.open(orig_path).convert('RGB')
                aug_img = Image.open(aug_path).convert('RGB')

                dino_global, dino_patch = dino_cosine_similarity(model, orig_img, aug_img)
                ssim_val = compute_ssim_quick(orig_img, aug_img)

                all_results.append({
                    'test': test_name,
                    'method': method_name,
                    'img_id': img_id,
                    'dino_global': dino_global,
                    'dino_patch': dino_patch,
                    'ssim': ssim_val,
                })

                print(f'    {img_id}: DINO_global={dino_global:.3f}  DINO_patch={dino_patch:.3f}  SSIM={ssim_val:.3f}')

                torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'SUMMARY: All Methods Ranked by DINO Patch Similarity')
    print(f'{"="*60}')

    # Group by method
    method_scores = defaultdict(lambda: {'dino_global': [], 'dino_patch': [], 'ssim': []})
    for r in all_results:
        m = r['method']
        method_scores[m]['dino_global'].append(r['dino_global'])
        method_scores[m]['dino_patch'].append(r['dino_patch'])
        method_scores[m]['ssim'].append(r['ssim'])

    # Sort by mean DINO patch similarity
    ranked = sorted(method_scores.items(), key=lambda x: np.mean(x[1]['dino_patch']), reverse=True)

    print(f'\n{"Method":<30} {"DINO_global":>12} {"DINO_patch":>12} {"SSIM":>8} {"N":>4}')
    print('-' * 70)
    for method, scores in ranked:
        dg = np.mean(scores['dino_global'])
        dp = np.mean(scores['dino_patch'])
        ss = np.mean(scores['ssim'])
        n = len(scores['dino_patch'])
        print(f'{method:<30} {dg:>12.3f} {dp:>12.3f} {ss:>8.3f} {n:>4}')

    # ── Per-sample detail for controlnet test ─────────────────
    print(f'\n{"="*60}')
    print(f'DETAIL: ControlNet Test (per sample)')
    print(f'{"="*60}')
    cn_results = [r for r in all_results if r['test'] == 'controlnet_test']
    if cn_results:
        ids = sorted(set(r['img_id'] for r in cn_results))
        methods_cn = sorted(set(r['method'] for r in cn_results))

        print(f'\n{"ID":<12}', end='')
        for m in methods_cn:
            print(f'  {m[:20]:>22}', end='')
        print()
        print('-' * (12 + 24 * len(methods_cn)))

        for iid in ids:
            print(f'{iid:<12}', end='')
            for m in methods_cn:
                match = [r for r in cn_results if r['img_id'] == iid and r['method'] == m]
                if match:
                    r = match[0]
                    print(f'  D={r["dino_patch"]:.2f} S={r["ssim"]:.2f}', end='')
                else:
                    print(f'  {"N/A":>22}', end='')
            print()

    # ── Visualization: DINO vs SSIM scatter ───────────────────
    print(f'\nGenerating scatter plot...')

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        'A) Vanilla IP2P': '#2d8a4e',
        'B) ControlNet IP2P': '#e74c3c',
        'C) Canny CN+img2img': '#3498db',
        'OLD prompt IP2P': '#e67e22',
        'NEW prompt IP2P': '#27ae60',
        'IP2P SD1.5': '#2ecc71',
        'SDXL IP2P': '#9b59b6',
        'FLUX Kontext': '#e74c3c',
        'IP2P SD1.5 (flux cmp)': '#95a5a6',
    }

    for r in all_results:
        m = r['method']
        c = colors.get(m, '#7f8c8d')
        ax.scatter(r['ssim'], r['dino_patch'], c=c, s=80, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Legend with means
    for method, scores in ranked:
        c = colors.get(method, '#7f8c8d')
        dp = np.mean(scores['dino_patch'])
        ss = np.mean(scores['ssim'])
        ax.scatter([], [], c=c, s=80, label=f'{method} (DINO={dp:.2f}, SSIM={ss:.2f})')

    ax.set_xlabel('SSIM', fontsize=12)
    ax.set_ylabel('DINO Patch Similarity', fontsize=12)
    ax.set_title('DINO vs SSIM: Weather Augmentation Method Comparison\n(Higher DINO = better semantic preservation)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Ideal zone annotation
    ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.5)
    ax.text(0.05, 0.72, 'Good semantic preservation (DINO > 0.7)', fontsize=8, color='green', alpha=0.7)

    plt.tight_layout()
    out_path = OUT_DIR / 'dino_vs_ssim.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.pdf'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
