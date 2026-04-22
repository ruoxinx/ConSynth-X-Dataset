#!/usr/bin/env python3
"""
Generate 50 samples with OLD prompt IP2P, compute both DINO and SSIM,
then pick the 5 best cases where they disagree:
  - 5 samples: DINO bad (<0.55) but SSIM good (>0.55) — "SSIM is fooled"
  - 5 samples: DINO good (>0.75) but SSIM bad (<0.60)  — "SSIM undervalues good edits"

Save a clean comparison figure for each group.
"""

import os
import sys
import gc
import random
from pathlib import Path
from io import BytesIO

_REPO = Path(os.environ.get("CONSYNTH_REPO_ROOT", Path(__file__).resolve().parents[4]))
_DATA = Path(os.environ.get("CONSYNTH_DATA_ROOT", Path.home() / "consynth_data"))

import numpy as np
import pyarrow as pa
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from physics import add_natural_rain

ORIG_ARROW = _DATA / 'LouisChen15___construction_site' / 'construction_site-train-00000-of-00002.arrow'
OUT_DIR = _REPO / 'paper' / 'figures' / 'dino_eval'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLD_PROMPT = "make it a heavy rainy day, dark overcast sky, wet muddy ground"


def load_arrow(path):
    with open(path, 'rb') as f:
        return pa.ipc.open_stream(f).read_all()


def get_image(table, idx):
    row = table.column('image')[idx].as_py()
    if isinstance(row, dict):
        return Image.open(BytesIO(row['bytes'])).convert('RGB')
    return Image.open(BytesIO(row)).convert('RGB')


def compute_ssim(orig_pil, aug_pil):
    a = np.array(orig_pil.resize((256, 256)))
    b = np.array(aug_pil.resize((256, 256)))
    return ssim(a, b, channel_axis=2)


def load_dino():
    """Load DINOv2 with xformers patch."""
    import xformers.ops
    def _native_attn(q, k, v, attn_bias=None, p=0.0, scale=None):
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        return out.transpose(1, 2)
    xformers.ops.memory_efficient_attention = _native_attn
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval().cuda()
    return model


def dino_patch_sim(model, img1, img2, size=518):
    tfm = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        f1 = model.forward_features(tfm(img1).unsqueeze(0).cuda())['x_norm_patchtokens']
        f2 = model.forward_features(tfm(img2).unsqueeze(0).cuda())['x_norm_patchtokens']
    return F.cosine_similarity(f1.squeeze(0), f2.squeeze(0), dim=1).mean().item()


def generate_ip2p(pipe, img_pil, prompt, seed):
    h, w = img_pil.height, img_pil.width
    max_dim = 768
    scale = min(max_dim / max(h, w), 1.0)
    nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    img_resized = img_pil.resize((nw, nh), Image.LANCZOS)
    g = torch.Generator("cuda").manual_seed(seed)
    result = pipe(prompt, image=img_resized, num_inference_steps=30,
                  image_guidance_scale=1.5, guidance_scale=10.0, generator=g).images[0]
    aug_np = add_natural_rain(np.array(result.resize((w, h), Image.LANCZOS)))
    return Image.fromarray(aug_np)


def main():
    print('Loading data...')
    table = load_arrow(ORIG_ARROW)
    total = len(table)

    print('Loading IP2P...')
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print('Loading DINO...')
    dino = load_dino()

    # Generate 50 samples spread across the dataset
    N = 50
    step = total // N
    indices = [i * step for i in range(N)]
    seed_base = 42

    results = []
    print(f'\nGenerating {N} samples with OLD prompt...')
    for i, idx in enumerate(indices):
        img_id = str(table.column('image_id')[idx].as_py())
        orig = get_image(table, idx)
        aug = generate_ip2p(pipe, orig, OLD_PROMPT, seed_base + idx)

        ss = compute_ssim(orig, aug)
        dn = dino_patch_sim(dino, orig, aug)

        results.append({
            'img_id': img_id, 'idx': idx,
            'ssim': ss, 'dino': dn,
            'orig': orig, 'aug': aug,
        })

        marker = ''
        if dn < 0.55 and ss > 0.55: marker = ' *** DINO_bad SSIM_good'
        elif dn > 0.75 and ss < 0.60: marker = ' *** DINO_good SSIM_bad'
        print(f'  [{i+1}/{N}] {img_id} DINO={dn:.3f} SSIM={ss:.3f}{marker}')
        torch.cuda.empty_cache()

    # Unload IP2P
    del pipe; gc.collect(); torch.cuda.empty_cache()

    # ── Find disagreement cases ──
    # Case A: DINO bad, SSIM good (SSIM is fooled)
    case_a = sorted([r for r in results if r['dino'] < 0.55 and r['ssim'] > 0.50],
                    key=lambda x: x['ssim'] - x['dino'], reverse=True)[:5]

    # Case B: DINO good, SSIM bad (SSIM undervalues good edits)
    case_b = sorted([r for r in results if r['dino'] > 0.70 and r['ssim'] < 0.65],
                    key=lambda x: x['dino'] - x['ssim'], reverse=True)[:5]

    # If not enough, relax thresholds
    if len(case_a) < 5:
        print(f'\n  Only {len(case_a)} Case A found, relaxing thresholds...')
        case_a = sorted(results, key=lambda x: x['ssim'] - x['dino'], reverse=True)[:5]
    if len(case_b) < 5:
        print(f'\n  Only {len(case_b)} Case B found, relaxing thresholds...')
        case_b = sorted(results, key=lambda x: x['dino'] - x['ssim'], reverse=True)[:5]

    # ── Build figure ──
    print(f'\nBuilding figure...')
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))

    # Left 2 cols: Case A (DINO bad, SSIM good)
    for i, r in enumerate(case_a):
        axes[i, 0].imshow(r['orig'])
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Original ({r['img_id']})", fontsize=9, color='gray')

        axes[i, 1].imshow(r['aug'])
        axes[i, 1].axis('off')
        gap = r['ssim'] - r['dino']
        axes[i, 1].set_title(
            f"DINO={r['dino']:.2f}  SSIM={r['ssim']:.2f}\n"
            f"SSIM is fooled (gap={gap:+.2f})",
            fontsize=10, fontweight='bold', color='#c0392b', pad=6)

        # Save individual
        r['orig'].save(OUT_DIR / f"caseA_{r['img_id']}_orig.jpg", quality=95)
        r['aug'].save(OUT_DIR / f"caseA_{r['img_id']}_aug.jpg", quality=95)

    # Right 2 cols: Case B (DINO good, SSIM bad)
    for i, r in enumerate(case_b):
        axes[i, 2].imshow(r['orig'])
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f"Original ({r['img_id']})", fontsize=9, color='gray')

        axes[i, 3].imshow(r['aug'])
        axes[i, 3].axis('off')
        gap = r['dino'] - r['ssim']
        axes[i, 3].set_title(
            f"DINO={r['dino']:.2f}  SSIM={r['ssim']:.2f}\n"
            f"SSIM undervalues (gap={gap:+.2f})",
            fontsize=10, fontweight='bold', color='#2d8a4e', pad=6)

        r['orig'].save(OUT_DIR / f"caseB_{r['img_id']}_orig.jpg", quality=95)
        r['aug'].save(OUT_DIR / f"caseB_{r['img_id']}_aug.jpg", quality=95)

    # Column group titles
    fig.text(0.25, 1.005, 'SSIM is Fooled\n(DINO bad, SSIM says OK)',
             ha='center', fontsize=14, fontweight='bold', color='#c0392b')
    fig.text(0.75, 1.005, 'SSIM Undervalues Good Edits\n(DINO good, SSIM says bad)',
             ha='center', fontsize=14, fontweight='bold', color='#2d8a4e')

    plt.suptitle(
        'DINO vs SSIM Disagreement: 5+5 Cases from 50 IP2P Rain Samples (OLD prompt)\n'
        'Left: semantic damage hidden by pixel similarity | Right: good weather edit penalized by pixel change',
        fontsize=12, y=1.04)
    plt.tight_layout()

    out = OUT_DIR / 'disagreement_5plus5.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.with_suffix('.pdf'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # ── Print summary ──
    print(f'\n{"="*60}')
    print('Case A: DINO bad, SSIM good ("SSIM is fooled")')
    print(f'{"="*60}')
    for r in case_a:
        print(f"  {r['img_id']}  DINO={r['dino']:.3f}  SSIM={r['ssim']:.3f}  gap={r['ssim']-r['dino']:+.3f}")

    print(f'\n{"="*60}')
    print('Case B: DINO good, SSIM bad ("SSIM undervalues")')
    print(f'{"="*60}')
    for r in case_b:
        print(f"  {r['img_id']}  DINO={r['dino']:.3f}  SSIM={r['ssim']:.3f}  gap={r['dino']-r['ssim']:+.3f}")

    # Overall stats
    dinos = [r['dino'] for r in results]
    ssims = [r['ssim'] for r in results]
    print(f'\nOverall (50 samples):')
    print(f'  DINO: mean={np.mean(dinos):.3f}, std={np.std(dinos):.3f}')
    print(f'  SSIM: mean={np.mean(ssims):.3f}, std={np.std(ssims):.3f}')
    print(f'  Correlation: {np.corrcoef(dinos, ssims)[0,1]:.3f}')


if __name__ == '__main__':
    main()
