#!/usr/bin/env python3
"""
CLIP-based filter for downloaded construction weather images.

Uses CLIP zero-shot classification to keep only images that are
actually construction sites under adverse weather conditions.

Two-stage filter:
  1. Must be a construction/building site (vs non-construction)
  2. Must show the target weather condition (vs clear weather)

Usage:
  python validation/filter_construction_weather_clip.py
  python validation/filter_construction_weather_clip.py --condition rain
  python validation/filter_construction_weather_clip.py --threshold 0.25 --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Config ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "reference_data" / "construction_weather"

# CLIP prompts for zero-shot classification
CONSTRUCTION_POSITIVE = [
    "a construction site",
    "a building under construction",
    "construction workers on a job site",
    "a crane at a construction site",
    "scaffolding at a building site",
    "an excavator at a construction site",
    "road construction work",
    "a construction zone",
    "heavy machinery at a building site",
]

CONSTRUCTION_NEGATIVE = [
    "a painting or artwork",
    "a sculpture or statue",
    "ancient ruins or archaeological site",
    "a natural landscape without construction",
    "a finished building or architecture",
    "a portrait of a person",
    "food or restaurant",
    "an animal or wildlife",
    "a diagram or chart",
    "text or document",
]

WEATHER_PROMPTS = {
    "rain": {
        "positive": [
            "rain falling on a construction site",
            "a wet construction site in the rain",
            "rainy weather at a building site",
            "construction in heavy rain",
        ],
        "negative": [
            "a construction site on a clear sunny day",
            "dry weather at a construction site",
        ],
    },
    "snow": {
        "positive": [
            "snow at a construction site",
            "a construction site covered in snow",
            "snowy winter weather at a building site",
            "construction in snowfall",
        ],
        "negative": [
            "a construction site on a clear sunny day",
            "summer at a construction site",
        ],
    },
    "fog": {
        "positive": [
            "fog at a construction site",
            "a foggy construction site",
            "misty morning at a building site",
            "construction in haze and fog",
        ],
        "negative": [
            "a construction site on a clear sunny day",
            "clear visibility at a construction site",
        ],
    },
    "night": {
        "positive": [
            "a construction site at night",
            "nighttime construction with lights",
            "dark construction site with floodlights",
            "evening construction work",
        ],
        "negative": [
            "a construction site during daytime",
            "bright daylight at a construction site",
        ],
    },
}


def load_clip(device="cuda"):
    """Load CLIP model."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval().to(device)
    return model, preprocess, tokenizer, device


def encode_texts(model, tokenizer, texts, device):
    """Encode text prompts with CLIP."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def encode_image(model, preprocess, img_path, device):
    """Encode a single image with CLIP."""
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat
    except Exception as e:
        print(f"    Error loading {img_path.name}: {e}")
        return None


def clip_score(image_feat, pos_text_feats, neg_text_feats):
    """Compute relative CLIP score: mean(pos) - mean(neg)."""
    pos_sim = (image_feat @ pos_text_feats.T).mean().item()
    neg_sim = (image_feat @ neg_text_feats.T).mean().item()
    return pos_sim - neg_sim, pos_sim, neg_sim


def filter_condition(condition, data_dir, model, preprocess, tokenizer, device,
                     construction_threshold=0.02, weather_threshold=0.01,
                     dry_run=False):
    """Filter images for one condition using CLIP."""
    cond_dir = data_dir / condition
    if not cond_dir.exists():
        print(f"  {condition}: directory not found, skipping")
        return 0, 0

    # Prepare text embeddings
    cons_pos = encode_texts(model, tokenizer, CONSTRUCTION_POSITIVE, device)
    cons_neg = encode_texts(model, tokenizer, CONSTRUCTION_NEGATIVE, device)

    weather_cfg = WEATHER_PROMPTS[condition]
    weath_pos = encode_texts(model, tokenizer, weather_cfg["positive"], device)
    weath_neg = encode_texts(model, tokenizer, weather_cfg["negative"], device)

    # Setup rejected directory
    reject_dir = cond_dir / "rejected"
    reject_dir.mkdir(exist_ok=True)

    # Load metadata
    metadata_path = cond_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"images": {}}

    # Process images
    image_files = [f for f in cond_dir.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                   and f.is_file()]

    kept = 0
    rejected = 0
    results = []

    print(f"\n  {condition.upper()}: {len(image_files)} images to filter")

    for i, img_path in enumerate(sorted(image_files)):
        feat = encode_image(model, preprocess, img_path, device)
        if feat is None:
            if not dry_run:
                shutil.move(str(img_path), str(reject_dir / img_path.name))
            rejected += 1
            continue

        # Stage 1: Is it a construction site?
        cons_score, cons_pos_sim, cons_neg_sim = clip_score(feat, cons_pos, cons_neg)

        # Stage 2: Does it show the target weather?
        weath_score, weath_pos_sim, weath_neg_sim = clip_score(feat, weath_pos, weath_neg)

        is_construction = cons_score >= construction_threshold
        is_weather = weath_score >= weather_threshold
        keep = is_construction and is_weather

        results.append({
            "file": img_path.name,
            "construction_score": round(cons_score, 4),
            "weather_score": round(weath_score, 4),
            "keep": keep,
        })

        if keep:
            kept += 1
        else:
            rejected += 1
            reason = []
            if not is_construction:
                reason.append(f"not_construction({cons_score:.3f})")
            if not is_weather:
                reason.append(f"not_{condition}({weath_score:.3f})")

            if not dry_run:
                shutil.move(str(img_path), str(reject_dir / img_path.name))

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(image_files)} | kept={kept} rejected={rejected}")

    # Update metadata
    if not dry_run:
        # Remove rejected images from metadata
        rejected_files = {r["file"] for r in results if not r["keep"]}
        for img_id in list(metadata.get("images", {}).keys()):
            info = metadata["images"][img_id]
            if info.get("filename") in rejected_files:
                metadata["images"][img_id]["status"] = "rejected_by_clip"

        metadata["clip_filter"] = {
            "construction_threshold": construction_threshold,
            "weather_threshold": weather_threshold,
            "total_processed": len(image_files),
            "kept": kept,
            "rejected": rejected,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save detailed scores
    scores_path = cond_dir / "clip_scores.json"
    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"    RESULT: {kept} kept, {rejected} rejected")
    print(f"    Scores saved to {scores_path}")
    return kept, rejected


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based filter for construction weather images"
    )
    parser.add_argument("--condition", choices=list(WEATHER_PROMPTS.keys()))
    parser.add_argument("--construction-threshold", type=float, default=0.02,
                        help="Min construction score (default: 0.02)")
    parser.add_argument("--weather-threshold", type=float, default=0.01,
                        help="Min weather score (default: 0.01)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 60)
    print("CLIP Filter: Construction Weather Images")
    print("=" * 60)
    print(f"Data:    {data_dir}")
    print(f"Thresholds: construction={args.construction_threshold}, "
          f"weather={args.weather_threshold}")

    model, preprocess, tokenizer, device = load_clip()

    conditions = [args.condition] if args.condition else list(WEATHER_PROMPTS.keys())
    total_kept = 0
    total_rejected = 0

    for condition in conditions:
        kept, rejected = filter_condition(
            condition, data_dir, model, preprocess, tokenizer, device,
            args.construction_threshold, args.weather_threshold, args.dry_run
        )
        total_kept += kept
        total_rejected += rejected

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_kept} kept, {total_rejected} rejected")
    print(f"{'='*60}")
    if args.dry_run:
        print("(DRY RUN — no files moved)")
    else:
        print("Rejected images moved to 'rejected/' subfolder in each condition.")
        print("You can review and recover any false rejections from there.")


if __name__ == "__main__":
    main()
