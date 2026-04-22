#!/usr/bin/env python3
"""
Quick validation of 20 strong snow samples using:
  1. Weather classifier (SigLIP2) → snow recognition accuracy
  2. Texture fidelity (GLCM/LBP/DCT/Haralick + Dempster-Shafer) → artifact detection

Compares strong (gs=12) vs current (gs=8) configs.
"""

import sys
import json
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import numpy as np
from PIL import Image

SNOW_DIR = (_REPO_ROOT / "validation/results/snow_intensity_test")
OUT_PATH = (_REPO_ROOT / "validation/results/snow_strong_metrics.json")


def load_images(subdir: str):
    d = SNOW_DIR / subdir
    return [Image.open(d / f"{i:03d}.jpg").convert("RGB") for i in range(20)]


def run_weather_classifier(images_dict: dict) -> dict:
    """Classify images with pre-trained weather classifier."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch

    print("\n=== Weather Classifier (SigLIP2) ===")
    model_name = "prithivMLmods/Weather-Image-Classification"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).eval().cuda()

    ID2LABEL = {0: "cloudy/overcast", 1: "foggy/hazy", 2: "rain/storm",
                3: "snow/frosty", 4: "sun/clear"}

    results = {}
    for name, images in images_dict.items():
        preds = []
        confs = []
        for img in images:
            inputs = processor(images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inputs)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            preds.append(ID2LABEL[pred_idx])
            confs.append(float(probs[pred_idx]))

        snow_correct = sum(1 for p in preds if p == "snow/frosty")
        snow_conf = np.mean([c for p, c in zip(preds, confs) if p == "snow/frosty"]) if snow_correct else 0.0

        results[name] = {
            "snow_accuracy": snow_correct / len(images),
            "snow_confidence": float(snow_conf),
            "predictions": preds,
            "mean_confidence": float(np.mean(confs)),
        }
        print(f"  {name:<10} snow_accuracy = {results[name]['snow_accuracy']:.1%}  "
              f"(mean_conf={results[name]['mean_confidence']:.2f})")
        from collections import Counter
        pred_dist = Counter(preds).most_common()
        print(f"             distribution: {pred_dist}")

    return results


def extract_texture_features(images: list) -> dict:
    """Extract 4 texture feature channels per image."""
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    import cv2

    def glcm_features(gray):
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)
        feats = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            feats.extend(graycoprops(glcm, prop).flatten())
        return np.array(feats)  # 24-dim

    def lbp_features(gray):
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        return hist / (hist.sum() + 1e-9)  # 10-dim

    def dct_features(gray):
        dct = cv2.dct(np.float32(gray))
        energies = []
        h, w = dct.shape
        for i, j in [(0, 0), (h//8, 0), (0, w//8), (h//4, w//4), (h//2, w//2)]:
            block = dct[i:i+16, j:j+16]
            energies.append(float(np.sum(block**2)))
        return np.array(energies)  # 5-dim

    all_glcm = []
    all_lbp = []
    all_dct = []
    for img in images:
        gray = np.array(img.convert("L").resize((256, 256)))
        all_glcm.append(glcm_features(gray))
        all_lbp.append(lbp_features(gray))
        all_dct.append(dct_features(gray))

    return {
        "glcm": np.array(all_glcm),
        "lbp": np.array(all_lbp),
        "dct": np.array(all_dct),
    }


def wasserstein_distance(a, b):
    """Multi-dim Wasserstein via mean+variance matching."""
    mean_diff = np.abs(a.mean(axis=0) - b.mean(axis=0))
    std_diff = np.abs(a.std(axis=0) - b.std(axis=0))
    return float(np.mean(mean_diff) + np.mean(std_diff))


def run_texture_fidelity(images_dict: dict) -> dict:
    """Compare texture features of augmented vs original."""
    print("\n=== Texture Fidelity ===")
    feats = {name: extract_texture_features(imgs) for name, imgs in images_dict.items()}

    orig_feats = feats["original"]
    results = {}
    for name in ["current", "strong", "extreme"]:
        if name not in feats:
            continue
        f = feats[name]
        w_glcm = wasserstein_distance(orig_feats["glcm"], f["glcm"])
        w_lbp = wasserstein_distance(orig_feats["lbp"], f["lbp"])
        w_dct = wasserstein_distance(orig_feats["dct"], f["dct"])
        composite = (w_glcm + w_lbp * 100 + w_dct / 1000) / 3

        results[name] = {
            "glcm_wasserstein": w_glcm,
            "lbp_wasserstein": w_lbp,
            "dct_wasserstein": w_dct,
            "composite_distance": composite,
        }
        print(f"  {name:<10} GLCM={w_glcm:>6.2f}  LBP={w_lbp:>6.4f}  DCT={w_dct:>10.0f}")
    return results


def main():
    print("Loading 20 samples from each config...")
    images = {
        "original": load_images("original"),
        "current": load_images("current"),
        "strong": load_images("strong"),
        "extreme": load_images("extreme"),
    }
    print(f"  {len(images)} configs x {len(images['original'])} images")

    # 1. Weather classifier
    cls_results = run_weather_classifier({k: v for k, v in images.items() if k != "original"})
    # Also evaluate original for baseline
    orig_cls = run_weather_classifier({"original": images["original"]})
    cls_results["original"] = orig_cls["original"]

    # 2. Texture fidelity
    tex_results = run_texture_fidelity(images)

    # Save
    output = {
        "weather_classifier": cls_results,
        "texture_fidelity": tex_results,
    }
    # Convert numpy to list for JSON
    def to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json(x) for x in obj]
        return obj

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(to_json(output), f, indent=2)

    print(f"\nResults saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
