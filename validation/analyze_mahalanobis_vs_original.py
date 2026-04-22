#!/usr/bin/env python3
"""
Analyze relative Mahalanobis results vs original clear-day baseline.

For each augmentation condition, reports:
  • Baseline (real ACDC vs ACDC)    — ideal reference, expected ≈ 0
  • Original (clear-day vs target)  — untreated baseline, expected far from real
  • Aug                              — augmented images after DINO filtering
  • Gap to base                      — remaining distance from aug to real
  • Closure %                        — (aug - original) / (baseline - original)
                                        0% = no improvement, 100% = matches real
  • Δ vs orig                        — + = closer to real than clear-day

Usage:
  python validation/analyze_mahalanobis_vs_original.py \
      --json validation/results/relative_mahalanobis/dino_ge_0p8/relative_mahalanobis_results.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default="validation/results/relative_mahalanobis/dino_ge_0p8/"
                "relative_mahalanobis_results.json",
    )
    args = parser.parse_args()

    data = json.load(open(args.json))
    orig = data.get("original")
    if orig is None:
        raise SystemExit("'original' condition missing from JSON — re-run script")

    # Augmentation → target ACDC condition
    target_map = {
        "weather_style_rain_0":  "rain",
        "weather_style_rain_1":  "rain",
        "weather_style_rain_2":  "rain",
        "weather_style_snow_0":  "snow",
        "weather_style_snow_1":  "snow",
        "weather_style_snow_2":  "snow",
        "diffusion_rain":        "rain",
        "diffusion_snow_light":  "snow",
        "diffusion_snow_heavy":  "snow",
        "diffusion_fog_light":   "fog",
        "diffusion_fog_medium":  "fog",
        "diffusion_fog_heavy":   "fog",
        "night":                 "night",
    }

    print("=" * 100)
    print(f"Source: {args.json}")
    print("Closeness to real ACDC weather — higher -d_rel (closer to 0) = more realistic")
    print("=" * 100)

    for emb in ("clip", "dinov3"):
        # Detect whether this embedding is present
        sample_key = f"baseline_fog_{emb}"
        if sample_key not in orig:
            continue

        print(f"\n### {emb.upper()} ###")
        header = (f"{'Condition':<24} {'Tgt':>5} "
                  f"{'Baseline':>10} {'Original':>10} {'Aug':>10} "
                  f"{'GapBase':>9} {'Closure':>8} {'Δvs orig':>10} {'N':>6}")
        print(header)
        print("-" * len(header))

        # Group by target for nice ordering
        for tgt_order in ("rain", "snow", "fog", "night"):
            for aug_name, tgt in target_map.items():
                if tgt != tgt_order or aug_name not in data:
                    continue
                aug_entry = data[aug_name].get(f"d_rel_{tgt}_{emb}")
                if aug_entry is None:
                    continue

                baseline = orig[f"baseline_{tgt}_{emb}"]["neg_d_rel_mean"]
                original = orig[f"d_rel_{tgt}_{emb}"]["neg_d_rel_mean"]
                aug_val = aug_entry["neg_d_rel_mean"]
                n = aug_entry["n_images"]

                gap_base = baseline - aug_val
                total = baseline - original
                pct = 100 * (aug_val - original) / total if total else 0.0
                delta = aug_val - original

                print(f"{aug_name:<24} {tgt:>5} "
                      f"{baseline:>10.2f} {original:>10.2f} {aug_val:>10.2f} "
                      f"{gap_base:>9.2f} {pct:>7.1f}% {delta:>+10.2f} {n:>6}")

    print()
    print("  Baseline = real ACDC vs ACDC  (lý tưởng, gần 0)")
    print("  Original = ảnh clear-day vs target weather (chưa augment)")
    print("  Aug      = augmentation đã qua filter DINO ≥ threshold")
    print("  Gap      = baseline − aug (residual distance to real)")
    print("  Closure  = (aug − original) / (baseline − original) — % khoảng cách đã đóng")


if __name__ == "__main__":
    main()
