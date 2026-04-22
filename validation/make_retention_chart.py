#!/usr/bin/env python3
"""
Retention rate chart: DINO + SSIM thresholds for snow_heavy (3004 images).

3-panel figure:
  (a) Keep rate vs DINO threshold (line)
  (b) Keep rate vs SSIM threshold (line)
  (c) Joint heatmap: retention vs (DINO, SSIM)
"""

import csv
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = (_REPO_ROOT / "validation/results/snow_strong_dino_ssim.csv")
OUT_PATH = (_REPO_ROOT / "validation/results/snow_heavy_retention_chart.pdf")


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["ssim"]), float(r["dino_sim"])))
    ssim = np.array([r[0] for r in rows])
    dino = np.array([r[1] for r in rows])
    N = len(rows)
    print(f"Loaded {N} images")
    print(f"  SSIM range: [{ssim.min():.3f}, {ssim.max():.3f}]")
    print(f"  DINO range: [{dino.min():.3f}, {dino.max():.3f}]")

    # Thresholds
    dino_thr = np.linspace(0.3, 0.95, 100)
    ssim_thr = np.linspace(0.2, 0.9, 100)

    keep_dino = np.array([(dino >= t).sum() for t in dino_thr]) / N * 100
    keep_ssim = np.array([(ssim >= t).sum() for t in ssim_thr]) / N * 100

    # Joint heatmap (coarser grid)
    dino_bins = np.linspace(0.3, 0.95, 14)
    ssim_bins = np.linspace(0.2, 0.9, 14)
    grid = np.zeros((len(ssim_bins), len(dino_bins)))
    for i, st in enumerate(ssim_bins):
        for j, dt in enumerate(dino_bins):
            grid[i, j] = ((ssim >= st) & (dino >= dt)).sum() / N * 100

    # ─── Figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})

    # (a) DINO threshold
    ax = axes[0]
    ax.plot(dino_thr, keep_dino, color="#2563eb", linewidth=2.5)
    ax.fill_between(dino_thr, 0, keep_dino, alpha=0.15, color="#2563eb")
    # Highlight 0.75 threshold
    ax.axvline(0.75, color="#dc2626", linestyle="--", alpha=0.6, linewidth=1.5,
               label="DINO = 0.75 (tested)")
    keep_075 = (dino >= 0.75).mean() * 100
    ax.scatter([0.75], [keep_075], color="#dc2626", zorder=5, s=60)
    ax.annotate(f"{keep_075:.0f}%", xy=(0.75, keep_075),
                xytext=(0.77, keep_075 + 5), fontsize=11, color="#dc2626",
                fontweight="bold")
    ax.set_xlabel("DINO similarity threshold", fontsize=11)
    ax.set_ylabel("Images retained (%)", fontsize=11)
    ax.set_title("(a) Retention vs DINO threshold", fontsize=12, fontweight="bold")
    ax.set_xlim(0.3, 0.95)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    # (b) SSIM threshold
    ax = axes[1]
    ax.plot(ssim_thr, keep_ssim, color="#059669", linewidth=2.5)
    ax.fill_between(ssim_thr, 0, keep_ssim, alpha=0.15, color="#059669")
    ax.axvline(0.5, color="#dc2626", linestyle="--", alpha=0.6, linewidth=1.5,
               label="SSIM = 0.5 (prev. filter low)")
    keep_05 = (ssim >= 0.5).mean() * 100
    ax.scatter([0.5], [keep_05], color="#dc2626", zorder=5, s=60)
    ax.annotate(f"{keep_05:.0f}%", xy=(0.5, keep_05),
                xytext=(0.52, keep_05 + 5), fontsize=11, color="#dc2626",
                fontweight="bold")
    ax.set_xlabel("SSIM threshold", fontsize=11)
    ax.set_ylabel("Images retained (%)", fontsize=11)
    ax.set_title("(b) Retention vs SSIM threshold", fontsize=12, fontweight="bold")
    ax.set_xlim(0.2, 0.9)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    # (c) Joint heatmap
    ax = axes[2]
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[dino_bins[0], dino_bins[-1], ssim_bins[0], ssim_bins[-1]],
                   vmin=0, vmax=100)
    # Contour lines for milestone %
    contours = ax.contour(dino_bins, ssim_bins, grid,
                          levels=[25, 50, 75, 90], colors="white",
                          alpha=0.6, linewidths=1.2)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%d%%")
    ax.set_xlabel("DINO similarity threshold", fontsize=11)
    ax.set_ylabel("SSIM threshold", fontsize=11)
    ax.set_title("(c) Joint retention (DINO ≥ x AND SSIM ≥ y)",
                 fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Retention (%)", fontsize=10)

    # Supertitle
    fig.suptitle(
        f"Snow-Heavy Retention Analysis ($N$ = {N} augmented images, no pre-filter)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.savefig(OUT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Saved: {OUT_PATH.with_suffix('.png')}")

    # Print key retention numbers
    print("\n" + "=" * 60)
    print("Key retention points:")
    print("=" * 60)
    print(f"  DINO ≥ 0.70: {(dino >= 0.70).mean()*100:.1f}%")
    print(f"  DINO ≥ 0.75: {(dino >= 0.75).mean()*100:.1f}%")
    print(f"  DINO ≥ 0.80: {(dino >= 0.80).mean()*100:.1f}%")
    print(f"  DINO ≥ 0.85: {(dino >= 0.85).mean()*100:.1f}%")
    print(f"  SSIM ≥ 0.4:  {(ssim >= 0.4).mean()*100:.1f}%")
    print(f"  SSIM ≥ 0.5:  {(ssim >= 0.5).mean()*100:.1f}%")
    print(f"  SSIM ≥ 0.6:  {(ssim >= 0.6).mean()*100:.1f}%")
    print(f"  Joint (DINO ≥ 0.75 & SSIM ≥ 0.5): "
          f"{((dino >= 0.75) & (ssim >= 0.5)).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
