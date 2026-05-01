#!/usr/bin/env python3
"""
Retention charts for ALL augmentation conditions.

Two outputs:
  1. dino_retention_all_conditions.pdf — 1 line per condition, DINO threshold vs retention
  2. retention_grid.pdf                 — 3-panel grid (DINO, SSIM, joint heatmap) per condition
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

IN_DIR = (_REPO_ROOT / "validation/results/dino_ssim")
OUT_DIR = (_REPO_ROOT / "validation/results")

# Display order + colors (grouped by pipeline)
DISPLAY_ORDER = [
    # Style transfer — warm colors (reds)
    ("st_rain_a",       "ST Rain A",        "#fca5a5"),
    ("st_rain_b",       "ST Rain B",        "#f87171"),
    ("st_rain_c",       "ST Rain C",        "#dc2626"),
    ("st_snow_a",       "ST Snow A",        "#fdba74"),
    ("st_snow_b",       "ST Snow B",        "#fb923c"),
    ("st_snow_c",       "ST Snow C",        "#ea580c"),
    # Diffusion — cool colors (blues/greens)
    ("ip2p_rain",       "IP2P Rain (light)", "#60a5fa"),
    ("ip2p_rain_heavy", "IP2P Rain (heavy)", "#1d4ed8"),
    ("ip2p_snow_light", "IP2P Snow (light)", "#93c5fd"),
    ("ip2p_snow_heavy", "IP2P Snow (heavy)", "#2563eb"),
    # Fog — purples
    ("fog_light",       "Fog (light)",      "#c4b5fd"),
    ("fog_medium",      "Fog (medium)",     "#8b5cf6"),
    ("fog_heavy",       "Fog (heavy)",      "#6d28d9"),
    # Night — dark
    ("night",           "Night (CycleGAN)", "#1f2937"),
    # Night + weather (Order B) — pinks/magentas
    ("night_rain",      "Night + Rain",     "#db2777"),
    ("night_snow",      "Night + Snow",     "#f472b6"),
]


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "ssim": float(r["ssim"]),
                "dino_sim": float(r["dino_sim"]),
            })
    return rows


def retention_curve(values, thresholds):
    values = np.asarray(values)
    return np.array([(values >= t).mean() * 100 for t in thresholds])


def main():
    data = {}
    for cond, label, color in DISPLAY_ORDER:
        p = IN_DIR / f"{cond}.csv"
        if not p.exists():
            print(f"  SKIP {cond}: no CSV")
            continue
        rows = load_csv(p)
        data[cond] = {
            "label": label,
            "color": color,
            "ssim": np.array([r["ssim"] for r in rows]),
            "dino": np.array([r["dino_sim"] for r in rows]),
            "n": len(rows),
        }
        print(f"  {cond:<20} {len(rows):>5} images  "
              f"DINO mean={data[cond]['dino'].mean():.3f}  "
              f"SSIM mean={data[cond]['ssim'].mean():.3f}")

    if not data:
        print("No data found.")
        return

    # ─── Chart 1: DINO retention curves ──────────────────────────
    dino_thr = np.linspace(0.2, 0.98, 200)
    ssim_thr = np.linspace(0.1, 0.95, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # DINO panel
    ax = axes[0]
    for cond, info in data.items():
        curve = retention_curve(info["dino"], dino_thr)
        ax.plot(dino_thr, curve, label=info["label"],
                color=info["color"], linewidth=2)
    ax.axvline(0.75, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0.751, 2, "DINO = 0.75", fontsize=8, color="gray")
    ax.set_xlabel("DINO similarity threshold", fontsize=11)
    ax.set_ylabel("Images retained (%)", fontsize=11)
    ax.set_title("Retention vs DINO threshold", fontsize=12, fontweight="bold")
    ax.set_xlim(0.2, 0.98)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.9)

    # SSIM panel
    ax = axes[1]
    for cond, info in data.items():
        curve = retention_curve(info["ssim"], ssim_thr)
        ax.plot(ssim_thr, curve, label=info["label"],
                color=info["color"], linewidth=2)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0.501, 2, "SSIM = 0.5", fontsize=8, color="gray")
    ax.set_xlabel("SSIM threshold", fontsize=11)
    ax.set_ylabel("Images retained (%)", fontsize=11)
    ax.set_title("Retention vs SSIM threshold", fontsize=12, fontweight="bold")
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.9)

    plt.suptitle("Retention curves across augmentation conditions",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    out1 = OUT_DIR / "dino_ssim_retention_all.pdf"
    out1_png = out1.with_suffix(".png")
    plt.savefig(out1, bbox_inches="tight", dpi=150)
    plt.savefig(out1_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nSaved: {out1}")
    print(f"Saved: {out1_png}")

    # ─── Chart 2: summary table at key thresholds ───────────────
    key_dino = [0.70, 0.75, 0.80, 0.85, 0.90]
    key_ssim = [0.3, 0.4, 0.5, 0.6, 0.7]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    cols_dino = [f"≥{t:.2f}" for t in key_dino]
    cols_ssim = [f"≥{t:.1f}" for t in key_ssim]
    rows_label = [data[c]["label"] for c in data]

    mat_dino = np.array([[((data[c]["dino"] >= t).mean() * 100) for t in key_dino]
                         for c in data])
    mat_ssim = np.array([[((data[c]["ssim"] >= t).mean() * 100) for t in key_ssim]
                         for c in data])

    for ax, mat, cols, title in [
        (axes[0], mat_dino, cols_dino, "Retention % at DINO thresholds"),
        (axes[1], mat_ssim, cols_ssim, "Retention % at SSIM thresholds"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=10)
        ax.set_yticks(range(len(rows_label)))
        ax.set_yticklabels(rows_label, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                color = "black" if 30 < v < 80 else "white"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)

    plt.suptitle("Retention rate across conditions and thresholds",
                 fontsize=13, fontweight="bold")
    out2 = OUT_DIR / "dino_ssim_retention_table.pdf"
    out2_png = out2.with_suffix(".png")
    plt.savefig(out2, bbox_inches="tight", dpi=150)
    plt.savefig(out2_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out2}")
    print(f"Saved: {out2_png}")

    # Print summary table to console
    print(f"\n{'='*70}")
    print(f"Retention summary at DINO thresholds:")
    print(f"{'='*70}")
    header = f"{'Condition':<22}" + "".join(f"{c:>8}" for c in cols_dino) + f"{'mean':>8}"
    print(header)
    print("-" * len(header))
    for i, cond in enumerate(data):
        label = data[cond]["label"]
        vals = "".join(f"{mat_dino[i,j]:>7.0f}%" for j in range(len(key_dino)))
        print(f"{label:<22}{vals}  {data[cond]['dino'].mean():>7.3f}")


if __name__ == "__main__":
    main()
