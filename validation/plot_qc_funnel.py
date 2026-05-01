#!/usr/bin/env python3
"""
Plot the image-quality-control funnel for ConSynth-X.

Reads:  validation/results/qc_funnel/funnel_counts.csv
Writes: validation/results/qc_funnel/figures/
  • funnel_exemplar_cs_test_rain_light.pdf  (3-stage funnel)
  • funnel_overview_yields.pdf              (yield per source × condition)
  • funnel_dino_distributions.pdf           (DINO retention curve per CS-test cond)

Run after  validation/build_qc_funnel.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

REPO = Path(__file__).resolve().parents[1]
QC = REPO / "validation" / "results" / "qc_funnel"
FIG = QC / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DINO_DIR = REPO / "validation" / "results" / "dino_ssim"


def read_funnel():
    rows = []
    with open(QC / "funnel_counts.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in list(row.items()):
                if v == "":
                    row[k] = None
            rows.append(row)
    return rows


def to_int(v):
    return int(v) if v not in (None, "") else None


def to_float(v):
    return float(v) if v not in (None, "") else None


# ── Figure 1: 3-stage funnel for CS test rain_light ──
def plot_exemplar(rows):
    target = next((r for r in rows if r["source"] == "cs_test" and r["condition"] == "rain_light"), None)
    if not target:
        print("exemplar row missing")
        return
    n_in   = to_int(target["n_input"])
    n_proc = to_int(target["n_processed_by_filter"])
    n_kept = to_int(target["n_kept"])
    p75 = to_float(target.get("dino_ge_75_pct"))
    n_after_dino = int(round(n_kept * p75 / 100.0)) if p75 else n_kept

    stages = [
        ("Source\n(CS test)",           n_in,   "#cccccc"),
        ("After IP2P\ngeneration",      n_proc, "#88c0d0"),
        ("After SSIM+LPIPS\nfilter",    n_kept, "#5e81ac"),
        (f"After DINO≥0.75\n(audit, post-hoc)", n_after_dino, "#a3be8c"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    max_w = 6.0
    y_step = 1.05
    for i, (label, n, color) in enumerate(stages):
        w = max_w * (n / n_in)
        x = (max_w - w) / 2
        y = -i * y_step
        rect = patches.FancyBboxPatch((x, y), w, 0.85,
                                      boxstyle="round,pad=0.02,rounding_size=0.08",
                                      linewidth=1.2, edgecolor="#222",
                                      facecolor=color)
        ax.add_patch(rect)
        ax.text(max_w / 2, y + 0.42, f"{label}",
                ha="center", va="center", fontsize=10, fontweight="bold")
        pct = n / n_in * 100.0
        ax.text(max_w + 0.15, y + 0.42, f"N = {n:,}  ({pct:.1f}%)",
                ha="left", va="center", fontsize=10, fontfamily="monospace")
        # Drop annotation between stages
        if i > 0:
            n_prev = stages[i - 1][1]
            d = n_prev - n
            d_pct = d / n_prev * 100 if n_prev else 0
            if d > 0:
                ax.text(max_w / 2, y + 0.95, f"↓ drop {d:,} ({d_pct:.1f}%)",
                        ha="center", va="bottom", fontsize=8.5,
                        color="#bf616a", style="italic")

    ax.set_xlim(-0.5, max_w + 3.5)
    ax.set_ylim(-(len(stages) - 0.2) * y_step, 1.0)
    ax.axis("off")
    ax.set_title("Image-quality-control funnel — CS test, rain_light\n"
                 "(SSIM+LPIPS = generation-time filter; DINO≥0.75 = post-hoc audit)",
                 fontsize=11)
    plt.tight_layout()
    out = FIG / "funnel_exemplar_cs_test_rain_light.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=180)
    plt.close()
    print(f"wrote {out}")


RAIN_CONDS = {"rain_light", "rain_heavy"}


# ── Figure 2: overview yields for every (source, condition) ──
def plot_overview(rows):
    # Order: group by source, then by condition
    groups = ["cs_train", "cs_test", "soda_voc", "soda_ktsh"]
    cond_order = ["rain_light", "rain_heavy", "snow_light", "snow_heavy",
                  "fog_light", "fog_medium", "fog_heavy",
                  "night", "night_rain", "night_snow"]

    bars = []  # (label, n_input, n_kept, n_after_dino_75, color, src, filtered)
    color_map = {"cs_train": "#5e81ac", "cs_test": "#81a1c1",
                 "soda_voc": "#a3be8c", "soda_ktsh": "#b48ead"}
    for src in groups:
        for cond in cond_order:
            r = next((x for x in rows if x["source"] == src and x["condition"] == cond), None)
            if r is None:
                continue
            n_in = to_int(r["n_input"])
            n_kept = to_int(r["n_kept"])
            p75 = to_float(r.get("dino_ge_75_pct"))
            n_dino = int(round(n_kept * p75 / 100.0)) if (p75 and n_kept) else None
            filtered = cond in RAIN_CONDS
            bars.append((f"{src}/{cond}", n_in, n_kept, n_dino, color_map[src], src, filtered))

    fig, ax = plt.subplots(figsize=(12, 0.36 * len(bars) + 1.4))
    y = np.arange(len(bars))[::-1]

    max_n = max(b[1] for b in bars)
    for i, (lbl, n_in, n_kept, n_dino, color, src, filtered) in enumerate(bars):
        # Faint background = N_input only meaningful when full source was processed.
        # For unfiltered/subset rows, we still show N_input as reference but mark
        # the bar as a "coverage" stripe rather than a filter funnel.
        ax.barh(y[i], n_in, height=0.78, color=color, alpha=0.15,
                edgecolor=color, linewidth=0.6)
        ax.barh(y[i], n_kept, height=0.78, color=color, alpha=0.78,
                edgecolor=color, linewidth=0.8,
                hatch=None if filtered else "...")
        if n_dino is not None:
            ax.barh(y[i], n_dino, height=0.78, color=color, alpha=1.0,
                    edgecolor="#222", linewidth=0.8, hatch="///")

        # Annotate
        if filtered:
            note = f"  filter retention {n_kept/n_in*100:.1f}%"
        else:
            note = f"  coverage {n_kept/n_in*100:.1f}% (no filter — partial gen)"
        text = f"  {n_kept:,}/{n_in:,}{note}"
        if n_dino is not None:
            text += f"  · DINO≥0.75: {n_dino:,}"
        ax.text(n_in + max_n * 0.005, y[i], text,
                va="center", fontsize=8, fontfamily="monospace")

    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8.5, fontfamily="monospace")
    ax.set_xlabel("Number of images")
    ax.set_title("ConSynth-X QC funnel — solid bars = rain (SSIM+LPIPS filtered);  "
                 "dotted bars = non-rain (no filter, partial generation);  "
                 "hatched ‘///’ overlay = DINO≥0.75 audit", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    handles = [
        patches.Patch(facecolor="#888", alpha=0.15, edgecolor="#888", label="N_input (source)"),
        patches.Patch(facecolor="#888", alpha=0.78, edgecolor="#888",
                      label="N_kept — rain (after SSIM+LPIPS filter)"),
        patches.Patch(facecolor="#888", alpha=0.78, edgecolor="#888", hatch="...",
                      label="N_kept — non-rain (no filter; partial gen)"),
        patches.Patch(facecolor="#888", alpha=1.0, edgecolor="#222", hatch="///",
                      label="DINO≥0.75 (post-hoc audit, all conditions)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    plt.tight_layout()
    out = FIG / "funnel_overview_yields.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=180)
    plt.close()
    print(f"wrote {out}")


# ── Figure 3: DINO retention curves (per CS test condition) ──
DINO_NAME_MAP = {
    ("cs_test", "rain_light"):  "ip2p_rain",
    ("cs_test", "rain_heavy"):  "ip2p_rain_heavy",
    ("cs_test", "snow_light"):  "ip2p_snow_light",
    ("cs_test", "snow_heavy"):  "ip2p_snow_heavy",
    ("cs_test", "fog_light"):   "fog_light",
    ("cs_test", "fog_medium"):  "fog_medium",
    ("cs_test", "fog_heavy"):   "fog_heavy",
    ("cs_test", "night"):       "night",
    ("cs_test", "night_rain"):  "night_rain",
    ("cs_test", "night_snow"):  "night_snow",
    # extended (filled in by extract_dino_ssim_extended.py)
    **{(s, c): f"{s}_{c}"
       for s in ("cs_train", "soda_voc", "soda_ktsh")
       for c in ("rain_light", "rain_heavy", "snow_light", "snow_heavy")},
}


def _read_dino_sims(p: Path):
    sims = []
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                sims.append(float(row["dino_sim"]))
            except (KeyError, ValueError):
                pass
    return np.asarray(sims)


def plot_dino_curves_v2(rows):
    """One panel per source — retention curves stacked for comparison."""
    sources = ["cs_test", "cs_train", "soda_voc", "soda_ktsh"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    thresholds = np.linspace(0.30, 0.95, 27)
    cmap = plt.get_cmap("tab10")
    style = {
        "rain_light": ("-",  0), "rain_heavy": ("--", 0),
        "snow_light": ("-",  1), "snow_heavy": ("--", 1),
        "fog_light":  ("-",  2), "fog_medium": ("--", 2), "fog_heavy": (":", 2),
        "night":      ("-",  3), "night_rain": ("-",  4), "night_snow": ("-",  5),
    }

    for ax, src in zip(axes.flat, sources):
        any_drawn = False
        for r in rows:
            if r["source"] != src:
                continue
            cond = r["condition"]
            key = DINO_NAME_MAP.get((src, cond))
            if not key:
                continue
            p = DINO_DIR / f"{key}.csv"
            if not p.exists():
                continue
            sims = _read_dino_sims(p)
            if len(sims) == 0:
                continue
            retention = np.array([(sims >= t).mean() * 100 for t in thresholds])
            ls, ci = style.get(cond, ("-", 9))
            ax.plot(thresholds, retention, ls, color=cmap(ci),
                    linewidth=1.8, label=cond)
            any_drawn = True

        ax.axvline(0.75, color="#bf616a", linestyle=":", linewidth=1)
        ax.set_title(src, fontsize=11, fontweight="bold")
        ax.set_xlim(0.30, 0.95)
        ax.set_ylim(0, 102)
        ax.grid(linestyle=":", alpha=0.5)
        if any_drawn:
            ax.legend(loc="lower left", fontsize=8, ncol=2)
        else:
            ax.text(0.625, 50, "no DINO data yet", ha="center",
                    va="center", color="#888", fontsize=10, style="italic")

    for ax in axes[1, :]:
        ax.set_xlabel("DINO cosine threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Retention (%)")
    fig.suptitle("DINO retention curves — per source × condition", fontsize=12)
    plt.tight_layout()
    out = FIG / "funnel_dino_distributions.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=180)
    plt.close()
    print(f"wrote {out}")


def main():
    rows = read_funnel()
    plot_exemplar(rows)
    plot_overview(rows)
    plot_dino_curves_v2(rows)


if __name__ == "__main__":
    main()
