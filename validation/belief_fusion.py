#!/usr/bin/env python3
"""
Dempster-Shafer Belief Theory fusion for texture fidelity scores.

Takes the 4 Wasserstein distance scores (GLCM, LBP, DCT, Haralick) from
compute_texture_fidelity.py and fuses them into a global fidelity assessment
with uncertainty quantification and conflict detection.

Implements the multi-criteria combination from:
  Duminil, Ieng & Gruyer (2025). "Fidelity assessment of synthetic images
  with multi-criteria combination under adverse weather conditions."
  Scientific Reports. DOI: 10.1038/s41598-025-15480-0

Key equations:
  - BBA generation: Eq. 14 (BBFs Φ₁, Φ₂ with α₀ reliability, τ threshold)
  - CRC combination: Eq. 20-23 (generalized conjunctive rule for N sources)
  - Output: H (fidelity), H̄ (not fidelity), Ω (uncertainty), ∅ (conflict)

Usage:
  # Run on existing texture fidelity results
  python validation/belief_fusion.py

  # With custom parameters
  python validation/belief_fusion.py --tau 0.6 --alpha-default 0.8
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────
TEXTURE_RESULTS = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/validation/results/texture_fidelity/texture_fidelity_results.json")
OUT_DIR = Path("/users/PGS0407/binben14/VietHuy/ConSynth-X/validation/results/texture_fidelity")

# 4 criteria (observations) matching paper's 4 processing channels
CRITERIA = ["glcm", "lbp", "dct", "haralick"]
CRITERIA_LABELS = ["GLCM", "LBP", "DCT", "Haralick"]


# ══════════════════════════════════════════════════════════════════
# STEP 1: Wasserstein → Similarity Score [0, 1]
# ══════════════════════════════════════════════════════════════════

def wasserstein_to_similarity(w_scores: dict, ref_ranges: dict) -> dict:
    """Convert Wasserstein distances to similarity scores in [0, 1].

    Uses exponential decay: Sc = exp(-λ * W) where λ is calibrated per
    feature type so that the median observed distance maps to ~0.5.

    This replaces the CNN prediction scores (Sc_k) in the paper.
    The CNN outputs [0, 100] accuracy; we output [0, 1] similarity.
    """
    similarity = {}
    for feat, w in w_scores.items():
        # λ = ln(2) / median_distance → median maps to 0.5
        median_d = ref_ranges[feat]
        lam = np.log(2) / max(median_d, 1e-10)
        similarity[feat] = float(np.exp(-lam * w))
    return similarity


# ══════════════════════════════════════════════════════════════════
# STEP 2: Generate BBAs (Basic Belief Assignments)
# ══════════════════════════════════════════════════════════════════

def generate_bba(sc: float, alpha_0: float, tau: float) -> dict:
    """Generate BBA triplet {m(H), m(H̄), m(Ω)} from a similarity score.

    Implements Eq. 14 from Duminil et al. (2025):
      - H: "image is faithful to reality"
      - H̄: "image is NOT faithful to reality"
      - Ω: uncertainty (ignorance)

    BBFs (Basic Belief Functions):
      Φ₁(α₀, Sc) = α₀ * (Sc - τ) / (1 - τ)   for Sc > τ  (supports H)
      Φ₂(α₀, Sc) = α₀ * (τ - Sc) / τ           for Sc < τ  (supports H̄)

    Args:
        sc: similarity score in [0, 1] (higher = more similar to real)
        alpha_0: reliability coefficient in (0, 1]
        tau: threshold dividing H/H̄ regions (τ > 0.5 = pessimistic)

    Returns:
        dict with keys 'H', 'not_H', 'omega' summing to 1.0
    """
    sc = np.clip(sc, 0.0, 1.0)

    if sc > tau:
        # Score supports fidelity hypothesis H
        phi1 = alpha_0 * (sc - tau) / (1.0 - tau) if tau < 1.0 else 0.0
        m_H = phi1
        m_not_H = 0.0
        m_omega = 1.0 - phi1
    elif sc < tau:
        # Score supports non-fidelity hypothesis H̄
        phi2 = alpha_0 * (tau - sc) / tau if tau > 0.0 else 0.0
        m_H = 0.0
        m_not_H = phi2
        m_omega = 1.0 - phi2
    else:
        # Exactly at threshold — maximum uncertainty
        m_H = 0.0
        m_not_H = 0.0
        m_omega = 1.0

    return {"H": m_H, "not_H": m_not_H, "omega": m_omega}


# ══════════════════════════════════════════════════════════════════
# STEP 3: Conjunctive Rule of Combination (CRC)
# ══════════════════════════════════════════════════════════════════

def combine_bbas_crc(bbas: list) -> dict:
    """Combine N BBAs using the Conjunctive Rule of Combination.

    Implements Eq. 20-23 from Duminil et al. (2025):
      m(H)  = ∏(1 - m_j(H̄)) - ∏(m_j(Ω))           (Eq. 20)
      m(H̄) = ∏(1 - m_j(H))  - ∏(m_j(Ω))           (Eq. 21)
      m(Ω)  = ∏(m_j(Ω))                              (Eq. 22)
      m(∅)  = 1 - m(H) - m(H̄) - m(Ω)  [conflict]   (Eq. 23)

    Args:
        bbas: list of BBA dicts, each with keys 'H', 'not_H', 'omega'

    Returns:
        dict with 'H' (fidelity), 'not_H' (not fidelity),
        'omega' (uncertainty), 'conflict' (empty set)
    """
    n = len(bbas)

    # Product terms
    prod_1_minus_not_H = 1.0
    prod_1_minus_H = 1.0
    prod_omega = 1.0

    for bba in bbas:
        prod_1_minus_not_H *= (1.0 - bba["not_H"])
        prod_1_minus_H *= (1.0 - bba["H"])
        prod_omega *= bba["omega"]

    # Combined masses (Eq. 20-23)
    m_H = prod_1_minus_not_H - prod_omega
    m_not_H = prod_1_minus_H - prod_omega
    m_omega = prod_omega
    m_conflict = 1.0 - m_H - m_not_H - m_omega

    # Numerical safety
    m_H = max(0.0, m_H)
    m_not_H = max(0.0, m_not_H)
    m_omega = max(0.0, m_omega)
    m_conflict = max(0.0, m_conflict)

    return {
        "H": float(m_H),
        "not_H": float(m_not_H),
        "omega": float(m_omega),
        "conflict": float(m_conflict),
    }


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def plot_belief_results(all_belief: dict, output_dir: Path):
    """Generate paper-style visualizations: bar plots + radar charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    conditions = [c for c in all_belief if not c.startswith("acdc_") and c != "original"]
    if not conditions:
        return

    # ── Figure 1: BBA bar plot + radar chart per condition (like Fig. 10/11) ──
    n_conds = len(conditions)
    n_cols = min(3, n_conds)
    n_rows = (n_conds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, cond in enumerate(conditions):
        row = idx // n_cols
        col = idx % n_cols
        belief = all_belief[cond]

        # ── Left: BBA bar plot per criterion ──
        ax_bar = axes[row, col * 2]
        x = np.arange(len(CRITERIA))
        width = 0.25

        h_vals = [belief["per_criterion"][c]["H"] for c in CRITERIA]
        nh_vals = [belief["per_criterion"][c]["not_H"] for c in CRITERIA]
        om_vals = [belief["per_criterion"][c]["omega"] for c in CRITERIA]

        ax_bar.bar(x - width, h_vals, width, label="H (fidelity)", color="#2ecc71")
        ax_bar.bar(x, nh_vals, width, label="not H", color="#e74c3c")
        ax_bar.bar(x + width, om_vals, width, label="Ω (uncertainty)", color="#3498db")

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(CRITERIA_LABELS, fontsize=8)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_ylabel("Mass", fontsize=8)
        short_name = cond.replace("weather_style_", "style_").replace("diffusion_", "diff_")
        ax_bar.set_title(f"{short_name}\nBBA per criterion", fontsize=9, fontweight="bold")
        if idx == 0:
            ax_bar.legend(fontsize=6, loc="upper right")

        # ── Right: Radar chart (H, not_H, Ω, Conflict) ──
        ax_radar = fig.add_subplot(n_rows, n_cols * 2, row * (n_cols * 2) + col * 2 + 2,
                                    polar=True)
        # Remove the default subplot
        axes[row, col * 2 + 1].set_visible(False)

        categories = ["H\n(Fidelity)", "not H\n(Not Fidelity)", "Ω\n(Uncertainty)", "Conflict"]
        combined = belief["combined"]
        values = [combined["H"], combined["not_H"], combined["omega"], combined["conflict"]]
        values += values[:1]  # close the polygon

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax_radar.plot(angles, values, "o-", linewidth=2, markersize=5, color="#2c3e50")
        ax_radar.fill(angles, values, alpha=0.2, color="#3498db")
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=7)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
        ax_radar.set_title(f"{short_name}\nCombined", fontsize=9, fontweight="bold", pad=15)

    # Hide unused axes
    for idx in range(n_conds, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col * 2].set_visible(False)

    fig.suptitle("Dempster-Shafer Belief Fusion: Per-Criterion BBAs + Combined Assessment\n"
                 "(H = faithful texture, not H = artifact detected, Ω = uncertain, Conflict = criteria disagree)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "belief_bba_per_condition.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "belief_bba_per_condition.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: belief_bba_per_condition.png")

    # ── Figure 2: Summary comparison (all conditions) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: stacked bar chart
    ax = axes[0]
    cond_labels = [c.replace("weather_style_", "style_").replace("diffusion_", "diff_")
                   for c in conditions]

    h_vals = [all_belief[c]["combined"]["H"] for c in conditions]
    nh_vals = [all_belief[c]["combined"]["not_H"] for c in conditions]
    om_vals = [all_belief[c]["combined"]["omega"] for c in conditions]
    cf_vals = [all_belief[c]["combined"]["conflict"] for c in conditions]

    # Sort by H (fidelity) descending
    sort_idx = np.argsort(h_vals)[::-1]
    cond_labels = [cond_labels[i] for i in sort_idx]
    h_vals = [h_vals[i] for i in sort_idx]
    nh_vals = [nh_vals[i] for i in sort_idx]
    om_vals = [om_vals[i] for i in sort_idx]
    cf_vals = [cf_vals[i] for i in sort_idx]

    y = np.arange(len(cond_labels))
    ax.barh(y, h_vals, color="#2ecc71", label="H (Fidelity)", edgecolor="white")
    ax.barh(y, om_vals, left=h_vals, color="#3498db", label="Ω (Uncertainty)", edgecolor="white")
    ax.barh(y, cf_vals, left=[h + o for h, o in zip(h_vals, om_vals)],
            color="#f39c12", label="Conflict", edgecolor="white")
    ax.barh(y, nh_vals, left=[h + o + c for h, o, c in zip(h_vals, om_vals, cf_vals)],
            color="#e74c3c", label="not H (Artifact)", edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(cond_labels, fontsize=9)
    ax.set_xlabel("Mass Distribution")
    ax.set_title("Belief Fusion: Fidelity Assessment\n(sorted by H — higher = more natural texture)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # Annotate H values
    for i, h in enumerate(h_vals):
        ax.text(h / 2, i, f"H={h:.3f}", ha="center", va="center", fontsize=7,
                fontweight="bold", color="white" if h > 0.1 else "black")

    # Right: scatter plot H vs Conflict
    ax = axes[1]
    for c in conditions:
        combined = all_belief[c]["combined"]
        short = c.replace("weather_style_", "style_").replace("diffusion_", "diff_")

        if "style" in c:
            color, marker = "#e74c3c", "s"
        elif "diffusion" in c or "diff" in c:
            color, marker = "#3498db", "^"
        elif "night" in c:
            color, marker = "#9b59b6", "D"
        else:
            color, marker = "#f39c12", "o"

        ax.scatter(combined["H"], combined["conflict"], c=color, marker=marker,
                   s=100, edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(short, (combined["H"], combined["conflict"]),
                    fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    ax.set_xlabel("H (Fidelity) →", fontsize=10)
    ax.set_ylabel("Conflict →", fontsize=10)
    ax.set_title("Fidelity vs Conflict\n(ideal = high H, low conflict)",
                 fontsize=11, fontweight="bold")
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.3, label="Low conflict zone")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3, label="50% fidelity")

    legend_elements = [
        Patch(facecolor="#e74c3c", label="Style Transfer"),
        Patch(facecolor="#3498db", label="Diffusion (IP2P)"),
        Patch(facecolor="#9b59b6", label="Night (CycleGAN)"),
        Patch(facecolor="#f39c12", label="Fog"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, max(cf_vals) * 1.3 + 0.05)

    plt.tight_layout()
    plt.savefig(output_dir / "belief_fusion_summary.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "belief_fusion_summary.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: belief_fusion_summary.png")

    # ── Figure 3: Radar comparison (selected conditions) ──
    selected = conditions[:6]  # top 6
    fig, axes_radar = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
    axes_radar = axes_radar.flatten()

    categories = ["H\n(Fidelity)", "not H", "Ω\n(Uncertainty)", "Conflict"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for idx, cond in enumerate(selected):
        if idx >= len(axes_radar):
            break
        ax = axes_radar[idx]
        combined = all_belief[cond]["combined"]
        values = [combined["H"], combined["not_H"], combined["omega"], combined["conflict"]]
        values += values[:1]

        if "style" in cond:
            color = "#e74c3c"
        elif "diffusion" in cond or "diff" in cond:
            color = "#3498db"
        elif "night" in cond:
            color = "#9b59b6"
        else:
            color = "#f39c12"

        ax.plot(angles, values, "o-", linewidth=2, markersize=6, color=color)
        ax.fill(angles, values, alpha=0.2, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_rticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=6)
        short = cond.replace("weather_style_", "style_").replace("diffusion_", "diff_")
        ax.set_title(f"{short}\nH={combined['H']:.3f}",
                     fontsize=10, fontweight="bold", pad=15)

    for idx in range(len(selected), len(axes_radar)):
        axes_radar[idx].set_visible(False)

    fig.suptitle("Dempster-Shafer Radar: Multi-Criteria Fidelity Assessment\n"
                 "(following Duminil et al. 2025, Fig. 10-11 format)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "belief_radar_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "belief_radar_comparison.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: belief_radar_comparison.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dempster-Shafer belief fusion for texture fidelity"
    )
    parser.add_argument("--input", type=str, default=str(TEXTURE_RESULTS),
                        help="Path to texture_fidelity_results.json")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--tau", type=float, default=0.6,
                        help="BBF threshold (>0.5 = pessimistic, paper uses 0.6)")
    parser.add_argument("--alpha-default", type=float, default=0.8,
                        help="Default reliability for GLCM/LBP/DCT criteria")
    parser.add_argument("--alpha-haralick", type=float, default=0.5,
                        help="Reliability for Haralick (paper: 0.5 for statistical score)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 65}")
    print(f"DEMPSTER-SHAFER BELIEF FUSION")
    print(f"Method: Duminil et al. (2025), Eq. 14-23")
    print(f"Parameters: τ={args.tau}, α₀={args.alpha_default}, α₀_haralick={args.alpha_haralick}")
    print(f"{'=' * 65}")

    # Load texture fidelity results
    with open(args.input) as f:
        texture_results = json.load(f)

    # ── Collect all Wasserstein scores to calibrate similarity mapping ──
    # Need median distance per feature to set λ for exp(-λ*W)
    all_w_scores = {feat: [] for feat in CRITERIA}

    for cond, data in texture_results.items():
        if cond == "original" or cond.startswith("acdc_"):
            continue
        for feat in CRITERIA:
            key = f"vs_original_{feat}"
            if key in data:
                all_w_scores[feat].append(data[key]["wasserstein_mean"])

    # Median Wasserstein per feature (calibration reference)
    ref_ranges = {}
    for feat in CRITERIA:
        if all_w_scores[feat]:
            ref_ranges[feat] = float(np.median(all_w_scores[feat]))
        else:
            ref_ranges[feat] = 1.0

    print(f"\n  Calibration (median Wasserstein → Sc=0.5):")
    for feat in CRITERIA:
        print(f"    {feat.upper():>8}: median_W = {ref_ranges[feat]:.4f}")

    # ── Per-criterion reliability (α₀) ──
    alphas = {
        "glcm": args.alpha_default,
        "lbp": args.alpha_default,
        "dct": args.alpha_default,
        "haralick": args.alpha_haralick,  # Paper: S_H criterion = 0.5
    }
    print(f"\n  Reliability coefficients (α₀):")
    for feat in CRITERIA:
        print(f"    {feat.upper():>8}: α₀ = {alphas[feat]}")

    # ── Process each condition ──
    all_belief = {}

    for cond, data in texture_results.items():
        if cond == "original" or cond.startswith("acdc_"):
            continue

        # Check if vs_original scores exist
        w_scores = {}
        for feat in CRITERIA:
            key = f"vs_original_{feat}"
            if key in data:
                w_scores[feat] = data[key]["wasserstein_mean"]

        if len(w_scores) < len(CRITERIA):
            continue

        # Step 1: Wasserstein → Similarity [0, 1]
        similarities = wasserstein_to_similarity(w_scores, ref_ranges)

        # Step 2: Generate BBAs per criterion
        bbas = []
        per_criterion = {}
        for feat in CRITERIA:
            bba = generate_bba(similarities[feat], alphas[feat], args.tau)
            bbas.append(bba)
            per_criterion[feat] = {
                "wasserstein": w_scores[feat],
                "similarity": similarities[feat],
                "H": bba["H"],
                "not_H": bba["not_H"],
                "omega": bba["omega"],
            }

        # Step 3: CRC combination
        combined = combine_bbas_crc(bbas)

        all_belief[cond] = {
            "description": data.get("description", ""),
            "n_images": data.get("n_images", 0),
            "per_criterion": per_criterion,
            "combined": combined,
        }

    # ── Print results ──
    print(f"\n{'=' * 65}")
    print(f"BELIEF FUSION RESULTS")
    print(f"{'=' * 65}")

    # Per-criterion detail
    for cond in sorted(all_belief.keys()):
        belief = all_belief[cond]
        short = cond.replace("weather_style_", "style_").replace("diffusion_", "diff_")
        print(f"\n  [{short}] {belief['description']}")

        for feat in CRITERIA:
            pc = belief["per_criterion"][feat]
            print(f"    {feat.upper():>8}: W={pc['wasserstein']:.4f} → Sc={pc['similarity']:.3f}"
                  f"  →  m(H)={pc['H']:.3f}  m(H̄)={pc['not_H']:.3f}  m(Ω)={pc['omega']:.3f}")

        c = belief["combined"]
        print(f"    {'COMBINED':>8}: H={c['H']:.4f}  not_H={c['not_H']:.4f}"
              f"  Ω={c['omega']:.4f}  Conflict={c['conflict']:.4f}")

    # ── Summary table ──
    print(f"\n{'=' * 65}")
    print(f"SUMMARY TABLE (sorted by H — higher = more faithful texture)")
    print(f"{'=' * 65}")

    header = f"{'Condition':<28} {'H(Fidelity)':>12} {'not_H':>8} {'Ω(Uncert)':>10} {'Conflict':>10} {'Verdict':<15}"
    print(f"\n{header}")
    print("-" * len(header))

    sorted_conds = sorted(all_belief.keys(),
                           key=lambda c: all_belief[c]["combined"]["H"], reverse=True)

    for cond in sorted_conds:
        c = all_belief[cond]["combined"]
        short = cond.replace("weather_style_", "style_").replace("diffusion_", "diff_")

        # Verdict
        if c["H"] > 0.5 and c["conflict"] < 0.1:
            verdict = "FAITHFUL"
        elif c["H"] > 0.3:
            verdict = "moderate"
        elif c["not_H"] > 0.5:
            verdict = "ARTIFACT"
        elif c["omega"] > 0.5:
            verdict = "uncertain"
        else:
            verdict = "mixed"

        print(f"{short:<28} {c['H']:>12.4f} {c['not_H']:>8.4f} {c['omega']:>10.4f} "
              f"{c['conflict']:>10.4f} {verdict:<15}")

    # ── Save JSON ──
    json_path = output_dir / "belief_fusion_results.json"
    with open(json_path, "w") as f:
        json.dump(all_belief, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Generate plots ──
    print(f"\nGenerating visualizations...")
    plot_belief_results(all_belief, output_dir)

    # ── LaTeX table ──
    print(f"\n{'=' * 65}")
    print("Paper-ready table (LaTeX):")
    print(f"{'=' * 65}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Augmentation & $m(H)$ $\uparrow$ & $m(\bar{H})$ $\downarrow$ & $m(\Omega)$ & Conflict $\downarrow$ \\")
    print(r"\hline")

    for cond in sorted_conds:
        c = all_belief[cond]["combined"]
        cond_tex = cond.replace("_", r"\_")
        print(f"{cond_tex} & {c['H']:.4f} & {c['not_H']:.4f} & {c['omega']:.4f} & {c['conflict']:.4f} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")

    print(f"\nDone! Results at {output_dir}/")


if __name__ == "__main__":
    main()
