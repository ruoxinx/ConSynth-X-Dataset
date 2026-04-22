#!/usr/bin/env python3
"""
Analyze VLM Jury results: aggregate 3 judges, compute agreement, generate tables.

Usage:
  python -m validation.vlm_jury.analyze_results
"""

import json
from collections import defaultdict
from pathlib import Path
import os as _os
from pathlib import Path as _Path
_DATA_ROOT = _Path(_os.environ.get("CONSYNTH_DATA_ROOT", str(_Path.home() / "consynth_data")))
_REPO_ROOT = _Path(_os.environ.get("CONSYNTH_REPO_ROOT", str(_Path(__file__).resolve().parents[1])))
_BR_ROOT = _Path(_os.environ.get("CONSYNTH_BENCHMARK_RUNNER", str(_REPO_ROOT.parent / "Benchmark_runner")))

import numpy as np

RESULT_DIR = (_REPO_ROOT / "validation/results/vlm_jury")

JUDGES = ["qwen2.5-vl-7b", "internvl2.5-8b", "phi-4-multimodal"]
JUDGE_SHORT = {"qwen2.5-vl-7b": "Qwen", "internvl2.5-8b": "InternVL", "phi-4-multimodal": "Phi-4"}


def load_results():
    """Load results from all judges."""
    all_results = {}
    for judge in JUDGES:
        path = RESULT_DIR / f"{judge}_results.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {judge}")
            continue
        with open(path) as f:
            data = json.load(f)
        all_results[judge] = data
        print(f"  {judge}: {len(data)} evaluations")
    return all_results


def compute_acceptance_rates(all_results):
    """Compute per-condition acceptance rates for each judge."""
    # Build: {condition: {judge: [decisions]}}
    cond_decisions = defaultdict(lambda: defaultdict(list))

    for judge, results in all_results.items():
        for r in results:
            if r["decision"] is not None:
                cond_decisions[r["condition"]][judge].append(r["decision"])

    return cond_decisions


def wilson_ci(n_success, n_total, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return 0, 0, 0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return p, max(0, center - margin), min(1, center + margin)


def cohens_kappa(decisions_a, decisions_b):
    """Compute Cohen's kappa for two judges."""
    assert len(decisions_a) == len(decisions_b)
    n = len(decisions_a)
    if n == 0:
        return 0

    agree = sum(1 for a, b in zip(decisions_a, decisions_b) if a == b)
    p_o = agree / n

    # Expected agreement
    p_a_true = sum(decisions_a) / n
    p_b_true = sum(decisions_b) / n
    p_e = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)

    if p_e == 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def analyze():
    """Main analysis."""
    print("=" * 70)
    print("VLM Jury Analysis")
    print("=" * 70)

    all_results = load_results()
    if len(all_results) < 2:
        print("Need at least 2 judges to analyze. Exiting.")
        return

    cond_decisions = compute_acceptance_rates(all_results)

    # ── Per-judge acceptance rates ──
    print(f"\n{'='*70}")
    print("Per-Judge Acceptance Rates")
    print(f"{'='*70}")

    header = f"{'Condition':<20}"
    for j in JUDGES:
        if j in all_results:
            header += f" {JUDGE_SHORT[j]:>10}"
    header += f" {'Majority':>10}"
    print(header)
    print("-" * len(header))

    summary = {}
    for cond in sorted(cond_decisions.keys()):
        row = f"{cond:<20}"
        majority_decisions = []

        for j in JUDGES:
            if j not in all_results:
                continue
            decisions = cond_decisions[cond].get(j, [])
            if decisions:
                rate = sum(decisions) / len(decisions)
                row += f" {rate:>9.1%}"
            else:
                row += f" {'N/A':>10}"

        # Majority vote (2/3 or 3/3)
        available_judges = [j for j in JUDGES if j in all_results]
        n_samples = max(len(cond_decisions[cond].get(j, [])) for j in available_judges)

        majority_true = 0
        for idx in range(n_samples):
            votes = []
            for j in available_judges:
                decs = cond_decisions[cond].get(j, [])
                if idx < len(decs):
                    votes.append(decs[idx])
            if votes and sum(votes) > len(votes) / 2:
                majority_true += 1

        if n_samples > 0:
            maj_rate, ci_lo, ci_hi = wilson_ci(majority_true, n_samples)
            row += f" {maj_rate:>9.1%}"
            summary[cond] = {
                "majority_rate": maj_rate,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": n_samples,
            }
        print(row)

    # ── Inter-judge agreement ──
    print(f"\n{'='*70}")
    print("Inter-Judge Agreement (Cohen's κ)")
    print(f"{'='*70}")

    available = [j for j in JUDGES if j in all_results]
    for i in range(len(available)):
        for k in range(i + 1, len(available)):
            j1, j2 = available[i], available[k]
            # Align decisions by (condition, sample_idx)
            idx_map_1 = {(r["condition"], r["sample_idx"]): r["decision"]
                         for r in all_results[j1] if r["decision"] is not None}
            idx_map_2 = {(r["condition"], r["sample_idx"]): r["decision"]
                         for r in all_results[j2] if r["decision"] is not None}
            common = set(idx_map_1.keys()) & set(idx_map_2.keys())
            if not common:
                continue
            d1 = [idx_map_1[k] for k in sorted(common)]
            d2 = [idx_map_2[k] for k in sorted(common)]
            kappa = cohens_kappa(d1, d2)
            print(f"  {JUDGE_SHORT[j1]:>10} vs {JUDGE_SHORT[j2]:<10}: κ = {kappa:.3f} (n={len(common)})")

    # ── Parse error rates ──
    print(f"\n{'='*70}")
    print("Parse Error Rates")
    print(f"{'='*70}")
    for j in available:
        errors = sum(1 for r in all_results[j] if r.get("parse_error", False))
        total = len(all_results[j])
        print(f"  {JUDGE_SHORT[j]:<10}: {errors}/{total} = {errors/total:.1%}")

    # ── LaTeX table ──
    print(f"\n{'='*70}")
    print("LaTeX Table")
    print(f"{'='*70}")
    print("\\begin{tabular}{l" + "c" * (len(available) + 1) + "}")
    print("\\toprule")
    cols = " & ".join(f"\\textbf{{{JUDGE_SHORT[j]}}}" for j in available)
    print(f"\\textbf{{Condition}} & {cols} & \\textbf{{Majority}} \\\\")
    print("\\midrule")
    for cond in sorted(summary.keys()):
        vals = []
        for j in available:
            decs = cond_decisions[cond].get(j, [])
            if decs:
                rate = sum(decs) / len(decs)
                vals.append(f"{rate:.1%}")
            else:
                vals.append("--")
        s = summary[cond]
        maj = f"\\textbf{{{s['majority_rate']:.1%}}}"
        print(f"{cond} & {' & '.join(vals)} & {maj} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # Save summary
    out_path = RESULT_DIR / "vlm_jury_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out_path}")


if __name__ == "__main__":
    analyze()
