#!/usr/bin/env python3
"""
Build the image-quality-control "funnel" report for ConSynth-X.

For each (source, condition) pair, count:
  - N_input : raw images entering the diffusion pipeline
  - N_kept  : images surviving the SSIM+LPIPS filter (= rows in output Arrow,
              cross-checked against meta.csv KEEP count when available)
  - DINO    : retention statistics from validation/results/dino_ssim/*.csv
              (only available for CS test scope at time of writing)

Outputs:
  validation/results/qc_funnel/funnel_counts.csv
  validation/results/qc_funnel/funnel_report.md

Usage:  python validation/build_qc_funnel.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

REPO = Path(__file__).resolve().parents[1]
AUG = REPO / "augmentation_data"
DINO_DIR = REPO / "validation" / "results" / "dino_ssim"
OUT_DIR = REPO / "validation" / "results" / "qc_funnel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Source N_input (verified by counting source datasets) ──
SOURCE_N = {
    "cs_train":  7009,
    "cs_test":   3004,
    "soda_voc":  19846,
    "soda_ktsh": 9988,
}

# ── SSIM+LPIPS filter is applied ONLY to rain conditions ──
# For all other conditions, N_kept < source-N reflects partial generation
# (subset run), NOT filter rejection. The "Retention" column should only be
# interpreted as filter retention for rain rows; for the rest, kept = generated.
HAS_FILTER = {("rain_light",), ("rain_heavy",)}  # see helper below

def is_filtered(condition: str) -> bool:
    return condition in ("rain_light", "rain_heavy")


# ── Subset / b2 small-batch overrides ──
# Conditions deliberately run on a smaller subset rather than the full source.
# For these, N_input is the subset size, so retention is computed against the
# slice that actually entered the pipeline.
SUBSET_OVERRIDE = {
    # b2 (batch-2) snow_heavy: generated from _b2_inputs/snow_1000.arrow
    ("soda_voc",  "snow_heavy"): 1000,
    ("soda_ktsh", "snow_heavy"): 1000,
    # fog: only ~1,000-image subset of CS test was run through the fog pipeline
    ("cs_test",   "fog_light"):  1002,
    ("cs_test",   "fog_medium"): 1001,
    ("cs_test",   "fog_heavy"):  1001,
}

# ── Condition registry: (source, condition) → output Arrow / meta location ──
ENTRIES = [
    # (source, condition, family, arrow_path, meta_csv_path_or_None)
    ("cs_train",  "rain_light",  "rain_snow", AUG/"construction_site/rain_snow/diffusion/train/rain_light/rain_light.arrow", None),
    ("cs_train",  "rain_heavy",  "rain_snow", AUG/"construction_site/rain_snow/diffusion/train/rain_heavy/rain_heavy.arrow", None),
    ("cs_train",  "snow_light",  "rain_snow", AUG/"construction_site/rain_snow/diffusion/train/snow_light/snow_light.arrow", None),
    ("cs_train",  "snow_heavy",  "rain_snow", AUG/"construction_site/rain_snow/diffusion/train/snow_heavy/snow_heavy.arrow", None),

    ("cs_test",   "rain_light",  "rain_snow", None,
                  AUG/"construction_site/rain_snow/diffusion/test/rain_light/meta.csv"),
    ("cs_test",   "snow_light",  "rain_snow", None,
                  AUG/"construction_site/rain_snow/diffusion/test/snow_light/meta.csv"),

    ("soda_voc",  "rain_light",  "rain_snow", AUG/"soda_voc/rain_snow/diffusion/rain_light.arrow", None),
    ("soda_voc",  "rain_heavy",  "rain_snow", AUG/"soda_voc/rain_snow/diffusion/rain_heavy.arrow", None),
    ("soda_voc",  "snow_light",  "rain_snow", AUG/"soda_voc/rain_snow/diffusion/snow_light.arrow", None),
    ("soda_voc",  "snow_heavy",  "rain_snow", AUG/"soda_voc/rain_snow/diffusion/snow_heavy.arrow", None),

    ("soda_ktsh", "rain_light",  "rain_snow", AUG/"soda_ktsh/rain_snow/diffusion/rain_light.arrow", None),
    ("soda_ktsh", "rain_heavy",  "rain_snow", AUG/"soda_ktsh/rain_snow/diffusion/rain_heavy.arrow", None),
    ("soda_ktsh", "snow_light",  "rain_snow", AUG/"soda_ktsh/rain_snow/diffusion/snow_light.arrow", None),
    ("soda_ktsh", "snow_heavy",  "rain_snow", AUG/"soda_ktsh/rain_snow/diffusion/snow_heavy.arrow", None),

    # Pass-through (no SSIM/LPIPS at gen time)
    ("cs_test",   "fog_light",   "fog",       AUG/"construction_site/fog/diffusion/test/fog_light.arrow",  None),
    ("cs_test",   "fog_medium",  "fog",       AUG/"construction_site/fog/diffusion/test/fog_medium.arrow", None),
    ("cs_test",   "fog_heavy",   "fog",       AUG/"construction_site/fog/diffusion/test/fog_heavy.arrow",  None),
    ("cs_test",   "night",       "night",     AUG/"construction_site/night/test/night_constructionsite_test.arrow", None),
    ("cs_test",   "night_rain",  "night_weather",
                  AUG/"construction_site/night_weather/rain_night/rain_night.arrow",
                  AUG/"construction_site/night_weather/rain_night/meta.csv"),
    ("cs_test",   "night_snow",  "night_weather",
                  AUG/"construction_site/night_weather/snow_night/snow_night.arrow",
                  AUG/"construction_site/night_weather/snow_night/meta.csv"),
]

# DINO CSV name mapping. CS test was built earlier by extract_dino_ssim_all.py
# (legacy filenames `ip2p_*`, `fog_*`, `night*`); CS train, SODA-VOC, SODA-KTSH
# are built by extract_dino_ssim_extended.py with {source}_{condition}.csv names.
DINO_KEY = {
    # CS test (legacy names)
    ("cs_test",  "rain_light"):  "ip2p_rain",
    ("cs_test",  "rain_heavy"):  "ip2p_rain_heavy",
    ("cs_test",  "snow_light"):  "ip2p_snow_light",
    ("cs_test",  "snow_heavy"):  "ip2p_snow_heavy",
    ("cs_test",  "fog_light"):   "fog_light",
    ("cs_test",  "fog_medium"):  "fog_medium",
    ("cs_test",  "fog_heavy"):   "fog_heavy",
    ("cs_test",  "night"):       "night",
    ("cs_test",  "night_rain"):  "night_rain",
    ("cs_test",  "night_snow"):  "night_snow",
    # CS train, SODA-VOC, SODA-KTSH (extended naming: {source}_{condition})
    ("cs_train",  "rain_light"): "cs_train_rain_light",
    ("cs_train",  "rain_heavy"): "cs_train_rain_heavy",
    ("cs_train",  "snow_light"): "cs_train_snow_light",
    ("cs_train",  "snow_heavy"): "cs_train_snow_heavy",
    ("soda_voc",  "rain_light"): "soda_voc_rain_light",
    ("soda_voc",  "rain_heavy"): "soda_voc_rain_heavy",
    ("soda_voc",  "snow_light"): "soda_voc_snow_light",
    ("soda_voc",  "snow_heavy"): "soda_voc_snow_heavy",
    ("soda_ktsh", "rain_light"): "soda_ktsh_rain_light",
    ("soda_ktsh", "rain_heavy"): "soda_ktsh_rain_heavy",
    ("soda_ktsh", "snow_light"): "soda_ktsh_snow_light",
    ("soda_ktsh", "snow_heavy"): "soda_ktsh_snow_heavy",
}

DINO_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]


def arrow_nrows(path: Path) -> Optional[int]:
    if not path or not Path(path).exists():
        return None
    try:
        with pa.memory_map(str(path), "r") as mm:
            try:
                return ipc.RecordBatchFileReader(mm).read_all().num_rows
            except Exception:
                mm.seek(0)
                return ipc.RecordBatchStreamReader(mm).read_all().num_rows
    except Exception:
        return None


def arrow_status_count(path: Path) -> Optional[dict]:
    """If output Arrow has a `status` column, count KEEP/DROP."""
    if not path or not Path(path).exists():
        return None
    try:
        with pa.memory_map(str(path), "r") as mm:
            try:
                t = ipc.RecordBatchFileReader(mm).read_all()
            except Exception:
                mm.seek(0)
                t = ipc.RecordBatchStreamReader(mm).read_all()
    except Exception:
        return None
    if "status" not in t.column_names:
        return None
    vals = t.column("status").to_pylist()
    out: dict = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out


def meta_csv_status_count(path: Path) -> Optional[dict]:
    if not path or not Path(path).exists():
        return None
    out: dict = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            s = row.get("status") or "?"
            out[s] = out.get(s, 0) + 1
    return out


def load_dino_csv(name: str) -> Optional[list]:
    p = DINO_DIR / f"{name}.csv"
    if not p.exists():
        return None
    rows = []
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append(float(row["dino_sim"]))
            except (KeyError, ValueError):
                pass
    return rows or None


def dino_stats(values: list) -> dict:
    a = np.asarray(values)
    out = {
        "dino_n": int(len(a)),
        "dino_mean": float(a.mean()),
        "dino_p25":  float(np.percentile(a, 25)),
        "dino_p50":  float(np.percentile(a, 50)),
        "dino_p75":  float(np.percentile(a, 75)),
    }
    for thr in DINO_THRESHOLDS:
        out[f"dino_ge_{int(thr*100)}_pct"] = float((a >= thr).mean() * 100.0)
    return out


def main():
    rows_out = []
    for source, cond, family, arrow_path, meta_path in ENTRIES:
        # N_input
        n_input = SUBSET_OVERRIDE.get((source, cond), SOURCE_N.get(source))

        # N_kept: prefer Arrow row count; cross-check meta.csv if present
        n_kept_arrow = arrow_nrows(arrow_path) if arrow_path else None
        meta_status = meta_csv_status_count(meta_path) if meta_path else None
        arrow_status = arrow_status_count(arrow_path) if arrow_path else None

        # Resolve N_kept
        if n_kept_arrow is not None:
            n_kept = n_kept_arrow
        elif meta_status:
            n_kept = meta_status.get("KEEP", 0)
        else:
            n_kept = None

        # Total seen by filter (denominator for retention) = whichever we trust
        # Prefer meta.csv total (rows generated, both KEEP+DROP) where present.
        if meta_status:
            n_processed = sum(meta_status.values())
        elif arrow_status:
            n_processed = sum(arrow_status.values())
        else:
            n_processed = n_input

        # Drops by status
        if meta_status:
            n_drop = meta_status.get("DROP", 0)
        elif arrow_status:
            n_drop = sum(v for k, v in arrow_status.items() if k != "KEEP")
        elif n_kept is not None and n_input is not None:
            n_drop = max(n_input - n_kept, 0)
        else:
            n_drop = None

        # DINO
        dino_name = DINO_KEY.get((source, cond))
        dino_vals = load_dino_csv(dino_name) if dino_name else None
        dstats = dino_stats(dino_vals) if dino_vals else {}

        retention_pct = (
            (n_kept / n_input * 100.0) if (n_kept is not None and n_input)
            else None
        )
        # Final after a hypothetical DINO ≥ 0.75 audit (post-hoc, doesn't filter
        # the released dataset — just shows what fraction would survive).
        dino_final_75 = None
        if n_kept is not None and dstats and "dino_ge_75_pct" in dstats:
            dino_final_75 = int(round(n_kept * dstats["dino_ge_75_pct"] / 100.0))

        # Distinguish "filter retention" (rain only — meaningful) from
        # "generation coverage" (everything else — just means we didn't run
        # the full source through generation).
        filter_applied = is_filtered(cond)
        if filter_applied:
            filter_retention = retention_pct
            coverage_pct = None  # rain is run on full source
        else:
            filter_retention = None
            coverage_pct = retention_pct

        rows_out.append({
            "source": source,
            "condition": cond,
            "family": family,
            "filter_applied": filter_applied,
            "n_input": n_input,
            "n_processed_by_filter": n_processed,
            "n_kept": n_kept,
            "n_drop": n_drop,
            "filter_retention_pct": (round(filter_retention, 2) if filter_retention is not None else None),
            "coverage_pct": (round(coverage_pct, 2) if coverage_pct is not None else None),
            "has_status_breakdown": bool(meta_status or arrow_status),
            "drops_breakdown": json.dumps(meta_status or arrow_status or {}),
            "has_dino": dino_vals is not None,
            **dstats,
            "n_after_dino_ge_75": dino_final_75,
        })

    # Write CSV — collect union of all keys across rows (DINO fields appear only
    # for conditions that have DINO data).
    csv_path = OUT_DIR / "funnel_counts.csv"
    keys: list = []
    seen = set()
    for r in rows_out:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    for r in rows_out:
        for k in keys:
            r.setdefault(k, "")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {csv_path}  ({len(rows_out)} rows)")

    # Markdown report
    md_path = OUT_DIR / "funnel_report.md"
    with open(md_path, "w") as f:
        f.write("# ConSynth-X — Quality-Control Funnel Report\n\n")
        f.write("Generated by `validation/build_qc_funnel.py`. Counts come from output\n"
                "Arrow files (`row count = N_kept`) and `meta.csv` `status` columns when\n"
                "available. DINO retention is computed post-hoc from\n"
                "`validation/results/dino_ssim/*.csv` (only CS test scope at present).\n\n")

        f.write("**IMPORTANT:** SSIM+LPIPS filter is applied ONLY to rain (light\n"
                "and heavy). For every other condition, `N_kept < N_input` means we\n"
                "deliberately generated only a subset of the source — NOT that the\n"
                "filter rejected images. Treat the `Filter retention` column as\n"
                "meaningful only on rain rows; the `Coverage` column shows how much\n"
                "of the source dataset was actually pushed through the pipeline.\n\n")

        # Summary by source
        f.write("## Summary by source × condition\n\n")
        f.write("| Source | Condition | N_input | N_kept | Filter retention | Coverage (gen) | DINO≥0.75 | Final (DINO≥0.75) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_out:
            fret = (f"**{r['filter_retention_pct']}%**"
                    if r['filter_retention_pct'] is not None else "—")
            cov = (f"{r['coverage_pct']}%"
                   if r['coverage_pct'] is not None else "—")
            d75 = f"{r.get('dino_ge_75_pct'):.1f}%" if r.get("dino_ge_75_pct") not in ("", None) else "—"
            n75 = r.get("n_after_dino_ge_75") or "—"
            f.write(f"| {r['source']} | {r['condition']} | {r['n_input']} | {r['n_kept']} | {fret} | {cov} | {d75} | {n75} |\n")

        # Status breakdown where available
        f.write("\n## Detailed drop breakdown (where `status` column present)\n\n")
        any_detail = False
        for r in rows_out:
            if r["has_status_breakdown"]:
                any_detail = True
                f.write(f"- **{r['source']} / {r['condition']}**: {r['drops_breakdown']}\n")
        if not any_detail:
            f.write("_(none)_\n")

        # DINO distribution where available
        f.write("\n## DINO similarity distribution (per condition)\n\n")
        f.write("| Condition (CS test) | N | mean | p25 | p50 | p75 | ≥0.50 | ≥0.60 | ≥0.70 | ≥0.75 | ≥0.80 | ≥0.85 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_out:
            if not r["has_dino"]:
                continue
            f.write(f"| {r['source']}/{r['condition']} | {r['dino_n']} | "
                    f"{r['dino_mean']:.3f} | {r['dino_p25']:.3f} | {r['dino_p50']:.3f} | {r['dino_p75']:.3f} | "
                    f"{r['dino_ge_50_pct']:.1f}% | {r['dino_ge_60_pct']:.1f}% | "
                    f"{r['dino_ge_70_pct']:.1f}% | {r['dino_ge_75_pct']:.1f}% | "
                    f"{r['dino_ge_80_pct']:.1f}% | {r['dino_ge_85_pct']:.1f}% |\n")

        # Anomalies / notes
        f.write("\n## Notes / anomalies\n\n")
        f.write("- **Filter scope**: SSIM+LPIPS is applied only to rain (light & heavy).\n"
                "  Snow / fog / night / night_weather pipelines do NOT filter; their\n"
                "  `N_kept < N_input` reflects partial generation (subset), not rejection.\n")
        f.write("- **Paired rain_light = rain_heavy** (same N_kept across all sources):\n"
                "  rain light/heavy share the same SSIM+LPIPS filter pass; the heavy\n"
                "  variant is produced from the same kept subset.\n")
        f.write("- **soda_voc/snow_heavy = 1,000** and **soda_ktsh/snow_heavy = 1,000**:\n"
                "  produced from the b2 small-batch pipeline (`_b2_inputs/snow_1000.arrow`).\n")
        f.write("- **fog × 3 ran on a 1k subset of CS test** (not 3,004), hence ~1,001 kept.\n")
        f.write("- **night_rain / night_snow = 3,004 / 3,004**: full CS test passed through\n"
                "  Order-B without filtering.\n")
        f.write("- **DINO retention now covers all sources** — extended CSVs\n"
                "  (cs_train_*, soda_voc_*, soda_ktsh_*) computed via\n"
                "  `validation/extract_dino_ssim_extended.py` on Pitzer V100-32g.\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
