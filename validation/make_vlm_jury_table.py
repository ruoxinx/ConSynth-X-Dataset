#!/usr/bin/env python3
"""Render the VLM jury results as a PDF/PNG table, grouped by augmentation family.

Reads:
  validation/results/vlm_jury/vlm_jury_summary.json
  validation/results/vlm_jury/*_results.json
Outputs:
  validation/results/vlm_jury_table.pdf
  validation/results/vlm_jury_table.png
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path(__file__).resolve().parents[1]
JURY_DIR = REPO / 'validation/results/vlm_jury'
OUT_DIR = REPO / 'validation/results'


ORDER = [
    ('ACDC (real reference)', ['acdc_rain', 'acdc_snow', 'acdc_fog', 'acdc_night']),
    ('Style transfer',        ['st_rain_a', 'st_rain_b', 'st_rain_c', 'st_snow_b']),
    ('IP2P diffusion',        ['ip2p_rain', 'ip2p_snow', 'ip2p_snow_heavy_full']),
    ('Fog physics',           ['fog_heavy']),
    ('Night',                 ['night']),
]

# Auxiliary full-dataset runs that are stored OUTSIDE vlm_jury_summary.json
# ({model} -> file in vlm_jury dir)
AUX_FULL = {
    'ip2p_snow_heavy_full': {
        'internvl2.5-8b': 'snow_strong_full_internvl.json',
        'phi-4-multimodal': 'snow_strong_full_phi4.json',
        'qwen2.5-vl-7b':    'snow_strong_full_qwen.json',
        'label':            'ip2p_snow_heavy (full, 3004)',
    },
}

# DINO >= 0.75 filtered re-run: shared results file per model.
DINO075_FILES = {
    'internvl2.5-8b':   'internvl2.5-8b_dino0.75_results.json',
    'phi-4-multimodal': 'phi-4-multimodal_dino0.75_results.json',
    'qwen2.5-vl-7b':    'qwen2.5-vl-7b_dino0.75_results.json',
    'qwen3-vl-8b':      'qwen3-vl-8b_dino0.75_results.json',
}

# DINO >= 0.80 filtered re-run (Claude Sonnet 4.6 only so far).
DINO080_FILES = {
    'claude-sonnet-4.6': 'claude-sonnet-4.6_dino0.8_results.json',
}

# Conditions to show in the DINO-filtered section (any that have >=1 row
# in any _dino0.75 file will be included automatically).
DINO075_ORDER = [
    # ACDC real-reference (single-judge fallback; Qwen3 + Claude only)
    'acdc_rain', 'acdc_snow', 'acdc_fog', 'acdc_night',
    # Synthetic
    'ip2p_rain',
    'ip2p_rain_heavy',
    'ip2p_snow_light',
    'ip2p_snow_heavy',
    'night',
    'fog_light',
    'fog_medium',
    'fog_heavy',
    'st_rain_a', 'st_rain_b', 'st_rain_c', 'st_snow_b',
    'night_rain',
    'night_snow',
]

# Master (comprehensive) table groups
MASTER_GROUPS = [
    ('ACDC (real reference)', ['acdc_rain', 'acdc_snow', 'acdc_fog', 'acdc_night']),
    ('Style transfer',        ['st_rain_a', 'st_rain_b', 'st_rain_c', 'st_snow_b']),
    ('IP2P diffusion',        ['ip2p_rain', 'ip2p_rain_heavy', 'ip2p_snow_light', 'ip2p_snow_heavy']),
    ('Fog physics',           ['fog_light', 'fog_medium', 'fog_heavy']),
    ('Night',                 ['night']),
    ('Night + weather (Order B)', ['night_rain', 'night_snow']),
]
MASTER_GROUP_COLORS = {
    'ACDC (real reference)':     '#e5e7eb',
    'Style transfer':            '#fee2e2',
    'IP2P diffusion':            '#dbeafe',
    'Fog physics':               '#ede9fe',
    'Night':                     '#f3f4f6',
    'Night + weather (Order B)': '#fce7f3',
}

# Rename legacy condition keys in display (keys unchanged in JSON)
DISPLAY_RENAME = {
    'ip2p_rain': 'ip2p_rain (light)',
    'ip2p_snow': 'ip2p_snow (light)',
}

GROUP_COLOR = {
    'ACDC (real reference)': '#e5e7eb',
    'Style transfer':        '#fee2e2',
    'IP2P diffusion':        '#dbeafe',
    'Fog physics':           '#ede9fe',
    'Night':                 '#f3f4f6',
}


def load_per_model():
    per_model = {}
    for f in sorted(JURY_DIR.glob('*_results.json')):
        model = f.stem.replace('_results', '')
        rows = json.load(open(f))
        per = {}
        for r in rows:
            c = r['condition']
            dec = r.get('decision')
            if dec is None and 'raw_response' in r:
                m = re.search(r'"decision"\s*:\s*(true|false)', r['raw_response'], re.I)
                if m:
                    dec = m.group(1).lower() == 'true'
            if dec is not None:
                per.setdefault(c, []).append(dec)
        per_model[model] = {c: sum(v) / len(v) if v else None for c, v in per.items()}
    return per_model


def colour_for_rate(rate, is_reference=False):
    if rate is None:
        return '#f3f4f6'
    if is_reference:
        return '#e5e7eb'
    if rate >= 0.85:
        return '#bbf7d0'   # green
    if rate >= 0.70:
        return '#fef08a'   # amber
    if rate >= 0.40:
        return '#fed7aa'   # orange
    return '#fecaca'       # red


SHORT_MODEL = {
    'internvl2.5-8b':   'InternVL2.5-8B',
    'phi-4-multimodal': 'Phi-4 MM',
    'qwen2.5-vl-7b':    'Qwen2.5-VL-7B',
    'qwen3-vl-8b':      'Qwen3-VL-8B',
    'claude-sonnet-4.6': 'Claude Sonnet 4.6',
}

# Threshold used per model in the combined DINO table
MODEL_THRESHOLD = {
    'internvl2.5-8b':   0.75,
    'phi-4-multimodal': 0.75,
    'qwen2.5-vl-7b':    0.75,
    'qwen3-vl-8b':      0.75,
    'claude-sonnet-4.6': 0.80,
}


def short_model(name, with_threshold=False):
    s = SHORT_MODEL.get(name, name)
    if with_threshold and name in MODEL_THRESHOLD:
        s += f"\n(D>={MODEL_THRESHOLD[name]})"
    return s


def _stat_from_decisions(decs):
    """Wilson 95% CI for a binary accept rate."""
    import math
    n = len(decs)
    if n == 0:
        return None
    p = sum(decs) / n
    z = 1.96
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return {'majority_rate': p, 'ci_lo': max(0, (centre - adj) / denom),
            'ci_hi': min(1, (centre + adj) / denom), 'n': n}


def load_dino(files_map):
    """Load DINO-filtered jury runs.

    Returns (per_judge, summary) where:
      per_judge[model][cond] = fraction accepted
      summary[cond] = {majority_rate, ci_lo, ci_hi, n, votes}
    """
    per_judge = {m: {} for m in files_map}
    # condition -> image_id -> model -> bool
    votes = {}
    for model, fname in files_map.items():
        path = JURY_DIR / fname
        if not path.exists():
            continue
        d = json.load(open(path))
        for r in d:
            cond = r['condition']
            dec = r.get('decision')
            if dec is None and 'raw_response' in r:
                m = re.search(r'"decision"\s*:\s*(true|false)', r.get('raw_response', ''), re.I)
                if m:
                    dec = m.group(1).lower() == 'true'
            if dec is None:
                continue
            votes.setdefault(cond, {}).setdefault(r['image_id'], {})[model] = dec
        # per-judge per-condition rate
        by_cond = {}
        for r in d:
            c = r['condition']
            dec = r.get('decision')
            if dec is None and 'raw_response' in r:
                m = re.search(r'"decision"\s*:\s*(true|false)', r.get('raw_response', ''), re.I)
                if m:
                    dec = m.group(1).lower() == 'true'
            if dec is not None:
                by_cond.setdefault(c, []).append(dec)
        per_judge[model] = {c: sum(v) / len(v) for c, v in by_cond.items() if v}

    summary = {}
    for cond, img_map in votes.items():
        n_judges_voting = max((len(v) for v in img_map.values()), default=0)
        maj_decs = []
        for img_id, mvotes in img_map.items():
            if n_judges_voting >= 2:
                # Standard majority: >=2/3 accept among judges that voted on
                # this image. Require at least 2 judges to have voted.
                if len(mvotes) >= 2:
                    maj_decs.append(sum(mvotes.values()) >= 2)
            elif len(mvotes) >= 1:
                # Only one judge ran — fall back to that judge's decision
                # so single-judge conditions (e.g. ACDC) still appear.
                maj_decs.append(list(mvotes.values())[0])
        stat = _stat_from_decisions(maj_decs)
        if stat is not None:
            stat['models_voted'] = sorted({m for v in img_map.values() for m in v})
            summary[cond] = stat
    return per_judge, summary


def load_aux(summary, per_model):
    """Merge AUX_FULL experimental runs into summary + per_model."""
    for cond, spec in AUX_FULL.items():
        per_judge = {}
        all_decs_by_id = {}
        for model, fname in spec.items():
            if model == 'label':
                continue
            path = JURY_DIR / fname
            if not path.exists():
                continue
            d = json.load(open(path))
            decs = []
            for r in d:
                dec = r.get('decision')
                if dec is None and 'raw_response' in r:
                    m = re.search(r'"decision"\s*:\s*(true|false)', r.get('raw_response', ''), re.I)
                    if m:
                        dec = m.group(1).lower() == 'true'
                if dec is not None:
                    decs.append(dec)
                    all_decs_by_id.setdefault(r.get('image_id', len(decs)), {})[model] = dec
            per_judge[model] = sum(decs) / len(decs) if decs else None
            per_model.setdefault(model, {})[cond] = per_judge[model]

        # Majority = >=2/3 accept
        majority_decs = []
        for img_id, votes in all_decs_by_id.items():
            if len(votes) >= 2:  # at least 2 judges voted
                majority_decs.append(sum(votes.values()) >= 2)
        stat = _stat_from_decisions(majority_decs)
        if stat:
            summary[cond] = stat


def _render_table(display_rows, summary, per_model, models, missing,
                   out_stem, title, subtitle, use_groups=True):
    """Render a single PDF/PNG table and return nothing."""
    n_rows = len(display_rows)
    show_thresh = all(m in MODEL_THRESHOLD for m in models) and len(models) > 1 and \
        len(set(MODEL_THRESHOLD[m] for m in models)) > 1
    headers = ['Condition', 'N', 'Majority', '95% CI Wilson'] + \
        [short_model(m, with_threshold=show_thresh) for m in models]
    n_cols = len(headers)

    n_groups = len({g for g, _ in display_rows}) if use_groups else 0
    row_h = 0.38
    fig_h = 1.2 + row_h * (n_rows + n_groups + 4)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Title
    title_y = 0.99
    ax.text(0.5, title_y, title,
            ha='center', va='top', fontsize=15, weight='bold')
    ax.text(0.5, title_y - 0.025, subtitle,
            ha='center', va='top', fontsize=9, color='#555')

    # Column x positions (relative fractions)
    base = [0.22, 0.05, 0.10, 0.18]
    model_frac = (1.0 - sum(base)) / max(len(models), 1)
    col_fracs = base + [model_frac] * len(models)
    col_x = [0]
    for f in col_fracs:
        col_x.append(col_x[-1] + f)

    # Layout: compute y positions top-down
    top = title_y - 0.07
    bottom_reserved = 0.12  # for legend + missing notes
    avail = top - bottom_reserved
    total_rows = n_rows + n_groups + 1  # headers
    dy = avail / total_rows
    y = top

    # Header row
    for i, h in enumerate(headers):
        x0 = col_x[i]; x1 = col_x[i + 1]
        ax.add_patch(mpatches.Rectangle((x0, y - dy), x1 - x0, dy,
                                         facecolor='#374151', edgecolor='white'))
        ax.text((x0 + x1) / 2, y - dy / 2, h, ha='center', va='center',
                color='white', fontsize=9.5, weight='bold')
    y -= dy

    # Data rows grouped
    current_group = None
    for group, cond in display_rows:
        if use_groups and group != current_group:
            # group header band
            ax.add_patch(mpatches.Rectangle((0, y - dy), 1, dy,
                                             facecolor=GROUP_COLOR.get(group, '#e5e7eb'),
                                             edgecolor='white'))
            ax.text(0.008, y - dy / 2, group, ha='left', va='center',
                    fontsize=9.5, weight='bold', color='#1f2937')
            y -= dy
            current_group = group

        s = summary[cond]
        is_ref = group.startswith('ACDC')
        mr = s['majority_rate']; lo = s['ci_lo']; hi = s['ci_hi']; n = s['n']

        display_name = (AUX_FULL[cond]['label'] if cond in AUX_FULL
                        else DISPLAY_RENAME.get(cond, cond))
        cells = [
            display_name,
            str(n),
            f'{mr:.3f}',
            f'[{lo:.3f}, {hi:.3f}]',
        ]
        cell_colors = ['white', 'white', colour_for_rate(mr, is_ref), 'white']

        for m in models:
            v = per_model[m].get(cond)
            cells.append(f'{v:.3f}' if v is not None else '—')
            cell_colors.append(colour_for_rate(v, is_ref))

        for i, (txt, bg) in enumerate(zip(cells, cell_colors)):
            x0 = col_x[i]; x1 = col_x[i + 1]
            ax.add_patch(mpatches.Rectangle((x0, y - dy), x1 - x0, dy,
                                             facecolor=bg, edgecolor='#e5e7eb'))
            align = 'left' if i == 0 else 'center'
            tx = x0 + 0.008 if i == 0 else (x0 + x1) / 2
            ax.text(tx, y - dy / 2, txt, ha=align, va='center', fontsize=9)
        y -= dy

    # Legend for colors
    leg_y = bottom_reserved - 0.04
    legend_items = [
        ('>= 0.85',    '#bbf7d0'),
        ('0.70-0.85',  '#fef08a'),
        ('0.40-0.70',  '#fed7aa'),
        ('< 0.40',     '#fecaca'),
        ('reference',  '#e5e7eb'),
    ]
    x = 0.02
    ax.text(x, leg_y + 0.025, 'Colour = majority rate:', fontsize=9, weight='bold')
    for label, col in legend_items:
        ax.add_patch(mpatches.Rectangle((x, leg_y - 0.005), 0.02, 0.02,
                                         facecolor=col, edgecolor='#6b7280'))
        ax.text(x + 0.024, leg_y + 0.005, label, fontsize=9, va='center')
        x += 0.12

    # Missing conditions note
    if missing:
        note_y = 0.02
        ax.text(0.02, note_y + 0.03,
                'Not yet run through VLM jury:',
                fontsize=9, weight='bold', color='#991b1b')
        ax.text(0.02, note_y,
                ', '.join(missing),
                fontsize=9, color='#991b1b')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    pdf_path = OUT_DIR / f'{out_stem}.pdf'
    png_path = OUT_DIR / f'{out_stem}.png'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    print(f'Saved: {pdf_path}')
    print(f'Saved: {png_path}')


def main():
    summary = json.load(open(JURY_DIR / 'vlm_jury_summary.json'))
    per_model = load_per_model()
    models = ['internvl2.5-8b', 'phi-4-multimodal', 'qwen2.5-vl-7b']
    models = [m for m in models if m in per_model]
    load_aux(summary, per_model)

    # --- Section 1: original random-50 jury ---
    display_rows = []
    for group, conds in ORDER:
        for c in conds:
            if c in summary:
                display_rows.append((group, c))

    missing = ['night_rain', 'night_snow',
               'ip2p_rain_heavy',
               'fog_light', 'fog_medium']

    _render_table(
        display_rows, summary, per_model, models, missing,
        out_stem='vlm_jury_table',
        title='VLM Jury (random sampling) — Realism / Scene-preservation majority vote',
        subtitle=f'{len(models)} models: {", ".join(short_model(m) for m in models)} | '
                  'majority = fraction of samples judged "acceptable" by >=2/3 models',
    )

    # --- Section 2: combined DINO-filtered jury (all judges) ---
    # Open-source judges use DINO>=0.75; Claude uses DINO>=0.80 (stricter).
    # Per-judge per-condition accept rates are shown together; majority is
    # computed across the open-source 0.75 judges only because they share a
    # common image set.
    per_model_75, summary_75 = load_dino(DINO075_FILES)
    per_model_80, summary_80 = load_dino(DINO080_FILES)

    # Combined per_model (different image sets, but same condition keys)
    per_model_all = {**per_model_75, **per_model_80}

    # Conditions = union of those that have at least one judge with results
    seen_conds = set(summary_75) | set(summary_80)
    dino_rows = [('DINO-filtered', c) for c in DINO075_ORDER if c in seen_conds]
    extras = sorted(seen_conds - set(DINO075_ORDER))
    for c in extras:
        dino_rows.append(('DINO-filtered', c))

    # Per-row summary uses the 0.75 majority (open-source consensus); if a
    # condition only exists in 0.80 (Claude-only), fall back to that.
    summary_combined = {}
    for c in seen_conds:
        if c in summary_75:
            summary_combined[c] = summary_75[c]
        else:
            summary_combined[c] = summary_80[c]

    all_models = ['internvl2.5-8b', 'phi-4-multimodal', 'qwen2.5-vl-7b',
                  'qwen3-vl-8b', 'claude-sonnet-4.6']
    models_present = [m for m in all_models if any(per_model_all.get(m, {}).values())]

    if dino_rows:
        incomplete = []
        for m in models_present:
            done = sum(1 for _, c in dino_rows if per_model_all.get(m, {}).get(c) is not None)
            if done < len(dino_rows):
                incomplete.append(f'{SHORT_MODEL.get(m,m)} {done}/{len(dino_rows)}')
        subtitle_bits = [
            f'{len(models_present)} judges',
            'open-source: DINO sim >= 0.75 ; Claude Sonnet 4.6: DINO sim >= 0.80',
            'majority = >=2/3 accept across open-source judges (Claude excluded from majority)',
        ]
        if incomplete:
            subtitle_bits.append('partial: ' + '; '.join(incomplete))

        # Use threshold-aware judge labels
        per_model_labelled = per_model_all
        # Patch _render_table by passing models with annotated short names
        _render_table(
            dino_rows, summary_combined, per_model_labelled, models_present,
            missing=[],
            out_stem='vlm_jury_table_dino',
            title='VLM Jury (DINO-filtered) - 50 samples / condition',
            subtitle=' | '.join(subtitle_bits),
            use_groups=True,
        )


def _master_table():
    """Build a single comprehensive table: all conditions x all judges, with
    per-cell data pulled from the best available run for that (condition,
    judge). Missing cells are shown as — .

    Priority per cell:
      1. DINO>=0.75 run (open-source judges) or DINO>=0.80 (Claude)
      2. Random-sampling jury (legacy)
      3. AUX_FULL (e.g. snow_strong_full 3004)
    """
    per_model_75, summary_75 = load_dino(DINO075_FILES)
    per_model_80, summary_80 = load_dino(DINO080_FILES)

    # Legacy random-sampling summary (3 judges on 8 synthetic + 4 acdc)
    random_summary = json.load(open(JURY_DIR / 'vlm_jury_summary.json'))
    random_per_model = load_per_model()
    load_aux(random_summary, random_per_model)

    all_models = ['internvl2.5-8b', 'phi-4-multimodal', 'qwen2.5-vl-7b',
                  'qwen3-vl-8b', 'claude-sonnet-4.6']

    # per_model_master[model][cond] = (rate, source_tag)
    per_model_master = {m: {} for m in all_models}
    for m in all_models:
        for cond in set(per_model_75.get(m, {})) | set(per_model_80.get(m, {})) | \
                    set(random_per_model.get(m, {})):
            r = per_model_75.get(m, {}).get(cond)
            if r is not None:
                per_model_master[m][cond] = (r, '0.75')
                continue
            r = per_model_80.get(m, {}).get(cond)
            if r is not None:
                per_model_master[m][cond] = (r, '0.80')
                continue
            r = random_per_model.get(m, {}).get(cond)
            if r is not None:
                per_model_master[m][cond] = (r, 'rand')

    # Master summary (majority): prefer DINO-0.75 majority, else DINO-0.80,
    # else random.
    master_summary = {}
    for cond in set(summary_75) | set(summary_80) | set(random_summary):
        if cond in summary_75:
            s = summary_75[cond]; src = 'D>=0.75'
        elif cond in summary_80:
            s = summary_80[cond]; src = 'D>=0.80'
        else:
            s = random_summary[cond]; src = 'rand-50'
        master_summary[cond] = {**s, 'src': src}

    # Build display_rows from MASTER_GROUPS order, including any extras
    display_rows = []
    seen = set()
    for group, conds in MASTER_GROUPS:
        for c in conds:
            if c in master_summary:
                display_rows.append((group, c))
                seen.add(c)
    extras = sorted(set(master_summary) - seen)
    for c in extras:
        display_rows.append(('Other', c))

    # Custom render: modified to pass (rate, tag) per cell
    _render_master(display_rows, master_summary, per_model_master, all_models,
                   out_stem='vlm_jury_table_master',
                   title='VLM Jury (comprehensive) - all judges x all conditions',
                   subtitle='per-cell source: 0.75 = DINO>=0.75, 0.80 = DINO>=0.80, '
                            'rand = random-50 | Majority column uses best available: '
                            'DINO>=0.75 > DINO>=0.80 > random | "-" = not evaluated')


def _render_master(display_rows, summary, per_model_master, models,
                   out_stem, title, subtitle):
    """Variant of _render_table where per-cell values come from
    per_model_master[model][cond] = (rate, source_tag).
    """
    n_rows = len(display_rows)
    n_groups = len({g for g, _ in display_rows})
    headers = ['Condition', 'N', 'Majority', 'CI (Wilson)'] + \
        [SHORT_MODEL.get(m, m) for m in models]
    row_h = 0.34
    fig_h = 1.3 + row_h * (n_rows + n_groups + 4)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    title_y = 0.99
    ax.text(0.5, title_y, title,
            ha='center', va='top', fontsize=15, weight='bold')
    ax.text(0.5, title_y - 0.02, subtitle,
            ha='center', va='top', fontsize=8.5, color='#555')

    base = [0.17, 0.05, 0.09, 0.15]
    mf = (1.0 - sum(base)) / max(len(models), 1)
    col_fracs = base + [mf] * len(models)
    col_x = [0]
    for f in col_fracs:
        col_x.append(col_x[-1] + f)

    top = title_y - 0.06
    bottom_reserved = 0.10
    avail = top - bottom_reserved
    total_rows = n_rows + n_groups + 1
    dy = avail / total_rows
    y = top

    # Header
    for i, h in enumerate(headers):
        x0 = col_x[i]; x1 = col_x[i + 1]
        ax.add_patch(mpatches.Rectangle((x0, y - dy), x1 - x0, dy,
                                         facecolor='#374151', edgecolor='white'))
        ax.text((x0 + x1) / 2, y - dy / 2, h, ha='center', va='center',
                color='white', fontsize=9, weight='bold')
    y -= dy

    current_group = None
    for group, cond in display_rows:
        if group != current_group:
            ax.add_patch(mpatches.Rectangle((0, y - dy), 1, dy,
                                             facecolor=GROUP_COLOR.get(group, '#e5e7eb'),
                                             edgecolor='white'))
            ax.text(0.008, y - dy / 2, group, ha='left', va='center',
                    fontsize=9, weight='bold', color='#1f2937')
            y -= dy
            current_group = group

        s = summary[cond]
        is_ref = group.startswith('ACDC')
        mr = s['majority_rate']; lo = s['ci_lo']; hi = s['ci_hi']
        n = s['n']; src = s.get('src', '')

        display_name = (AUX_FULL[cond]['label'] if cond in AUX_FULL
                        else DISPLAY_RENAME.get(cond, cond))
        cells = [
            display_name,
            f'{n}',
            f'{mr:.3f}\n({src})',
            f'[{lo:.3f}, {hi:.3f}]',
        ]
        cell_colors = ['white', 'white', colour_for_rate(mr, is_ref), 'white']

        for m in models:
            entry = per_model_master.get(m, {}).get(cond)
            if entry is None:
                cells.append('-')
                cell_colors.append('#f9fafb')
            else:
                rate, tag = entry
                cells.append(f'{rate:.3f}\n({tag})')
                cell_colors.append(colour_for_rate(rate, is_ref))

        for i, (txt, bg) in enumerate(zip(cells, cell_colors)):
            x0 = col_x[i]; x1 = col_x[i + 1]
            ax.add_patch(mpatches.Rectangle((x0, y - dy), x1 - x0, dy,
                                             facecolor=bg, edgecolor='#e5e7eb'))
            align = 'left' if i == 0 else 'center'
            tx = x0 + 0.008 if i == 0 else (x0 + x1) / 2
            ax.text(tx, y - dy / 2, txt, ha=align, va='center', fontsize=8.5)
        y -= dy

    # Legend
    leg_y = bottom_reserved - 0.04
    legend_items = [
        ('>= 0.85',    '#bbf7d0'),
        ('0.70-0.85',  '#fef08a'),
        ('0.40-0.70',  '#fed7aa'),
        ('< 0.40',     '#fecaca'),
        ('reference',  '#e5e7eb'),
        ('not run',    '#f9fafb'),
    ]
    x = 0.02
    ax.text(x, leg_y + 0.025, 'Colour = accept rate:', fontsize=9, weight='bold')
    for label, col in legend_items:
        ax.add_patch(mpatches.Rectangle((x, leg_y - 0.005), 0.02, 0.02,
                                         facecolor=col, edgecolor='#6b7280'))
        ax.text(x + 0.024, leg_y + 0.005, label, fontsize=9, va='center')
        x += 0.11

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    pdf_path = OUT_DIR / f'{out_stem}.pdf'
    png_path = OUT_DIR / f'{out_stem}.png'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    print(f'Saved: {pdf_path}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    # Register master group colors now that GROUP_COLOR is defined
    for _g, _c in MASTER_GROUP_COLORS.items():
        GROUP_COLOR[_g] = _c
    main()
    _master_table()
