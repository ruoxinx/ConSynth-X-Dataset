#!/usr/bin/env python3
"""
Generate all slide figures from Benchmark_runner eval results.
Outputs to slide/figures/ for LaTeX inclusion.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

OUT_DIR = Path(__file__).parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)

BR = _BR_ROOT

# ── Color scheme ──
COND_COLORS = {
    'original': '#2196F3',
    'weather':  '#4CAF50',
    'night':    '#9C27B0',
    'small':    '#FF9800',
}
CONDITIONS = ['original', 'weather', 'night', 'small']

MODEL_DISPLAY = {
    'InternVL2_5-26B-MPO': 'InternVL2.5-26B',
    'Phi-4-multimodal-instruct': 'Phi-4',
    'Qwen3-VL-8B-Instruct': 'Qwen3-VL-8B',
    'gpt-4o-mini': 'GPT-4o-mini',
    'gemma-3-27b-it': 'Gemma-3-27B',
    'llava-1.5-7b-hf': 'LLaVA-1.5-7B',
}

MODEL_COLORS = {
    'InternVL2.5-26B': '#1565C0',
    'Phi-4':           '#E65100',
    'Qwen3-VL-8B':     '#2E7D32',
    'GPT-4o-mini':      '#6A1B9A',
    'Gemma-3-27B':     '#C62828',
    'LLaVA-1.5-7B':    '#78909C',
}

# ════════════════════════════════════════
# 1. VQA SAFETY FIGURES
# ════════════════════════════════════════

print("Loading VQA eval results...")
with open(BR / 'output' / 'vqa_eval_results.json') as f:
    vqa_raw = json.load(f)

# Build DataFrame
rows = []
for entry in vqa_raw:
    model = MODEL_DISPLAY.get(entry['model'], entry['model'])
    cond = entry['condition']
    n_errors = entry['n_errors']
    n_samples = entry['n_samples']
    cls = entry.get('classification', {})
    iou = entry.get('bbox_iou', {})

    row = {'model': model, 'condition': cond, 'n_samples': n_samples, 'n_errors': n_errors}
    for rule_id in ['rule_1', 'rule_2', 'rule_3', 'rule_4']:
        rc = cls.get(rule_id, {})
        ri = iou.get(rule_id, {})
        row[f'{rule_id}_P'] = rc.get('precision', 0)
        row[f'{rule_id}_R'] = rc.get('recall', 0)
        row[f'{rule_id}_F1'] = rc.get('f1', 0)
        row[f'{rule_id}_IoU'] = ri.get('mean_iou', 0) if isinstance(ri, dict) else 0

    # Overall micro
    tp_all = sum(cls.get(r, {}).get('tp', 0) for r in ['rule_1','rule_2','rule_3','rule_4'])
    fp_all = sum(cls.get(r, {}).get('fp', 0) for r in ['rule_1','rule_2','rule_3','rule_4'])
    fn_all = sum(cls.get(r, {}).get('fn', 0) for r in ['rule_1','rule_2','rule_3','rule_4'])
    row['overall_P'] = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
    row['overall_R'] = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
    row['overall_F1'] = 2 * row['overall_P'] * row['overall_R'] / (row['overall_P'] + row['overall_R']) if (row['overall_P'] + row['overall_R']) > 0 else 0

    # Overall IoU (mean of rules that have values)
    iou_vals = [row[f'{r}_IoU'] for r in ['rule_1','rule_2','rule_3','rule_4'] if row[f'{r}_IoU'] > 0]
    row['overall_IoU'] = np.mean(iou_vals) if iou_vals else 0

    rows.append(row)

vqa_df = pd.DataFrame(rows)

# Filter out LLaVA (all errors)
vqa_df = vqa_df[vqa_df['model'] != 'LLaVA-1.5-7B']
models_vqa = [m for m in ['Gemma-3-27B', 'InternVL2.5-26B', 'Qwen3-VL-8B', 'Phi-4', 'GPT-4o-mini'] if m in vqa_df['model'].values]

# ── Fig 1: VQA F1 Radar by Condition ──
print("  Generating VQA F1 radar by condition...")
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2 * pi, len(CONDITIONS), endpoint=False).tolist()
angles += angles[:1]

for model in models_vqa:
    vals = []
    for cond in CONDITIONS:
        row = vqa_df[(vqa_df['model'] == model) & (vqa_df['condition'] == cond)]
        vals.append(row['overall_F1'].values[0] * 100 if len(row) > 0 else 0)
    vals += vals[:1]
    color = MODEL_COLORS.get(model, 'gray')
    ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=color, markersize=5)
    ax.fill(angles, vals, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([c.capitalize() for c in CONDITIONS], fontsize=11)
ax.set_ylim(0, 16)
ax.set_yticks([4, 8, 12, 16])
ax.set_yticklabels(['4%', '8%', '12%', '16%'], fontsize=8, color='gray')
ax.set_title('VQA Safety F1 by Condition', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / 'vqa_f1_radar_conditions.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'vqa_f1_radar_conditions.png', bbox_inches='tight', dpi=200)
plt.close()

# ── Fig 2: VQA Per-Rule F1 Heatmap ──
print("  Generating VQA per-rule F1 heatmap...")
fig, ax = plt.subplots(figsize=(8, 4.5))

rules = ['rule_1', 'rule_2', 'rule_3', 'rule_4']
rule_labels = ['Rule 1\n(PPE)', 'Rule 2\n(Harness)', 'Rule 3\n(Edge)', 'Rule 4\n(Excavator)']

# Use original condition
pivot_data = []
for model in models_vqa:
    row_data = []
    for rule in rules:
        r = vqa_df[(vqa_df['model'] == model) & (vqa_df['condition'] == 'original')]
        row_data.append(r[f'{rule}_F1'].values[0] * 100 if len(r) > 0 else 0)
    pivot_data.append(row_data)

pivot_arr = np.array(pivot_data)
im = ax.imshow(pivot_arr, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(rule_labels)))
ax.set_xticklabels(rule_labels, fontsize=10)
ax.set_yticks(range(len(models_vqa)))
ax.set_yticklabels(models_vqa, fontsize=10)

for i in range(len(models_vqa)):
    for j in range(len(rules)):
        val = pivot_arr[i, j]
        color = 'white' if val > pivot_arr.mean() + pivot_arr.std() else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, label='F1 %')
ax.set_title('Per-Rule F1% on Original Condition', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'vqa_per_rule_heatmap.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'vqa_per_rule_heatmap.png', bbox_inches='tight', dpi=200)
plt.close()

# ── Fig 3: VQA F1 Grouped Bar ──
print("  Generating VQA F1 grouped bar chart...")
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(models_vqa))
width = 0.18

for i, cond in enumerate(CONDITIONS):
    vals = []
    for model in models_vqa:
        r = vqa_df[(vqa_df['model'] == model) & (vqa_df['condition'] == cond)]
        vals.append(r['overall_F1'].values[0] * 100 if len(r) > 0 else 0)
    bars = ax.bar(x + i * width, vals, width, label=cond.capitalize(), color=COND_COLORS[cond])
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('F1 %')
ax.set_title('VQA Safety Overall F1 by Model and Condition', fontweight='bold', fontsize=13)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models_vqa, fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0, 18)
plt.tight_layout()
plt.savefig(OUT_DIR / 'vqa_f1_bar.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'vqa_f1_bar.png', bbox_inches='tight', dpi=200)
plt.close()

# ════════════════════════════════════════
# 2. DESCRIPTION FIGURES
# ════════════════════════════════════════

print("\nLoading Description eval results...")
with open(BR / 'output' / 'description_eval_results.json') as f:
    desc_raw = json.load(f)

desc_df = pd.DataFrame(desc_raw)
desc_df['model'] = desc_df['model'].map(lambda x: MODEL_DISPLAY.get(x, x))
models_desc = [m for m in ['Phi-4', 'InternVL2.5-26B', 'Gemma-3-27B', 'Qwen3-VL-8B', 'LLaVA-1.5-7B'] if m in desc_df['model'].values]

# ── Fig 4: Description Radar (Original condition) ──
print("  Generating description radar chart...")
radar_metrics = ['bleu', 'rouge_l', 'meteor', 'bertscore_f1', 'sbert_similarity', 'clipscore']
radar_labels = ['BLEU', 'ROUGE-L', 'METEOR', 'BERT-F1', 'SBERT-Sim', 'CLIPScore']

df_orig = desc_df[desc_df['condition'] == 'original']
max_vals = {m: df_orig[m].max() for m in radar_metrics}

angles = np.linspace(0, 2 * pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for model in models_desc:
    row = df_orig[df_orig['model'] == model]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    raw_vals = [row[m] for m in radar_metrics]
    norm_vals = [row[m] / max_vals[m] * 100 if max_vals[m] > 0 else 0 for m in radar_metrics]
    norm_vals += norm_vals[:1]
    color = MODEL_COLORS.get(model, 'gray')
    ax.plot(angles, norm_vals, 'o-', linewidth=2, label=model, color=color, markersize=5)
    ax.fill(angles, norm_vals, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0, 110)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='gray')
ax.set_title('Description Metrics — Original\n(normalized by best)', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / 'desc_radar_original.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'desc_radar_original.png', bbox_inches='tight', dpi=200)
plt.close()

# ── Fig 5: Description Heatmap (all metrics x conditions) ──
print("  Generating description heatmap...")
all_metrics = ['bleu', 'rouge_l', 'meteor', 'bertscore_f1', 'sbert_similarity', 'clipscore']
all_labels = ['BLEU', 'ROUGE-L', 'METEOR', 'BERT-F1', 'SBERT-Sim', 'CLIPScore']

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
flat_axes = axes.flatten()

for idx, (ax, metric, label) in enumerate(zip(flat_axes, all_metrics, all_labels)):
    pivot = desc_df.pivot(index='model', columns='condition', values=metric)
    pivot = pivot.reindex(columns=CONDITIONS, index=models_desc)

    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([c.capitalize() for c in CONDITIONS], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(models_desc)))
    ax.set_yticklabels(models_desc, fontsize=9)
    ax.set_title(label, fontweight='bold', fontsize=11)

    for i in range(len(models_desc)):
        for j in range(len(CONDITIONS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > np.nanmean(pivot.values) + np.nanstd(pivot.values) else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)

fig.suptitle('Description Metrics — All Models x Conditions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'desc_heatmap.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'desc_heatmap.png', bbox_inches='tight', dpi=200)
plt.close()

# ── Fig 6: Description Degradation Bar ──
print("  Generating description degradation chart...")
inv_metrics = ['bertscore_f1', 'sbert_similarity', 'clipscore']
inv_labels = ['BERT-F1', 'SBERT-Sim', 'CLIPScore']
aug_conds = ['weather', 'night', 'small']

fig, axes = plt.subplots(1, len(models_desc), figsize=(3.5 * len(models_desc), 4.5))
if len(models_desc) == 1:
    axes = [axes]

for col_idx, model in enumerate(models_desc):
    ax = axes[col_idx]
    orig = desc_df[(desc_df['model'] == model) & (desc_df['condition'] == 'original')]
    if len(orig) == 0:
        continue
    orig = orig.iloc[0]
    x = np.arange(len(inv_metrics))
    width = 0.25
    for i, cond in enumerate(aug_conds):
        row = desc_df[(desc_df['model'] == model) & (desc_df['condition'] == cond)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        drops = [(orig[m] - row[m]) / orig[m] * 100 if orig[m] != 0 else 0 for m in inv_metrics]
        ax.bar(x + i * width, drops, width, label=cond.capitalize(), color=COND_COLORS[cond])

    ax.set_ylabel('% Drop' if col_idx == 0 else '')
    ax.set_title(model, fontweight='bold', fontsize=10)
    ax.set_xticks(x + width)
    ax.set_xticklabels(inv_labels, fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    if col_idx == 0:
        ax.legend(fontsize=8)

fig.suptitle('Description Degradation (% drop from original)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'desc_degradation.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'desc_degradation.png', bbox_inches='tight', dpi=200)
plt.close()

print(f"\nAll figures saved to {OUT_DIR}/")
print("Files:", sorted([f.name for f in OUT_DIR.glob('*.pdf')]))
