# Rain-heavy re-validation runbook

> Created 2026-04-21. Use after `ip2p_rain_heavy` (physics-overlay-on-light variant)
> is generated and before Batch B narrative rewrite.

## Context

User decided (DEVLOG 2026-04-21) that rain gets a 2-level scheme like snow:
- `rain` (light) = existing IP2P v4 output, g=10, LPIPS+SSIM filtered, 1,652 rows CS test
- `rain_heavy` = same light output + `add_natural_rain(intensity='heavy_fog')` physics overlay, no extra diffusion

Paper (Batch B) needs fresh validation numbers for `rain_heavy` across 6 metrics,
comparable to existing `rain` light and to snow_light/snow_heavy.

## Patches already applied (2026-04-21)

| File | Change |
|---|---|
| [`validation/extract_dino_ssim_all.py`](../validation/extract_dino_ssim_all.py) | Added `ip2p_rain_heavy` entry (`IP2P_DATA / "rain_heavy"`) |
| [`validation/compute_relative_mahalanobis.py`](../validation/compute_relative_mahalanobis.py) | Added `diffusion_rain_heavy` → target `rain`, `dino_csv="ip2p_rain_heavy"` |
| [`validation/weather_classifier.py`](../validation/weather_classifier.py) | Added `diffusion_rain_heavy` → expected `rain/storm` |
| [`validation/compute_fid_kid.py`](../validation/compute_fid_kid.py) | Added `diffusion_rain_heavy` → reference `rain` |
| [`validation/compute_texture_fidelity.py`](../validation/compute_texture_fidelity.py) | Added `diffusion_rain_heavy` |
| [`validation/vlm_jury/data_loader.py`](../validation/vlm_jury/data_loader.py) | Added `ip2p_rain_heavy` → weather `rain`, `type=dir` |
| [`validation/make_retention_charts_all.py`](../validation/make_retention_charts_all.py) | Added `ip2p_rain_heavy` to DISPLAY_ORDER, relabelled existing `ip2p_rain` as "IP2P Rain (light)" |

Each script now has both `diffusion_rain` / `ip2p_rain` (light, unchanged) and the new heavy entry.

## Stale-path flag (separate issue, not addressed here)

`weather_classifier.py`, `compute_fid_kid.py`, `compute_texture_fidelity.py`, and
`vlm_jury/data_loader.py` still reference `diffusion_rain` at the OLD path
`$CONSYNTH_DATA_ROOT/output/construction_site_test/diffusion_rain_heavy/`, which is
the pre-2026-04-20 location. The data there is the same v4 data now canonically
mirrored at `augmentation_data/.../test/rain/` (DEVLOG 2026-04-20 fixed
`extract_dino_ssim_all.py` + `compute_relative_mahalanobis.py` but not these
four). Only a concern if the OLD path disappears (disk reorg).

## Submission order

### Step 1 — DINO/SSIM extraction (GPU, ~15-25 min)

```bash
sbatch jobs/extract_dino_ssim_rain_heavy.sh
```

Produces `validation/results/dino_ssim/ip2p_rain_heavy.csv`.
Uses `vlm-new` env (DINOv3). Re-runs `extract_dino_ssim_all.py` — it skips
existing CSVs (logic at line 147), so compute is only for `ip2p_rain_heavy`.

**Why first**: Downstream scripts (Mahalanobis filter, retention chart, sample
selection for Kaggle) depend on this CSV existing.

### Step 2 — Retention chart refresh (CPU, <1 min, on login node)

After Step 1 done:
```bash
cd /users/PGS0407/binben14/VietHuy/ConSynth-X
conda activate VLM
python validation/make_retention_charts_all.py
```

Updates `validation/results/dino_ssim_retention_all.{png,pdf}` with 15 conditions
(adds IP2P Rain heavy). Relabels existing "IP2P Rain" → "IP2P Rain (light)".

### Step 3 — Parallel batch of independent validations (GPU, ~2-4 hrs each)

Can submit in parallel — no dependencies between them:

```bash
sbatch jobs/rerun_weather_cls_rain_heavy.sh          # Weather classifier (SigLIP2)
sbatch jobs/rerun_fid_kid_rain_heavy.sh              # FID/KID vs 3 real-weather references
sbatch jobs/rerun_mahalanobis_rain_heavy.sh          # CLIP + DINOv3 relative Mahalanobis
sbatch jobs/rerun_texture_fidelity_rain_heavy.sh     # GLCM+LBP+DCT+Haralick + belief fusion
```

Each recomputes ALL conditions (overwrites existing JSON). If existing numbers
are already cited in figures, back up `validation/results/{weather_cls,fid_kid,relative_mahalanobis,texture_fidelity}/`
before submitting.

### Step 4 — VLM Jury (GPU, ~2-4 hrs × 3 judges)

```bash
bash jobs/rerun_vlm_jury_rain_heavy.sh
```

Submits 3 judge jobs (Qwen2.5-VL, InternVL2.5, Phi-4). Uses `--resume` flag to
extend existing checkpoint with new rain_heavy samples (50 synthetic). After
all 3 done:

```bash
python validation/vlm_jury/analyze_results.py
```

Updates `validation/results/vlm_jury/vlm_jury_summary.json`.

### Step 5 — Sanity check

```bash
ls validation/results/dino_ssim/ip2p_rain_heavy.csv
grep '"diffusion_rain_heavy"\|"ip2p_rain_heavy"' validation/results/**/*.json | head
```

Expect each of the 5 metric JSON files to contain the new condition.

## Expected outcomes (rough priors)

| Metric | rain (light) existing | rain_heavy expected | Why |
|---|---|---|---|
| DINO mean | 0.871 (2026-04-20 v4) | **lower** (0.75-0.82) | physics overlay adds pixel-level noise, reduces DINO CLS similarity |
| SSIM mean | 0.756 | **lower** (0.55-0.65) | blur + streaks + fog degrade structural similarity |
| Weather classifier rain% | 63.5% | **higher** (80-92%) | heavier streaks + fog → more "rain/storm"-like |
| FID vs ACDC rain | inconclusive (+1%) | **lower** gap | heavier atmosphere closer to real wet driving scenes |
| Mahalanobis CLIP | 25% gap closed | **higher** (35-45%) | more adverse-atmosphere-like |
| Texture fidelity H | 0.854 | **lower** (0.3-0.6) | blur removes HF content → larger DCT/Haralick Wasserstein |
| VLM Jury accept | 72% | **unknown**; possibly higher (~80%) | stronger visual rain, but blur may trigger "unrealistic" flag |

## After re-validation: Batch B paper rewrite

1. Update Table 1 dataset statistics — add `IP2P Rain heavy` row
2. Update §3.1 Weather Augmentation — frame: "rain has 2 intensities; heavy uses
   physics overlay on top of light (not a 2nd diffusion pass) due to diffusion
   hallucination at high guidance — document in DEVLOG 2026-04-21."
3. Update §4 Technical Validation tables — add rain_heavy column to all 6 metric
   tables, note expected pattern (trade-off: stronger weather recognizability vs
   texture fidelity)
4. Move `Weather (style)` into Supplementary Ablation (not main Table 1)
5. Update Figure 5 retention chart caption — 15 conditions
6. Abstract + Intro: recount pipelines ("three main pipelines: weather [rain
   2-level / snow 2-level / fog 3-level], day-to-night, outpainting. Weather
   augmentation uses IP2P diffusion for light intensity and physics overlay for
   heavy intensity; style transfer kept as ablation baseline.")

## Separate user task (2026-04-21): Kaggle sample filtered by DINO threshold

```bash
# After Step 1 (DINO CSV for rain_heavy exists):
python scripts/update_kaggle_rain_dino_filtered.py --dry-run  # preview counts
python scripts/update_kaggle_rain_dino_filtered.py            # overwrite samples

cd augmentation_data_sample
kaggle datasets version -m "Rain samples filtered by DINO threshold (light>=0.75, heavy>=0.70, n=300 each)"
```

Replaces `cs_diff_rain_test.arrow` and `cs_diff_rain_heavy_test.arrow` in the
Kaggle sample release with 300 DINO-filtered rows each. Other sample files
untouched.
