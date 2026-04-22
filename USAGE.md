# Usage Guide — Processing ConSynth-X Data

> Step-by-step workflow for loading ConSynth-X Arrow data and producing the outputs reported in the paper
> (object-detection robustness tables, realism validation, cross-condition comparisons).
>
> Reviewer-facing. Assumes you already completed [`INSTALL.md`](INSTALL.md) (clone + submodules + `.env` +
> `bash weights/download.sh`).

---

## 0. Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.10 |
| GPU | 1× A100 for IP2P / FLUX / VLM; not required for loading / filtering / YOLO inference |
| Disk | ~15 GB (repo + weights); +94 GB if downloading the full HF dataset |
| Env vars | `CONSYNTH_REPO_ROOT` and `CONSYNTH_DATA_ROOT` exported (see `.env.example`) |
| Conda env | `VLM` (default) or `vlm-new` (for DINOv3 / Qwen3-VL) |

---

## 1. Dataset layout

ConSynth-X stores all augmented images as **HuggingFace Arrow (IPC stream)** files with JPEG-encoded
image bytes embedded in the row. No loose images; no manifest CSV required.

```
$CONSYNTH_DATA_ROOT/augmentation_data/           # ← set this to the data root
├── construction_site/        # Construction Site 10k (HF, 3 object classes)
│   ├── night/test/           # CycleGAN-Turbo day→night
│   ├── fog/diffusion/test/{heavy,medium,light}/    # Koschmieder fog, 3 zones
│   ├── small/test/           # FLUX.1-Fill-dev outpainting
│   └── rain_snow/diffusion/
│       ├── test/rain/        # IP2P rain LIGHT (g=10)
│       ├── test/rain_heavy/  # IP2P light + physics-only overlay (2026-04-21)
│       ├── test/snow_light/  # IP2P snow LIGHT (g=8)
│       ├── test/snow_heavy/  # IP2P snow HEAVY (g=12)
│       └── train/…           # Same layout for train split
├── soda_voc/                 # SODA VOC (15 object classes, Pascal VOC annotations)
└── soda_ktsh/                # SODA KTSH (image captions)
```

Each condition folder contains either a single `*.arrow` file or multiple `batch_<start>-<end>.arrow`
shards. Meta CSVs (`meta_<start>-<end>.csv`) carry the per-image `ssim`/`lpips`/`status` for rain.

Per-image DINOv3 + SSIM scores (versus the original clean image) are released in
`validation/results/dino_ssim/*.csv` — use these for quality filtering.

---

## 2. Step-by-step processing

### Step 2.1 — Load a single Arrow file

```python
import io
import pyarrow as pa
from PIL import Image

def load_arrow(path):
    """Read an Arrow IPC stream into a pa.Table."""
    with open(path, "rb") as f:
        return pa.ipc.open_stream(f).read_all()

def get_image(table, row_idx):
    """Decode an image from the 'image' column (struct<bytes, path>)."""
    cell = table.column("image")[row_idx].as_py()
    data = cell["bytes"] if isinstance(cell, dict) else cell
    return Image.open(io.BytesIO(data)).convert("RGB")

# Example: IP2P rain heavy, first shard of Construction Site test
t = load_arrow("augmentation_data/construction_site/rain_snow/diffusion/test/rain_heavy/batch_0-500.arrow")
print(f"{t.num_rows} rows; schema fields: {[f.name for f in t.schema]}")
img = get_image(t, 0)
img.save("/tmp/sample_rain_heavy.jpg")
```

### Step 2.2 — Concatenate all shards of a condition

Some conditions (`rain/`, `rain_heavy/`, `snow_light/`, fog zones) ship as multiple `batch_*.arrow`
shards. Concatenate before iterating:

```python
from pathlib import Path

def concat_shards(dir_path):
    shards = sorted(Path(dir_path).glob("*.arrow"))
    tables = [load_arrow(p) for p in shards]
    return pa.concat_tables(tables, promote_options="default")

rain_heavy_all = concat_shards(
    "augmentation_data/construction_site/rain_snow/diffusion/test/rain_heavy/"
)
print(f"rain_heavy test: {rain_heavy_all.num_rows} rows")  # → 1,652
```

### Step 2.3 — Read annotations

Schema differs by source dataset. Example for Construction Site (3 classes + 4 safety rules +
scene attributes):

```python
row = rain_heavy_all.slice(0, 1).to_pylist()[0]
print("image_id:  ", row["image_id"])
print("caption:   ", row["image_caption"])
print("rebar bboxes (normalized xyxy):", row["rebar"])  # [[x1,y1,x2,y2], ...]
print("worker bboxes:                  ", row["worker_with_white_hard_hat"])
print("rule_3_violation:               ", row["rule_3_violation"])  # {"bounding_box": ..., "reason": ...}
print("illumination / view / camera:    ",
      row["illumination"], "/", row["view"], "/", row["camera_distance"])
```

SODA VOC rows expose `objects_name`, `objects_bbox`, `filename`; SODA KTSH rows expose `filename`,
`captions`. Schema reference: [`augmentation_data/README.md`](augmentation_data/README.md#schemas).

**Bounding box convention**: normalised `[x1, y1, x2, y2]` in image-width/height units (all float in
`[0, 1]`). Bounding boxes are preserved verbatim from the source image across all pixel-level
augmentations (weather, night, rain_heavy); only `small/` (FLUX outpainting) re-maps them — already
applied at generation time, no user-side transform needed.

### Step 2.4 — Pair with the clean original

Every augmented row's `image_id` (and `ref_id`) matches the clean original. Use this to build
paired `(original, augmented)` inputs — required for SSIM, LPIPS, DINO, FID reference loading, and
VLM paired prompts.

```python
orig_path = f"{CONSYNTH_DATA_ROOT}/augmentation_data_arrow/construction_site_test.arrow"
orig = load_arrow(orig_path)
orig_map = {
    str(orig.column("image_id")[i].as_py()): i
    for i in range(orig.num_rows)
}

aug_row_idx = 0
aug_id = rain_heavy_all.column("image_id")[aug_row_idx].as_py()
orig_idx = orig_map[str(aug_id)]

aug_img  = get_image(rain_heavy_all, aug_row_idx)
orig_img = get_image(orig, orig_idx)
# → pass to paired metrics (SSIM, DINO, VLM jury, etc.)
```

### Step 2.5 — Filter by quality metric

Two strategies are supported; apply either at load time.

**(a) Use the per-row SSIM/LPIPS already computed at generation time** (rain variants only; applies to
the filter snapshot):

```python
import numpy as np

ssims = np.array(rain_heavy_all.column("ssim").to_pylist())
# Keep rows with SSIM ≥ 0.5 (loose) or use the paper's default (rain ≥ 0.60, snow ≥ 0.50)
keep = ssims >= 0.50
print(f"{keep.sum()} / {len(keep)} rows pass SSIM ≥ 0.5")
```

`rain_heavy` rows inherit the SSIM / LPIPS / `status` fields from the light stage (physics
overlay does not alter these).

**(b) Use the released DINOv3 + SSIM CSVs** (recommended for paper-grade filtering):

```python
import csv

dino = {}
with open("validation/results/dino_ssim/ip2p_rain_heavy.csv") as f:
    for r in csv.DictReader(f):
        dino[r["image_id"]] = float(r["dino_sim"])

# e.g., heavy threshold 0.70 (matches Kaggle v6 sample release)
eligible = {k for k, v in dino.items() if v >= 0.70}
print(f"{len(eligible)} / {len(dino)} eligible rows at DINO ≥ 0.70")
```

Thresholds we used for the Kaggle v6 sample release:

| Variant | CSV | Threshold | Pass rate |
|---|---|---|---|
| rain (light) | `ip2p_rain.csv` | DINO ≥ 0.75 | 1,515 / 1,652 = 91.7% |
| rain_heavy | `ip2p_rain_heavy.csv` | DINO ≥ 0.70 | 1,219 / 1,652 = 73.8% |

Reproduce the exact 300-row Kaggle subsets with
[`scripts/update_kaggle_rain_dino_filtered.py`](scripts/update_kaggle_rain_dino_filtered.py).

---

## 3. Downstream task recipes

Each recipe is the exact pipeline that produced a specific table/figure in the paper.

### 3.1 — Object-detection robustness (YOLOv8, paper Tables 7–8)

Input: augmented Arrow + originals; output: `metrics.json` with per-class mAP@0.5 and mAP@0.5:0.95
for each condition.

```bash
conda activate VLM

# Export Arrow → YOLO txt format
python benchmarks/detection/extract_val_from_arrow.py \
    --arrow-glob "augmentation_data/construction_site/rain_snow/diffusion/test/rain_heavy/*.arrow" \
    --output     "$CONSYNTH_DATA_ROOT/validation_data/rain_heavy_yolo/"

# Train YOLOv8n on the clean+augmented mix (reported "All conditions" config)
python benchmarks/detection/train.py \
    --config benchmarks/detection/configs/all_conditions.yaml

# Evaluate a trained checkpoint on each condition
python benchmarks/detection/evaluate.py \
    --weights  validation_data/downstream_detection/all_conditions/weights/best.pt \
    --test-dir validation_data/rain_heavy_yolo/
```

Cross-condition comparison (one detector, multiple test sets) is orchestrated by
[`experiments/cross_condition_eval.py`](experiments/cross_condition_eval.py); SLURM submitter in
`benchmarks/detection/SODA/submit_yolo_soda.py`.

### 3.2 — VLM benchmarking (Tasks: captioning / VQA / detection)

```bash
# Single VLM on one condition
python benchmarks/vlm/run.py \
    --model qwen2.5-vl-7b \
    --task  description \
    --condition rain_heavy

# Batch all 11 models × all conditions (SLURM)
python benchmarks/vlm/scripts/submit_jobs.py
```

Output: `benchmarks/vlm/results/<model>/<task>/<condition>.json` with per-image predictions +
aggregate BERTScore / BLEU / mAP.

### 3.3 — Realism validation (Tables 4–8)

Full re-run of the 7 validation approaches (rain_heavy runbook:
[`docs/rain_heavy_revalidation_runbook.md`](docs/rain_heavy_revalidation_runbook.md)):

```bash
# DINO + SSIM retention (GPU, ~25 min)
sbatch jobs/extract_dino_ssim_rain_heavy.sh

# FID/KID vs ACDC + WeatherNet + WeatherBench (GPU, ~30 min)
sbatch jobs/rerun_fid_kid_rain_heavy.sh

# Weather classifier SigLIP2 (GPU, ~1 hr)
sbatch jobs/rerun_weather_cls_rain_heavy.sh

# CLIP + DINOv3 relative Mahalanobis (GPU, ~1 hr)
sbatch jobs/rerun_mahalanobis_rain_heavy.sh

# Texture fidelity + Dempster-Shafer belief fusion (CPU, ~10 min)
sbatch jobs/rerun_texture_fidelity_rain_heavy.sh

# VLM jury (3 judges × ~3 hr each, can run in parallel)
bash jobs/rerun_vlm_jury_rain_heavy.sh

# After all above: regenerate retention chart
python validation/make_retention_charts_all.py
```

Output locations:

| Metric | Results |
|---|---|
| DINO/SSIM | `validation/results/dino_ssim/<condition>.csv` |
| FID/KID | `validation/results/fid_kid/<reference>/results.json` + barplot PDFs |
| Weather classifier | `validation/results/weather_cls/{results.json, heatmap.pdf}` |
| Mahalanobis | `validation/results/relative_mahalanobis/{results.json, plots.pdf}` |
| Texture fidelity | `validation/results/texture_fidelity/{texture_fidelity_results.json, belief_fusion_results.json}` |
| VLM jury | `validation/results/vlm_jury/{<model>_results.json, vlm_jury_summary.json}` |

### 3.4 — Reproduce the augmentation itself (regenerate augmented data from originals)

Not required for dataset use, but documented for verifiability:

```bash
# IP2P rain (light) on Construction Site test set
cd generation/weather/rain_snow/diffusion
sbatch submit_test.py     # internally runs batch_worker.py with g=10, v4 SSIM+LPIPS filter

# Rain heavy = light + physics-only overlay (CPU, no GPU; deterministic)
python apply_heavy_physics_to_light.py \
    --input-dir  ../../../augmentation_data/construction_site/rain_snow/diffusion/test/rain \
    --output-dir ../../../augmentation_data/construction_site/rain_snow/diffusion/test/rain_heavy \
    --seed 42
```

Per-parameter justifications (guidance scale, physics config, SSIM/LPIPS thresholds): see
[`docs/methods.md`](docs/methods.md) §1.2.

---

## 4. Common pitfalls

1. **Don't forget `set -a; source .env; set +a`** before running any Python entry point. Scripts
   resolve paths via `CONSYNTH_REPO_ROOT` / `CONSYNTH_DATA_ROOT` — a silent wrong path is the #1
   source of "file not found" errors (see `DEVLOG.md` 2026-04-22, job 5023889 failure).
2. **Arrow open mode**: use `pa.ipc.open_stream(f)` for ConSynth-X files (IPC streaming format).
   HuggingFace's `Dataset.from_file()` also works and is more convenient for integrating with
   `datasets.Features`.
3. **Large-binary slicing**: when filtering rows by id, slice row-by-row (`[table.slice(i, 1) for i in idx]`)
   and `pa.concat_tables` — a single `table.take(idx)` on a huge `image` column can overflow int32
   offsets. See `scripts/update_kaggle_rain_dino_filtered.py:filter_and_sample` for a safe implementation.
4. **`rain_heavy` is NOT a second diffusion pass**. It is CPU physics overlay applied to the
   already-filtered `rain` (light) output; metadata (bboxes, ssim, lpips, status) is inherited
   verbatim from light. See `docs/methods.md` §1.2 IP2P Rain 2-Intensity.
5. **`fog/diffusion/`** is physically named but holds **Koschmieder physics** output, not diffusion.
   Legacy folder naming.
6. **Style-transfer rain/snow is ablation-only** (moved to `experiments/ablation_style_transfer/`
   on 2026-04-21). Main-release filtering + paper primary tables use IP2P only.
7. **Deterministic seeds**: generation workers use `seed_base=42` + `image_index`; for exact
   bit-for-bit reproduction, do not override `--seed`.

---

## 5. What "output" looks like

Running the recipes above produces the following artifacts, corresponding to paper content:

| Recipe | Output file(s) | Paper artifact |
|---|---|---|
| 3.1 detection | `validation_data/downstream_detection/<config>/metrics.json` | Tables 7–8 cross-condition mAP |
| 3.2 VLM | `benchmarks/vlm/results/…/*.json` + CSVs | VLM benchmark tables |
| 3.3 DINO/SSIM | `validation/results/dino_ssim/*.csv` + `dino_ssim_retention_all.{pdf,png}` | Fig. 4–5 retention charts |
| 3.3 FID/KID | `validation/results/fid_kid/*.pdf` + LaTeX | Table 5 distributional fidelity |
| 3.3 Weather cls | `validation/results/weather_cls/heatmap.pdf` + JSON | Table 4 recognizability |
| 3.3 Mahalanobis | `validation/results/relative_mahalanobis/*.pdf` | Table 6 CLIP / DINOv3 gap |
| 3.3 Texture | `validation/results/texture_fidelity/belief_fusion_results.json` | Table 8 belief H/H̄ |
| 3.3 VLM jury | `validation/results/vlm_jury/vlm_jury_summary.json` | Table 7 jury acceptance |
| 3.4 regenerate | `augmentation_data/construction_site/.../{rain,rain_heavy,…}/` | The dataset itself |

Backups of pre-rerun results: `validation/results/<metric>_backup_20260421_2329/`.

---

## 6. Reproducibility

- Code — Apache 2.0, pinned via git commit hash in the repository
- Dataset — CC BY-NC 4.0, released at (a) HuggingFace `Ben11304/ConSynth-X-augmentation` (full ~94 GB),
  (b) Kaggle `viethuyduong/consynth-x-augmentation-sample` (CC0-1.0, ~100-300 rows/condition)
- Third-party weights — Apache-2.0 (VGG) + MIT (day2night) redistributed unchanged; SHA256 in
  [`weights/checksums.sha256`](weights/checksums.sha256)
- Submodule — `img2img-turbo @ 86f5414` (MIT)
- Vendored — `Weather_Effect_Generator @ 7d62b67` (Apache-2.0), attribution in
  [`generation/weather/libs/Weather_Effect_Generator/NOTICE.md`](generation/weather/libs/Weather_Effect_Generator/NOTICE.md)
- HuggingFace / torch.hub models (IP2P, FLUX, MiDaS, Depth-Anything, weather classifier) — loaded
  at runtime without `revision=` pinning (deliberate trade-off documented in
  [`docs/data_sources.md`](docs/data_sources.md) §6.2). The authoritative record of exact bytes
  used is the released dataset itself.

For the full developer log (day-by-day decisions, pilot iterations, incident post-mortems):
[`DEVLOG.md`](DEVLOG.md).

---

## 7. Where to ask questions

- Issue tracker: GitHub `<repo-url>/issues`
- Dataset questions: dataset card contact (see [`data_card.md`](data_card.md))
- Method / parameter justifications: [`docs/methods.md`](docs/methods.md)
- Literature citations: [`docs/literature.md`](docs/literature.md)
