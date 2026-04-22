---
license: cc0-1.0
task_categories:
  - object-detection
  - image-classification
  - image-to-image
tags:
  - construction-site
  - data-augmentation
  - weather
  - synthetic
  - diffusion
  - fog
  - rain
  - snow
  - night
  - soda
  - voc
pretty_name: ConSynth-X Augmentation Data
size_categories:
  - 10K<n<100K
---

# ConSynth-X — Weather-Augmented Construction-Site Data

**ConSynth-X** is a multi-condition synthetic-augmentation dataset for construction-site computer
vision. Each clean daytime image from two source datasets (Construction Site 10k, SODA) is
re-rendered under adverse field conditions (**rain**, **snow**, **fog**, **night**, **scale
variation**) while preserving the original bounding boxes, captions, and scene attributes. A
legacy **neural-style-transfer** pipeline is retained in the `experiments/ablation_style_transfer/`
archive of the parent repository for baseline comparison (see provenance).

---

## Table of contents

1. [Dataset description](#1-dataset-description)
2. [Preview](#2-preview)
3. [Directory layout](#3-directory-layout)
4. [File inventory & row counts](#4-file-inventory--row-counts)
5. [Feature schema (per variant)](#5-feature-schema-per-variant)
6. [Quickstart — how to load and use](#6-quickstart--how-to-load-and-use)
7. [Quality filtering](#7-quality-filtering)
8. [Provenance (how each condition was generated)](#8-provenance-how-each-condition-was-generated)
9. [Licence](#9-licence)
10. [Citation](#10-citation)

---

## 1. Dataset description

| Attribute | Value |
|---|---|
| **Name** | ConSynth-X Augmentation Data |
| **Purpose** | Robustness benchmarking of construction CV models under adverse field conditions |
| **Source datasets** | Construction Site 10k (`LouisChen15/ConstructionSite`, CC BY-NC 4.0), SODA VOC (Duan et al. 2022, *Automation in Construction*), SODA-KTSH (Deng et al. 2025, *Buildings*) |
| **Augmented conditions** | `rain` (light + heavy), `snow` (light + heavy), `fog` (light/medium/heavy), `night`, `small` (scale/outpainting), `original` (clean baseline) |
| **File format** | Apache Arrow IPC stream (`.arrow`), JPEG-encoded image bytes embedded in rows |
| **Total size** | ~94 GB across 97 Arrow files |
| **Splits** | `train`, `test` inherited from source datasets |
| **Annotation formats** | Bounding boxes (normalised xyxy), image captions, safety-rule violations, scene attributes (illumination, camera distance, view) — preserved verbatim from the source per image, **NOT re-annotated on the augmented images** |
| **Image resolution** | preserved from source (varies; typically ≤ 1920 × 1080 px for CS, ≤ 4 K for SODA) |
| **Compression** | JPEG quality 95 |

### What makes this dataset distinctive

- **Multi-label condition annotation** — each image can carry multiple condition tags
  (e.g. `rain + low_light`). See `taxonomy/extreme_conditions_definition.md` in the parent repo.
- **Paired clean / augmented** — every augmented row's `image_id` (and `ref_id`) matches an entry
  in the clean `original` split. This enables paired evaluation (SSIM, LPIPS, DINO, VLM jury).
- **Two-intensity rain and snow** — each of rain and snow has a `light` and `heavy` variant.
  *Note:* rain heavy is produced by **physics-only overlay** on the rain-light output (NOT a
  second diffusion pass); snow heavy uses IP2P diffusion at higher guidance. See
  [provenance](#8-provenance-how-each-condition-was-generated).
- **Released with per-image quality metrics** — DINOv3-ViT-L/16 cosine similarity + SSIM
  versus the clean original, one CSV per condition under `validation/results/dino_ssim/`
  in the parent repo. Threshold to taste.

---

## 2. Preview

| Clean | Night | Fog (heavy) | Fog (medium) | Fog (light) |
| :---: | :---: | :---: | :---: | :---: |
| ![](samples/soda_original.jpg) | ![](samples/soda_night.jpg) | ![](samples/soda_fog_heavy.jpg) | ![](samples/soda_fog_medium.jpg) | ![](samples/soda_fog_light.jpg) |

| Rain diffusion (light) | Snow diffusion | Rain style-transfer (ablation) | Snow style-transfer (ablation) | KTSH rain |
| :---: | :---: | :---: | :---: | :---: |
| ![](samples/soda_rain_diffusion.jpg) | ![](samples/soda_snow_diffusion.jpg) | ![](samples/soda_rain_style.jpg) | ![](samples/soda_snow_style.jpg) | ![](samples/ktsh_rain.jpg) |

**Construction-site variant (with rule-violation annotations):**

| CS night | CS fog | CS rain (diffusion) | CS snow (diffusion) | CS rain (ST, ablation) |
| :---: | :---: | :---: | :---: | :---: |
| ![](samples/cs_night.jpg) | ![](samples/cs_fog_heavy.jpg) | ![](samples/cs_rain_diffusion.jpg) | ![](samples/cs_snow_diffusion.jpg) | ![](samples/cs_rain_style.jpg) |

---

## 3. Directory layout

Three root directories, one per source dataset:

```
<dataset-root>/
├── construction_site/   # Construction Site 10k (HF, 3 object classes)
├── soda_voc/            # SODA (Duan et al. 2022, 15 object classes, VOC format)
└── soda_ktsh/           # SODA-KTSH (Deng et al. 2025, image captions)
```

Within each root:

```
<root>/
├── original/                                 # clean baseline (soda_voc has full copy; CS references parent-repo arrow)
├── night/test/                               # CycleGAN-Turbo day→night
├── fog/diffusion/test/{heavy,medium,light}/  # Koschmieder physics fog at 3 visibility zones
├── small/test/                               # FLUX.1-Fill-dev outpainting (scale variation)
└── rain_snow/diffusion/
    ├── {train,test}/rain/                    # IP2P rain LIGHT (g=10)
    ├── {train,test}/rain_heavy/              # physics overlay on light (NO second diffusion pass)
    ├── {train,test}/snow_light/              # IP2P snow LIGHT (g=8)
    └── {train,test}/snow_heavy/              # IP2P snow HEAVY (g=12)
```

*Note on folder name `fog/diffusion/`:* historical naming; fog is actually produced by the
**Koschmieder atmospheric-scattering model** + Depth Anything V2 monocular depth estimation, not a
diffusion model.

---

## 4. File inventory & row counts

- **97 Arrow files**, total ~94 GB
- **56 CSV files** carrying per-image SSIM scores for the ablation style-transfer data
- **0 loose images** — all image bytes packed inside Arrow

### Headline row counts

| File | Rows | Conditions |
|---|---:|---|
| `soda_voc/original/soda_voc_original_first3000.arrow` | 3,000 | clean baseline |
| `soda_voc/night/soda_day2night.arrow` | 19,846 | CycleGAN night |
| `soda_voc/rain_snow/diffusion/rain.arrow` | 4,790 | IP2P rain LIGHT |
| `soda_voc/rain_snow/diffusion/rain_heavy.arrow` | 4,790 | rain LIGHT + physics overlay |
| `soda_voc/rain_snow/diffusion/snow.arrow` | 19,623 | IP2P snow |
| `construction_site/night/test/night_constructionsite_test.arrow` | 3,004 | CycleGAN night, CS test |
| `construction_site/rain_snow/diffusion/test/rain/` (7 shards) | 1,652 | IP2P rain LIGHT, CS test |
| `construction_site/rain_snow/diffusion/test/rain_heavy/` (7 shards) | 1,652 | rain HEAVY physics overlay, CS test (paired with light) |
| `construction_site/rain_snow/diffusion/test/snow_light/` (7 shards) | 2,940 | IP2P snow LIGHT, CS test |
| `construction_site/rain_snow/diffusion/test/snow_heavy/` (3 shards) | 3,004 | IP2P snow HEAVY, CS test |
| `construction_site/rain_snow/diffusion/train/rain_heavy/train_rain_heavy.arrow` | 3,627 | rain HEAVY, CS train |
| `soda_ktsh/rain_snow/diffusion/rain.arrow` | 2,112 | IP2P rain LIGHT, KTSH |
| `soda_ktsh/rain_snow/diffusion/rain_heavy.arrow` | 2,112 | rain HEAVY physics overlay, KTSH |

---

## 5. Feature schema (per variant)

All Arrow tables share two common columns:

| Feature | Type | Description |
|---|---|---|
| `image_id` | `string` | Stable identifier matching the source image (e.g. `"0000001"`). Use to pair with `original`. |
| `image` | `struct<bytes: binary, path: string>` (or `binary`) | JPEG-encoded image bytes. Decode with `PIL.Image.open(io.BytesIO(bytes))`. |

Per-variant extra columns are listed below.

### 5.1 — `construction_site/` (CS 10k)

Fields carried from the Construction Site 10k annotation schema. Each field type is the exact Arrow
schema type.

| Feature | Arrow type | Description |
|---|---|---|
| `image_caption` | `string` | Natural-language scene description (VLM-generated on the source dataset; copied over) |
| `illumination` | `string` | Enum: `normal lighting`, `low light`, `backlit`, `glare`, … (preserved from CS) |
| `camera_distance` | `string` | Enum: `short distance`, `medium distance`, `long distance`, `close-up`, … |
| `view` | `string` | Enum: `elevation view`, `aerial view`, `ground view`, `oblique view`, … |
| `quality_of_info` | `string` | Enum: `good info`, `poor info` — source-image informativeness tag |
| `rule_1_violation` | `struct<bounding_box: list<list<float64>>, reason: string>` or `null` | Hand-pose safety violation (e.g. worker near excavator boom). `bounding_box` is a list of `[x1, y1, x2, y2]` in normalised coords. `null` if no violation. |
| `rule_2_violation` | same as rule_1 or `null` | Equipment-proximity violation |
| `rule_3_violation` | same as rule_1 or `null` | Rebar-handling safety violation |
| `rule_4_violation` | same as rule_1 or `null` | PPE-compliance violation (hard-hat colour mismatch) |
| `excavator` | `list<list<float64>>` | Zero or more normalised xyxy bboxes for excavators in the image |
| `rebar` | `list<list<float64>>` | Zero or more normalised xyxy bboxes for rebar bundles |
| `worker_with_white_hard_hat` | `list<list<float64>>` | Zero or more normalised xyxy bboxes for compliant workers |
| `ref_id` | `string` | Alias of `image_id` (for legacy join compatibility) |

**Variants `rain` / `rain_heavy` / `snow_light` / `snow_heavy`** additionally carry the SSIM+LPIPS
record from the quality-filter stage:

| Feature | Arrow type | Description |
|---|---|---|
| `ssim` | `float32` | Structural similarity versus the clean source image at 256 × 256, range `[-1, 1]`. Higher ⇒ closer to clean. |
| `lpips` | `float32` or `null` | LPIPS perceptual distance (AlexNet backbone) versus clean source at 256 × 256; `null` for snow (not computed). Lower ⇒ closer to clean. |
| `status` | `string` | `"KEEP"` for all released rows — dropped rows were excluded at generation time. |

**Variant `fog/diffusion/test/<zone>/`** adds:

| Feature | Arrow type | Description |
|---|---|---|
| `fog_label` | `string` | `"heavy"`, `"medium"`, or `"light"` — the zone the Koschmieder visibility sample fell into |
| `fog_visibility` | `float32` | Sampled visibility in metres (300 – 500 heavy, 500 – 750 medium, 750 – 1000 light) |

### 5.2 — `soda_voc/` (SODA, VOC-style)

Follows Pascal-VOC conventions with fields decoded from the original `.xml` annotations.

| Feature | Arrow type | Description |
|---|---|---|
| `filename` | `string` | Original filename from SODA (e.g. `"00001.jpg"`) |
| `width` | `int32` | Image width in pixels |
| `height` | `int32` | Image height in pixels |
| `objects_name` | `list<string>` | Class name per object in this image (15-class SODA vocabulary: `person`, `helmet`, `vest`, `hook`, `fence`, `board`, `slogan`, `rebar`, `handcart`, `ebox`, `hopper`, `wood`, `scaffold`, `brick`, `cutter`) |
| `objects_bbox` | `list<list<float32>>` | One xyxy bounding box per object, aligned with `objects_name` |
| `objects_difficult` | `list<int8>` | VOC `difficult` flag (0/1) per object |
| `objects_truncated` | `list<int8>` | VOC `truncated` flag (0/1) per object |
| `ref_id` | `string` | Source-image identifier, matches entry in `original/` |
| `weather` | `string` | Canonical condition label (`"rain"`, `"snow"`, `"fog"`, `"night"`, `"clear"`, …) |
| `style` | `string` | For style-transfer ablation only: style-reference name (`"style_rain_0"`, …); absent for IP2P diffusion variants |

Variants `soda_voc/{original,small,night}.arrow` replace the per-object columns with two
pre-serialised strings for compactness:

| Feature | Arrow type | Description |
|---|---|---|
| `annotation` | `string` | JSON-serialised annotation dictionary (decode with `json.loads`) |
| `meta` | `string` | JSON-serialised metadata dictionary (image dimensions, source filename, etc.) |

### 5.3 — `soda_ktsh/` (SODA-KTSH captioned)

Caption-centric extension of SODA:

| Feature | Arrow type | Description |
|---|---|---|
| `filename` | `string` | Source filename from SODA-KTSH |
| `captions` | `string` | Multi-sentence scene caption (natural language, used for VLM description benchmarks) |
| `ref_id` | `string` | Source-image identifier |
| `weather` | `string` | Condition label |

### 5.4 — Ablation: `rain_snow/style_transfer/` (legacy VGG neural style transfer)

*Moved to `experiments/ablation_style_transfer/` in the parent repository on 2026-04-21; retained
on HuggingFace for historical reproducibility only.*

Same VOC-style columns as 5.2, plus:

| Feature | Arrow type | Description |
|---|---|---|
| `ssim_score` | `float32` | Per-image SSIM vs. clean source. Style-transfer outputs were filtered at `SSIM ∈ [0.5, 0.95]` (snow) or `[0.6, 0.95]` (rain). |

---

## 6. Quickstart — how to load and use

### 6.1 — Install

```bash
pip install pyarrow pillow datasets  # minimum — for loading and decoding
pip install huggingface_hub           # for streaming / snapshot-download
```

### 6.2 — Load a single Arrow file

```python
import io
import pyarrow as pa
from PIL import Image

with open("construction_site/rain_snow/diffusion/test/rain_heavy/batch_0-500.arrow", "rb") as f:
    table = pa.ipc.open_stream(f).read_all()

print(f"{table.num_rows} rows")            # e.g. 262
print([f.name for f in table.schema])      # ['image', 'image_id', 'image_caption', ...]

# Decode image in row 0
cell = table.column("image")[0].as_py()
img_bytes = cell["bytes"] if isinstance(cell, dict) else cell
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
img.save("/tmp/preview.jpg")
```

### 6.3 — Concatenate all shards of a condition

Some conditions ship as multiple `batch_*.arrow` files (rain, rain_heavy, snow_light, snow_heavy,
fog zones). Concatenate them before iterating:

```python
from pathlib import Path

shards = sorted(Path("construction_site/rain_snow/diffusion/test/rain_heavy/").glob("*.arrow"))
tables = []
for p in shards:
    with open(p, "rb") as f:
        tables.append(pa.ipc.open_stream(f).read_all())
full = pa.concat_tables(tables, promote_options="default")
print(f"rain_heavy (CS test): {full.num_rows} rows")  # → 1,652
```

### 6.4 — Load via HuggingFace `datasets`

```python
from datasets import Dataset
from datasets.features import Image as HFImage

ds = Dataset.from_file("soda_voc/rain_snow/diffusion/rain.arrow")
ds = ds.cast_column("image", HFImage())   # decode bytes → PIL on access
sample = ds[0]
sample["image"].show()
print("Classes in this image:", sample["objects_name"])
```

### 6.5 — Pair augmented rows with their clean originals

```python
orig = pa.ipc.open_stream(open("construction_site_test.arrow", "rb")).read_all()
orig_map = {
    str(orig.column("image_id")[i].as_py()): i
    for i in range(orig.num_rows)
}

aug_id = str(full.column("image_id")[0].as_py())
orig_idx = orig_map[aug_id]

# → now you have a paired sample for SSIM / LPIPS / DINO / VLM jury / etc.
```

### 6.6 — Decode bounding boxes to pixel coordinates

```python
row = table.slice(0, 1).to_pylist()[0]
w, h = img.size
for cls in ["excavator", "rebar", "worker_with_white_hard_hat"]:
    for (x1, y1, x2, y2) in row[cls]:
        px = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        print(cls, px)
```

### 6.7 — Stream only what you need from HuggingFace

```python
from huggingface_hub import snapshot_download

local = snapshot_download(
    repo_id="Ben11304/ConSynth-X-augmentation",
    repo_type="dataset",
    allow_patterns=[
        "construction_site/rain_snow/diffusion/test/rain_heavy/*.arrow",
        "construction_site/rain_snow/diffusion/test/rain/*.arrow",
    ],
)
print("downloaded to", local)
```

---

## 7. Quality filtering

Two filters are already baked in at generation time:

- **SSIM filter** — rain rows have `ssim ∈ [0.60, 0.95]`; snow rows have `ssim ∈ [0.50, 0.95]`.
  Dropped rows are not included in the release.
- **LPIPS filter** — rain rows have `lpips < 0.35` (AlexNet backbone). Not applied to snow.

For *additional* quality thresholding (e.g. to match published paper tables), use the released
per-image **DINOv3 + SSIM** CSVs in the parent repository:

```
<parent-repo>/validation/results/dino_ssim/
├── ip2p_rain.csv            # 1,652 rows, DINO mean 0.871, SSIM mean 0.756
├── ip2p_rain_heavy.csv      # 1,652 rows, DINO mean 0.754, SSIM mean 0.522
├── ip2p_snow_light.csv      # 2,940 rows
├── ip2p_snow_heavy.csv      # 3,004 rows
├── fog_{light,medium,heavy}.csv
├── night.csv
└── st_{rain,snow}_{a,b,c}.csv   # style-transfer ablation
```

Each CSV has columns `image_id`, `ssim`, `dino_sim`, `dino_dist`. Thresholds used in the paper
(*Kaggle sample release*):

| Variant | DINO threshold | Fraction eligible |
|---|---|---|
| rain light | `dino_sim ≥ 0.75` | 91.7% (1,515 / 1,652) |
| rain heavy | `dino_sim ≥ 0.70` | 73.8% (1,219 / 1,652) |

Example filter:

```python
import csv
dino = {}
with open("validation/results/dino_ssim/ip2p_rain_heavy.csv") as f:
    for r in csv.DictReader(f):
        dino[r["image_id"]] = float(r["dino_sim"])

eligible_ids = {k for k, v in dino.items() if v >= 0.70}
keep_indices = [
    i for i in range(full.num_rows)
    if str(full.column("image_id")[i].as_py()) in eligible_ids
]
print(f"keeping {len(keep_indices)} / {full.num_rows}")
```

---

## 8. Provenance (how each condition was generated)

| Condition | Pipeline | Key parameters |
|---|---|---|
| `original` | Source as-is, no modification | — |
| `rain` (light) | **InstructPix2Pix diffusion** + physics overlay + SSIM+LPIPS filter | prompt `"a rainy day with dark overcast sky, rain falling, grey clouds"`; `guidance_scale=10.0`, `image_guidance_scale=1.5`, 30 inference steps; physics overlay at default intensity (fog α 0.15–0.30, 3 streak layers, 2 800 – 5 200 streaks total) |
| `rain_heavy` | **Physics-only overlay on light** — `add_natural_rain(image, intensity='heavy_fog')`. NO second diffusion pass. | fog α 0.30–0.45 (2× default); 3 streak layers `n ∈ {3000–4500, 2200–3600, 900–1600}`, length 12–60 px, α 0.35–0.55; Gaussian blur 5×5 σ=1.3 post-processing; deterministic per-image seed = 42 + image_index; CPU-only. Rationale in parent repo's [`DEVLOG.md`](../DEVLOG.md) 2026-04-21. |
| `snow_light` | IP2P diffusion + physics overlay | `guidance_scale=8.0`, prompt `"a cold winter day with snow, frost on surfaces, grey sky, snow on the ground"` |
| `snow_heavy` | IP2P diffusion + physics overlay | `guidance_scale=12.0`, prompt `"a cold winter day with heavy snow, thick snow covering the ground and surfaces, grey overcast sky, snowfall"` |
| `fog` (3 zones) | **Koschmieder atmospheric scattering** + Depth Anything V2 monocular depth | visibility sampled uniform in [300, 1000] m, 3 labelled zones; fog colour random in [210, 245] (grey–white); `I_fog(x) = I(x)·exp(-β·d(x)) + A·(1 - exp(-β·d(x)))` |
| `night` | **CycleGAN-Turbo** `day_to_night` checkpoint ([img2img-turbo](https://github.com/GaParmar/img2img-turbo), MIT license) | FP16 inference at 512×512, LANCZOS upscale to source resolution; 1:1 mapping |
| `small` | **FLUX.1-Fill-dev outpainting** | scale factor ~ truncated-Gaussian(μ=0.25, σ=0.05, range [0.20, 0.40]); prompt `"an outdoor construction site with buildings, roads, and open sky in the background"`; bounding boxes re-mapped with offset compensation |
| `rain` / `snow` **style-transfer** (ablation only) | VGG19 neural style transfer + MiDaS depth + physics particle overlay | 3 style references per hazard, `style_weight=10 000 or 100 000`, 10 or 50 LBFGS steps (parent-repo pipelines); post-filtered SSIM (snow ≥ 0.50 rain ≥ 0.60, upper 0.95) |

**Models used at runtime** (no hard-pinning of HF `revision=`; authoritative record is the dataset
bytes themselves):
`timbrooks/instruct-pix2pix`, `intel-isl/MiDaS:DPT_Large`, `black-forest-labs/FLUX.1-Fill-dev`,
`DepthAnything/Depth-Anything-V2`, `prithivMLmods/Weather-Image-Classification` (validation),
`GaParmar/img2img-turbo @ 86f5414` (day-to-night checkpoint).

**Deterministic seeds** — all workers use `seed_base=42 + image_index`. Re-running the pipelines
on the same inputs produces bit-identical outputs (except for non-determinism in HuggingFace model
loading, which affects IP2P / FLUX / DINO — the authoritative record is this release's SHA256).

---

## 9. Licence

- **Images, annotations, and Arrow rows in this release** — **CC0 1.0 Universal** (public domain
  dedication, to the extent that any new copyright arose from the synthetic augmentation).
- **Upstream source images** —
  - Construction Site 10k: **CC BY-NC 4.0** (Chen & Zou 2025, arXiv:2508.11011)
  - SODA: no formal licence stated by the authors; dataset is publicly released for research use
    (Duan et al. 2022, DOI: 10.1016/j.autcon.2022.104499). Confirm with the authors before any
    commercial re-use.
- **Third-party models** — distributed under their own licences (MIT for IP2P / img2img-turbo;
  Apache-2.0 for MiDaS and Weather_Effect_Generator VGG weights; FLUX.1 [dev] Non-Commercial for
  FLUX.1-Fill-dev; Apache-2.0 / CC-BY-NC-4.0 for Depth Anything V2 depending on variant).

When redistributing or citing derived work, please respect the upstream image licences in addition
to this dataset's CC0 dedication.

---

## 10. Citation

```bibtex
@dataset{consynthx_2026,
  title  = {ConSynth-X: Multi-Condition Synthetic Augmentation of Construction-Site Imagery},
  author = {Duong, Viet Huy and collaborators},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Ben11304/ConSynth-X-augmentation},
  note   = {CC0 1.0 release of synthetic augmentations; upstream images remain under their
            original licences (CC BY-NC 4.0 for Construction Site 10k)}
}
```

Please also cite the upstream datasets:

```bibtex
@misc{chen2025vlm,
  title         = {Are Large Pre-trained Vision Language Models Effective Construction Safety Inspectors?},
  author        = {Chen, Xuezheng and Zou, Zhengbo},
  year          = {2025},
  eprint        = {2508.11011},
  archivePrefix = {arXiv}
}

@article{duan2022soda,
  title   = {SODA: A large-scale open site object detection dataset for deep learning in construction},
  author  = {Duan, Rui and Deng, Hui and Tian, Mao and Deng, Yichuan and Lin, Jiarui},
  journal = {Automation in Construction},
  volume  = {142},
  pages   = {104499},
  year    = {2022},
  doi     = {10.1016/j.autcon.2022.104499}
}
```

---

## Acknowledgments

This work was supported in part by an allocation of computing time from the Ohio Supercomputer
Center (OSC). See the parent repository [`INSTALL.md`](../INSTALL.md) for end-to-end reproduction,
[`docs/methods.md`](../docs/methods.md) for per-parameter justifications, and
[`USAGE.md`](../USAGE.md) for downstream-task recipes.
