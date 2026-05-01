# Data Augmentation Report — ConSynth-X

This document describes the synthetic data augmentation pipelines used in ConSynth-X to generate extreme-condition variants of construction site images for robustness benchmarking.

> **Scope note (2026-04-21):** Main rain/snow pipeline in this dataset is **IP2P diffusion** (documented under §2 of `docs/methods.md`). The VGG Neural Style Transfer pipeline described in §2 below is retained as an **ablation baseline only** (see [`experiments/ablation_style_transfer/README.md`](../experiments/ablation_style_transfer/README.md)). §2 of this report documents the NST pipeline for historical / reproducibility reasons; the current main-pipeline description for rain/snow lives in [`docs/methods.md`](../docs/methods.md) §1.2 Method 2, and primary output statistics are summarized in [`dataset_card.md`](../dataset_card.md).

## 1. Overview

ConSynth-X applies synthetic augmentation pipelines to base datasets, producing the following conditions for benchmarking:

| Condition | Method | Annotation Strategy |
|---|---|---|
| `original` | No augmentation (baseline) | Unchanged |
| `rain` (light) | IP2P diffusion (g=10) + physics overlay (default intensity) | Copied directly (pixel-level transform) |
| `rain_heavy` | `rain` (light) + `add_natural_rain(intensity='heavy_fog')` **physics-only overlay** — no additional diffusion pass | Copied verbatim from light stage (same `image_id`, `bboxes`, `ssim`, `lpips`, `status`) |
| `snow_light` | IP2P diffusion (g=8) + physics overlay | Copied directly (pixel-level transform) |
| `snow_heavy` | IP2P diffusion (g=12, strong prompt) + physics overlay | Copied directly (pixel-level transform) |
| `fog` (light/medium/heavy) | Koschmieder atmospheric scattering + Depth Anything V2 depth (3 visibility zones) | Copied directly (pixel-level transform) |
| `night` | img2img-turbo (CycleGAN-Turbo `day_to_night` checkpoint) | Copied directly (pixel-level transform) |
| `night_rain` | Night → IP2P (rain) → Physics rain streaks | Copied directly (pixel-level transform) |
| `night_snow` | Night → IP2P (snow) → Physics snowflakes | Copied directly (pixel-level transform) |
| `small` | FLUX.1-Fill-dev outpainting | Bounding boxes re-mapped with offset compensation |
| `weather_style_*` (ablation) | Legacy VGG Neural Style Transfer (MiDaS + VGG19), moved to [`experiments/ablation_style_transfer/`](../experiments/ablation_style_transfer/) on 2026-04-21 | Copied directly (pixel-level transform) |

### Base Datasets

| Dataset | Format | Classes | Images |
|---|---|---|---|
| Construction Site (CS10k) | HuggingFace Arrow | 3 (excavator, rebar, worker_with_white_hard_hat) | 10,013 (train 7,009 + test 3,004) |
| SODA-VOC | Pascal VOC (XML) | 15 | 19,846 |
| SODA-KTSH (auxiliary) | KTSH | — | 9,988 |

---

## 2. Weather Augmentation — Ablation Baseline (Legacy Neural Style Transfer)

> **STATUS:** Ablation-only since 2026-04-21. Main rain/snow = IP2P (see [`docs/methods.md`](../docs/methods.md) §1.2 Method 2). Data files archived at [`experiments/ablation_style_transfer/`](../experiments/ablation_style_transfer/). Output-statistics tables in §2.6 and §8 below describe this **ablation archive**, not the main dataset.

### 2.1 Motivation (ablation)

Early iterations of ConSynth-X used VGG neural style transfer to synthesize rain and snow. The approach was evaluated against IP2P diffusion under 7 validation approaches (§Technical Validation). IP2P was chosen as main pipeline due to higher texture fidelity (belief H=0.85-0.86 vs 0.00 for ST) and fewer scene hallucinations; NST is retained as ablation because it yields stronger weather recognizability on the SigLIP2 classifier (62-95% vs IP2P 26-64%), making it useful for reporting the realism-vs-recognizability trade-off [cite: Ruck et al. 2026].

### 2.2 Pipeline

```
Input Image
    │
    ▼
MiDaS DPT_Large ──► Depth Map
    │
    ▼
VGG19 Style Transfer ──► Weather-toned Image
    (style_weight=10000,           │
     steps=10)                     ▼
                        Weather Particle Generator
                        (Snow or Rain particles    ──► Output Image
                         intensity=light)
```

**Step 1 — Depth Estimation (MiDaS DPT_Large)**
- Generates a monocular depth map for each input image.
- Used by the particle generator to place weather effects with depth-aware density (closer = more particles).

**Step 2 — Neural Style Transfer (VGG19)**
- Transfers the color tone and texture of a weather style reference image onto the input.
- Uses `torch.optim.LBFGS` optimizer with `style_weight=10,000` and `10` optimization steps.
- Pre-trained VGG19 classifiers (fog-clear, rain-clear, snow-clear) guide the transfer.

**Step 3 — Weather Particle Overlay**
- **Snow**: `SnowEffectGenerator` adds depth-aware snowflake particles with configurable intensity (light/medium/heavy/extreme).
- **Rain**: `RainEffectGenerator` adds depth-aware rain streaks with directional motion.

### 2.3 Style References

6 style reference images (3 rain, 3 snow) stored in `generation/weather/rain_style/` and `generation/weather/snow_style/`:

| Style | Description |
|---|---|
| `style_rain_0` | Light rain tone |
| `style_rain_1` | Moderate rain tone |
| `style_rain_2` | Heavy rain tone |
| `style_snow_0` | Light snow tone |
| `style_snow_1` | Moderate snow tone |
| `style_snow_2` | Heavy snow tone |

### 2.4 SSIM Quality Filtering

After generation, an SSIM-based filter removes samples that are too similar to the original (unchanged, SSIM > 0.95) or too degraded (unrecognizable, SSIM < 0.5):

- **Threshold**: 0.5 <= SSIM <= 0.95
- **Implementation**: `skimage.metrics.structural_similarity` on full RGB images
- **Purpose**: Ensures augmented images are meaningfully different but still recognizable

### 2.5 Annotation Handling

Weather augmentation is a pixel-level transform that does not alter object positions or sizes. All bounding box annotations, rule violation labels, and metadata are **copied directly** from the original without modification.

### 2.6 Output Statistics

**Construction Site (test set, 3,004 original images):**

| Style | Images After SSIM Filtering |
|---|---|
| `style_rain_0` | 2,211 |
| `style_rain_1` | 1,243 |
| `style_rain_2` | 1,208 |
| `style_snow_0` | 2,159 |
| `style_snow_1` | 1,649 |
| `style_snow_2` | 1,796 |
| **Total** | **10,266** |

**SODA (19,846 original images):**

| Style | Images After SSIM Filtering |
|---|---|
| `style_rain_0` | 8,392 |
| `style_rain_1` | 8,059 |
| `style_rain_2` | 9,541 |
| `style_snow_0` | 9,818 |
| `style_snow_1` | 7,485 |
| `style_snow_2` | 3,874 |
| **Total** | **47,169** |

---

## 3. Day-to-Night Augmentation (img2img-turbo)

### 3.1 Motivation

Nighttime construction sites have drastically different lighting — artificial lights, high contrast, deep shadows. Most construction datasets are daytime-only. Day-to-night translation creates realistic nighttime variants to evaluate model robustness under low-light conditions.

### 3.2 Pipeline

```
Input Image (Day)
    │
    ▼
Resize to 512x512
    │
    ▼
CycleGAN-Turbo (day_to_night)
    │  - pretrained_name="day_to_night"
    │  - FP16 inference on GPU
    │  - xformers memory-efficient attention
    │
    ▼
Resize back to Original Resolution (LANCZOS)
    │
    ▼
Output Image (Night)
```

**Model**: CycleGAN-Turbo `day_to_night` checkpoint from the [img2img-turbo](https://github.com/GaParmar/img2img-turbo) project.

**Key characteristics**:
- 1:1 mapping: every input image produces exactly one nighttime variant
- Operates at 512x512 internal resolution, then upscales back with LANCZOS interpolation
- FP16 precision for memory efficiency on A100 GPUs

### 3.3 Annotation Handling

Day-to-night is a global illumination transform. Object positions and sizes remain unchanged. All bounding box annotations are **copied directly** from the original.

### 3.4 Output Statistics

| Dataset | Original | Night |
|---|---|---|
| Construction Site (test) | 3,004 | 3,004 |
| SODA-VOC | 19,846 | 19,846 |

---

## 4. Night Weather Augmentation (IP2P + Physics on Night Images)

### 4.1 Motivation

Real construction sites experience combined adverse conditions — rain at night, snow at night. These compound conditions are harder for vision models than individual conditions alone (night-only or weather-only). To simulate these, we chain our day-to-night pipeline with weather augmentation.

### 4.2 Pipeline

```
Night Image (from CycleGAN-Turbo day2night)
    │
    ▼
Resize to max 768px (divisible by 8)
    │
    ▼
InstructPix2Pix (IP2P)
    │  - Rain prompt: "a rainy night with rain falling,
    │    wet reflections on surfaces, dark overcast sky"
    │  - Snow prompt: "a cold winter night with snow falling,
    │    frost on surfaces, snow on the ground"
    │  - image_guidance_scale=1.5, steps=30
    │  - guidance_scale: 10.0 (rain), 8.0 (snow)
    │
    ▼
Resize back to Original Resolution (LANCZOS)
    │
    ▼
Physics Particle Overlay
    │  - Rain: 3-layer streaks (far/mid/near) + atmospheric fog
    │  - Snow: 3-layer flakes (far/mid/near) + screen blend
    │
    ▼
SSIM/LPIPS Quality Filter
    │  - Compare against ORIGINAL (not night) image
    │  - Rain: SSIM [0.6, 0.95] + LPIPS < 0.35
    │  - Snow: SSIM [0.5, 0.95]
    │
    ▼
Output Arrow + Metadata CSV
```

### 4.3 Design Decisions

1. **Night-first, then weather**: CycleGAN-Turbo produces the night base, then IP2P adds weather atmosphere (overcast, reflections) and physics adds particles. This order is chosen because IP2P prompts can reference night-specific elements (wet reflections, artificial lights).

2. **Night-specific prompts**: Prompts differ from day-weather prompts to reference night phenomena (reflections, frost under artificial light) rather than day phenomena (overcast sky, grey clouds).

3. **Quality comparison against original**: SSIM/LPIPS are computed against the **original day image** (not the intermediate night image), measuring total visual distance from source. This ensures the compound augmentation doesn't drift too far from the recognizable scene.

4. **Same IP2P model & physics module**: Reuses `timbrooks/instruct-pix2pix` and `physics.py` from the day-weather pipeline for consistency.

### 4.4 Annotation Handling

Night weather is a pixel-level transform chain (CycleGAN → IP2P → physics overlay). Object positions and sizes remain unchanged. All annotations are **copied directly** from the original.

### 4.5 Output Statistics

| Dataset | Night Input | Rain Night | Snow Night |
|---|---|---|---|
| Construction Site (test) | 3,004 | 3,004 | 3,004 |
| SODA-VOC | 19,846 | Pending | Pending |

### 4.6 Configuration

| Parameter | Rain Night | Snow Night |
|---|---|---|
| IP2P prompt | "a rainy night with rain falling, wet reflections on surfaces, dark overcast sky" | "a cold winter night with snow falling, frost on surfaces, snow on the ground" |
| guidance_scale | 10.0 | 8.0 |
| image_guidance_scale | 1.5 | 1.5 |
| num_inference_steps | 30 | 30 |
| Physics overlay | `add_natural_rain()` — 3-layer streaks + fog | `add_natural_snow()` — 3-layer flakes + screen blend |
| SSIM range | [0.6, 0.95] | [0.5, 0.95] |
| LPIPS threshold | < 0.35 | N/A |

---

## 5. Outpainting — Scale/Distance Augmentation (FLUX.1-Fill-dev)

### 5.1 Motivation

Construction objects at distance appear small and are harder to detect. To simulate this without losing image quality, we use outpainting: expanding the image canvas around the original, making objects appear smaller and more distant. This is more realistic than simple downscaling because the surrounding context is AI-generated.

### 5.2 Pipeline

```
Input Image (W x H)
    │
    ▼
Generate Gaussian Scale Factor
    │  mean=0.25, std=(max-min)/4
    │  range: [0.20, 0.40]
    │
    ▼
Create Canvas ((1+2s)*W x (1+2s)*H)
    │  - Gray (128,128,128) background
    │  - Original pasted at center
    │  - Binary mask: white=outpaint, black=keep
    │
    ▼
FLUX.1-Fill-dev Inpainting
    │  - prompt: "an outdoor construction site with
    │    buildings, roads, and open sky in the background"
    │  - guidance_scale=30.0
    │  - num_inference_steps=28
    │  - max_sequence_length=512
    │  - max internal resolution=728px
    │
    ▼
Transfer Bounding Boxes
    │  - Apply paste offset (expand_w, expand_h)
    │  - Re-normalize to new canvas size
    │
    ▼
Output: Outpainted Image + Transferred Annotations + Metadata
```

### 5.3 Scale Factor Distribution

Scale factors are sampled from a **truncated Gaussian**:
- Mean: 0.25 (25% expansion on each side)
- Std: (0.4 - 0.2) / 4 = 0.05
- Clamped to [0.20, 0.40]

This means the original image occupies between ~40% and ~70% of the final canvas area, making objects appear proportionally smaller.

### 5.4 Bounding Box Transfer

Unlike weather/night augmentation, outpainting changes the coordinate system. Bounding boxes are transferred using:

```
x1_new = (x1_orig * orig_w * resize_factor + paste_x) / new_w
y1_new = (y1_orig * orig_h * resize_factor + paste_y) / new_h
```

This accounts for:
1. Original normalized coords → pixel coords
2. Resize factor (if image was resized for memory)
3. Paste offset (center placement on canvas)
4. Re-normalization to new canvas dimensions

All annotation types are transferred: object bounding boxes (excavator, rebar, worker), rule violation bounding boxes (rules 1-4), and metadata fields.

### 5.5 Model Configuration

- **Model**: `black-forest-labs/FLUX.1-Fill-dev` (bfloat16)
- **Memory optimizations**: CPU offload, VAE slicing, VAE tiling
- **Prompt**: `"an outdoor construction site with buildings, roads, and open sky in the background"`
- **GPU requirement**: 1x A100 (80GB)

### 5.6 Output Statistics

| Dataset | Original | Outpainted (small) |
|---|---|---|
| Construction Site (test) | 3,004 | 1,323 |
| SODA | 19,846 | 1,001 |

---

## 6. Dual Format Support

All pipelines support two data formats:

| Format | Used By | Annotation Format |
|---|---|---|
| **HuggingFace Arrow** | Construction Site dataset | Inline JSON in Arrow columns |
| **Pascal VOC (XML)** | SODA dataset | Separate XML files per image |

Each pipeline has separate worker scripts for each format:
- Arrow: `arrow_augmentation_worker.py`, `day2night_batch_worker.py`, `flux_pipeline_worker.py`
- VOC: `soda_augmentation_worker_*.py`, `day2night_soda_worker.py`, `flux_pipeline_worker_voc.py`

---

## 7. Compute Infrastructure

All augmentation jobs run on the OSC SLURM cluster:

| Pipeline | GPU | Memory | Time/Batch | Batch Size |
|---|---|---|---|---|
| Weather (style transfer) | 1x V100 | 32 GB | ~4 hours | 100 images |
| Day-to-Night | 1x A100 | 32 GB | ~2 hours | 100 images |
| Night Weather (IP2P) | 1x A100 | 32 GB | ~6 hours | 500 images |
| Outpainting (FLUX) | 1x A100 | 64 GB | ~4 hours | 50 images |

- **Account**: `pgs0407`
- **Conda env**: `VLM` (Python 3.10, PyTorch, transformers, diffusers)
- **CUDA**: 12.8.1

---

## 8. Summary of Generated Data

> **Scope:** Tables below separate **main pipeline** (IP2P + CycleGAN + FLUX + Koschmieder fog) from the **ablation archive** (legacy NST). Canonical per-variant row counts are maintained in [`dataset_card.md`](../dataset_card.md) version history; ground truth is the Arrow file directory under `$CONSYNTH_DATA_ROOT/augmentation_data/`.

### Construction Site Dataset — Main Pipeline

Counts verified from Arrow files in `augmentation_data/construction_site/` on 2026-04-25.

| Condition | Method | Images | Annotations |
|---|---|---|---|
| `original` | Baseline (CS10k full) | 10,013 (train 7,009 + test 3,004) | Unchanged |
| `rain` (light) | IP2P g=10 + physics overlay | test 1,652 + train 3,627 = 5,279 | Copied |
| `rain_heavy` | Light + heavy_fog physics overlay (no 2nd diffusion) | test 1,652 + train 3,627 = 5,279 | Copied |
| `snow_light` | IP2P g=8 + physics | test 2,940 + train 6,755 = 9,695 | Copied |
| `snow_heavy` | IP2P g=12 + physics | test 3,004 (train pending) | Copied |
| `fog` (light/medium/heavy) | Koschmieder + Depth Anything V2 | 1,002 + 1,001 + 1,001 = 3,004 (test) | Copied |
| `night` | CycleGAN-Turbo | 3,004 (test) | Copied |
| `night_rain` | Night → IP2P rain → Physics | 3,004 (test) | Copied |
| `night_snow` | Night → IP2P snow → Physics | 3,004 (test) | Copied |
| `small` | FLUX outpainting | 1,323 (test) | Bbox transferred |
| **Augmented total (excluding original)** | | **36,596** | |
| **Total including baseline** | | **46,609** | |

### Construction Site Dataset — Ablation Archive (legacy NST)

| Condition | Method | Images | Annotations |
|---|---|---|---|
| `weather_style_rain_{0,1,2}` | VGG style transfer + physics + SSIM filter | 2,211 + 1,243 + 1,208 = 4,662 | Copied |
| `weather_style_snow_{0,1,2}` | VGG style transfer + physics + SSIM filter | 2,159 + 1,649 + 1,796 = 5,604 | Copied |
| **Ablation total** | | **10,266** | Archived at `experiments/ablation_style_transfer/construction_site/` |

### SODA Dataset — Main Pipeline

Counts verified from Arrow files in `augmentation_data/soda_voc/` and `augmentation_data/soda_ktsh/` on 2026-04-25.

**SODA-VOC** (15-class, full 19,846-image corpus unless noted):

| Condition | Method | Images | Annotations |
|---|---|---|---|
| `original` (VOC) | Baseline | 19,846 | Unchanged |
| IP2P rain (light) | IP2P + physics | 4,790 | Copied (XML) |
| IP2P rain_heavy | physics-only overlay | 4,790 | Copied (XML) |
| IP2P snow | IP2P + physics | 19,623 | Copied (XML) |
| `fog` (light/medium/heavy) | Koschmieder + Depth Anything V2 (3,000 subset) | 1,000 × 3 = 3,000 | Copied (XML) |
| `night` | CycleGAN-Turbo | 19,846 | Copied (XML) |
| `small` | FLUX outpainting (3,000 subset) | 1,000 | Bbox transferred (XML) |
| **SODA-VOC augmented total** | | **53,049** | |
| **SODA-VOC total including baseline** | | **72,895** | |

**SODA-KTSH** (auxiliary split, 9,988 source images):

| Condition | Method | Images | Annotations |
|---|---|---|---|
| IP2P rain (light) | IP2P + physics | 2,112 | Copied |
| IP2P rain_heavy | physics-only overlay | 2,112 | Copied |
| IP2P snow | IP2P + physics | 9,671 | Copied |
| **SODA-KTSH augmented total** | | **13,895** | |

### SODA Dataset — Ablation Archive (legacy NST)

| Condition | Method | Images | Annotations |
|---|---|---|---|
| `weather_style_rain_{0,1,2}` | VGG style transfer + SSIM filter (VOC) | 8,392 + 8,059 + 9,541 = 25,992 | Copied (XML) |
| `weather_style_snow_{0,1,2}` | VGG style transfer + SSIM filter (VOC) | 9,818 + 7,485 + 3,874 = 21,177 | Copied (XML) |
| **Ablation total** | | **47,169** | Archived at `experiments/ablation_style_transfer/soda_voc/` |

---

## 9. Key Design Decisions

1. **IP2P diffusion over style transfer for main weather (2026-04-21)**: After comparing VGG neural style transfer and IP2P across 7 validation approaches, IP2P was chosen as the main rain/snow method because (a) texture fidelity is substantially higher (Dempster-Shafer belief H=0.85-0.86 vs 0.00), (b) classifier-driven weather recognizability gap is smaller than the texture-artifact gap, and (c) IP2P yields fewer scene-level hallucinations once the "wet muddy ground" prompt issue was fixed (see `docs/methods.md` §1.2). NST is retained as an ablation baseline (`experiments/ablation_style_transfer/`) for reporting the realism-vs-recognizability trade-off.

2. **SSIM filtering**: Removes both failed augmentations (SSIM > 0.95, image barely changed) and over-corrupted samples (SSIM < 0.5, unrecognizable), ensuring augmented data is meaningfully different but usable.

3. **Outpainting over downscaling**: FLUX inpainting generates realistic surrounding context rather than simple zero-padding or interpolation artifacts, making the scale augmentation more representative of real distant-object scenarios.

4. **Annotation preservation by design**: Weather and night transforms are pixel-level operations that don't move objects, so annotations are copied verbatim. Outpainting requires explicit coordinate transfer with offset compensation.

5. **Truncated Gaussian for scale**: Avoids extreme outliers while centering augmentation around a moderate 25% expansion, balancing the diversity of scale factors with usability.

---

## 10. File Reference

| Component | Location |
|---|---|
| Weather workers | `generation/weather/arrow_augmentation_worker*.py`, `soda_augmentation_worker_*.py` |
| Weather SLURM submitters | `generation/weather/submit_arrow_augmentation.py`, `submit_soda_augmentation.py` |
| SSIM filter | `generation/weather/ssim_filter_worker*.py` |
| Style references | `generation/weather/rain_style/`, `generation/weather/snow_style/` |
| MiDaS + VGG19 lib | `generation/weather/Weather_Effect_Generator/` |
| Day2Night workers | `generation/day2night/day2night_batch_worker.py`, `day2night_soda_worker.py` |
| Day2Night SLURM | `generation/day2night/submit_day2night_batch.py`, `submit_day2night_soda.py` |
| Night Weather worker | `generation/weather/rain_snow/diffusion/night_weather_batch_worker.py` |
| Night Weather SLURM | `generation/weather/rain_snow/diffusion/submit_night_weather_test.py` |
| Outpainting workers | `generation/outpainting/flux_pipeline_worker.py`, `flux_pipeline_worker_voc.py` |
| Outpainting SLURM | `generation/outpainting/submit_outpainting_pipeline.py`, `submit_outpainting_soda.py` |
| Arrow export utility | `generation/utils/export_arrow_to_train.py` |
