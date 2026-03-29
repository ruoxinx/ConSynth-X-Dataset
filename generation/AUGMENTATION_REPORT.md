# Data Augmentation Report — ConSynth-X

This document describes the synthetic data augmentation pipelines used in ConSynth-X to generate extreme-condition variants of construction site images for robustness benchmarking.

## 1. Overview

ConSynth-X applies three augmentation pipelines to two base datasets, producing four dataset conditions for benchmarking:

| Condition | Method | Annotation Strategy |
|---|---|---|
| `original` | No augmentation (baseline) | Unchanged |
| `weather` | Neural Style Transfer (MiDaS + VGG19) | Copied directly (pixel-level transform) |
| `night` | img2img-turbo (CycleGAN-Turbo) | Copied directly (pixel-level transform) |
| `small` | FLUX.1-Fill-dev outpainting | Bounding boxes re-mapped with offset compensation |

### Base Datasets

| Dataset | Format | Classes | Images |
|---|---|---|---|
| Construction Site | HuggingFace Arrow | 3 (excavator, rebar, worker_with_white_hard_hat) | 3,004 (test) |
| SODA | Pascal VOC (XML) | 15 | 19,846 |

---

## 2. Weather Augmentation (Neural Style Transfer)

### 2.1 Motivation

Real construction sites experience rain and snow that degrade object detection. Training data rarely contains these conditions, so we synthetically generate them using neural style transfer to produce visually realistic weather effects while preserving scene geometry.

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
| SODA | 19,846 | Pending |

---

## 4. Outpainting — Scale/Distance Augmentation (FLUX.1-Fill-dev)

### 4.1 Motivation

Construction objects at distance appear small and are harder to detect. To simulate this without losing image quality, we use outpainting: expanding the image canvas around the original, making objects appear smaller and more distant. This is more realistic than simple downscaling because the surrounding context is AI-generated.

### 4.2 Pipeline

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

### 4.3 Scale Factor Distribution

Scale factors are sampled from a **truncated Gaussian**:
- Mean: 0.25 (25% expansion on each side)
- Std: (0.4 - 0.2) / 4 = 0.05
- Clamped to [0.20, 0.40]

This means the original image occupies between ~40% and ~70% of the final canvas area, making objects appear proportionally smaller.

### 4.4 Bounding Box Transfer

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

### 4.5 Model Configuration

- **Model**: `black-forest-labs/FLUX.1-Fill-dev` (bfloat16)
- **Memory optimizations**: CPU offload, VAE slicing, VAE tiling
- **Prompt**: `"an outdoor construction site with buildings, roads, and open sky in the background"`
- **GPU requirement**: 1x A100 (80GB)

### 4.6 Output Statistics

| Dataset | Original | Outpainted (small) |
|---|---|---|
| Construction Site (test) | 3,004 | 1,323 |
| SODA | 19,846 | 1,001 |

---

## 5. Dual Format Support

All pipelines support two data formats:

| Format | Used By | Annotation Format |
|---|---|---|
| **HuggingFace Arrow** | Construction Site dataset | Inline JSON in Arrow columns |
| **Pascal VOC (XML)** | SODA dataset | Separate XML files per image |

Each pipeline has separate worker scripts for each format:
- Arrow: `arrow_augmentation_worker.py`, `day2night_batch_worker.py`, `flux_pipeline_worker.py`
- VOC: `soda_augmentation_worker_*.py`, `day2night_soda_worker.py`, `flux_pipeline_worker_voc.py`

---

## 6. Compute Infrastructure

All augmentation jobs run on the OSC SLURM cluster:

| Pipeline | GPU | Memory | Time/Batch | Batch Size |
|---|---|---|---|---|
| Weather (style transfer) | 1x V100 | 32 GB | ~4 hours | 100 images |
| Day-to-Night | 1x A100 | 32 GB | ~2 hours | 100 images |
| Outpainting (FLUX) | 1x A100 | 64 GB | ~4 hours | 50 images |

- **Account**: `pgs0407`
- **Conda env**: `VLM` (Python 3.10, PyTorch, transformers, diffusers)
- **CUDA**: 12.8.1

---

## 7. Summary of Generated Data

### Construction Site Dataset

| Condition | Method | Images | Annotations |
|---|---|---|---|
| `original` | Baseline | 3,004 | Unchanged |
| `weather` (6 styles) | Neural Style Transfer + SSIM filter | 10,266 | Copied |
| `night` | CycleGAN-Turbo | 3,004 | Copied |
| `small` | FLUX outpainting | 1,323 | Bbox transferred |
| **Total** | | **17,597** | |

### SODA Dataset

| Condition | Method | Images | Annotations |
|---|---|---|---|
| `original` | Baseline | 19,846 | Unchanged |
| `weather` (6 styles) | Neural Style Transfer + SSIM filter | 47,169 | Copied (XML) |
| `small` | FLUX outpainting | 1,001 | Bbox transferred (XML) |
| **Total** | | **68,016** | |

---

## 8. Key Design Decisions

1. **Style transfer over GAN**: Neural style transfer with VGG19 was chosen over CycleGAN for weather because it provides finer control over intensity via style reference images and preserves more structural detail.

2. **SSIM filtering**: Removes both failed augmentations (SSIM > 0.95, image barely changed) and over-corrupted samples (SSIM < 0.5, unrecognizable), ensuring augmented data is meaningfully different but usable.

3. **Outpainting over downscaling**: FLUX inpainting generates realistic surrounding context rather than simple zero-padding or interpolation artifacts, making the scale augmentation more representative of real distant-object scenarios.

4. **Annotation preservation by design**: Weather and night transforms are pixel-level operations that don't move objects, so annotations are copied verbatim. Outpainting requires explicit coordinate transfer with offset compensation.

5. **Truncated Gaussian for scale**: Avoids extreme outliers while centering augmentation around a moderate 25% expansion, balancing the diversity of scale factors with usability.

---

## 9. File Reference

| Component | Location |
|---|---|
| Weather workers | `generation/weather/arrow_augmentation_worker*.py`, `soda_augmentation_worker_*.py` |
| Weather SLURM submitters | `generation/weather/submit_arrow_augmentation.py`, `submit_soda_augmentation.py` |
| SSIM filter | `generation/weather/ssim_filter_worker*.py` |
| Style references | `generation/weather/rain_style/`, `generation/weather/snow_style/` |
| MiDaS + VGG19 lib | `generation/weather/Weather_Effect_Generator/` |
| Day2Night workers | `generation/day2night/day2night_batch_worker.py`, `day2night_soda_worker.py` |
| Day2Night SLURM | `generation/day2night/submit_day2night_batch.py`, `submit_day2night_soda.py` |
| Outpainting workers | `generation/outpainting/flux_pipeline_worker.py`, `flux_pipeline_worker_voc.py` |
| Outpainting SLURM | `generation/outpainting/submit_outpainting_pipeline.py`, `submit_outpainting_soda.py` |
| Arrow export utility | `generation/utils/export_arrow_to_train.py` |
| Parameter ranges | `generation/parameter_ranges.json` |
