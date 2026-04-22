# Ablation Archive — VGG Neural Style Transfer (rain/snow)

This folder holds the **legacy neural style transfer weather augmentation** for rain/snow, relocated here on 2026-04-21 when IP2P diffusion was promoted to the sole main-pipeline method for weather augmentation in ConSynth-X.

## Why it was moved

Prior to 2026-04-21 the main dataset contained two parallel weather pipelines:

- **Method 1 — Neural Style Transfer** (VGG19 + MiDaS depth + physics overlay), produced `rain_{0,1,2}.arrow` / `snow_{0,1,2}.arrow` at 3 intensity levels with VOC-style object bboxes preserved.
- **Method 2 — InstructPix2Pix diffusion** (IP2P + physics + SSIM/LPIPS filter), produced single-file `rain.arrow` / `snow.arrow`.

After the IP2P pipeline reached higher visual fidelity and texture consistency, the decision was made to use IP2P as the sole main weather method. Style transfer is retained here for:

1. **Ablation / baseline comparison** — reviewers can reproduce the realism vs. recognizability trade-off reported in the paper (Style Transfer: strong weather recognizability, IP2P: high texture fidelity).
2. **Reproducibility** — the paper (`paper/main.tex`, §3.1) describes both pipelines; relocating the data preserves the ability to re-run any figure or table that references style transfer.

## Layout

```
ablation_style_transfer/
├── construction_site/rain_snow/style_transfer/   # ~5.9 GB
└── soda_voc/rain_snow/style_transfer/            # ~37 GB (includes ssim/ filter outputs)
```

Arrow schema (unchanged from original):
```
[image, image_id, filename, width, height,
 objects_name, objects_bbox, objects_difficult, objects_truncated,
 ref_id, weather, style]
```

- `rain_{0,1,2}.arrow` / `snow_{0,1,2}.arrow` — three intensity levels each.
- `ssim/` (soda_voc only) — SSIM-filtered subsets at various thresholds.

## Regeneration

Generation code stays under [`generation/weather/rain_snow/style_transfer/`](../../generation/weather/rain_snow/style_transfer/) and is unchanged. If you regenerate, write the output back into this folder (not `augmentation_data/`).

## Not to be confused with

- `augmentation_data/*/rain_snow/diffusion/` — the **current** main-pipeline IP2P output.
- `augmentation_data/soda_ktsh/rain_snow/` — never had style transfer (IP2P-only from the start).
