# Weather_Effect_Generator — Vendored Copy

This directory is a **vendored copy** (not a git submodule) of a modified Weather_Effect_Generator library used for style-transfer-based weather augmentation in ConSynth-X.

## Upstream

- **Repository**: https://github.com/hgupta01/Weather_Effect_Generator
- **Upstream commit**: `7d62b67` (base files: `Fog/Rain/Snow_Effect_Generator.py`, `MiDaS_Depth_Estimation.py`, `Neural_Style_Transfer.py`, `lib/`)
- **License**: Apache License 2.0 (see `LICENSE`)
- **Citation**: H. Gupta et al., *Robust Object Detection in Challenging Weather Conditions*.

## Local additions (first-party, Apache-2.0 under ConSynth-X)

The following files were added/modified by the ConSynth-X authors on top of the upstream base:

- `rain_pipeline.py` — orchestrates MiDaS depth → VGG style transfer → rain particle overlay
- `snow_pipeline.py` — same for snow
- `weather_pipeline.py` — unified entry point
- `style_transfer.py` — batch style transfer helper
- `lib/style_transfer_utils.py` — modified from upstream `lib/` (VGG checkpoint resolution + batch utilities)
- `Snow_Effect_Generator.py` — modified (depth-aware particle density)

## Weights (not included here)

Pretrained VGG and CycleGAN weights (`VGG/*.pth`, `Gan/*.pth`) are **not** checked in due to file size (~3 GB). Download via `ConSynth-X/weights/download.sh` or follow `ConSynth-X/docs/data_sources.md` § 6.

## Vendoring rationale

Upstream base is stable, but local additions (pipelines + modified generators) are tightly coupled to ConSynth-X's worker scripts. Submodule-only would split the code across two places and complicate reviewer reproduction; vendoring keeps the augmentation logic self-contained.
