# Data Card

- Dataset name: ConSynth-X
- Curators: Repository maintainers
- Contact: Use the repository issue tracker for questions or corrections.
- Version: Label schema `0.1`

Labeling and conditions:
- Each image may have multiple condition labels (multi-label); e.g., `rain` + `low_light`.
- Condition labeling approach: labels may be assigned manually from visual inspection or rule-based from simulator and capture metadata. The label schema records provenance as `manual` or `rule-based`.
- Label consistency: taxonomy definitions live in `taxonomy/extreme_conditions_definition.md`, and condition identifiers are tracked in `metadata/condition_labels.json`.

Train/test splits:
- Splits follow the source datasets. Construction Site 10k: train (7,009 images) / test (3,004 images). SODA: original VOC splits.
- All augmented data inherits the split of its source image. Augmented train images are for training; augmented test images are for cross-condition evaluation.
- Split information is encoded in the Arrow file names (e.g., `construction_site-train-*.arrow`, `construction_site-test.arrow`) and folder structure (`diffusion/train/`, `diffusion/test/`).
- Rain/snow augmentation in the main dataset uses IP2P diffusion only. The legacy VGG neural-style-transfer rain/snow outputs (3 intensity levels × 2 hazards × 2 datasets, ~43 GB with VOC bboxes preserved) are retained under [`experiments/ablation_style_transfer/`](experiments/ablation_style_transfer/) for baseline comparison and reproducibility of paper §3.1 (see that folder's `README.md`).

Rain/snow intensity levels (2026-04-21):
- **Rain**: 2 levels. `light` = IP2P (g=10, prompt "heavy rainy day...") with default physics overlay — files `rain.arrow` / `test/rain/` / `train_rain_v4.arrow`. `heavy` = `light` + CPU-only `add_natural_rain(intensity='heavy_fog')` overlay (denser streaks, stronger fog, Gaussian blur) — files `rain_heavy.arrow` / `test/rain_heavy/` / `train/rain_heavy/train_rain_heavy.arrow`. See `DEVLOG.md` 2026-04-21 and `docs/methods.md` for rationale and physics parameters.
- **Snow**: 2 levels on CS test. `light` = IP2P g=8, `heavy` = IP2P g=12 (both diffusion-based with different prompts). Train split and SODA datasets currently single-level; intensity parity for those is future work.
- **Fog**: 3 zones (heavy/medium/light) via Koschmieder visibility sampling — unchanged.

Included repository assets:
- Condition definitions: `taxonomy/extreme_conditions_definition.md`
- Condition configs: `conditions/*.yaml`
- Object annotations: embedded in Arrow tables (per-class bounding box columns)
- Evaluation utilities: `evaluation/`
- Generation pipelines (self-contained): `generation/` — see `INSTALL.md`
- Validation results (7 approaches): `validation/results/`
- Weight manifest + checksums: `weights/checksums.sha256`

Reproducibility:
- Every generation pipeline in this repository is self-contained. Reviewers can
  reproduce the augmented dataset by following `INSTALL.md`: clone with
  `--recurse-submodules`, set `CONSYNTH_DATA_ROOT`, run `bash weights/download.sh`,
  and invoke the pipelines under `generation/`.
- Pretrained weights are redistributed unchanged from their original upstream
  publishers (img2img-turbo @ CMU, Weather_Effect_Generator VGG @ Google Drive);
  `weights/checksums.sha256` pins the exact bytes used to produce this dataset.
- HuggingFace / torch.hub models (IP2P, FLUX, MiDaS, Depth-Anything, weather
  classifier) are loaded at runtime without `revision=` pinning; the
  authoritative record of the exact artifacts used is this released dataset
  plus the SHA256s above.
- Full upstream URLs, licenses, and citations: `docs/data_sources.md` §6.
- Per-parameter justifications (style transfer weights, IP2P guidance, physics
  overlay): `docs/methods.md`.

Recommended reporting tables for papers:
- Overall performance
- Per-condition performance (e.g., rain, fog, low_light)
- Cross-condition generalization (train on clean, test on each condition)
- Combined-condition performance (images with >1 extreme condition)
- Failure-case examples with visualizations and short analysis

Licensing and privacy:
- Code in this repository is released under Apache-2.0.
- The dataset card declares dataset licensing as CC BY-NC 4.0.
- Any release containing identifiable people, vehicles, or site-sensitive imagery should be reviewed before distribution and documented in the downstream dataset release notes.
