# Dataset Card

- Name: **ConSynth-X: A Large-Scale Synthetic Construction-Site Image Dataset for Challenging Field Conditions**
- Authors: Viet Huy Duong¹ · Ruoxin Xiong, Ph.D.²
- Short name: ConSynth-X
- Purpose: Provide controlled synthetic data generation and curated real splits for evaluating robustness of construction CV models under extreme field conditions.
- Generation: See `generation/` for engine, scenario templates, and parameter ranges. Setup walkthrough: [`INSTALL.md`](INSTALL.md).
- Conditions: YAML condition configurations live in `conditions/` and are used to generate labeled variants.
- Metadata: scene-level and object-level annotations plus condition labels are stored in `metadata/`.
- Splits: standard `train`, `validation`, and `test` manifests in `splits/`.
- Pretrained weights used during generation: documented in [`weights/README.md`](weights/README.md) and [`docs/data_sources.md`](docs/data_sources.md) §6. SHA256 for the 3 third-party weights: [`weights/checksums.sha256`](weights/checksums.sha256).
- Code license: Apache-2.0
- Dataset license: CC BY-NC 4.0
- Third-party weights: redistributed unchanged under their original licenses (img2img-turbo MIT; Weather_Effect_Generator Apache-2.0).

## Public sample release (Kaggle)

URL: https://www.kaggle.com/datasets/viethuyduong/consynth-x-augmentation-sample
License: CC0-1.0 (sample only; full dataset CC BY-NC 4.0)

Sample contents (row counts reflect current Kaggle version; legacy 100-row samples noted where changed):

| File | Condition | Rows | Notes |
|---|---|---|---|
| `cs_original.arrow` | Clear baseline | 1,191 | unchanged since v1 |
| `cs_style_rain_{test,train}.arrow` | Style Transfer rain | 100 each | legacy, ablation archive |
| `cs_style_snow_{test,train}.arrow` | Style Transfer snow | 100 each | legacy, ablation archive |
| `cs_diff_rain_test.arrow` | IP2P rain **LIGHT** (g=10, default physics) | **300** | **DINO-filtered ≥ 0.75 (v6 2026-04-22)** |
| `cs_diff_rain_heavy_test.arrow` | IP2P rain **HEAVY** (light + `add_natural_rain(intensity='heavy_fog')`) | **300** | **DINO-filtered ≥ 0.70, physics-only — no extra diffusion** |
| `cs_diff_rain_train.arrow` | IP2P rain train (light) | 100 | paired ids with test baseline |
| `cs_diff_rain_heavy_train.arrow` | IP2P rain train (heavy, physics-only) | 100 | added v3 |
| `cs_diff_snow_light_test.arrow` | IP2P snow LIGHT (g=8) | 100 | |
| `cs_diff_snow_heavy_test.arrow` | IP2P snow HEAVY (g=12, diffusion-based) | 100 | no pre-filter |
| `cs_diff_snow_train.arrow` | IP2P snow train (light) | 100 | |
| `cs_fog_{light,medium,heavy}.arrow` | Koschmieder fog + Depth Anything V2 depth | 100 each | |
| `cs_night.arrow` | CycleGAN-Turbo day-to-night | 100 | |
| `cs_small.arrow` | FLUX.1-Fill-dev outpainting | 100 | |
| `soda_voc_diff_rain.arrow` | SODA VOC IP2P rain light | 100 | added v3 |
| `soda_voc_diff_rain_heavy.arrow` | SODA VOC IP2P rain heavy (physics-only) | 100 | added v3 |
| `soda_ktsh_diff_rain.arrow` | SODA KTSH IP2P rain light | 100 | |
| `soda_ktsh_diff_rain_heavy.arrow` | SODA KTSH IP2P rain heavy (physics-only) | 100 | added v3 |
| `soda_voc_{night,small,diff_snow,original}.arrow` | other SODA VOC augmentations | 100 each | |
| `soda_ktsh_{diff_snow,original}.arrow` | other SODA KTSH augmentations | 100 each | |

**Version history:**
- **v1** (2026-04-16): Initial release — 22 Arrow files, all base augmentations.
- **v2** (2026-04-20): Added `cs_diff_snow_heavy_test.arrow` (g=12 variant, VLM Jury acceptance InternVL 89% / Phi-4 92% vs light variant 30% / 8%). Renamed old `cs_diff_snow_test.arrow` → `cs_diff_snow_light_test.arrow`.
- **v3** (2026-04-21): Added rain heavy variants (`cs_diff_rain_heavy_{test,train}.arrow`, `soda_voc_diff_rain_heavy.arrow`, `soda_ktsh_diff_rain_heavy.arrow`) produced by `scripts/update_kaggle_rain_heavy.py`. Also added `soda_voc_diff_rain.arrow` as previously-missing light baseline. Heavy = physics-only overlay on existing IP2P light output (no additional diffusion pass); see `DEVLOG.md` 2026-04-21 for rationale and `docs/methods.md` §1.3 for physics config.
- **v6** (2026-04-22): `cs_diff_rain_test.arrow` (light) and `cs_diff_rain_heavy_test.arrow` (heavy) replaced with DINO-threshold-filtered 300-row subsets. Thresholds: light ≥ 0.75 (selects 300 of 1,515 eligible), heavy ≥ 0.70 (selects 300 of 1,219 eligible). DINO scores from `validation/results/dino_ssim/ip2p_rain{,_heavy}.csv` (DINOv3 ViT-L/16 cosine similarity vs original). Filtering script: `scripts/update_kaggle_rain_dino_filtered.py`. Other files re-uploaded unchanged as part of snapshot.

## Citation

```bibtex
@dataset{duong2026consynthx,
  title  = {ConSynth-X: A Large-Scale Synthetic Construction-Site Image Dataset
            for Challenging Field Conditions},
  author = {Duong, Viet Huy and Xiong, Ruoxin},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Ben11304/ConSynth-X}
}
```
