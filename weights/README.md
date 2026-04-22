# Weights

Pretrained checkpoints required by the generation pipelines.

**These weights are NOT produced by the ConSynth-X authors.** They are
redistributed from upstream projects under their respective licenses. The
download script below pulls them directly from the original publishers so
reviewers can verify provenance end-to-end.

These files are **not tracked in git** (~2.7 GB combined) — see `.gitignore`.
`checksums.sha256` pins the exact bytes we used during dataset generation.

## Files

| File | Size | SHA256 (first 8) | Upstream source | License |
|---|---|---|---|---|
| `day2night.pkl` | 1.6 GB | `7c13bdf3` | https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl | MIT — `GaParmar/img2img-turbo` @ `86f5414` |
| `rain_vgg_512.pth` | 535 MB | `7640881b` | [Google Drive folder](https://drive.google.com/drive/folders/1MEVMLVhrv4t7efwAfCSk13yie8G-XcIB?usp=sharing) | Apache-2.0 — `hgupta01/Weather_Effect_Generator` @ `7d62b67` |
| `snow_vgg_512.pth` | 535 MB | `cfba3604` | same Google Drive folder | Apache-2.0 — `hgupta01/Weather_Effect_Generator` @ `7d62b67` |

Full SHA256 listed in `checksums.sha256`.

## Upstream citations

- **day2night.pkl** — Parmar, Park, Narasimhan & Zhu (2024), *One-Step Image Translation with Text-to-Image Models*, arXiv:2403.12036. Checkpoint hosted by the authors at the CMU URL above (see `src/cyclegan_turbo.py` in `img2img-turbo`).
- **rain/snow_vgg_512.pth** — Gupta et al., *Robust Object Detection in Challenging Weather Conditions* — VGG19 classifiers trained for rain-vs-clear and snow-vs-clear; distributed with the Weather_Effect_Generator repository README.

## How to fetch

### 1) Automatic (recommended)

```bash
pip install gdown          # only needed for the VGG Google Drive folder
bash weights/download.sh
```

- `day2night.pkl` is pulled directly from CMU via `curl`.
- The two VGG `.pth` files are pulled from the public Google Drive folder using
  `gdown --folder`; `gdown` handles file-ID enumeration so no manual ID lookup is
  needed. The folder also contains `fog_vgg_512.pth`, which the script ignores.
  The temporary download directory is removed after the two files are extracted.

### 2) Manual

If automatic download fails (firewall, Drive quota, etc.):
1. Open https://drive.google.com/drive/folders/1MEVMLVhrv4t7efwAfCSk13yie8G-XcIB?usp=sharing
2. Download `rain_vgg_512.pth` and `snow_vgg_512.pth` into `weights/`.
3. Download `day2night.pkl` from https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl into `weights/`.
4. Re-run `bash weights/download.sh` — it will skip existing files, verify
   checksums, and create the symlinks the pipelines expect.

### 3) Local cluster mirror

On hosts that already have the files on disk:
```bash
export CONSYNTH_WEIGHTS_SRC=/path/to/local/mirror   # directory holding the 3 files
bash weights/download.sh
```

### What the script does

1. Downloads (or copies) the three files into `weights/`.
2. Verifies SHA256 against `checksums.sha256` — fails loudly on mismatch.
3. Creates symlinks where the pipeline code looks them up:
   - `generation/day2night/checkpoints/day2night.pkl`
   - `generation/weather/libs/Weather_Effect_Generator/VGG/{rain,snow}_vgg_512.pth`
