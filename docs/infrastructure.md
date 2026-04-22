# Infrastructure & Compute

---

## SLURM Cluster

- **Cluster**: OSC, account `pgs0407`, partition `batch` (or `nextgen` for A100)
- **Modules**: `cuda/11.8.0` (or `cuda/12.8.1`), `miniconda3/24.1.2-py310`
- **Submit scripts**: tạo `.sh` files trong `jobs/`, logs vào `logs/`
- All SLURM submit scripts activate conda via `conda activate "${CONSYNTH_CONDA_ENV:-VLM}"` — export `CONSYNTH_CONDA_ENV` to override.

## Conda Environments

| Env | Transformers | Dùng cho |
|---|---|---|
| `VLM` | 4.49.0 | Default (generation pipelines, detection, most VLM inference: LLaVA, InternVL, Gemma, Phi) |
| `vlm-new` | 5.3.0 | Qwen2.5-VL, Qwen3-VL, **DINOv3** (`facebook/dinov3-*`) |

**Lưu ý**: `validation/extract_dino_ssim_all.py` và các script dùng DINOv3
**phải** dùng `vlm-new`. Env `VLM` (4.49.0) không nhận model type `dinov3`.

See `environment.yml` for the pinned dependency list (pip + conda).

## GPU Requirements

| Workload | GPU |
|---|---|
| VLM inference 7-8B (Qwen2.5-7B, LLaVA-7B, InternVL-8B) | 1× A100 |
| VLM inference 26-27B (InternVL-26B, Gemma-27B) | 3-4× A100 |
| VLM inference 72B (Qwen2.5-72B) | 4× A100 |
| Generation pipelines (IP2P, FLUX, day2night) | 1× A100 |
| Sensitivity YOLOv8 training (6 configs) | 1× A100 per job, 8h, 32GB |

## Environment Variables (Path Configuration)

ConSynth-X resolves every path through environment variables — **no hardcoded
`/users/PGS0407/...` anywhere in the Python code**. Defaults in
`generation/_paths.py`; template in `ConSynth-X/.env.example`.

| Env var | Default | Purpose |
|---|---|---|
| `CONSYNTH_REPO_ROOT` | auto-detect | Output paths for figures, results, augmented data written inside the repo |
| `CONSYNTH_DATA_ROOT` | `~/consynth_data` | Base datasets (Construction Site Arrow, SODA VOC), augmentation outputs, validation data, YOLO checkpoints |
| `CONSYNTH_CONDA_ENV` | `VLM` | Conda env name activated inside SLURM submit scripts |
| `CONSYNTH_CONDA_ENV_NEW` | `vlm-new` | Conda env for transformers 5.x models (Qwen3-VL, DINOv3) |
| `CONSYNTH_ANNOTATION_ROOT` | `$CONSYNTH_DATA_ROOT/../ConstructionSite-10k-Implementation/Annotations` | VLM benchmark annotation JSON/TSVs |
| `CONSYNTH_WEIGHTS_URL` | (unset) | Optional override base URL for `weights/download.sh` |
| `CONSYNTH_WEIGHTS_SRC` | (unset) | Local mirror directory (offline clusters) |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache (SLURM VLM scripts respect `$HF_HOME`) |

### Typical OSC setup

```bash
# In ~/.bashrc or a sourced .env
export CONSYNTH_REPO_ROOT=/users/PGS0407/binben14/VietHuy/ConSynth-X
export CONSYNTH_DATA_ROOT=/users/PGS0407/binben14/VietHuy/ConstructionSite
export CONSYNTH_CONDA_ENV=VLM
export CONSYNTH_ANNOTATION_ROOT=/users/PGS0407/binben14/VietHuy/ConstructionSite-10k-Implementation/Annotations
export HF_HOME=/users/PGS0407/binben14/.cache/huggingface
```

Then `source .env` (or `set -a; source .env; set +a` to export all at once) before running any pipeline or submit script.

## Data Locations

Data directories expected under `$CONSYNTH_DATA_ROOT`:

| Sub-path | Format | Source |
|---|---|---|
| `LouisChen15___construction_site/` | HuggingFace Arrow | Construction Site 10k (`LouisChen15/ConstructionSite` on HF) |
| `SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007/` | Pascal VOC | SODA (Duan et al., 2022) |
| `SODA/data/soda-ktsh/{images,captiondata}/` | JPEG + captions | SODA-KTSH extended |
| `augmentation_data/` | Arrow + JPG + metadata | Outputs of generation pipelines (rain_snow, fog, night, small) |
| `augmentation_data_arrow/` | Arrow shards | Merged augmented output (used by preview + validation) |
| `validation_data/downstream_detection/` | YOLO run dirs | Trained YOLOv8 checkpoints + metrics |

Weights (manifested by `weights/download.sh`, see also `docs/data_sources.md` §6):

| Target path | Source |
|---|---|
| `weights/day2night.pkl` ← `generation/day2night/checkpoints/day2night.pkl` (symlink) | `https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl` |
| `weights/rain_vgg_512.pth` ← `generation/weather/libs/Weather_Effect_Generator/VGG/rain_vgg_512.pth` (symlink) | Weather_Effect_Generator Google Drive folder |
| `weights/snow_vgg_512.pth` ← `generation/weather/libs/Weather_Effect_Generator/VGG/snow_vgg_512.pth` (symlink) | same folder |

## Bootstrap for reviewers (end-to-end)

See `INSTALL.md` for the full walkthrough. Short version:

```bash
git clone --recurse-submodules <repo-url> ConSynth-X
cd ConSynth-X
cp .env.example .env && $EDITOR .env        # set CONSYNTH_DATA_ROOT at minimum
set -a; source .env; set +a
conda env create -f environment.yml          # creates VLM env
conda activate VLM
pip install gdown                            # for VGG Google Drive folder
bash weights/download.sh                     # ~2.7 GB, verifies SHA256
```
