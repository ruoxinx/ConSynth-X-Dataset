# Installation

Reviewer-facing setup walkthrough for ConSynth-X. All dependencies either
live in this repository, come through a pinned git submodule, or are fetched
from their original upstream publishers by scripts in this repo — there is no
reliance on a sibling working tree.

## 0. Prerequisites

- Linux host with Python 3.10 and CUDA-capable GPU (A100 recommended).
- `git`, `curl`, `conda` (Miniconda or Anaconda).
- ~15 GB free disk for the repo + weights + augmented outputs.
- Network access to:
  - `github.com` (repo + submodule)
  - `https://www.cs.cmu.edu/~img2img-turbo/` (day2night weight)
  - `https://drive.google.com/` (Weather_Effect_Generator VGG weights)
  - `huggingface.co` (runtime model download: IP2P, FLUX, MiDaS, etc.)

## 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url> ConSynth-X
cd ConSynth-X
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

Submodules pinned:
- `generation/day2night/img2img-turbo` → `GaParmar/img2img-turbo@86f5414` (MIT)

## 2. Configure environment variables

ConSynth-X has no hardcoded `/users/PGS0407/...` paths. Everything resolves
through environment variables. Copy the template and edit:

```bash
cp .env.example .env
$EDITOR .env
```

Minimum required:
```bash
CONSYNTH_DATA_ROOT=/abs/path/to/data    # see step 3 for expected layout
```

Optional (defaults shown in `generation/_paths.py`):
```bash
CONSYNTH_REPO_ROOT=/abs/path/to/ConSynth-X   # else auto-detected
CONSYNTH_CONDA_ENV=VLM                        # default conda env name
CONSYNTH_CONDA_ENV_NEW=vlm-new                # for DINOv3, Qwen3-VL
CONSYNTH_ANNOTATION_ROOT=/abs/.../ConstructionSite-10k-Implementation/Annotations
HF_HOME=$HOME/.cache/huggingface
```

Then export into your shell:
```bash
set -a; source .env; set +a
```

## 3. Prepare base datasets

`$CONSYNTH_DATA_ROOT` should contain:

```
$CONSYNTH_DATA_ROOT/
├── LouisChen15___construction_site/          # HF Arrow shards (download from HF)
│   ├── construction_site-test.arrow
│   ├── construction_site-train-00000-of-00002.arrow
│   └── construction_site-train-00001-of-00002.arrow
├── SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007/   # Pascal VOC (SODA)
│   ├── Annotations/
│   ├── ImageSets/
│   └── JPEGImages/
├── SODA/data/soda-ktsh/
│   ├── images/
│   └── captiondata/
├── augmentation_data/          # created by generation pipelines
├── augmentation_data_arrow/    # merged arrow output
└── validation_data/            # YOLOv8 checkpoints + metrics
```

### Download Construction Site 10k

```python
from datasets import load_dataset
ds = load_dataset("LouisChen15/ConstructionSite")
# Requires HF login + accepting the dataset's terms.
# Then copy the parquet/arrow files into $CONSYNTH_DATA_ROOT/LouisChen15___construction_site/
```

Paper & license: Chen & Zou, 2025 (arXiv:2508.11011), CC BY-NC 4.0.

### Download SODA

- SharePoint link: see `docs/data_sources.md` §2. Requires manual agreement.
- Paper: Duan et al., *Automation in Construction* 142, 104499 (2022),
  DOI:10.1016/j.autcon.2022.104499.

## 4. Create conda environment

```bash
conda env create -f environment.yml
conda activate "${CONSYNTH_CONDA_ENV:-VLM}"
```

Optional second env for transformers 5.x models (Qwen3-VL, DINOv3):
```bash
conda create -n vlm-new python=3.10
conda activate vlm-new
pip install transformers==5.3.0 torch accelerate  # etc.
```

## 5. Fetch pretrained weights

These are third-party weights we redistribute from their **original upstream
publishers** — no re-hosting by the ConSynth-X authors.

```bash
pip install gdown
bash weights/download.sh
```

The script:
1. `curl` pulls `day2night.pkl` (1.6 GB) from CMU's img2img-turbo host.
2. `gdown --folder` pulls `rain_vgg_512.pth` and `snow_vgg_512.pth` (535 MB
   each) from the Weather_Effect_Generator Google Drive folder.
3. Verifies SHA256 against `weights/checksums.sha256`.
4. Creates symlinks at:
   - `generation/day2night/checkpoints/day2night.pkl`
   - `generation/weather/libs/Weather_Effect_Generator/VGG/{rain,snow}_vgg_512.pth`

Alternatives if automatic download fails — see `weights/README.md`.

### For OSC / offline clusters with an existing mirror

```bash
export CONSYNTH_WEIGHTS_SRC=/path/to/mirror/with/all/three/files
bash weights/download.sh
```

## 6. Smoke tests

```bash
# Weights resolved?
ls -l generation/day2night/checkpoints/day2night.pkl
ls -l generation/weather/libs/Weather_Effect_Generator/VGG/

# Submodule populated?
ls generation/day2night/img2img-turbo/src/cyclegan_turbo.py

# Env vars visible to Python?
python -c "from generation._paths import DATA_ROOT, REPO_ROOT; print(DATA_ROOT, REPO_ROOT)"
```

## 7. Run a pipeline (example: IP2P rain on one batch)

```bash
cd generation/weather/rain_snow/diffusion
python submit_test.py --dry-run          # print the SLURM script without submitting
```

SLURM submission (OSC):
```bash
python submit_test.py --weather rain --batch-size 500 --end 500
# check logs/ and jobs/ for the generated .sh + output
```

See each pipeline's README + `docs/plan.md` for the full end-to-end runbook.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: snow_pipeline` | Confirm `generation/weather/libs/Weather_Effect_Generator/` exists and contains `snow_pipeline.py`. If empty, re-run step 1 (submodule init) — the directory is vendored, not a submodule, so it ships inside the repo. |
| `FileNotFoundError: .../img2img-turbo/src/cyclegan_turbo.py` | `git submodule update --init --recursive` |
| SHA256 mismatch in `bash weights/download.sh` | Delete the offending file under `weights/` and re-run. If still mismatching, file at https://github.com/<your-fork>/issues — upstream may have rotated. |
| `gdown` rate-limit on Google Drive | Retry later, or download the two VGG files manually from https://drive.google.com/drive/folders/1MEVMLVhrv4t7efwAfCSk13yie8G-XcIB and place into `weights/`, then re-run `bash weights/download.sh` (it will skip existing files, verify SHA, and create symlinks). |
| `KeyError: 'CONSYNTH_DATA_ROOT'` | You forgot `set -a; source .env; set +a`. Python reads from `os.environ`. |
| Test on a different cluster | Set `CONSYNTH_CONDA_ENV` to match your local env name; remove the `module load` lines in `jobs/*.sh` if your cluster uses a different module system. |

## Reference docs

- [`docs/architecture.md`](docs/architecture.md) — full repo layout + design principles
- [`docs/infrastructure.md`](docs/infrastructure.md) — SLURM, conda envs, env vars, data locations
- [`docs/data_sources.md`](docs/data_sources.md) — upstream URLs, licenses, checksums for datasets + models
- [`docs/methods.md`](docs/methods.md) — pipeline algorithms + parameter justifications
- [`docs/checklist.md`](docs/checklist.md) — pre-publication checklist
- [`weights/README.md`](weights/README.md) — weight provenance + manual download
- [`DEVLOG.md`](DEVLOG.md) — dated development notes
