# Reproduction guide

This guide covers local regeneration. The public checkout contains code and
configuration only. Keep datasets, model checkpoints, logs, generated images,
and results outside it.

## 1. Dataset access boundary

The full ConSynth-X dataset is not redistributed through this repository or
through Hugging Face. The project page on Hugging Face is retained for
metadata, documentation, citation, and future access instructions. It is not a
public full-dataset mirror.

This repository therefore does not provide a `load_dataset` recipe. Any local
reproduction requires separately obtained upstream data and the permissions
described in [UPSTREAM_ATTRIBUTION.md](UPSTREAM_ATTRIBUTION.md).

## 2. Prepare a local generation checkout

The generation pipeline requires Linux-compatible GPU software and, for batch
workflows, a SLURM environment. It is not a lightweight CPU tutorial.

```bash
git clone --recurse-submodules https://github.com/ruoxinx/ConSynth-X-Dataset.git
cd ConSynth-X-Dataset
cp .env.example .env
```

Set `CONSYNTH_DATA_ROOT` in `.env` to a directory outside this checkout:

```text
$CONSYNTH_DATA_ROOT/
|-- LouisChen15___construction_site/       # source Arrow shards
|-- SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007/
|-- SODA/data/soda-ktsh/images/
|-- SODA/data/soda-ktsh/captiondata/
|-- augmentation_data/                     # generated intermediate files
`-- augmentation_data_arrow/                # merged Arrow output
```

Obtain upstream datasets directly from their owners. The SODA subsets require
separate permission. See [UPSTREAM_ATTRIBUTION.md](UPSTREAM_ATTRIBUTION.md)
before downloading anything.

No single environment file is included in this public release. The previous
development environment combined generation, validation, and benchmark-only
dependencies. Install the requirements for the selected upstream pipeline and
record the operating system, package versions, model revisions, and GPU used.

Download required model checkpoints directly from their upstream projects.
No model weights belong in this repository.

## 3. Run a worker

Workers take explicit input and output paths. Preview arguments before a run:

```bash
python generation/weather/fog/diffusion/arrow_fog_worker.py --help
python generation/weather/rain_snow/diffusion/batch_worker.py --help
python generation/day2night/day2night_batch_worker.py --help
python generation/outpainting/flux_pipeline_worker.py --help
```

| Pipeline | Entry point | Main requirement |
| --- | --- | --- |
| Fog | `generation/weather/fog/diffusion/arrow_fog_worker.py` | Depth model and GPU |
| Rain/snow | `generation/weather/rain_snow/diffusion/batch_worker.py` | IP2P weights and GPU |
| Day-to-night | `generation/day2night/day2night_batch_worker.py` | CycleGAN-Turbo submodule and checkpoint |
| Scale/outpainting | `generation/outpainting/flux_pipeline_worker.py` | FLUX checkpoint and GPU |

The compound night/weather workers have different input orders and should not
be treated as interchangeable:

- `night_weather_batch_worker.py` starts from a precomputed night Arrow,
  then applies IP2P, the physics overlay, and the quality filter. It does not
  invoke CycleGAN-Turbo itself; generate the night input first with a
  day-to-night worker.
- `weather_night_batch_worker.py` starts from an original daytime Arrow and
  applies IP2P, CycleGAN-Turbo day-to-night, the physics overlay, and the
  quality filter.
- `voc_b2_night_worker.py` starts from an already weather-augmented Arrow,
  skips IP2P, then applies CycleGAN-Turbo day-to-night and the physics overlay.

Run a small range first, inspect the output schema, and then scale to a full
source split. Keep submission scripts and logs outside the checkout; workers
are independent of a particular cluster account or filesystem layout.

## 4. Reproducibility boundary

No public dataset artifact is produced by this checkout. A local regeneration
is a separate research run and can differ across library versions, GPU
kernels, model revisions, and upstream source revisions. Record the input
snapshot, model revision, environment, seed, and output manifest for every new
run. Do not call a local regeneration byte-identical to an unavailable public
artifact.
