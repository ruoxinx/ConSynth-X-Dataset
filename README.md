# ConSynth-X — Code & Reproduction Pipeline

Code repository for **ConSynth-X: A Large-Scale Synthetic Construction-Site Image Dataset for Challenging Field Conditions** (Duong & Xiong, 2026). This repo hosts the *generation, validation, and benchmarking pipeline*. The released dataset bytes (~54 GB Arrow shards) live on HuggingFace — see [Dataset access](#dataset-access).

---

## Preview

| Clean | Night | Fog (heavy) | Fog (medium) | Fog (light) |
| :---: | :---: | :---: | :---: | :---: |
| ![](augmentation_data/samples/soda_original.jpg) | ![](augmentation_data/samples/soda_night.jpg) | ![](augmentation_data/samples/soda_fog_heavy.jpg) | ![](augmentation_data/samples/soda_fog_medium.jpg) | ![](augmentation_data/samples/soda_fog_light.jpg) |

| Rain (IP2P) | Snow (IP2P) | Rain (ST, ablation) | Snow (ST, ablation) | KTSH rain |
| :---: | :---: | :---: | :---: | :---: |
| ![](augmentation_data/samples/soda_rain_diffusion.jpg) | ![](augmentation_data/samples/soda_snow_diffusion.jpg) | ![](augmentation_data/samples/soda_rain_style.jpg) | ![](augmentation_data/samples/soda_snow_style.jpg) | ![](augmentation_data/samples/ktsh_rain.jpg) |

---

## What's in this repo

| Path | Purpose |
|---|---|
| [`generation/`](generation/) | Augmentation pipelines — IP2P rain/snow, Koschmieder fog, CycleGAN-Turbo night, FLUX outpainting |
| [`validation/`](validation/) | DINOv3 + SSIM + LPIPS quality metrics, FID/recognition checks |
| [`human_validation/`](human_validation/) | Flask app for Turing / MOS / recognition human studies |
| [`evaluation/`](evaluation/), [`benchmarks/`](benchmarks/) | Downstream detection / VLM benchmarks |
| [`release_pipeline/`](release_pipeline/) | Repack to Arrow shards + HF/Kaggle/GDrive upload jobs |
| [`experiments/ablation_style_transfer/`](experiments/ablation_style_transfer/) | Legacy VGG neural-style-transfer rain/snow (ablation only) |
| [`docs/`](docs/) | Methods, literature, infrastructure, plan, checklist |
| [`taxonomy/`](taxonomy/) | 40+ extreme-condition labels, 6 categories |

---

## Dataset access

The actual augmented data is **not** committed here. Pull from HuggingFace:

```python
from huggingface_hub import snapshot_download
local = snapshot_download(repo_id="Ben11304/ConSynth-X", repo_type="dataset")
```

For loading recipes (Arrow → PIL, pairing augmented↔clean, bbox decoding, streaming), see [`USAGE.md`](USAGE.md). For the full data card (schema, row counts, provenance, quality filters), see [`dataset_card.md`](dataset_card.md).

---

## Reproduce the augmentations

End-to-end setup (clone + submodules + `.env` + model weights) is in [`INSTALL.md`](INSTALL.md). After install:

```bash
# example: regenerate fog at three visibility zones for SODA-VOC
sbatch jobs/fog_diffusion_soda_voc.sh

# rain/snow IP2P + physics overlay
sbatch jobs/rain_snow_ip2p.sh

# day → night via CycleGAN-Turbo
sbatch jobs/night_cyclegan_turbo.sh
```

Scope decisions, parameter justifications, and integrity rules live in [`CLAUDE.md`](CLAUDE.md) and [`docs/methods.md`](docs/methods.md).

---

## Licence

- **Code (this repo)** — Apache 2.0 ([`LICENSE`](LICENSE))
- **Released augmented dataset** — CC0 1.0 (synthetic content)
- **Upstream source images** — Construction Site 10k under CC BY-NC 4.0; SODA / SODA-KTSH per authors' terms. Respect upstream licences when redistributing derived work.
- **Third-party models** — under their own licences (IP2P MIT, img2img-turbo MIT, FLUX.1 [dev] Non-Commercial, Depth Anything V2 Apache-2.0 / CC-BY-NC-4.0, MiDaS Apache-2.0).

---

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

Please also cite the upstream datasets:

```bibtex
@misc{chen2025vlm,
  title         = {Are Large Pre-trained Vision Language Models Effective
                   Construction Safety Inspectors?},
  author        = {Chen, Xuezheng and Zou, Zhengbo},
  year          = {2025},
  eprint        = {2508.11011},
  archivePrefix = {arXiv}
}

@article{duan2022soda,
  title   = {SODA: A large-scale open site object detection dataset for
             deep learning in construction},
  author  = {Duan, Rui and Deng, Hui and Tian, Mao and Deng, Yichuan and Lin, Jiarui},
  journal = {Automation in Construction},
  volume  = {142},
  pages   = {104499},
  year    = {2022},
  doi     = {10.1016/j.autcon.2022.104499}
}
```

---

## Acknowledgments

Computing time provided by the Ohio Supercomputer Center (OSC).

**Authors.** Viet Huy Duong¹ · Ruoxin Xiong, Ph.D.²
