# ConSynth-X

ConSynth-X is a construction-vision dataset built from paired synthetic
augmentations of construction-site images. The project covers weather,
lighting, and scale changes for robust computer vision.

This public repository provides the generation code and minimal configuration
needed to study the published pipelines. It does not provide the full dataset.

### Quick links

- 📊 **[View representative sample data](https://ben11304.github.io/ConSynth-X/gallery)**
- 🌐 **[Project website](https://ben11304.github.io/ConSynth-X)**
- 💻 **[Generation code](https://github.com/ruoxinx/ConSynth-X-Dataset)**

## Dataset access and distribution

The full ConSynth-X dataset will not be redistributed through this GitHub
repository or through Hugging Face. The Hugging Face page is provided for
project metadata, documentation, citation, and any future access instructions;
it should not be interpreted as a public full-dataset download.

The project dataset comprises:

| Item | Value |
| --- | --- |
| Synthetic rows | 34,199 |
| Source scenes | 3,109 |
| Source subsets | `cs10k`, `soda_voc`, `soda_ktsh` |
| Synthetic conditions | 11 |
| Quality gate | DINO cosine similarity >= 0.85 for nine gated conditions |

The full dataset, source images, generated outputs, and dataset shards are not
included in this repository. Source-image licenses also prohibit public
redistribution of the upstream originals.

## Reproduce the pipeline

This repository contains the generation workers and documentation for local
regeneration. Validation, benchmarking, and downstream evaluation code are
kept in the private development archive and are not included in the current
public release. The repository does not publish dataset shards, source
images, checkpoints, experiment outputs, notebooks, slides, or internal
development reports.

1. Read [REPRODUCE.md](REPRODUCE.md) for the public workflow and data layout.
2. Copy [.env.example](.env.example) to `.env` and set paths on your machine.
3. Obtain the required third-party checkpoints directly from their upstream
   projects, subject to their licenses and access terms.
4. Run workers with `python <worker> --help`; input data and outputs must live
   outside the repository checkout.

The generation workers are under [`generation/`](generation/).

## Project conditions

The project conditions include `fog_light`, `fog_medium`,
`fog_heavy`, `rain`, `rain_heavy`, `snow_light`, `snow_heavy`, `night`,
`night_rain`, `night_snow`, and `small`. The corresponding source code is
organized by pipeline rather than by a bundled data directory.

## License and citation

The code in this repository is Apache-2.0. Dataset and upstream-image terms
are separate: synthetic release terms, source annotations, and reconstructed
source pixels inherit the terms documented in the Hugging Face project page
and [UPSTREAM_ATTRIBUTION.md](UPSTREAM_ATTRIBUTION.md).

```bibtex
@dataset{consynth_x_2026,
  title  = {ConSynth-X: A paired synthetic construction-site image dataset for robust computer vision under adverse conditions},
  author = {Duong, Viet Huy and Xiong, Ruoxin and Al Forhad, Md Abdullah and Shi, Weishi},
  year   = {2026},
  doi    = {10.57967/hf/9597},
  url    = {https://huggingface.co/datasets/openconstruction/ConSynth-X}
}
```

Please also cite the upstream datasets listed in the Hugging Face card and
the repository's attribution file.

## Acknowledgements

The authors acknowledge the creators of ConstructionSite 10k, SODA, and
SODA-KTSH for making the upstream resources available, and the Ohio
Supercomputer Center for computational resources used in dataset generation
and validation.
