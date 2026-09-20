# Upstream attribution

ConSynth-X combines code, synthetic outputs, annotations, and third-party
models with different terms. The Hugging Face project page records public
metadata and documentation; it is not a full-data mirror. This file prevents
the different terms from being collapsed into one license.

## Source datasets

- **Construction-Site / cs10k**: Construction-Site by LouisChen15, under the
  terms stated by the upstream dataset.
- **SODA-D / soda_voc** and **SODA-KTSH / soda_ktsh**: research-only source
  material. Obtain permission from the SODA authors before redistribution or
  publication of derived artifacts.

## Generation models

The public generation pipelines use third-party models and code, including
InstructPix2Pix, CycleGAN-Turbo / img2img-turbo, FLUX.1-Fill-dev, and Depth
Anything V2. Private validation also used DINO; that validation code is not
part of the public release. Each component remains under its own upstream
license. Consult the relevant upstream project before downloading or
redistributing any checkpoint or derivative.

## Dataset release

The full dataset is not redistributed through GitHub or Hugging Face. Any
controlled sample, metadata release, or future access arrangement must retain
the terms of the relevant source dataset and third-party models. Reconstructed
pixels inherit the license of the upstream source dataset.

For complete conditions and citations, see
<https://huggingface.co/datasets/openconstruction/ConSynth-X>.
