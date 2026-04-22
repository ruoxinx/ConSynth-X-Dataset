"""
Centralised path resolution for ConSynth-X generation pipelines.

Configure via environment variables:
  CONSYNTH_REPO_ROOT    Path to the ConSynth-X checkout (auto-detected if unset)
  CONSYNTH_DATA_ROOT    Path to data directory. Expected layout (mirrors the
                        "ConstructionSite" working tree used during development):
                          $CONSYNTH_DATA_ROOT/
                            LouisChen15___construction_site/   # HF arrow shards
                            SODA/data/SODA VOCdevkit/VOCdevkit/VOC2007/
                            augmentation_data/
                            augmentation_data_arrow/
                            validation_data/
  CONSYNTH_CONDA_ENV    Conda env used by SLURM submit scripts (default: VLM)

See ConSynth-X/.env.example and ConSynth-X/INSTALL.md.
"""

from __future__ import annotations

import os
from pathlib import Path


def _auto_repo_root() -> Path:
    """Walk up from this file to find the repo root (presence of .git or README.md)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists() or (parent / "README.md").exists() and (parent / "generation").exists():
            return parent
    return here.parent.parent  # fallback: generation/ parent


REPO_ROOT: Path = Path(os.environ.get("CONSYNTH_REPO_ROOT", _auto_repo_root()))
DATA_ROOT: Path = Path(os.environ.get("CONSYNTH_DATA_ROOT", Path.home() / "consynth_data"))
CONDA_ENV: str = os.environ.get("CONSYNTH_CONDA_ENV", "VLM")

# Standard sub-paths (legacy layout inherited from ConstructionSite working tree)
ARROW_CS_DIR: Path = DATA_ROOT / "LouisChen15___construction_site"
SODA_VOC_ROOT: Path = DATA_ROOT / "SODA" / "data" / "SODA VOCdevkit" / "VOCdevkit" / "VOC2007"
SODA_KTSH_IMG: Path = DATA_ROOT / "SODA" / "data" / "soda-ktsh" / "images"
SODA_KTSH_CAP: Path = DATA_ROOT / "SODA" / "data" / "soda-ktsh" / "captiondata"
AUG_DATA_DIR: Path = DATA_ROOT / "augmentation_data"
AUG_DATA_ARROW_DIR: Path = DATA_ROOT / "augmentation_data_arrow"
VALIDATION_DATA_DIR: Path = DATA_ROOT / "validation_data"
