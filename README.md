# ConSynth-X

This repository provides dataset splits, evaluation code, and benchmark protocols for measuring robustness of construction computer vision models under extreme field conditions (weather, lighting, scale, occlusion, image quality, seasonal and domain shifts).

Structure overview:
- `taxonomy/`: explicit definitions for each condition label
- `splits/`: per-condition and combined-condition dataset splits
- `metadata/`: label and condition schemas
- `benchmarks/`: task-specific benchmark configs
- `evaluation/`: evaluation scripts and robustness metrics
- `baselines/`: baseline implementations and configs
- `examples/`: notebooks demonstrating visualization and analysis

Guidance:
- Provide per-condition, multi-label annotations where relevant.
- Report overall, per-condition, cross-condition, and combined-condition performance tables.
- Include failure-case visualizations in the paper.

See `data_card.md` and `taxonomy/extreme_conditions_definition.md` for labeling details.
