# Data Card

- Dataset name: ConSynth-X
- Curators: Repository maintainers
- Contact: Use the repository issue tracker for questions or corrections.
- Version: Label schema `0.1`

Labeling and conditions:
- Each image may have multiple condition labels (multi-label); e.g., `rain` + `low_light`.
- Condition labeling approach: labels may be assigned manually from visual inspection or rule-based from simulator and capture metadata. The label schema records provenance as `manual` or `rule-based`.
- Label consistency: taxonomy definitions live in `taxonomy/extreme_conditions_definition.md`, and condition identifiers are tracked in `metadata/condition_labels.json`.

Included repository assets:
- Condition definitions: `conditions/*.yaml`
- Scene, object, and condition metadata: `metadata/`
- Benchmark manifests: `splits/train.json`, `splits/validation.json`, `splits/test.json`
- Evaluation utilities: `evaluation/`

Recommended reporting tables for papers:
- Overall performance
- Per-condition performance (e.g., rain, fog, low_light)
- Cross-condition generalization (train on clean, test on each condition)
- Combined-condition performance (images with >1 extreme condition)
- Failure-case examples with visualizations and short analysis

Licensing and privacy:
- Code in this repository is released under Apache-2.0.
- The dataset card declares dataset licensing as CC BY-NC 4.0.
- Any release containing identifiable people, vehicles, or site-sensitive imagery should be reviewed before distribution and documented in the downstream dataset release notes.
