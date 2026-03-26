# Dataset Card

- Name: ConSynth-X
- Purpose: Provide controlled synthetic data generation and curated real splits for evaluating robustness of construction CV models under extreme field conditions.
- Generation: See `generation/` for engine, scenario templates, and parameter ranges.
- Conditions: YAML condition configurations live in `conditions/` and are used to generate labeled variants.
- Metadata: scene-level and object-level annotations plus condition labels are stored in `metadata/`.
- Splits: standard `train`, `validation`, and `test` manifests in `splits/`.
- Code license: Apache-2.0
- Dataset license: CC BY-NC 4.0
