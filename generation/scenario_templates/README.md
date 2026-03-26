# Scenario Templates

This directory is reserved for scenario template assets used by the synthetic generation pipeline. Template files can be added here without changing the directory name or repository structure.

Recommended template contents:
- Scene layout or asset references
- Camera rigs and viewpoints
- Object spawn distributions
- Agent paths or waypoints
- Scene-level labels needed for export

Suggested filenames:
- `vehicle_loading.yaml`
- `worker_team.yaml`
- `excavation_zone.yaml`

Templates in this directory should stay aligned with the controls documented in `generation/parameter_ranges.json` and the condition taxonomy used throughout the repository.
