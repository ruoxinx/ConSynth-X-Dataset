# Simulation Engine

This repository documents the synthetic generation interface but does not ship a renderer implementation. The files in `generation/` define the parameter ranges, expected template inputs, and export targets that a simulator should follow.

Expected simulator capabilities:
- Load scenario templates describing scene layout, object placement, and camera setup.
- Sample controllable factors from `parameter_ranges.json`.
- Apply condition modifiers for weather, lighting, camera, and noise settings.
- Export image outputs and annotations into the repository metadata structure.

Current parameter interface:
- Weather: `rain_intensity`, `fog_density`, `snow`
- Lighting: `sun_elevation_deg`, `sun_azimuth_deg`, `exposure_ev`
- Camera: `focal_length_mm`, `sensor_width_mm`, `motion_blur_amount`
- Noise: `iso`, `gaussian_noise_std`

Reference pipeline:
1. Load scenario template
2. Sample parameter set from `parameter_ranges.json`
3. Apply condition modifiers to environment and sensor
4. Render RGB, depth, segmentation, and instance masks
5. Export annotations to `metadata/` manifests

Reproducibility guidance:
- Record the simulator name and version in downstream experiment logs.
- Store random seeds alongside each generated batch.
- Preserve the exact parameter sample used for each scene so condition labels can be regenerated or audited later.
