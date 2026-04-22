# Extreme Conditions Taxonomy

This file defines the taxonomy used in ConSynth-X. Each condition is defined explicitly and notes how labels should be created and validated.

1. Weather
- rain: visible falling rain streaks or wet surfaces caused by rain. Labeling: manual (visual) or sensor metadata if available.
- snow: visible snowflakes or accumulation. Manual labeling preferred.
- fog: reduced contrast and visibility due to fog. Can be rule-based from visibility metrics if available.
- haze: atmospheric scattering causing low contrast (distinct from fog by density).
- wet_surface: ground or equipment visibly wet (may co-occur with rain).

2. Lighting
- low_light: overall underexposed/low luminance images; may include dusk/dawn.
- backlight: strong backlighting causing silhouettes or loss of foreground detail.
- glare: bright specular highlights causing local saturation.
- shadow_heavy: pronounced shadows that obscure objects.
- nighttime: images captured at night with artificial lighting.

3. Scale / Size
- small_distant: objects of interest occupy small pixel area and are far from camera.
- medium: moderate object sizes.
- large_closeup: objects occupy large fraction of image (close-up).

4. Image Quality
- motion_blur: directional blur from camera or object motion.
- defocus_blur: out-of-focus blur.
- noise: sensor noise, high ISO grain.
- low_resolution: images or crops with low pixel resolution.
- compression_artifacts: visible JPEG or other codec artifacts.

5. Scene Complexity
- clutter: many overlapping objects or busy backgrounds.
- occlusion: objects partially occluded by other objects or people.
- dense_equipment_workers: many machines or workers in the scene creating dense interactions.

6. Domain Shift / Seasonal
- different_site: data from sites not present in training set.
- different_country: cross-country domain shift (appearance, equipment differences).
- seasonal_change: snow vs. no-snow, foliage changes, etc.

Labeling notes:
- Multi-label: images may carry multiple condition labels (e.g., `rain` + `low_light` + `occlusion`).
- Label provenance: record whether each condition label is `manual` or `rule-based` in `condition_schema.json`.
- Consistency: provide annotation guidelines and run inter-annotator agreement checks (Cohen's kappa, percent agreement).
