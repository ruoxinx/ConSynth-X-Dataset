# ConSynth-X

## What is this project?

ConSynth-X is a benchmarking and dataset framework for evaluating **construction computer vision model robustness** under extreme field conditions (weather, lighting, scale, noise, scene complexity, domain shift). It provides:

- A taxonomy of 40+ condition labels across 6 categories
- Multi-label annotation system (images can have multiple simultaneous conditions)
- Synthetic data generation pipelines (weather, day2night, outpainting)
- VLM (Vision-Language Model) benchmarking framework with 11+ models
- YOLOv8 object detection benchmarks measuring augmentation impact
- Per-condition and cross-condition evaluation protocols
- Curated dataset splits for standardized benchmarking

## Project Structure

```
ConSynth-X/
├── taxonomy/                              # Condition definitions
│   └── extreme_conditions_definition.md   # 40+ conditions across 6 categories
├── conditions/                            # Condition parameter configs (YAML)
│   ├── weather.yaml, lighting.yaml, scale.yaml, sensor_noise.yaml
├── splits/                                # Dataset split manifests (JSON)
│   ├── train.json, validation.json, test.json
│   └── weather/, lighting/, scale_size/, occlusion/, seasonal/
├── metadata/                              # Annotation schemas (JSON)
│   ├── condition_labels.json, scene_metadata.json, object_annotations.json
│
├── generation/                            # === SYNTHETIC DATA GENERATION ===
│   ├── weather/                           # Weather augmentation (neural style transfer)
│   │   ├── submit_arrow_augmentation.py   # SLURM submitter (Arrow format)
│   │   ├── submit_soda_augmentation.py    # SLURM submitter (VOC format)
│   │   ├── arrow_augmentation_worker*.py  # Workers: rain/snow + style transfer
│   │   ├── soda_augmentation_worker*.py   # Workers: SODA VOC format
│   │   ├── ssim_filter_worker*.py         # SSIM quality filtering
│   │   ├── submit_ssim_filter_*.py        # Filter job submitters
│   │   ├── Weather_Effect_Generator/      # Third-party lib (MiDaS, VGG19, rain/snow gen)
│   │   ├── rain_style/, snow_style/       # Style reference images (3 each)
│   │   └── __init__.py
│   ├── day2night/                         # Day-to-night conversion (img2img-turbo)
│   │   ├── submit_day2night_batch.py      # SLURM submitter (Arrow)
│   │   ├── submit_day2night_soda.py       # SLURM submitter (SODA VOC)
│   │   ├── day2night_batch_worker.py      # Worker: batch Arrow processing
│   │   ├── day2night_soda_worker.py       # Worker: VOC processing
│   │   ├── merge_arrow_batches.py         # Merge batched Arrow outputs
│   │   └── run_day2night.py               # Single-image test script
│   ├── outpainting/                       # Outpainting (FLUX Fill inpainting)
│   │   ├── submit_outpainting_pipeline.py # SLURM submitter (Arrow)
│   │   ├── submit_outpainting_soda.py     # SLURM submitter (SODA VOC)
│   │   ├── flux_pipeline_worker.py        # Worker: outpaint + bbox transfer (Arrow)
│   │   └── flux_pipeline_worker_voc.py    # Worker: outpaint + bbox transfer (VOC)
│   ├── utils/
│   │   └── export_arrow_to_train.py       # Arrow → image/annotation folders
│   ├── parameter_ranges.json              # Simulator parameter bounds
│   ├── simulation_engine.md               # Simulator interface documentation
│   └── scenario_templates/                # Template assets for generation
│
├── benchmarks/                            # === BENCHMARKING ===
│   ├── vlm/                               # Vision-Language Model benchmarks
│   │   ├── run.py                         # Main CLI: --model, --task, --condition
│   │   ├── models/                        # VLM wrappers (11 models)
│   │   │   ├── base_model.py, registry.py
│   │   │   ├── qwen2_vl.py, qwen3_5.py   # Qwen family (7B-72B)
│   │   │   ├── internvl.py               # InternVL2.5 (8B, 26B)
│   │   │   ├── llava.py, llava_v15.py     # LLaVA family
│   │   │   ├── gemma3.py, phi4.py         # Gemma-3, Phi-4
│   │   │   └── openai_api.py              # GPT-4o, GPT-4o-mini
│   │   ├── tasks/                         # Task definitions
│   │   │   ├── description.py             # 0-shot image captioning
│   │   │   ├── vqa_safety.py              # 5-shot safety VQA (JSON + bboxes)
│   │   │   ├── vqa_rule1.py               # Binary PPE compliance
│   │   │   ├── object_detection.py        # Normalized bbox detection
│   │   │   └── prompts.py                 # Centralized prompts & few-shot examples
│   │   ├── evaluations/                   # Metric computation
│   │   │   ├── description_eval.py        # BERTScore, BLEU, ROUGE-L, METEOR
│   │   │   ├── vqa_eval.py                # Multi-label P/R/F1 + bbox IoU
│   │   │   └── detection_eval.py          # Binary grid-based IoU
│   │   ├── runners/
│   │   │   └── inference_runner.py        # TSV loading, checkpoint/resume, inference
│   │   ├── scripts/
│   │   │   ├── submit_jobs.py             # SLURM job generation & submission
│   │   │   ├── evaluate_description_gpu.py # Batch description eval (GPU)
│   │   │   └── evaluate_vqa_batch.py      # Batch VQA eval (CPU)
│   │   └── configs/                       # YAML configs per condition
│   │       ├── base.yaml                  # Paths & inference defaults
│   │       └── dataset_{original,weather,night,small}.yaml
│   ├── detection/                         # YOLOv8 object detection benchmarks
│   │   ├── train_yolo.py                  # 5 training configs (Construction Site)
│   │   ├── dataset.py                     # Arrow + folder dataset loader
│   │   ├── train.py, evaluate.py          # Custom training & evaluation
│   │   ├── validate_augmentation.py       # Augmentation quality validation
│   │   ├── extract_val_from_arrow.py      # Extract validation set
│   │   └── SODA/                          # SODA dataset benchmarks
│   │       ├── train_yolo_soda.py         # 3 configs with fcntl.flock race protection
│   │       └── submit_yolo_soda.py        # Parallel SLURM submission
│   ├── segmentation/                      # (placeholder)
│   └── tracking/                          # (placeholder)
│
├── evaluation/                            # === CROSS-CUTTING EVALUATION ===
│   ├── robustness_metrics.py              # Per-condition AP computation (TODO)
│   └── cross_condition_eval.py            # Cross-condition eval harness (TODO)
│
├── examples/                              # === NOTEBOOKS & ANALYSIS ===
│   ├── visualize_conditions.ipynb         # Condition label visualization
│   ├── analyze_vqa_safety.ipynb           # VQA safety: radar, heatmap, F1 pivot
│   ├── evaluate_description.ipynb         # Description metrics: bars, heatmaps, radar
│   ├── adaptive_scale_analysis.ipynb      # Outpainting scale analysis
│   ├── tracking_data_quality.ipynb        # Data quality monitoring
│   ├── visualize_outpainting_bbox.ipynb   # Bbox transfer verification
│   ├── validation_metrics.ipynb           # YOLO training metrics
│   ├── verify.ipynb                       # Data integrity checks
│   └── vqa_safety_results.csv             # Pre-computed VQA results table
│
├── environment.yml                        # Conda environment (Python 3.10, torch, ultralytics, etc.)
├── data_card.md, dataset_card.md          # Dataset documentation
├── README.md                              # Project overview
└── CLAUDE.md                              # This file
```

## Source Mapping

Files were integrated from two existing projects:

| ConSynth-X Location | Original Source |
|---|---|
| `generation/weather/` | `ConstructionSite/weather_aug/` |
| `generation/day2night/` | `ConstructionSite/day2night/` |
| `generation/outpainting/` | `ConstructionSite/outpainting/` |
| `generation/utils/` | `ConstructionSite/export_arrow_to_train.py` |
| `benchmarks/vlm/` | `Benchmark_runner/` |
| `benchmarks/detection/` | `ConstructionSite/validation_data/downstream_detection/` |
| `examples/*.ipynb` | Both `Benchmark_runner/notebooks/` and `ConstructionSite/*.ipynb` |
| `environment.yml` | `ConstructionSite/environment.yml` |

## Data Locations (External, not copied)

Data remains at original locations, referenced by absolute paths in code:

- **Construction Site images**: `~/LMUData/` (HuggingFace Arrow format, 3 classes: excavator, rebar, worker_with_white_hard_hat)
- **SODA dataset**: `/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/` (Pascal VOC, 15 classes)
- **Augmented data**: `/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data/`
- **Annotations**: `/users/PGS0407/binben14/VietHuy/ConstructionSite-10k-Implementation/Annotations/`
- **Day2night checkpoint**: `ConstructionSite/day2night/checkpoints/day2night.pkl`

## Key Design Principles

1. **Multi-label**: Real construction images have multiple simultaneous conditions (e.g., rain + low_light + occlusion). Never assume single-label.
2. **Dual format support**: Both HuggingFace Arrow and Pascal VOC (XML) formats throughout the pipeline.
3. **Annotation preservation**: Weather/day2night copy annotations directly; outpainting auto-scales bboxes with offset compensation.
4. **Provenance tracking**: Every condition label records "manual" or "rule-based" assignment.
5. **Reproducibility**: Record simulator versions, random seeds, exact parameter samples.
6. **Construction-specific**: Taxonomy targets construction equipment/workers/sites.

## Augmentation Conditions

| Condition | Method | Key Detail |
|---|---|---|
| `original` | Baseline | Unmodified construction site images |
| `weather` | Neural style transfer | 6 styles: rain_0-2, snow_0-2 (MiDaS + VGG19) |
| `night` | img2img-turbo | Day-to-night, 1:1 mapping, annotations preserved |
| `small` | FLUX Fill outpainting | Gaussian scale expansion, **bboxes redefined** |

## VLM Models (benchmarks/vlm/)

Qwen2.5-VL (7B/32B/72B), Qwen3-VL-8B, InternVL2.5 (8B/26B), LLaVA v1.5-7B, Llama-3.2-11B-Vision, Gemma-3-27B, Phi-4-multimodal, GPT-4o, GPT-4o-mini

## Condition Categories

| Category | Examples |
|---|---|
| Weather | rain, snow, fog, haze, wet_surface |
| Lighting | low_light, backlight, glare, shadow_heavy, nighttime |
| Scale/Size | small_distant, medium, large_closeup |
| Image Quality | motion_blur, defocus_blur, noise, low_resolution, compression_artifacts |
| Scene Complexity | clutter, occlusion, dense_equipment_workers |
| Domain Shift | different_site, different_country, seasonal_change |

## SLURM Cluster Setup

- **Cluster**: OSC, account `pgs0407`, partition `batch`
- **Conda envs**: `VLM` (transformers 4.46.x), `vlm-new` (transformers 5.x)
- **GPU requirements**: 7-8B models = 1x A100; 26-27B = 3-4x A100; 72B = 4x A100
- **Modules**: cuda/12.8.1, miniconda3/24.1.2-py310

## Conventions

- Condition configs: YAML (`conditions/*.yaml`)
- Metadata and splits: JSON
- Code: Python
- Submit scripts generate SLURM `.sh` files in `jobs/`, logs go to `logs/`
- Licensing: Code = Apache 2.0, Dataset = CC BY-NC 4.0
