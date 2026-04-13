# Project Architecture

---

## Project Structure

```
ConSynth-X/
├── CLAUDE.md                              # Hub: mục tiêu, quy tắc, điều hướng
├── docs/                                  # Tài liệu nghiên cứu
│   ├── plan.md                            # Kế hoạch 7 phase
│   ├── methods.md                         # Phương pháp & thuật toán
│   ├── literature.md                      # Trích dẫn & verify
│   ├── data_sources.md                    # Nguồn dữ liệu & provenance
│   ├── checklist.md                       # Pre-publication checklist
│   ├── infrastructure.md                  # SLURM, cluster, environment
│   └── architecture.md                    # File này
│
├── taxonomy/                              # Condition definitions
│   └── extreme_conditions_definition.md   # 40+ conditions, 6 categories
├── conditions/                            # Condition parameter configs (YAML)
├── splits/                                # Dataset split manifests (JSON)
├── metadata/                              # Annotation schemas (JSON)
│
├── generation/                            # === SYNTHETIC DATA GENERATION ===
│   ├── weather/                           # Weather augmentation (style transfer + IP2P)
│   │   ├── rain_snow/diffusion/           # IP2P-based weather
│   │   │   ├── batch_worker.py            # Day weather: IP2P + physics + filter
│   │   │   ├── night_weather_batch_worker.py  # Night weather: Night → IP2P + physics + filter
│   │   │   ├── physics.py                 # Rain/snow particle generators
│   │   │   ├── submit_test.py             # SLURM submitter (day weather)
│   │   │   └── submit_night_weather_test.py   # SLURM submitter (night weather)
│   │   ├── rain_snow/style_transfer/      # Style transfer-based weather
│   │   │   ├── *_worker*.py               # Workers (Arrow + VOC formats)
│   │   │   └── ssim_filter_worker*.py     # SSIM quality filtering
│   │   ├── libs/                          # Shared: MiDaS, VGG19, rain/snow gen
│   │   └── rain_style/, snow_style/       # 6 style reference images
│   ├── day2night/                         # Day-to-night (img2img-turbo)
│   ├── outpainting/                       # Outpainting (FLUX.1-Fill-dev)
│   ├── sensitivity/                       # Sensitivity analysis
│   │   ├── ssim_lpips_sweep.py            # Threshold sweep on pre-computed CSVs
│   │   ├── run_sensitivity.py             # All-in-one: filter → YOLO → train → metrics
│   │   ├── submit_all.py                  # Submit 6 SLURM jobs for YOLOv8 per config
│   │   ├── filter_arrow_by_threshold.py   # Filter Arrow files by threshold combo
│   │   └── results/                       # Sweep CSVs + figures
│   └── utils/                             # Arrow export, packing utilities
│
├── benchmarks/                            # === BENCHMARKING ===
│   ├── vlm/                               # VLM benchmarks (11+ models, 3 tasks)
│   │   ├── run.py                         # CLI: --model, --task, --condition
│   │   ├── models/                        # Model wrappers
│   │   ├── tasks/                         # description, vqa_safety, vqa_rule1, detection
│   │   ├── evaluations/                   # BERTScore, BLEU, P/R/F1, IoU
│   │   ├── runners/                       # Inference runner with checkpoint/resume
│   │   ├── scripts/                       # SLURM submission & batch eval
│   │   └── configs/                       # YAML per condition
│   ├── detection/                         # YOLOv8 benchmarks
│   │   └── SODA/                          # SODA-specific configs
│   ├── segmentation/                      # (placeholder)
│   └── tracking/                          # (placeholder)
│
├── validation/                            # === REALISM VALIDATION (4 Approaches) ===
│   ├── realism_detector.py                # Approach 1: UnivFD / CLIP zero-shot (abandoned)
│   ├── run_realism_validation.py          # Approach 1: inference runner
│   ├── compute_fid_kid.py                 # Approach 2: FID/KID vs 3 reference datasets
│   ├── weather_classifier.py              # Approach 3: SigLIP2 weather classification (primary)
│   ├── compute_texture_fidelity.py        # Approach 4: GLCM+LBP+DCT+Haralick texture
│   ├── belief_fusion.py                   # Approach 4b: Dempster-Shafer fusion
│   ├── reference_data/                    # Real weather reference images
│   │   ├── acdc/rgb_anon/                 # ACDC: fog, night, rain, snow (3.6K)
│   │   ├── weatherbench/                  # WeatherBench: rain, snow, haze (3K)
│   │   └── weathernet/                    # WeatherNet: snow, fog (3.1K)
│   ├── jobs/                              # SLURM submission scripts
│   ├── weights/                           # Pretrained weights (gitignored)
│   ├── results/                           # Output JSON/CSV/figures per approach
│   └── logs/                              # SLURM logs
│
├── evaluation/                            # Cross-cutting evaluation (TODO)
├── examples/                              # Notebooks & analysis
├── related_paper/                         # Literature references
└── slide/                                 # Presentation assets
```

## Source Mapping

Files integrated từ 2 dự án gốc:

| ConSynth-X Location | Original Source |
|---|---|
| `generation/weather/` | `ConstructionSite/weather_aug/` |
| `generation/day2night/` | `ConstructionSite/day2night/` |
| `generation/outpainting/` | `ConstructionSite/outpainting/` |
| `generation/utils/` | `ConstructionSite/export_arrow_to_train.py` |
| `benchmarks/vlm/` | `Benchmark_runner/` |
| `benchmarks/detection/` | `ConstructionSite/validation_data/downstream_detection/` |
| `examples/*.ipynb` | `Benchmark_runner/notebooks/` + `ConstructionSite/*.ipynb` |
| `environment.yml` | `ConstructionSite/environment.yml` |

## Key Design Principles

1. **Multi-label**: Real construction images có nhiều conditions đồng thời (rain + low_light + occlusion). Không bao giờ assume single-label.
2. **Dual format**: Hỗ trợ cả HuggingFace Arrow và Pascal VOC (XML) xuyên suốt pipeline.
3. **Annotation preservation**: Weather/day2night copy annotations; outpainting auto-scale bboxes với offset compensation.
4. **Provenance tracking**: Mỗi condition label ghi "manual" hoặc "rule-based".
5. **Reproducibility**: Record simulator versions, random seeds, exact parameters.
6. **Construction-specific**: Taxonomy nhắm vào construction equipment/workers/sites.

## Augmentation Conditions

| Condition | Method | Key Detail |
|---|---|---|
| `original` | Baseline | Unmodified construction site images |
| `weather` | Neural style transfer + IP2P | 6 styles: rain_0-2, snow_0-2 (MiDaS + VGG19) |
| `night` | img2img-turbo | Day-to-night, 1:1 mapping, annotations preserved |
| `night_rain` | Night → IP2P rain → Physics | Compound: CycleGAN night + IP2P rain + rain streaks |
| `night_snow` | Night → IP2P snow → Physics | Compound: CycleGAN night + IP2P snow + snowflakes |
| `small` | FLUX Fill outpainting | Gaussian scale expansion, bboxes redefined |

## VLM Models (benchmarks/vlm/)

Qwen2.5-VL (7B/32B/72B), Qwen3-VL-8B, InternVL2.5 (8B/26B), LLaVA v1.5-7B, Llama-3.2-11B-Vision, Gemma-3-27B, Phi-4-multimodal, GPT-4o, GPT-4o-mini

## Condition Categories

| Category | Examples |
|---|---|
| Weather | rain, snow, fog, haze, wet_surface |
| Lighting | low_light, backlight, glare, shadow_heavy, nighttime |
| Scale/Size | small_distant, medium, large_closeup |
| Image Quality | motion_blur, defocus_blur, noise, low_resolution |
| Scene Complexity | clutter, occlusion, dense_equipment_workers |
| Domain Shift | different_site, different_country, seasonal_change |
