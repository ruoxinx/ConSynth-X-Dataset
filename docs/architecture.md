# Project Architecture

---

## Project Structure

```
ConSynth-X/
├── CLAUDE.md                              # Hub: mục tiêu, quy tắc, điều hướng
├── INSTALL.md                             # Reviewer-facing setup (submodules, weights, env)
├── .env.example                           # Environment variables template
├── docs/                                  # Tài liệu nghiên cứu
│   ├── plan.md                            # Kế hoạch 7 phase
│   ├── methods.md                         # Phương pháp & thuật toán
│   ├── literature.md                      # Trích dẫn & verify
│   ├── data_sources.md                    # Nguồn dữ liệu & provenance (incl. weights §6)
│   ├── checklist.md                       # Pre-publication checklist
│   ├── infrastructure.md                  # SLURM, cluster, environment
│   └── architecture.md                    # File này
│
├── weights/                               # Pretrained weight manifest (files gitignored)
│   ├── download.sh                        # Pull from upstream: CMU + Google Drive
│   ├── checksums.sha256                   # SHA256 for day2night.pkl + rain/snow_vgg_512.pth
│   └── README.md                          # Provenance + manual fallback
│
├── taxonomy/                              # Condition definitions
│   └── extreme_conditions_definition.md   # 40+ conditions, 6 categories
├── conditions/                            # Condition parameter configs (YAML)
├── splits/                                # Dataset split manifests (JSON)
├── metadata/                              # Annotation schemas (JSON)
│
├── generation/                            # === SYNTHETIC DATA GENERATION ===
│   ├── _paths.py                          # Centralised path resolution via env vars
│   ├── weather/                           # Weather augmentation (style transfer + IP2P)
│   │   ├── libs/
│   │   │   └── Weather_Effect_Generator/  # Vendored (upstream hgupta01 @7d62b67 + local additions)
│   │   │       ├── NOTICE.md              # Provenance + attribution
│   │   │       ├── LICENSE                # Apache-2.0
│   │   │       ├── rain_pipeline.py       # Local: MiDaS + VGG ST + rain particles
│   │   │       ├── snow_pipeline.py       # Local: MiDaS + VGG ST + snow particles
│   │   │       ├── weather_pipeline.py    # Local: unified entry
│   │   │       ├── style_transfer.py      # Local: batch helper
│   │   │       ├── lib/                   # Upstream helpers + modified style_transfer_utils.py
│   │   │       ├── {Fog,Rain,Snow}_Effect_Generator.py  # Upstream particle generators
│   │   │       └── VGG/                   # Weights (gitignored, populated by weights/download.sh)
│   │   ├── rain_snow/diffusion/           # IP2P-based weather
│   │   │   ├── batch_worker.py            # Day weather: IP2P + physics + filter
│   │   │   ├── night_weather_batch_worker.py  # Night weather: Night → IP2P + physics + filter
│   │   │   ├── physics.py                 # Rain/snow particle generators
│   │   │   ├── submit_test.py             # SLURM submitter (day weather)
│   │   │   └── submit_night_weather_test.py   # SLURM submitter (night weather)
│   │   └── rain_snow/style_transfer/      # Style transfer-based weather
│   │       ├── *_worker*.py               # Workers (Arrow + VOC formats) — import from libs/
│   │       ├── ssim_filter_worker*.py     # SSIM quality filtering
│   │       └── rain_style/, snow_style/   # 6 style reference images
│   ├── day2night/                         # Day-to-night
│   │   ├── img2img-turbo/                 # Pinned git submodule @86f5414 (GaParmar, MIT)
│   │   ├── checkpoints/                   # day2night.pkl (symlinked by weights/download.sh)
│   │   └── day2night_batch_worker.py      # Loads from img2img-turbo/src + checkpoints/
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
├── validation/                            # === REALISM VALIDATION (7 Approaches) ===
│   ├── realism_detector.py                # Approach 1: UnivFD / CLIP zero-shot (abandoned)
│   ├── run_realism_validation.py          # Approach 1: inference runner
│   ├── compute_fid_kid.py                 # Approach 2: FID/KID vs 3 reference datasets
│   ├── weather_classifier.py              # Approach 3: SigLIP2 weather classification (primary)
│   ├── compute_texture_fidelity.py        # Approach 4: GLCM+LBP+DCT+Haralick texture
│   ├── belief_fusion.py                   # Approach 4b: Dempster-Shafer fusion
│   ├── compute_relative_mahalanobis.py    # Approach 5: CLIP + DINOv3 Mahalanobis
│   ├── vlm_jury/                          # Approach 6: VLM Jury (Ruck et al. 2026)
│   │   ├── run_vlm_jury.py                # Main runner (1 model × all conditions)
│   │   ├── data_loader.py                 # Load pairs + ACDC baseline
│   │   ├── prompts.py                     # Paired + baseline prompt templates
│   │   ├── result_parser.py               # Robust JSON extraction
│   │   ├── analyze_results.py             # Aggregate 3 judges, Cohen's κ
│   │   └── run_full_snow_strong.py        # Full 3004-image VLM eval
│   ├── extract_dino_ssim_all.py           # Approach 7: DINO + SSIM for 13 conditions
│   ├── make_retention_chart.py            # 3-panel snow_heavy retention chart
│   ├── make_retention_charts_all.py       # 13-condition comparison chart
│   ├── reference_data/                    # Real weather reference images
│   │   ├── acdc/rgb_anon/                 # ACDC: fog, night, rain, snow (3.6K)
│   │   ├── weatherbench/                  # WeatherBench: rain, snow, haze (3K)
│   │   └── weathernet/                    # WeatherNet: snow, fog (3.1K)
│   ├── jobs/                              # SLURM submission scripts
│   ├── weights/                           # Pretrained weights (gitignored)
│   ├── results/                           # Output JSON/CSV/figures per approach
│   │   ├── vlm_jury/                      # Per-judge JSON + summary
│   │   ├── dino_ssim/                     # 13 condition CSVs (DINO + SSIM per image)
│   │   ├── relative_mahalanobis/          # CLIP + DINOv3 distances
│   │   ├── snow_heavy_retention_chart.*   # Paper Figure 1
│   │   └── dino_ssim_retention_all.*      # Paper Figure 2
│   └── logs/                              # SLURM logs
│
├── release_pipeline/                      # === REPACK / PUBLIC RELEASE ===
│   ├── convert/
│   │   ├── release_schema.py              # Canonical pa.schema per sub + validate_table()
│   │   ├── voc_to_objects.py              # VOC XML parser (round-trip, hard-error on size mismatch)
│   │   ├── cs10k_to_objects.py            # cs10k per-class cols + rule_violations + image_attributes
│   │   ├── bbox_join.py                   # Join bbox from source for shards lacking annotation
│   │   ├── coco_emitter.py                # Streaming COCO writer (CocoStreamingEmitter)
│   │   └── integrity.py                   # SHA256, EXIF rotation, atomic write helpers
│   ├── scripts/
│   │   ├── repack_to_release.py           # Main runner — dry-run default, --execute writes
│   │   ├── validate_release.py            # Post-execute byte/bbox round-trip check
│   │   ├── pack_extra2k.py                # One-off: soda_voc/small extra2k FLUX → arrow
│   │   └── pack_soda_ktsh_original.py     # ktsh raw-jpg → arrow source builder
│   ├── metadata/
│   │   ├── condition_registry.yaml        # 35 shards mapped to format/condition/pipeline/dino csv
│   │   └── pending_dino_shards.yaml       # Historical (now empty after DINO compute)
│   ├── tests/test_parsers.py              # 12 unit tests (VOC round-trip, cs10k formats, etc.)
│   └── logs/                              # SLURM stdout/stderr (repack, hf_upload, kaggle_upload)
│
│   Output (gitignored):
│     /fs/scratch/PGS0407/binben14/ConSynth-X-release-v1/      → 50 GB Parquet
│     /fs/scratch/PGS0407/binben14/ConSynth-X-release-v1-coco/ → 53 GB COCO + JPEG
│   Public mirror: kaggle.com/datasets/viethuyduong/construction-site-augmentation-data
│
├── human_validation/                      # === HUMAN PERCEPTUAL VALIDATION ===
│   ├── app.py                             # Flask web app (3 tasks, admin dashboard)
│   ├── sample_images.py                   # Sample images from augmentation_data
│   ├── requirements.txt                   # flask>=3.0
│   ├── validation.db                      # SQLite database (auto-created, gitignored)
│   ├── static/
│   │   ├── css/style.css                  # UI styling
│   │   └── images/<condition>/            # Sampled images per condition
│   └── templates/                         # Jinja2 HTML templates
│       ├── base.html, login.html, index.html
│       ├── tutorial.html                  # Annotation guidelines
│       ├── task_turing.html               # Task 1: Real vs Synthetic
│       ├── task_realism.html              # Task 2: MOS rating 1-5
│       ├── task_recognition.html          # Task 3: Condition identification
│       ├── task_complete.html             # Completion screen
│       └── dashboard.html                 # Admin: stats, export CSV
│
├── evaluation/                            # Cross-cutting evaluation (TODO)
├── examples/                              # Notebooks & analysis
├── related_paper/                         # Literature references
└── slide/                                 # Presentation assets
```

## Source Mapping

### Internal integration (from sibling working trees)

| ConSynth-X Location | Original Source |
|---|---|
| `generation/weather/` | `ConstructionSite/weather_aug/` |
| `generation/day2night/` | `ConstructionSite/day2night/` (minus `img2img-turbo`, now a submodule) |
| `generation/outpainting/` | `ConstructionSite/outpainting/` |
| `generation/utils/` | `ConstructionSite/export_arrow_to_train.py` |
| `benchmarks/vlm/` | `Benchmark_runner/` |
| `benchmarks/detection/` | `ConstructionSite/validation_data/downstream_detection/` |
| `examples/*.ipynb` | `Benchmark_runner/notebooks/` + `ConstructionSite/*.ipynb` |
| `environment.yml` | `ConstructionSite/environment.yml` |

### External dependencies (self-contained)

Pulled automatically via `git clone --recurse-submodules` + `bash weights/download.sh`:

| Artifact | Form | Upstream | Pinned at |
|---|---|---|---|
| `generation/weather/libs/Weather_Effect_Generator/` | **Vendored** (Apache-2.0, `NOTICE.md` attributes upstream + local additions) | `hgupta01/Weather_Effect_Generator` | commit `7d62b67` |
| `generation/day2night/img2img-turbo/` | **Git submodule** (MIT) | `GaParmar/img2img-turbo` | commit `86f5414` |
| `weights/day2night.pkl` | Weight file (MIT) | `https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl` | SHA256 `7c13bdf3…` |
| `weights/{rain,snow}_vgg_512.pth` | Weight files (Apache-2.0) | Weather_Effect_Generator Google Drive folder | SHA256 `7640881b…` / `cfba3604…` |

Runtime-fetched from HuggingFace / torch.hub (no local artifact in this repo):
`timbrooks/instruct-pix2pix`, MiDaS DPT_Large, `black-forest-labs/FLUX.1-Fill-dev`,
`DepthAnything/Depth-Anything-V2`, `prithivMLmods/Weather-Image-Classification`,
Ultralytics YOLOv8. See `docs/data_sources.md` §6 for licenses.

### Path configuration

All Python code resolves dataset + repo paths through environment variables —
no hardcoded absolute paths. See `.env.example` and `generation/_paths.py`:

| Env var | Default | Used for |
|---|---|---|
| `CONSYNTH_REPO_ROOT` | auto-detect from file location | Output paths (figures, results, augmented data) |
| `CONSYNTH_DATA_ROOT` | `~/consynth_data` | Base datasets (Construction Site Arrow, SODA VOC), augmentation outputs, validation data |
| `CONSYNTH_CONDA_ENV` | `VLM` | Conda env name used in SLURM submit scripts |
| `CONSYNTH_WEIGHTS_URL` | (unset → use upstream URLs) | Optional override for `weights/download.sh` |
| `CONSYNTH_WEIGHTS_SRC` | (unset) | Local mirror directory for offline clusters |
| `CONSYNTH_ANNOTATION_ROOT` | `$CONSYNTH_DATA_ROOT/../ConstructionSite-10k-Implementation/Annotations` | VLM benchmark annotations |

## Key Design Principles

1. **Multi-label**: Real construction images có nhiều conditions đồng thời (rain + low_light + occlusion). Không bao giờ assume single-label.
2. **Dual format**: Hỗ trợ cả HuggingFace Arrow và Pascal VOC (XML) xuyên suốt pipeline.
3. **Annotation preservation**: Weather/day2night copy annotations; outpainting auto-scale bboxes với offset compensation.
4. **Provenance tracking**: Mỗi condition label ghi "manual" hoặc "rule-based".
5. **Reproducibility**: Record simulator versions, random seeds, exact parameters.
6. **Construction-specific**: Taxonomy nhắm vào construction equipment/workers/sites.
7. **Self-contained**: Clone + submodule update + weight download + conda env is enough to rerun every pipeline — no reliance on sibling working trees.

## Augmentation Conditions

| Condition | Method | Key Detail |
|---|---|---|
| `original` | Baseline | Unmodified construction site images |
| `rain` (light) | IP2P g=10 + physics overlay (default) | Text prompt "rainy day…"; SSIM+LPIPS filter |
| `rain_heavy` | `rain` (light) + `heavy_fog` physics overlay — **physics-only, no second diffusion pass** | Deterministic CPU pipeline; preserves bboxes/ssim/lpips from light stage |
| `snow_light` | IP2P g=8 + physics overlay | |
| `snow_heavy` | IP2P g=12 + physics overlay | Stronger prompt for thick snow coverage |
| `fog` (3 zones) | Koschmieder scattering + Depth Anything V2 | Visibility uniform 300-1000m, 3 labeled zones |
| `night` | CycleGAN-Turbo (img2img-turbo) | Day-to-night, 1:1 mapping, annotations preserved |
| `night_rain` | Night → IP2P rain → Physics | Compound: CycleGAN night + IP2P rain + rain streaks |
| `night_snow` | Night → IP2P snow → Physics | Compound: CycleGAN night + IP2P snow + snowflakes |
| `small` | FLUX.1-Fill-dev outpainting | Gaussian scale expansion, bboxes redefined |
| `weather_style_*` (ablation) | Legacy VGG Neural Style Transfer | Moved to `experiments/ablation_style_transfer/` 2026-04-21 |

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
