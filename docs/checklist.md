# Pre-Publication Checklist (Scientific Data)

> Audit ngày 2026-04-06 phát hiện các vấn đề cần fix.
> Chi tiết parameters: `docs/methods.md` | Chi tiết citations: `docs/literature.md`

---

## CRITICAL — Phải fix trước khi submit

- [ ] **Rule 2: Parameter justification** — 20+ parameters hardcode không có justification
  - [x] ~~Resolve mâu thuẫn style_weight (10,000 vs 100,000) và steps (10 vs 50)~~ — **RESOLVED 2026-04-21**: NST demoted to ablation archive, không còn canonical value cần thiết cho production. Mâu thuẫn được ghi nhận trong ablation note nếu supplementary yêu cầu.
  - [x] Chạy sensitivity analysis: SSIM/LPIPS sweep — data quantity done (`generation/sensitivity/results/`)
  - [ ] Chạy sensitivity analysis: downstream mAP per threshold — **ĐANG CHẠY** (6 SLURM jobs submitted)
  - [ ] Document tại sao LPIPS chỉ áp dụng cho rain, không snow (main pipeline IP2P)
  - [~] ~~Verify nguồn gốc MiDaS params (baseline=0.54, focal=721.09)~~ — **DEFERRED**: MiDaS chỉ dùng trong NST ablation, main IP2P pipeline không dùng depth estimation.
  - [ ] Document IP2P guidance parameters (image_guidance=1.5, guidance=10.0/8.0/12.0, steps=30) — **main**
  - [ ] Document physics overlay params hoặc cite literature — **main**
- [x] **Reproducibility: Self-contained repository + upstream model provenance** (2026-04-20)
  - [x] `Weather_Effect_Generator` vendored at `generation/weather/libs/Weather_Effect_Generator/` (upstream `hgupta01/Weather_Effect_Generator@7d62b67`, Apache-2.0, see `libs/.../NOTICE.md`)
  - [x] `img2img-turbo` pinned git submodule at `generation/day2night/img2img-turbo/` (`GaParmar/img2img-turbo@86f5414`, MIT)
  - [x] Pretrained weights: `weights/` with `checksums.sha256` + `download.sh` pulling from original upstream (CMU for `day2night.pkl`; Weather_Effect_Generator Google Drive for VGG `.pth`)
  - [x] VGG19 checkpoints: origin documented (`docs/data_sources.md` §6.1), SHA256 pinned
  - [x] day2night checkpoint: origin documented (`GaParmar/img2img-turbo@86f5414`, CMU URL), SHA256 pinned
  - [x] Hardcoded `/users/PGS0407/...` paths removed from all Python code (89 occurrences in 36 files) — replaced with `CONSYNTH_DATA_ROOT` / `CONSYNTH_REPO_ROOT` env vars via `generation/_paths.py` and `.env.example`
  - [x] HF models (`timbrooks/instruct-pix2pix`, FLUX, MiDaS, Depth-Anything, Weather-Image-Classification): URLs + licenses documented in `docs/data_sources.md` §6.2. Intentionally **not** hard-pinning `revision=` in code (third-party upstreams, decided trade-off — see §6.2 note).
  - [ ] Pin `ultralytics` version in `environment.yml` (MEDIUM, defer)
- [ ] **Data provenance** — hoàn thiện `docs/data_sources.md`
  - [x] Construction Site dataset: URL, license (CC-BY-NC-4.0), citation (Chen & Zou, 2025, ArXiv 2508.11011) — checksum còn thiếu
  - [x] SODA dataset: URL (SharePoint), citation (Duan et al., 2022, DOI 10.1016/j.autcon.2022.104499) — license + checksum còn thiếu
  - [ ] Style reference images: source, license
- [ ] **Documentation** — hoàn thiện docs/
  - [ ] `docs/literature.md`: verify tất cả TODO items (đã thêm [10]-[12]: ConstructionSite 10k, SODA, SODA-ktsh)
  - [ ] `docs/methods.md`: justify hoặc sensitivity-analyze tất cả parameters
  - [x] Reviewer-facing install doc: `INSTALL.md` (2026-04-20)

## MEDIUM — Nên fix trước submission

- [ ] Random seeds: unify (42, 77, 99) hoặc document lý do khác nhau
- [ ] GPU non-determinism: thêm `torch.backends.cudnn.deterministic = True` nếu claim reproducibility
- [ ] Data leakage verification script: tự động check train/val/test separation
- [ ] SODA outpainting 5% success rate: document nguyên nhân
- [ ] Tremblay et al. citation: thêm baseline/dataset context cho "21%" và "73%"
- [ ] `parameter_ranges.json`: cập nhật hoặc xoá (hiện outdated, không khớp code)

## NEW FINDINGS (2026-04-09)

- [x] **SSIM không phù hợp cho weather augmentation** — phạt cả thay đổi mong muốn. Dùng DINO patch similarity thay thế.
  - Evidence: 50 samples, 5+5 disagreement cases documented (`paper/figures/dino_eval/`)
  - DINO ranks Canny CN+img2img #1 (0.744) vs Vanilla IP2P #1 by SSIM (0.692)
  - Cần cập nhật paper: dùng DINO làm primary metric, SSIM làm secondary
- [x] **IP2P prompt "wet muddy ground" gây hallucinate** — fixed prompt v2, documented in methods.md
- [x] **Model comparison hoàn tất** — SD1.5 IP2P > SDXL > FLUX Kontext cho weather task (by both metrics)
- [ ] **Cần quyết định**: dùng Vanilla IP2P hay Canny CN+img2img cho final data generation?

## REALISM VALIDATION (2026-04-10 — 3 APPROACHES TESTED)

### Approach 1: UnivFD (ABANDONED)
- [x] ~~**AI-generated image detection (UnivFD)**~~ — **KHÔNG PHÙ HỢP**
  - Model: `WisconsinAIVision/UniversalFakeDetect` (CVPR 2023)
  - Full dataset tested (22,601 ảnh, 11 conditions)
  - Fooling rate 99.5-100% cho tất cả conditions → không phân biệt được quality
  - Lý do: ảnh augmented là edit trên ảnh thật, detector luôn thấy "real photograph"
  - Kết quả lưu tại `validation/results/full_univfd/`
  - Files: `validation/realism_detector.py`, `validation/run_realism_validation.py`

### Approach 2: FID/KID vs 3 Reference Datasets (VALIDATED)
- [x] **FID/KID vs ACDC + WeatherBench + WeatherNet** — **39 pairs, 4 conditions**
  - **3 reference datasets**: ACDC (3.6K driving, 4 conditions), WeatherBench (3K real-world, 3 conditions), WeatherNet (3.1K outdoor, 2 conditions)
  - Feature extractor: InceptionV3 pool3 (2048-dim, standard FID)
  - **Night validated**: FID 178.9 vs original 270.5 (−34% gap, ACDC only)
  - **Snow validated**: augmented snow FID lower than baseline across all 3 datasets (best: style_snow_2@WeatherNet 116.8 vs original 128.8)
  - **Fog validated**: fog_heavy best on WeatherBench (190.9 vs original 211.1) and WeatherNet (134.5 vs 143.5)
  - **Rain inconclusive**: augmentation ≈ baseline on ACDC, marginal on WeatherBench (domain gap dominates)
  - Files: `validation/compute_fid_kid.py`, `validation/download_reference.py`
  - Results: `validation/results/fid_kid/` (JSON + 3 barplot PNGs + LaTeX)

### Approach 3: Weather Classifier (SUCCESS — PRIMARY METRIC)
- [x] **Weather classification via SigLIP2** — **SUCCESS, primary metric cho paper**
  - Model: `prithivMLmods/Weather-Image-Classification` (SigLIP2, Apache-2.0, 85.89% test acc)
  - Full dataset tested (3004 ảnh/condition, 11 conditions)
  - **Style transfer rain**: 70-89% accuracy (validated well)
  - **Style transfer snow**: 62-95% accuracy (snow_1/snow_2 >92%)
  - **IP2P diffusion rain**: 63.5% (kém hơn style transfer)
  - **IP2P diffusion snow**: 25.7% (THẤT BẠI — 49% classify thành rain, cần investigate)
  - **Original/small**: 6-8% sun/clear (cross-domain bias, construction≠driving)
  - **Night**: 95% classify thành rain (no night class, different metric cần)
  - Files: `validation/weather_classifier.py`, `validation/jobs/run_weather_cls.sh`
  - Results: `validation/results/weather_cls/` (JSON + figures + LaTeX)

### Approach 4: Texture Fidelity (GLCM+LBP+DCT+Haralick) — SUCCESS
- [x] **Texture-based micro-level artifact detection** — **SUCCESS, complementary metric**
  - Method: Duminil et al. (2025) — 4 texture channels, Wasserstein + KS distribution comparison
  - Full test: 300 images/condition, 12 augmentation conditions + 4 ACDC references
  - **Diffusion (IP2P) best texture fidelity**: Composite 0.61-0.63 (texture gần original nhất)
  - **Style transfer higher artifacts**: Composite 2.96-5.96, 100% dimensions significant (p≈0.000)
  - **Night (CycleGAN) over-smoothed**: DCT_W=29.81 — mất HF content mạnh
  - **Style snow_1 outlier**: Composite 5.96 vs snow_0 (2.96), snow_2 (4.59) — cần kiểm tra style image
  - **Trade-off phát hiện**: Style transfer tạo weather mạnh hơn (classifier nhận ra) nhưng texture artifacts nhiều hơn. Diffusion tinh tế hơn (ít artifacts) nhưng weather effect nhẹ.
  - Files: `validation/compute_texture_fidelity.py`, `jobs/texture_fidelity_validation.sh`
  - Results: `validation/results/texture_fidelity/` (JSON + 6 plots + LaTeX)
  - **Dempster-Shafer belief fusion** completed on Wasserstein scores (Eq. 14-23, τ=0.6, α₀=0.8/0.5)
  - **Diffusion rain/snow = FAITHFUL** (H=0.85-0.86, 4 criteria unanimous, conflict=0)
  - **Style transfer = ARTIFACT/uncertain** (H=0, m(H̄)=0.25-0.83)
  - **Night CycleGAN = ARTIFACT + highest conflict** (0.078) — LBP disagrees with GLCM/DCT
  - Files: `validation/belief_fusion.py`
  - Results: `validation/results/texture_fidelity/belief_fusion_results.json` + 3 plots

### Approach 5: Relative Mahalanobis Distance (CLIP + DINOv3) — SUCCESS
- [x] **Per-image distributional proximity to real weather** — **SUCCESS, complementary to FID/KID**
  - Method: Ruck et al. (2026), Eq. 1-2 — relative formulation cancels background
  - Embeddings: CLIP ViT-L/14 (768-dim) + **DINOv3 ViT-L/16 (1024-dim)** — upgraded from DINOv2 (2026-04-15)
  - Reference: ACDC real weather (900 ref + 100 holdout per condition)
  - **All augmentations closer to ACDC than original** — validated across all 4 conditions
  - **Night = largest improvement** (38% gap closed in CLIP)
  - **CLIP vs DINOv3 divergence**: CLIP sensitive to style transfer, DINOv3 near zero shift
  - **Cross-domain limitation acknowledged**: ACDC driving ≠ construction → DINOv3 near-zero shift reflects domain mismatch, not augmentation failure
  - **Snow trade-off**: style transfer > diffusion (CLIP), diffusion > style transfer (DINOv3)
  - Files: `validation/compute_relative_mahalanobis.py`, `jobs/relative_mahalanobis.sh`
  - Results: `validation/results/relative_mahalanobis/` (JSON + 3 plots + LaTeX)

### Approach 6: VLM Jury Evaluation — SUCCESS (2026-04-17)
- [x] **3 local VLM judges: Qwen2.5-VL-7B, InternVL2.5-8B, Phi-4-multimodal** — **zero-shot, domain-agnostic (no reference needed)**
  - Method: Ruck et al. (2026) Section 3.3 — 2 criteria (Condition Realism + Semantic Preservation), binary accept/reject
  - Input: side-by-side pair (original | augmented), single image per inference
  - Sample: 50 synthetic/condition + 40 ACDC baseline/condition = 560 total × 3 judges = 1,680 inferences
  - **Key findings (majority vote 2/3)**:
    - Fog heavy: 98% (vượt real ACDC fog 97.5%)
    - IP2P Rain: 72% | ST Rain: 70% | Night: 58%
    - IP2P Snow light: 8% | ST Snow B: 10% — led to heavy variant introduction
  - **Inter-judge κ**: 0.16-0.36 (moderate). Qwen strict (57%), InternVL lenient (78%), Phi-4 middle (57%)
  - **Pattern**: Qwen strict specifically on snow (2-4% across all snow variants), similar to Gemini bias in Ruck et al.
  - Files: `validation/vlm_jury/{run_vlm_jury.py, data_loader.py, prompts.py, result_parser.py, analyze_results.py}`
  - Results: `validation/results/vlm_jury/` (per-judge JSON + summary)

### Approach 7: Retention Analysis (DINO + SSIM) — COMPLETE (2026-04-19)
- [x] **Per-image DINO similarity + SSIM for all 13 augmentation variants** — **informs filter design**
  - Computed across 13 conditions: 6 ST + 3 IP2P + 3 fog + 1 night
  - **3-panel retention chart** cho snow_heavy + **comparison chart** cho all conditions
  - Key findings:
    - Style transfer + light fog preserve DINO structure (>90% retention at DINO ≥ 0.80)
    - IP2P diffusion + night aggressive edits (retention ≤77% at DINO ≥ 0.80)
    - SSIM degrades faster than DINO for IP2P/night (pixel > structural changes)
    - **DINO threshold ≠ VLM Jury agreement** (~50% = random) — complementary metrics
  - Files: `validation/extract_dino_ssim_all.py`, `validation/make_retention_chart.py`, `validation/make_retention_charts_all.py`
  - Results: `validation/results/dino_ssim/*.csv` (13 files), `validation/results/{snow_heavy_retention_chart, dino_ssim_retention_all, dino_ssim_retention_table}.{pdf,png}`
  - Paper: `paper/figures/fig4_snow_heavy_retention.pdf` (Figure 1), `fig5_retention_curves_all.pdf` (Figure 2)

### Approach 8 (Platform Ready): Human Perceptual Validation
- [x] **Human validation web platform** — Flask app tại `human_validation/`
  - 3 tasks: Turing Test (fooling rate), Realism Rating (MOS 1-5, ITU-R BT.500), Condition Recognition (confusion matrix)
  - Admin dashboard: per-user progress, per-condition stats, CSV export
  - Keyboard shortcuts cho fast annotation, response time tracking
  - Image sampling script: `human_validation/sample_images.py` (50 imgs/condition default)
  - **Chạy:** `cd human_validation && python app.py` → `http://localhost:5000`
  - [ ] Sample ảnh từ augmentation_data (chạy `sample_images.py`)
  - [ ] Recruit 3+ annotators
  - [ ] Thu thập data → export CSV → tính fooling rate, MOS, recognition accuracy, Krippendorff's alpha

### Public Sample Release — DONE (2026-04-20)
- [x] **Kaggle sample dataset v2** uploaded — https://www.kaggle.com/datasets/viethuyduong/consynth-x-augmentation-sample
  - v1 (2026-04-16): 22 Arrow files, ~100 samples/condition (original, ST, IP2P rain/snow light, fog, night, small, SODA)
  - v2 (2026-04-20): Added `cs_diff_snow_heavy_test.arrow` (gs=12 variant); renamed old `cs_diff_snow_test.arrow` → `cs_diff_snow_light_test.arrow`
  - License: CC0-1.0 (sample) — matches sample-level license; full dataset CC BY-NC 4.0
  - Script: `scripts/update_kaggle_sample.py` + `kaggle datasets version -m "<msg>"`

### TODO tiếp theo
- [ ] Investigate tại sao IP2P diffusion_snow fail classifier (25.7% vs style transfer 92%)
- [ ] Document cross-domain bias cho construction images trong paper
- [ ] Tìm/train night-specific classifier (weather classifier không có night class)
- [ ] Viết Technical Validation section với kết quả Approach 3+4+5+6
- [ ] Kiểm tra style image snow_1 — texture artifacts cao bất thường
- [ ] Investigate night CycleGAN over-smoothing (DCT_W=29.81) — cần post-process?

## CROSS-VALIDATION SYNTHESIS (2026-04-12)

### Key findings từ tổng hợp 5 approaches

- [x] **Trade-off realism vs recognizability**: Diffusion giữ texture (H=0.85-0.86) nhưng weather effect nhẹ (classifier 26-64%). Style Transfer weather effect mạnh (classifier 62-95%) nhưng texture artifacts (H=0). Hai pipeline bổ sung nhau. → Document trong paper Technical Validation.
- [x] **Night CycleGAN**: best semantic (FID −34%, Mahalanobis −38%) nhưng worst texture (DCT_W=29.81, conflict=0.078). Known CycleGAN limitation — over-smoothing. → Report as-is.
- [x] **Diffusion Snow contradiction**: 26% classifier vs 86% texture fidelity. Cần visual inspection. → Xem TODO mới bên dưới.
- [x] **Detection augmentation neutral**: mAP50-95 +0.001, nhưng val set chỉ clean images. Chưa chứng minh robustness gain. → Xem CRITICAL gap bên dưới.

### CRITICAL GAPS cho Paper 1

- [ ] **GAP-1: Detection under weather conditions** — Tất cả detection eval hiện tại dùng clean val set. Cần test trained model trên weather/night test set để chứng minh augmentation tạo robustness, không chỉ "không gây hại".
- [ ] **GAP-2: SODA dataset validation thiếu** — 100% validation results chỉ cho Construction Site (3,004 images). SODA (19,846 images, ~47K augmented) chưa có validation nào. Thiếu 2/3 dataset validation.
- [ ] **GAP-3: Visual inspection diffusion snow** — Classifier accuracy 26% nhưng texture fidelity 86% và FID < baseline → contradiction. Cần visual comparison figure (diffusion snow vs style snow vs real ACDC snow) để giải thích trong paper.
- [ ] **GAP-4: Narrative cho paper** — Kết quả validation đủ mạnh cho câu chuyện "realistic augmentation pipeline with complementary methods". Cần viết Technical Validation section tổng hợp cả 5 approaches với narrative rõ ràng.

## PASSED — Không vi phạm

- [x] Data integrity: không fabrication, skip-on-error pattern
- [x] Annotation preservation: correct across all pipelines
- [x] Provenance tracking: ref_id consistent
- [x] Citation disclaimers: "abstract only" đúng chỗ
- [x] Fact/Literature/Decision separation: exemplary trong notebooks
- [x] JPEG quality: consistent (95), LANCZOS resampling
- [x] License: CC BY-NC 4.0 (data), Apache 2.0 (code)
