# Pre-Publication Checklist (Scientific Data)

> Audit ngày 2026-04-06 phát hiện các vấn đề cần fix.
> Chi tiết parameters: `docs/methods.md` | Chi tiết citations: `docs/literature.md`

---

## CRITICAL — Phải fix trước khi submit

- [ ] **Rule 2: Parameter justification** — 20+ parameters hardcode không có justification
  - [ ] Resolve mâu thuẫn style_weight (10,000 vs 100,000) và steps (10 vs 50)
  - [x] Chạy sensitivity analysis: SSIM/LPIPS sweep — data quantity done (`generation/sensitivity/results/`)
  - [ ] Chạy sensitivity analysis: downstream mAP per threshold — **ĐANG CHẠY** (6 SLURM jobs submitted)
  - [ ] Document tại sao LPIPS chỉ áp dụng cho rain, không snow
  - [ ] Verify nguồn gốc MiDaS params (baseline=0.54, focal=721.09) — có thể KITTI?
  - [ ] Document IP2P guidance parameters (image_guidance=1.5, guidance=10.0/8.0, steps=30)
  - [ ] Document physics overlay params hoặc cite literature
- [ ] **Reproducibility: Model versioning** — pin tất cả models
  - [ ] `timbrooks/instruct-pix2pix` → thêm `revision=` hash
  - [ ] MiDaS → pin version
  - [ ] FLUX.1-Fill-dev → thêm `revision=` hash
  - [x] img2img-turbo → commit `86f5414`, MIT license, ArXiv 2403.12036
  - [ ] VGG19 checkpoints → document origin + checksum
  - [ ] Pin `ultralytics` version trong `environment.yml`
- [ ] **Data provenance** — hoàn thiện `docs/data_sources.md`
  - [x] Construction Site dataset: URL, license (CC-BY-NC-4.0), citation (Chen & Zou, 2025, ArXiv 2508.11011) — checksum còn thiếu
  - [x] SODA dataset: URL (SharePoint), citation (Duan et al., 2022, DOI 10.1016/j.autcon.2022.104499) — license + checksum còn thiếu
  - [ ] Style reference images: source, license
  - [ ] day2night checkpoint: origin, checksum
- [ ] **Documentation** — hoàn thiện docs/
  - [ ] `docs/literature.md`: verify tất cả TODO items (đã thêm [10]-[12]: ConstructionSite 10k, SODA, SODA-ktsh)
  - [ ] `docs/methods.md`: justify hoặc sensitivity-analyze tất cả parameters

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

### Approach 5: Relative Mahalanobis Distance (CLIP + DINOv2) — SUCCESS
- [x] **Per-image distributional proximity to real weather** — **SUCCESS, complementary to FID/KID**
  - Method: Ruck et al. (2026), Eq. 1-2 — relative formulation cancels background
  - Embeddings: CLIP ViT-L/14 (768-dim) + DINOv2 ViT-L (1024-dim)
  - Reference: ACDC real weather (900 ref + 100 holdout per condition)
  - **All augmentations closer to ACDC than original** — validated across all 4 conditions
  - **Night = largest improvement** (38% gap closed in CLIP)
  - **CLIP vs DINOv2 divergence**: CLIP sensitive to style transfer, DINOv2 nearly unchanged
  - **Snow trade-off**: style transfer > diffusion (CLIP), diffusion > style transfer (DINOv2)
  - Files: `validation/compute_relative_mahalanobis.py`, `jobs/relative_mahalanobis.sh`
  - Results: `validation/results/relative_mahalanobis/` (JSON + 3 plots + LaTeX)

### TODO tiếp theo
- [ ] Investigate tại sao IP2P diffusion_snow fail classifier (25.7% vs style transfer 92%)
- [ ] Document cross-domain bias cho construction images trong paper
- [ ] Tìm/train night-specific classifier (weather classifier không có night class)
- [ ] Viết Technical Validation section với kết quả Approach 3+4+5
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
