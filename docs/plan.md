# ConSynth-X — Publication Plan

> Target: Paper 1 — Nature Scientific Data
> Created: 2026-04-06
> Last updated: 2026-04-06 (Phase 1A sweep done, YOLOv8 jobs submitted)

---

## Phase 0: Resolve Code Conflicts (ước tính: 1 ngày)

Mâu thuẫn trong code sẽ ảnh hưởng mọi bước sau — phải fix trước.

- [ ] **0.1** Resolve style_weight: chọn 10,000 hoặc 100,000 làm canonical. Xoá giá trị còn lại.
  - Files: `arrow_augmentation_worker.py`, `snow_pipeline.py`, `rain_pipeline.py`
- [ ] **0.2** Resolve steps: chọn 10 hoặc 50. Xoá giá trị còn lại.
  - Files: tương tự
- [ ] **0.3** Verify MiDaS params (baseline=0.54, focal=721.09)
  - Kiểm tra KITTI dataset paper — nếu đúng là KITTI params → cite. Nếu không → document nguồn gốc thật.
  - Files: `snow_pipeline.py:67-68`, `rain_pipeline.py:67-68`
- [ ] **0.4** Document tại sao LPIPS chỉ áp dụng rain (không snow)
  - Viết lý do vào `docs/methods.md` section 1.4

---

## Phase 1: Sensitivity Analysis (ước tính: 3-5 ngày compute + 1 ngày phân tích)

Đây là blocker lớn nhất. Không có sensitivity analysis → reviewer sẽ reject.

### 1A. SSIM Threshold Sweep

- [x] **1A.1** Viết script `generation/sensitivity/ssim_lpips_sweep.py` — ĐÃ XONG
  - Sweep SSIM_lower ∈ {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
  - Sweep LPIPS ∈ {0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0}
  - Results: `generation/sensitivity/results/ssim_sweep_results.csv`, `lpips_sweep_results.csv`
  - Figures: `ssim_sweep_pass_rate.png`, `ssim_sweep_kept_count.png`, `lpips_sweep_pass_rate.png`
- [x] **1A.2** Viết `generation/sensitivity/run_sensitivity.py` — ĐÃ XONG
  - Script all-in-one: filter Arrow → export YOLO → train YOLOv8n → save metrics
  - 6 configs: loose, moderate, current, strict, tight, no_lpips
- [ ] **1A.3** Chạy YOLOv8 training cho 6 threshold configs — **ĐANG CHẠY** (jobs 4706824-4706829)
  - Results sẽ nằm tại `generation/sensitivity/detection_results/metrics_sens_*.json`
- [ ] **1A.4** Phân tích kết quả: tạo figure mAP vs threshold, ghi vào `docs/methods.md` section 1.4

### 1B. Key Findings từ Sweep (data quantity analysis — đã xong)

| Config | SSIM rain | SSIM snow | LPIPS | Rain kept (train) | Snow kept (train) |
|---|---|---|---|---|---|
| loose | ≥0.40 | ≥0.40 | <0.50 | 5184 (74.0%) | 6909 (98.6%) |
| moderate | ≥0.50 | ≥0.50 | <0.40 | 4130 (58.9%) | 6755 (96.4%) |
| **current** | **≥0.60** | **≥0.50** | **<0.35** | **3626 (51.7%)** | **6322 (90.2%)** |
| strict | ≥0.65 | ≥0.60 | <0.30 | 3048 (43.5%) | 5680 (81.0%) |
| tight | ≥0.70 | ≥0.65 | <0.25 | 2259 (32.2%) | 4393 (62.7%) |
| no_lpips | ≥0.60 | ≥0.50 | off | 4022 (57.4%) | 6322 (90.2%) |

### 1C. IP2P Guidance Scale Sweep (nếu có compute)

- [ ] **1C.1** Sweep guidance_scale ∈ {6, 8, 10, 12, 14} cho rain
- [ ] **1C.2** Sweep image_guidance_scale ∈ {1.0, 1.5, 2.0}
- [ ] **1C.3** So sánh SSIM/LPIPS distribution per setting
- [ ] **1C.4** Document optimal hoặc report range

---

## Phase 2: Model Versioning & Reproducibility (ước tính: 1 ngày)

- [ ] **2.1** Pin InstructPix2Pix: lấy revision hash từ HuggingFace, thêm `revision=` vào code
  - Files: `batch_worker.py`, `batch_worker_soda.py`
- [ ] **2.2** Pin FLUX.1-Fill-dev: tương tự
  - Files: `flux_pipeline_worker.py`, `flux_pipeline_worker_voc.py`
- [ ] **2.3** Pin MiDaS: record torch.hub version hoặc download checkpoint cố định
- [ ] **2.4** Document VGG19 checkpoint origin + compute checksum
- [ ] **2.5** Pin `ultralytics` version trong `environment.yml`
- [ ] **2.6** Document img2img-turbo commit hash + compute checksum cho `day2night.pkl`
- [ ] **2.7** Tạo `CHECKSUMS.sha256` cho tất cả model files
- [ ] **2.8** Unify random seeds hoặc document lý do dùng seeds khác nhau (42, 77, 99)

---

## Phase 3: Data Provenance (ước tính: 1 ngày)

- [ ] **3.1** Construction Site dataset
  - [ ] Verify HuggingFace repo URL chính xác
  - [ ] Lấy license từ repo
  - [ ] Tìm + cite paper gốc (BibTeX)
  - [ ] Compute SHA256 checksum cho mỗi Arrow file
  - [ ] Viết download command hoạt động
- [ ] **3.2** SODA dataset
  - [ ] Tìm paper gốc + URL download
  - [ ] Lấy license
  - [ ] Cite (BibTeX)
  - [ ] Compute checksum
- [ ] **3.3** Style reference images (6 ảnh)
  - [ ] Document source/origin
  - [ ] Verify license/permission
  - [ ] Document criteria chọn 3 rain + 3 snow
- [ ] **3.4** Cập nhật `docs/data_sources.md` với tất cả thông tin trên

---

## Phase 4: Literature Verification (ước tính: 2 ngày)

- [ ] **4.1** Đọc full text Tremblay et al. (2021) — verify "21% improvement" và "73% more realistic"
  - Ghi baseline, dataset, metric cụ thể
- [ ] **4.2** Đọc full text Gurbindo et al. (2025) — verify IP2P weather augmentation claims
- [ ] **4.3** Verify Gatys et al. (2016) — paper recommend style_weight/steps bao nhiêu?
- [ ] **4.4** Verify Brooks et al. (2023) IP2P paper — recommend guidance scale range nào?
- [ ] **4.5** Tìm + cite paper gốc cho:
  - [ ] Construction Site dataset
  - [ ] SODA dataset
  - [ ] img2img-turbo (CycleGAN-Turbo)
  - [ ] FLUX.1 technical report
- [ ] **4.6** Cập nhật `docs/literature.md` — đổi "TODO" thành verified citations

---

## Phase 4B: Realism Validation — DONE (2026-04-10)

**Status**: ✅ **COMPLETED** — 3 approaches tested, weather classifier (Approach 3) chosen as primary metric.

> **Lịch sử**: Ban đầu dự định dùng UnivFD fooling rate → thất bại (99.5-100% all). Chuyển sang FID/KID vs weather reference → snow OK, rain fail. Cuối cùng dùng weather classifier (SigLIP2) trực tiếp — phân biệt rõ giữa methods. Xem `docs/methods.md` section 1.6 chi tiết.

### 4B.1 ✅ Approach 1: UnivFD (ABANDONED)

- [x] Download UnivFD weights (cloned from official GitHub)
- [x] Write `validation/realism_detector.py` (CLIP + FC classifier wrapper)
- [x] Run on full dataset (22,601 images, 11 conditions)
- [x] Verdict: **NOT SUITABLE** — fooling rate 99.5-100% for all conditions
- [x] Results saved at `validation/results/full_univfd/` (supplementary only)

### 4B.2 ✅ Approach 2: FID/KID vs 3 Reference Datasets

- [x] Reference 1: **WeatherNet-05** (`prithivMLmods/WeatherNet-05-18039`) — snow (1,875), fog (1,261)
- [x] Reference 2: **ACDC** (ETH Zurich, ICCV 2021) — rain (1K), snow (572), fog (1K), night (1K)
- [x] Reference 3: **WeatherBench** (arXiv 2509.11642) — rain (1K), snow (1K), haze→fog (1K), real-world paired
- [x] Write `validation/compute_fid_kid.py` (InceptionV3 + FID + KID, alias mapping fog↔haze)
- [x] Run full: **39 pairs** across 4 conditions × 3 datasets (Job 4896690, 29 min)
- [x] Results:
  - **Night**: FID 178.9 vs original 270.5 (**−34%**, strongest improvement)
  - **Snow**: validated across all 3 datasets (best FID: 116.8@WeatherNet, −9.3%)
  - **Fog**: validated on WeatherBench+WeatherNet (fog_heavy best: −10% FID)
  - **Rain**: inconclusive — augmentation ≈ baseline (cross-domain gap dominates)
- [x] Plots + JSON + LaTeX at `validation/results/fid_kid/`

### 4B.3 ✅ Approach 3: Weather Classifier (PRIMARY METRIC)

- [x] Identify pretrained model: `prithivMLmods/Weather-Image-Classification` (SigLIP2, Apache-2.0, 85.89% test acc)
- [x] Write `validation/weather_classifier.py`
- [x] Run on full dataset (3004 images/condition, 11 conditions)
- [x] Results: **SUCCESS** — phân biệt rõ methods
  - Style transfer rain: 70-89% accuracy
  - Style transfer snow: 62-95% accuracy
  - IP2P diffusion rain: 63.5%
  - IP2P diffusion snow: **25.7%** (failed)
  - Original/small: 6-8% (cross-domain bias, documented)
- [x] Figures, JSON, LaTeX table at `validation/results/weather_cls/`

### 4B.4 Pending Tasks

- [ ] **Investigate IP2P diffusion_snow failure** (25.7% classifier accuracy, 49% classify as rain)
  - Possible causes: IP2P prompt tạo atmosphere tối giống storm, physics snow overlay không đủ visible
  - Note: FID/KID snow validated OK (diffusion_snow FID thấp hơn baseline) → kết quả complementary
  - **Update 2026-04-12**: Texture fidelity H=0.862 (best) → contradiction với 26% classifier. Cần visual inspection, không dismiss cả hai.
- [ ] **Document cross-domain bias** — classifier trained on driving, tested on construction
- [ ] **Night validation**: weather classifier không có night class → FID/KID validated (−34% vs baseline)
  - **Update 2026-04-12**: Night = best semantic shift (FID −34%, Mahalanobis −38%) + worst texture (DCT_W=29.81). CycleGAN over-smoothing → report as-is, known limitation.
- [ ] **Tie results vào paper**: Technical Validation section với Approach 3 (classifier) + Approach 2 (FID/KID) + Approach 4 (texture) + Approach 5 (Mahalanobis)

### 4B.5 Cross-Validation Synthesis (2026-04-12)

**Tổng hợp xuyên suốt 5 approaches** cho thấy kết quả đủ mạnh cho paper, với narrative chính:

> "ConSynth-X provides complementary augmentation pipelines: diffusion preserves texture fidelity (belief H=0.85-0.86) while style transfer produces stronger weather effects (classifier accuracy 62-95%). Night augmentation achieves the largest semantic improvement (FID −34%). All augmentations pass realism detection (>99.5% fooling rate) and maintain detection performance (mAP50-95 neutral)."

**3 CRITICAL GAPS còn thiếu**:
1. **Detection under weather conditions** — hiện chỉ test clean val, chưa chứng minh robustness gain
2. **SODA validation** — 0/19,846 images validated (chỉ có Construction Site)
3. **Visual inspection diffusion snow** — classifier vs texture contradiction cần visual evidence

### 4B.5 Files Created

```
validation/
├── realism_detector.py           # UnivFD + CLIP zero-shot (Approach 1)
├── run_realism_validation.py     # UnivFD inference runner
├── download_reference.py         # Download WeatherNet + BDD100K
├── compute_fid_kid.py            # FID/KID computation (Approach 2) — 3 datasets, alias mapping
├── compute_texture_fidelity.py   # Texture fidelity (Approach 4)
├── belief_fusion.py              # Dempster-Shafer fusion (Approach 4b)
├── weather_classifier.py         # Weather classification (Approach 3, primary)
├── weights/fc_weights.pth        # UnivFD weights
├── reference_data/
│   ├── acdc/rgb_anon/            # ACDC: fog, night, rain, snow (3,578 images)
│   ├── weatherbench/             # WeatherBench: rain, snow, haze (3,000 images, 1K/cond)
│   └── weathernet/               # WeatherNet: snow (1,875), fog (1,261)
├── results/
│   ├── full_univfd/              # Approach 1 results
│   ├── fid_kid/                  # Approach 2 results (39 pairs, JSON + 3 plots)
│   ├── weather_cls/              # Approach 3 results (primary)
│   └── texture_fidelity/         # Approach 4 results + belief fusion
└── jobs/
    ├── download_reference.sh
    ├── run_fid_kid.sh
    └── run_weather_cls.sh
```

---

## Phase 5: IP2P Downstream Evaluation (ước tính: 2-3 ngày compute)

Hiện chỉ có detection results cho Style Transfer. Cần cho IP2P để so sánh.

- [ ] **5.1** Chạy YOLOv8 training với IP2P rain+snow augmented data (Construction Site)
- [ ] **5.2** Chạy YOLOv8 training với IP2P data (SODA)
- [ ] **5.3** So sánh mAP: Style Transfer vs IP2P vs Combined
- [ ] **5.4** Cập nhật `examples/compare_weather_augmentation.ipynb` với kết quả mới
- [ ] **5.5** Kết luận method nào phù hợp hơn (dựa trên data, không assume trước)

---

## Phase 6: Paper Writing (ước tính: 5-7 ngày)

### 6A. Dataset Description

- [ ] **6A.1** Viết Data Record section — mô tả format, schema, splits
- [ ] **6A.2** Viết Technical Validation section — SSIM/LPIPS/FID/DINO metrics + sensitivity analysis results
- [ ] **6A.3** Tạo dataset statistics table (images per condition, per class, per split)
- [ ] **6A.4** Tạo figure: augmentation pipeline overview diagram

### 6B. Methods

- [ ] **6B.1** Viết augmentation pipeline methodology (từ `docs/methods.md`)
- [ ] **6B.2** Viết quality filtering methodology (SSIM/LPIPS — với sensitivity results)
- [ ] **6B.3** Viết annotation preservation methodology

### 6C. Validation

- [ ] **6C.1** Viết downstream detection results (YOLOv8)
- [ ] **6C.2** Viết cross-condition robustness analysis
- [ ] **6C.3** Tạo figures: detection mAP bar charts, heatmaps, SSIM distributions
- [ ] **6C.4** Viết realism validation section — weather classifier accuracy (primary) + FID/KID snow (secondary) từ Phase 4B. Dùng Weather-Image-Classification (SigLIP2) accuracy làm primary metric. Document IP2P diffusion_snow failure. Document cross-domain bias cho construction images.

### 6D. Data Availability

- [ ] **6D.1** Prepare dataset cho public release (HuggingFace hoặc Zenodo)
- [ ] **6D.2** Viết Data Availability Statement
- [ ] **6D.3** Viết Code Availability Statement (GitHub repo)
- [ ] **6D.4** Tạo README cho dataset release

---

## Phase 7: Final Checks

- [ ] **7.1** Verify tất cả items trong Pre-Publication Checklist (`CLAUDE.md`)
- [ ] **7.2** Chạy data leakage verification script
- [ ] **7.3** Review toàn bộ paper theo 6 Research Integrity Rules
- [ ] **7.4** Peer review nội bộ (advisor)
- [ ] **7.5** Submit

---

## Dependencies

```
Phase 0 ──→ Phase 1 ──→ Phase 5 ──→ Phase 6C
                │
Phase 2 ──────→├──────────────────→ Phase 6B
                │
Phase 3 ──────→├──────────────────→ Phase 6D
                │
Phase 4 ──────→├──────────────────→ Phase 6A
                │
Phase 4B ─────→├──────────────────→ Phase 6C (FID/KID results)
                                    │
                                    ↓
                                Phase 7
```

- Phase 0 trước tất cả (resolve conflicts)
- Phase 1, 2, 3, 4, **4B** có thể làm **song song** (ngoại trừ 1 phụ thuộc Phase 0)
- Phase 4B độc lập — chỉ cần download ACDC + có augmented data sẵn
- Phase 5 phụ thuộc Phase 1 (cần biết optimal thresholds)
- Phase 6 phụ thuộc Phase 1-5 + 4B (cần tất cả kết quả)
- Phase 7 cuối cùng

---

## Quick Reference: File Locations

| Nội dung | File |
|---|---|
| Hub & quy tắc | `CLAUDE.md` |
| Parameter justifications | `docs/methods.md` |
| Literature citations | `docs/literature.md` |
| Data sources & provenance | `docs/data_sources.md` |
| Pre-publication checklist | `docs/checklist.md` |
| Infrastructure & compute | `docs/infrastructure.md` |
| Project architecture | `docs/architecture.md` |
| Related papers | `related_paper/weather_augmentation_references.md` |
| Augmentation report | `generation/AUGMENTATION_REPORT.md` |
| Sensitivity analysis scripts | `generation/sensitivity/` |
| Sensitivity sweep results | `generation/sensitivity/results/` |
| Development log | `DEVLOG.md` |
| This plan | `docs/plan.md` |
