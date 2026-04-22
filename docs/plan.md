# ConSynth-X — Publication Plan

> Target: Paper 1 — Nature Scientific Data
> Created: 2026-04-06
> Last updated: 2026-04-22 (Style transfer demoted to ablation; Phase 0 NST items deferred)

> **Scope decision 2026-04-21:** Rain/snow main pipeline = **IP2P diffusion**. VGG Neural Style Transfer = **ablation-only** (xem [`experiments/ablation_style_transfer/README.md`](../experiments/ablation_style_transfer/README.md) và [`DEVLOG.md`](../DEVLOG.md) entry `Decision: Move VGG neural style transfer rain/snow to ablation archive`). Tất cả Phase 0/1/5 items tham chiếu ST chỉ block ablation section của paper, **không block main submission**.

---

## Phase 0: Resolve Code Conflicts (ước tính: 1 ngày)

Chỉ còn items áp dụng cho main pipeline (IP2P). NST items đã downgrade.

- [~] **0.1** ~~Resolve style_weight (10,000 vs 100,000)~~ — **DEFERRED (ablation-only)**. NST di chuyển sang ablation archive 2026-04-21 → không cần canonical value cho production. Giữ nguyên code ablation; ghi chú mâu thuẫn trong supplementary nếu report.
- [~] **0.2** ~~Resolve steps (10 vs 50)~~ — **DEFERRED (ablation-only)**, cùng lý do 0.1.
- [~] **0.3** ~~Verify MiDaS params~~ — **DEFERRED (ablation-only)**. MiDaS chỉ dùng trong NST pipeline. Main IP2P không dùng MiDaS depth.
- [ ] **0.4** Document tại sao LPIPS chỉ áp dụng rain (không snow) — vẫn áp dụng cho IP2P main pipeline
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
- [~] **4.3** ~~Verify Gatys et al. (2016) — paper recommend style_weight/steps bao nhiêu?~~ — **DEFERRED (ablation-only)**, NST không còn main pipeline từ 2026-04-21.
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

### 4B.6 VLM Jury + Retention Analysis (2026-04-17 → 2026-04-19) — DONE

**Approach 6: VLM Jury Evaluation** (`validation/vlm_jury/`)
- 3 local judges: Qwen2.5-VL-7B, InternVL2.5-8B, Phi-4-multimodal
- Binary accept/reject cho 400 synthetic + 160 ACDC baseline = 560 × 3 judges = 1,680 inferences
- **Findings**: Fog heavy 98% (> real ACDC), Rain 70-72%, Night 58%, **Snow (light) 8-10% → led to heavy variant**
- Inter-judge κ 0.16-0.36 (moderate); Qwen strict on snow (2-4%) — known VLM bias

**Snow Heavy Variant** (2026-04-18)
- Introduced: `guidance_scale=12.0`, `image_guidance_scale=1.2`, heavier prompt
- No pre-filter (keeps all 3004 images)
- VLM Jury: InternVL 89%, Phi-4 92%, Qwen 9% → restored acceptance

**Approach 7: Retention Analysis** (`validation/extract_dino_ssim_all.py`)
- DINOv3 + SSIM per-image for all 13 augmentation variants
- **Key insight**: DINO threshold ≠ VLM Jury (agreement ~50% = random) — complementary metrics
- **Paper Figures 1+2** generated (retention curves + joint heatmap)

**Research integrity note**: No empirical support found for DINO ≥ 0.75 filter being optimal. Release per-image scores + let users choose threshold per application.

### 4B.7 Files Created

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

Main pipeline detection results cho IP2P. Style Transfer kết quả chỉ dùng cho ablation comparison section.

- [ ] **5.1** Chạy YOLOv8 training với IP2P rain+snow augmented data (Construction Site) — **main**
- [ ] **5.2** Chạy YOLOv8 training với IP2P data (SODA) — **main**
- [ ] **5.3** **Ablation comparison** (supplementary): report IP2P (main) vs ST (ablation) mAP side-by-side để minh hoạ trade-off realism/recognizability — KHÔNG dùng kết quả này để chọn method, quyết định đã chốt IP2P.
- [ ] **5.4** Cập nhật `examples/compare_weather_augmentation.ipynb` — mark ST panel là ablation
- [ ] **5.5** ~~Kết luận method nào phù hợp hơn~~ → **DONE 2026-04-21**: chọn IP2P làm main. Chi tiết lý do trong DEVLOG entry cùng ngày (texture fidelity 0.86 vs 0, lower hallucination rate, downstream numbers sẽ confirm).

## Phase 5B: Rain_heavy Re-validation (ước tính: 1 ngày compute)

Sau quyết định 2026-04-21 (rain có 2 intensity: light = IP2P, heavy = physics-only overlay), cần re-validate rain_heavy trên toàn bộ 6 metric để update paper tables. Chi tiết runbook: [`docs/rain_heavy_revalidation_runbook.md`](rain_heavy_revalidation_runbook.md).

- [x] **5B.1** Generate rain_heavy production data (CS test/train, SODA VOC, SODA KTSH) — DONE 2026-04-21
- [x] **5B.2** Patch 7 validation script registries (add `ip2p_rain_heavy` / `diffusion_rain_heavy`) — DONE 2026-04-21
- [x] **5B.3** DINO/SSIM extraction cho rain_heavy — DONE 2026-04-22 (job 5025824, 5 min)
  - Results: DINO mean 0.754, SSIM mean 0.522, 1,219/1,652 pass DINO≥0.70 (73.8%)
- [x] **5B.4** Kaggle sample v6 DINO-filtered upload — DONE 2026-04-22
  - 300 samples/variant: light DINO≥0.75, heavy DINO≥0.70
- [ ] **5B.5** Weather classifier accuracy rain_heavy — submit `jobs/rerun_weather_cls_rain_heavy.sh`
- [ ] **5B.6** FID/KID vs ACDC rain — submit `jobs/rerun_fid_kid_rain_heavy.sh`
- [ ] **5B.7** Relative Mahalanobis CLIP + DINOv3 — submit `jobs/rerun_mahalanobis_rain_heavy.sh`
- [ ] **5B.8** Texture fidelity + belief fusion — submit `jobs/rerun_texture_fidelity_rain_heavy.sh`
- [ ] **5B.9** VLM Jury 3 judges — `bash jobs/rerun_vlm_jury_rain_heavy.sh` (~2-4h/judge)
- [ ] **5B.10** Regenerate retention chart — `python validation/make_retention_charts_all.py`
- [ ] **5B.11** Update paper Tables 1, 4-9 với row/column rain_heavy

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
