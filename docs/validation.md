# Validation trong ConSynth-X

> Tài liệu này tổng hợp **toàn bộ chiến lược validation** của dự án ConSynth-X (Paper 1 — Nature Scientific Data). Mục tiêu: chứng minh synthetic augmentation (IP2P + physics + CycleGAN-Turbo) thực sự có giá trị khoa học, không chỉ là ảnh bị biến đổi random.
>
> Tài liệu đồng bộ với [`docs/methods.md`](methods.md) (§1.6 Realism Validation) và [`docs/checklist.md`](checklist.md) (mục REALISM VALIDATION). Khi có mâu thuẫn, `methods.md` là nguồn chính xác cho parameter/kết quả; tài liệu này tóm tắt kiến trúc và diễn giải.

> **🔄 Changelog 2026-04-23 22:18 — VLM Jury PRIMARY metric, kết quả cuối cùng:**
> - **Judge panel 3 → 5 (4 local + 1 API), ĐẦY ĐỦ:** InternVL2.5-8B, Phi-4-multimodal, Qwen2.5-VL-7B, Qwen3-VL-8B, **Claude Sonnet 4.6** (đã xong 2026-04-23 22:18, 860 rows, 18 conditions, DINO≥0.80, **0 parse errors**).
> - **Protocol 1 → 3 tracks song song:** Track A (random paired, extended 8→15 conditions), Track B (DINO≥0.75 pre-filter), Track C (full 3,004 snow_strong sweep).
> - **Qwen2.5 broadened 560 → 910 rows**; backup tại `*_results.backup_20260423.json`.
> - **~13,900 inferences tích luỹ tổng** (~43 GPU-hours + Claude API).
> - **Phát hiện cuối:** Claude Sonnet **lenient hơn đáng kể** tất cả local VLMs (ví dụ `ip2p_snow_light` Claude 88% vs Qwen3 6%; `night` Claude 100% vs InternVL 22%) — nhưng **agreement trên conditions hardest** (`night_snow`: Claude 6%, Qwen3 0%) → hard cases robust across model scale.
> - DINO ≥ 0.75 vẫn **không predict** jury accept (~50% agreement ≈ random).
> - Chi tiết đầy đủ trong §10 (A6 — VLM Jury).

---

## 1. Tại sao cần validation nhiều tầng?

Synthetic dataset muốn được một journal như *Nature Scientific Data* accept phải trả lời được 3 câu hỏi tách biệt:

| Câu hỏi | Không trả lời được bằng | Approach phù hợp |
|---|---|---|
| **Q1. Ảnh augmented có giữ được cấu trúc scene gốc không?** (objects, bbox, layout) | Ảnh đẹp nhưng hallucinate → không dùng để train detector | SSIM, DINO cosine, Texture fidelity |
| **Q2. Ảnh augmented có trông giống weather thật không?** (perceptual realism) | SSIM 0.99 nhưng không có mưa → vô dụng | FID/KID vs real, Weather classifier, VLM Jury, Human MOS |
| **Q3. Ảnh augmented có được người/robot nhận ra đúng condition không?** (recognizability) | FID thấp nhưng người xem không biết là mưa hay sương | Weather classifier accuracy, Condition recognition study |

Không có một metric nào trả lời đủ cả ba. Vì vậy ConSynth-X áp dụng **multi-tier validation** với 7 approaches automated + 3 human tasks + VLM jury, và coi **convergent evidence** (nhiều approach đồng ý) là bằng chứng mạnh hơn bất kỳ single number nào.

**Nguyên tắc Research Integrity (bắt buộc — xem `CLAUDE.md`):**
- Không chọn threshold trước rồi justify ngược. Threshold phải đến từ literature hoặc sensitivity analysis.
- Khi kết quả bất ngờ (ví dụ IP2P snow chỉ 25.7% weather-classifier accuracy) → **report as-is**, không explain away.
- Phân biệt rõ: fact from data (không cite) / claim from literature (phải cite DOI) / design decision (ghi vào `methods.md`).

---

## 2. Kiến trúc validation — 3 tầng

```
                    ConSynth-X Validation
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  Tier 1 — Automated Metrics (scalable, reproducible)     │
  │  ┌────────────────────────────────────────────────────┐  │
  │  │  A1 UnivFD (abandoned — không phân biệt được)      │  │
  │  │  A2 FID/KID × 3 reference datasets       CO-PRIMARY│  │
  │  │  A3 Weather Classifier (SigLIP2)         CO-PRIMARY│  │
  │  │  A4 Texture Fidelity (GLCM/LBP/DCT/Haralick)       │  │
  │  │  A4b Dempster-Shafer Belief Fusion       COMPLEMENT│  │
  │  │  A5 Relative Mahalanobis (CLIP + DINOv3)           │  │
  │  │  A6 VLM Jury (Qwen+InternVL+Phi-4)       CO-PRIMARY │  │
  │  │  A7 Per-image DINO/SSIM Retention                  │  │
  │  └────────────────────────────────────────────────────┘  │
  │                                                          │
  │  Tier 2 — Semi-automated (VLM-as-judge)                  │
  │  ┌────────────────────────────────────────────────────┐  │
  │  │  Majority vote 2/3 trên side-by-side image pair    │  │
  │  └────────────────────────────────────────────────────┘  │
  │                                                          │
  │  Tier 3 — Human perceptual study (Flask app)             │
  │  ┌────────────────────────────────────────────────────┐  │
  │  │  T1 Turing Test      → fooling rate                │  │
  │  │  T2 Realism MOS       → ITU-R BT.500 1-5           │  │
  │  │  T3 Condition Recog.  → confusion matrix           │  │
  │  └────────────────────────────────────────────────────┘  │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

**Logic phân tầng:** Tier 1 chạy trên **toàn bộ dataset** (≈22k ảnh) → scalable, cho bảng tổng quát. Tier 2 chạy trên **subset 50 ảnh/condition** → ground-truth-free. Tier 3 chạy trên **sample nhỏ** do expensive → ground-truth người, dùng để calibrate Tier 1/2.

---

## 3. Reference datasets — cross-domain caveat

Để đo "gần real weather", cần benchmark chống lại ảnh weather thật. Vấn đề: **không tồn tại real construction-site weather dataset**. Đây chính là gap mà ConSynth-X fill. Ta mượn 3 dataset **driving/outdoor** và chấp nhận cross-domain bias.

| Dataset | Domain | Conditions | N | License | Dùng cho |
|---|---|---|---|---|---|
| **ACDC** (Sakaridis et al., ICCV 2021) | Driving (Switzerland) | rain 1K, snow 572, fog 1K, night 1K | 3,578 | CC BY-NC-SA 4.0 | A2 FID/KID, A5 Mahalanobis, A6 VLM baseline |
| **WeatherBench** (Guan et al., arXiv 2509.11642) | Outdoor real-world | rain, snow, haze→fog | ~3K | Research-only | A2 FID/KID |
| **WeatherNet-05** | Generic outdoor | snow 1,875, fog 1,261 | 3,136 | Apache-2.0 | A2 FID/KID, weather classifier train src |

**Download script:** [`validation/download_reference.py`](../validation/download_reference.py), [`validation/download_acdc.py`](../validation/download_acdc.py).

**Cross-domain caveat — phải ghi rõ trong paper:**
- Wet road ≠ wet concrete; streetlight ≠ floodlight; traffic sign ≠ construction barrier.
- Relative Mahalanobis (A5) dùng công thức `d_rel = d_k - d_0` để **triệt tiêu shared background features**, nhưng chỉ hiệu quả ở CLIP space (weather-style agnostic); DINOv3 space vẫn còn domain gap (<10% gap closed).
- Kết luận validation phải framed là "augmented **closer to real weather than clear baseline**", KHÔNG phải "indistinguishable from real weather".

---

## 4. Approach A1 — UnivFD (ABANDONED)

**Ý tưởng:** Frozen CLIP:ViT-L/14 + FC classifier (`WisconsinAIVision/UniversalFakeDetect`, Ojha et al., CVPR 2023). Fooling rate cao ⇒ augmentation trông thật.

**Kết quả (22,601 ảnh, 11 conditions):** fooling rate 99.5–100% cho **tất cả** conditions, bao gồm cả ảnh original.

**Lý do fail:** UnivFD train để detect ảnh AI-generated **từ đầu** (faces, scenes hoàn toàn synthetic). Ảnh ConSynth-X là **edit trên photograph thật** — nền vẫn là photograph, chỉ thêm weather effect. Detector luôn thấy "real photograph".

**Status:** Lưu kết quả tại [`validation/results/full_univfd/`](../validation/results/full_univfd/), **chỉ dùng trong supplementary** để giải thích tại sao fake-detection không phù hợp cho edit-based augmentation.

**Files:** [`validation/realism_detector.py`](../validation/realism_detector.py), [`validation/run_realism_validation.py`](../validation/run_realism_validation.py), [`validation/setup_univfd.sh`](../validation/setup_univfd.sh).

---

## 5. Approach A2 — FID/KID vs 3 Reference Datasets (CO-PRIMARY)

**Ý tưởng:** Distribution-level validation. FID/KID giữa augmented distribution và real weather distribution càng thấp càng gần real.

**Setup:**
- Feature: **InceptionV3 pool3, 2048-dim** (standard FID backbone)
- KID: polynomial kernel `(x·y/d + 1)^3`, 100 subsets × min(500, N) samples
- Condition alias: fog ↔ haze (WeatherBench)
- **39 pairs evaluated** = 16 ACDC + 14 WeatherBench + 9 WeatherNet

**Kết quả (tóm tắt, Δ FID vs original baseline):**

| Condition | Best method | ACDC Δ FID | WBench Δ FID | WNet Δ FID | Verdict |
|---|---|---|---|---|---|
| Night | CycleGAN-Turbo | **−91.6 (−34%)** | — | — | ✅ strong |
| Snow | diffusion / style_2 | −10.7 (−5%) | −1.8 (−1%) | −12.0 (−9%) | ✅ validated 3/3 datasets |
| Fog | diffusion_heavy | +9.1 (+4%) | **−20.2 (−10%)** | **−9.0 (−6%)** | ✅ 2/3 datasets |
| Rain | ≈ baseline | +2.5 (+1%) | −0.2 (0%) | — | ⚠ inconclusive |

**Diễn giải:**
- **Night thuyết phục nhất** (−34%). CycleGAN-Turbo rất hiệu quả về distribution.
- **Snow nhất quán across 3 datasets** — mạnh vì cross-reference.
- **Fog ACDC ngược lại** (original gần hơn) — lý do: ACDC fog là driving fog khác construction fog (dust + haze).
- **Rain inconclusive** — cross-domain gap dominates weather effect. Cần pair với approach khác (A3, A6) để kết luận.

**Files:** [`validation/compute_fid_kid.py`](../validation/compute_fid_kid.py), [`validation/jobs/run_fid_kid.sh`](../validation/jobs/run_fid_kid.sh).
**Results:** [`validation/results/fid_kid/`](../validation/results/fid_kid/) — JSON, 3 barplot PDFs, LaTeX table.

---

## 6. Approach A3 — Weather Classifier (CO-PRIMARY)

**Ý tưởng:** Pretrained weather classifier (5 class: cloudy / foggy / rain-storm / snow-frosty / sun-clear). Accuracy cao ⇒ classifier nhận ra condition đúng ⇒ augmentation có tín hiệu weather rõ ở semantic level.

**Model:** `prithivMLmods/Weather-Image-Classification` — SigLIP2 (`google/siglip2-base-patch16-224`) fine-tune trên WeatherNet-05-18039, test accuracy 85.89%, Apache-2.0.

**Kết quả (3,004 ảnh/condition, CS dataset):**

| Condition | Expected | Accuracy | Top predicted | Verdict |
|---|---|---|---|---|
| original | sun/clear | **6.1%** | rain/storm (57.8%) | ⚠ cross-domain bias |
| small (outpaint) | sun/clear | 8.4% | rain/storm | ⚠ same bias |
| **weather_style_rain_0** | rain | **88.6%** | rain | ✅ excellent |
| weather_style_rain_1/2 | rain | 70.8–71.3% | rain | ✅ good |
| **weather_style_snow_1** | snow | **95.0%** | snow | ✅ excellent |
| weather_style_snow_2 | snow | 92.7% | snow | ✅ excellent |
| weather_style_snow_0 | snow | 61.8% | snow | ✓ moderate |
| diffusion_rain (IP2P) | rain | 63.5% | rain | ✓ moderate |
| **diffusion_snow (IP2P light)** | snow | **25.7%** | rain/storm (49.1%) | ✗ **failed — led to heavy variant** |
| night (CycleGAN) | (no night class) | — | rain/storm 95.1% | ⚠ N/A |

**Phát hiện quan trọng:**
1. **Construction domain bias** — original clear chỉ 6.1% vì classifier train trên driving scenes, nhầm dust/concrete thành overcast. Phải note explicitly trong paper.
2. **IP2P diffusion snow (light) thất bại rõ rệt** — motivated addition of `snow_heavy` variant (g=12, prompt "heavy snow, thick snow covering ground") — xem `methods.md` §1.2 IP2P Snow Intensity Variants.
3. **Trade-off realism vs recognizability** — Style transfer có accuracy cao hơn (70–95%) nhưng texture artifacts nhiều hơn (A4 composite 2.96–5.96); IP2P texture tinh tế hơn (composite 0.61–0.63) nhưng weather effect nhẹ hơn.

**Files:** [`validation/weather_classifier.py`](../validation/weather_classifier.py), [`validation/jobs/run_weather_cls.sh`](../validation/jobs/run_weather_cls.sh).
**Results:** [`validation/results/weather_cls/`](../validation/results/weather_cls/).

---

## 7. Approach A4 — Texture Fidelity (GLCM + LBP + DCT + Haralick)

**Ý tưởng:** 4 kênh texture truyền thống phát hiện **micro-level artifacts** mà semantic metrics (FID, classifier) bỏ qua: rain streaks lặp bất thường, over-smooth do diffusion, HF anomaly từ particle overlay.

**Reference:** Duminil, Ieng & Gruyer (2025). *Fidelity assessment of synthetic images with multi-criteria combination under adverse weather conditions.* Scientific Reports. — **[V] Verified** (full method section read).

**4 kênh features:**

| Feature | Dim | Phát hiện gì | Implementation |
|---|---|---|---|
| GLCM | 48 | Thiếu đa dạng gray-tone, texture discontinuity | 6 props × 2 distances × 4 angles, quantize 64 levels |
| LBP | 26 | Micro-texture artifact (streaks lặp) | Uniform LBP, radius=3, 24 pts → 26-bin |
| DCT | 12 | Thiếu natural HF noise / over-smooth | HF/LF energy ratio + DC stats + spectral entropy × 3 channels |
| Haralick | 16 | Statistical texture anomalies tổng hợp | 8 metrics × (mean+std) trên patches 64×64, ≤25/image |

**Distance measure:** Wasserstein per-dim (mean aggregated), KS test cho significance, L2 mean shift, Frobenius covariance divergence. **Không cần GT** — chỉ so distribution original vs augmented.

**Kết quả (300 images/condition):**

| Augmentation | GLCM_W | LBP_W | DCT_W | Haralick_W | **Composite** | Đánh giá |
|---|---|---|---|---|---|---|
| Diffusion snow (IP2P) | 0.807 | 0.003 | 1.337 | 0.288 | **0.609** | Best |
| Diffusion rain (IP2P) | 0.823 | 0.004 | 1.411 | 0.273 | **0.628** | Best |
| Style snow_0 / rain_1 | 2.1–2.1 | 0.005–0.009 | 8.9–10.6 | 0.78–1.04 | 2.96–3.43 | Moderate |
| Style rain_2 | 2.05 | 0.005 | 11.41 | 0.91 | 3.60 | Moderate |
| Style rain_0 / snow_2 | 2.5 | 0.007–0.009 | 12.6–14.8 | 1.06–1.06 | 4.05–4.59 | Significant |
| Fog light | 3.80 | 0.003 | 12.9 | 1.74 | 4.61 | Expected (HF removal) |
| **Style snow_1** | **4.71** | 0.009 | **17.19** | **1.93** | **5.96** | ⚠ Outlier |
| Fog heavy | 5.22 | 0.006 | 19.93 | 2.41 | 6.89 | Expected |
| **Night (CycleGAN)** | 4.77 | 0.003 | **29.81** | 1.97 | **9.14** | ⚠ DCT over-smooth |

**Key findings:**
- **Diffusion (IP2P) ≫ Style Transfer về texture fidelity** (composite 0.61–0.63 vs 2.96–5.96). IP2P giữ gần nguyên micro-texture (KS p > 0.05 ở nhiều dim).
- **CycleGAN-Turbo over-smoothed** — DCT_W = 29.81, gấp 21× diffusion rain. Mất HF content tự nhiên → có thể cần post-processing thêm noise.
- **Style snow_1 outlier** → kiểm tra lại style reference image.
- **Fog thay đổi texture nhiều là expected** (atmospheric scattering physically removes HF).

**Complementary vs A3:** Style transfer thắng A3 (classifier recognizability), IP2P thắng A4 (texture fidelity) → **trade-off realism vs recognizability**, giống Ruck et al. (2026).

**Files:** [`validation/compute_texture_fidelity.py`](../validation/compute_texture_fidelity.py), [`jobs/texture_fidelity_validation.sh`](../jobs/texture_fidelity_validation.sh).
**Results:** [`validation/results/texture_fidelity/`](../validation/results/texture_fidelity/).

---

## 8. Approach A4b — Dempster-Shafer Belief Fusion

**Ý tưởng:** Thay vì report 4 Wasserstein đơn giản, fuse bằng D-S belief theory → **fidelity mass m(H)** + **artifact mass m(H̄)** + **uncertainty m(Ω)** + **conflict**. Đây là contribution chính của Duminil et al. (2025).

**Implementation (adapted — skip CNN, dùng Wasserstein):**

| Tầng | Paper gốc | ConSynth-X |
|---|---|---|
| 1. Feature | GLCM, LBP, DCT, Haralick | ✓ giữ nguyên |
| 2. Score | 3 Xception + HaMeC | **Thay bằng `exp(-λ·W)`, λ calibrated per feature** |
| 3. BBA | Eq. 14 (α₀, τ) | ✓ implement đúng paper |
| 4. CRC combination | Eq. 20–23 (N-source conjunctive) | ✓ implement đúng paper |

**Parameters (theo paper section "Implementation on datasets"):**
- τ = **0.6** (pessimistic threshold)
- α₀ = 0.8 cho GLCM/LBP/DCT, **0.5** cho Haralick (paper note)
- λ = **ln(2) / median_W** per feature (Sc=0.5 at τ boundary)

**Kết quả:**

| Augmentation | m(H) ↑ | m(H̄) ↓ | m(Ω) | Conflict | Verdict |
|---|---|---|---|---|---|
| **Diffusion snow** | **0.862** | 0.000 | 0.138 | 0.000 | **FAITHFUL** |
| **Diffusion rain** | **0.854** | 0.000 | 0.146 | 0.000 | **FAITHFUL** |
| Fog light | 0.054 | 0.491 | 0.385 | 0.069 | mixed |
| Night (CycleGAN) | 0.018 | 0.734 | 0.170 | **0.078** | ARTIFACT + conflict |
| Style rain_1/2 | 0.000 | 0.26–0.29 | **0.71–0.74** | 0.000 | uncertain (near τ boundary) |
| Style snow_1 | 0.000 | **0.831** | 0.169 | 0.000 | ARTIFACT (strongest) |

**Insight D-S thêm được so với Wasserstein đơn giản:**
1. **Conflict phát hiện criteria disagreement** — Night conflict=0.078: LBP nói texture OK (Sc=0.648 > τ) nhưng GLCM/DCT nói hỏng. Wasserstein trung bình hoá, belief fusion **expose được mâu thuẫn**.
2. **Uncertainty quantifies confidence** — Style rain_1/2 Ω > 0.7 ⇒ chưa đủ evidence; khác với snow_1 (Ω=0.169) ⇒ chắc chắn artifact.
3. **Diffusion: 4 criteria unanimous** (conflict=0) → kết luận mạnh hơn single score.

**Files:** [`validation/belief_fusion.py`](../validation/belief_fusion.py).
**Results:** `validation/results/texture_fidelity/belief_fusion_results.json` + 3 figures.

---

## 9. Approach A5 — Relative Mahalanobis (CLIP + DINOv3)

**Ý tưởng:** Per-image metric đo proximity tới real adverse-condition distribution trong embedding space, dùng **relative formulation `d_rel = d_k - d_0`** để triệt tiêu shared background (construction scene giữ nguyên giữa original và augmented).

**Reference:** Ruck, Vautravers, Chalkley & Thomas (2026). *Scalable Evaluation of the Realism of Synthetic Environmental Augmentations in Images.* ArXiv 2603.04325. — **[V] Verified** (full paper read, Eq. 1-2 implemented, Section 3.4).

**Khác gì FID/KID (A2)?**

| Khía cạnh | FID/KID | Relative Mahalanobis |
|---|---|---|
| Granularity | Distribution (1 score/tập) | **Per-image** |
| Backbone | InceptionV3 (2048) | **CLIP ViT-L/14 (768) + DINOv3 ViT-L/16 (1024)** |
| Reference | WeatherNet (generic) | **ACDC per-condition Gaussian** |
| Background handling | Không tách | **d_rel = d_k − d_0** |

**Pipeline (4 steps):**
1. Extract CLIP + DINOv3 features cho ACDC real weather (fog/rain/snow/night), split reference/holdout.
2. Fit multivariate Gaussian `N(μₖ, Σₖ)` per condition + `N(μ₀, Σ₀)` background (pooled).
3. Baseline: 100 held-out ACDC/condition → expected near-zero `d_rel` (upper bound).
4. Per augmented image: `d_rel = d_k(x) − d_0(x)`, report as `−d_rel` (higher = closer to real).

**Parameters:**
- CLIP model: `openai/ViT-L-14` (via open_clip) — per paper.
- DINOv3: `facebook/dinov3-vitl16-pretrain-lvd1689m` (transformers 5.x) — per Section 3.4.1.
- Holdout: 100 per condition (per paper).
- Regularization: `Σ + 1e-5·I` for stability.

**Kết quả (300 images/condition):**

**CLIP −d_rel (% gap closed = better):**

| Condition | ACDC baseline | Original | Best augmentation | Best −d_rel | **% Gap Closed** |
|---|---|---|---|---|---|
| **Night** | −2.97 | −39.32 | CycleGAN-Turbo | **−25.60** | **38%** |
| **Snow** | −8.89 | −53.83 | style_snow_2 | **−41.60** | **27%** |
| **Rain** | −3.63 | −22.17 | style_rain_1 | **−17.57** | **25%** |
| **Fog** | −3.65 | −28.07 | fog_heavy | **−23.57** | **18%** |

**DINOv3 −d_rel:** Gap closed <10% cho tất cả conditions. Đây là **metric limitation** (structural features driving vs construction), không phải augmentation failure. CLIP đáng tin hơn vì weather style domain-agnostic.

**Files:** [`validation/compute_relative_mahalanobis.py`](../validation/compute_relative_mahalanobis.py), [`validation/analyze_mahalanobis_vs_original.py`](../validation/analyze_mahalanobis_vs_original.py), [`jobs/relative_mahalanobis.sh`](../jobs/relative_mahalanobis.sh).
**Results:** [`validation/results/relative_mahalanobis/`](../validation/results/relative_mahalanobis/).

---

## 10. Approach A6 — VLM Jury (CO-PRIMARY, đang mở rộng mạnh)

> **Status (cập nhật 2026-04-23):** VLM Jury đã trở thành approach **trọng tâm** của validation. Từ 3 judges ban đầu mở rộng thành **4 judges local + 1 judge API** (5 tổng), chạy **3 protocols song song**, bao phủ **13+ conditions** (tăng từ 8), và phát hiện mới về disagreement với DINO threshold đã motivate một dòng research riêng. Đây là approach quan trọng nhất cho Technical Validation section.

### 10.1 Ý tưởng & vai trò
3 (→ nay 4–5) VLM judges independent đánh giá binary accept/reject dựa trên (1) condition realism + (2) semantic preservation. **Zero-shot, domain-agnostic** — không cần reference dataset ⇒ không bị cross-domain issue như A2/A5.

**Reference:** Ruck et al. (2026), Section 3.3 + Appendix A. — **[V] Verified.**

**Tại sao VLM Jury là trọng tâm (vượt A2/A3/A5)?**
- A2 FID/KID rain **inconclusive** (cross-domain dominate) → VLM Jury trả lời trực tiếp.
- A3 Weather Classifier có cross-domain bias (original 6.1%) → VLM tolerant với domain shift.
- A5 Mahalanobis DINOv3 gap closed <10% do domain mismatch → VLM không cần reference.
- A4 texture mạnh về artifact detection nhưng **không** trả lời "ảnh có giống weather không".
- **VLM Jury là single approach duy nhất mà convergent evidence có thể inspect per-image explanation** (text JSON).

### 10.2 Judge roster (hiện tại)

| Judge | Model | GPU / Env | Status | Role |
|---|---|---|---|---|
| **Qwen2.5** | Qwen2.5-VL-7B-Instruct | 1× A100 / `vlm-new` | ✅ done, **broadened** 2026-04-23 | Strict judge (anchor low) |
| **InternVL** | InternVL2.5-8B | 1× A100 / `VLM` | ✅ done | Lenient judge (anchor high) |
| **Phi-4** | Phi-4-multimodal-instruct | 1× A100 / `VLM` | ✅ done | Middle judge |
| **Qwen3** *(mới)* | Qwen3-VL-8B | 1× A100 / `vlm-new` | ✅ done 2026-04-23, DINO≥0.75 protocol | Newer-gen cross-check |
| **Claude Sonnet 4.6** *(API)* | `claude-sonnet-4.6` | Anthropic API | ✅ **done 2026-04-23 22:18** (DINO≥0.80, 14 synthetic + 4 ACDC, **860 rows, 0 parse errors**) | Commercial-grade cross-check |

Job script Claude: [`jobs/vlm_jury_claude_sonnet.sh`](../jobs/vlm_jury_claude_sonnet.sh). Dùng `ANTHROPIC_API_KEY` từ `.env`, protocol identical (paired side-by-side, same prompt template). Kết quả tại [`claude-sonnet-4.6_dino0.8_results.json`](../validation/results/vlm_jury/claude-sonnet-4.6_dino0.8_results.json) (794KB). Panel 5-judge → chạy được majority ≥3/5 cho subset overlap conditions.

**Module implementation:** [`validation/vlm_jury/`](../validation/vlm_jury/)
- [`run_vlm_jury.py`](../validation/vlm_jury/run_vlm_jury.py) — main entry point, `--model` selector, `--n-synthetic`, `--n-acdc`, `--dino-threshold`, `--resume` (checkpoint-based restart)
- [`data_loader.py`](../validation/vlm_jury/data_loader.py) — Arrow shards + DINO-filtered sampling
- [`prompts.py`](../validation/vlm_jury/prompts.py) — condition-specific templates
- [`result_parser.py`](../validation/vlm_jury/result_parser.py) — JSON extraction + fallback regex
- [`analyze_results.py`](../validation/vlm_jury/analyze_results.py) — aggregation + Cohen's κ
- [`test_dino_threshold_vs_jury.py`](../validation/vlm_jury/test_dino_threshold_vs_jury.py), [`run_full_snow_strong.py`](../validation/vlm_jury/run_full_snow_strong.py), [`test_snow_strong.py`](../validation/vlm_jury/test_snow_strong.py)

### 10.3 Protocol hiện tại — 3 tracks song song

VLM Jury **không còn là single-pass evaluation**. Hiện có 3 tracks riêng biệt:

#### Track A — "Original" paired jury (baseline 2026-04-15 → extended 2026-04-23)
- **Sampling:** random 50 synthetic/condition + 40 ACDC/condition real baseline.
- **Input:** side-by-side (orig | aug) → 1 image per eval.
- **Initial conditions (8):** `fog_heavy`, `ip2p_rain`, `ip2p_snow`, `night`, `st_rain_a/b/c`, `st_snow_b`.
- **Extended conditions (2026-04-23, Qwen2.5 → 910 rows):** thêm `fog_light`, `fog_medium`, `ip2p_rain_heavy`, `ip2p_snow_heavy`, `ip2p_snow_light`, `night_rain`, `night_snow` → **15 conditions**.
- **Backup:** [`*_results.backup_20260423.json`](../validation/results/vlm_jury/) preserved trước khi extend.

#### Track B — DINO≥0.75 pre-filter jury (2026-04-23, **mới**)
- **Motivation:** Track A sample random không consider chất lượng pre-filter. Track B **chỉ sample ảnh đã pass DINO≥0.75** → đo jury rate trên subset "quality-gated".
- **Sampling:** 50 synthetic/condition với ràng buộc `dino_sim ≥ 0.75`.
- **Conditions (7 cho 3 judges gốc):** `fog_light`, `fog_medium`, `ip2p_rain_heavy`, `ip2p_snow_light`, `night`, `night_rain`, `night_snow`.
- **Conditions (18 cho Qwen3):** full sweep bao gồm ACDC baselines + 15 synthetic conditions.
- **Rationale:** thử nghiệm "nếu user filter DINO trước rồi mới dùng, VLM jury còn cao không?" — câu trả lời: **DINO filter không boost jury rate** (xem 10.6).
- **Results:** [`*_dino0.75_results.json`](../validation/results/vlm_jury/).

#### Track C — Full snow_strong sweep (2026-04-18, 9,012 inferences)
- **Sampling:** **all 3,004** snow_heavy images × 3 judges (InternVL + Phi-4 + Qwen2.5) = **9,012 inferences**.
- **Purpose:** dense validation cho snow_heavy variant quyết định (decision scope chọn diffusion-based heavy thay vì physics-only — xem `methods.md` §1.2).
- **Results:** [`snow_strong_full_{internvl,phi4,qwen}.json`](../validation/results/vlm_jury/) (≈ 1MB mỗi file).

### 10.4 Kết quả Track A — main results (extended 2026-04-23)

**Majority vote ≥2/3 (InternVL + Phi-4 + Qwen2.5):**

| Condition | Majority | ACDC baseline (same judge panel) | Notes |
|---|---|---|---|
| **Fog heavy** | **98%** | 97.5% | ✅ Vượt real ACDC fog |
| IP2P Rain | 72% | 87.5% | ✅ Good |
| ST Rain (A/B/C) | 70% | 87.5% | ✅ Good |
| Night | 58% | 90.0% | ⚠ Moderate — judges disagree strongly |
| ST Snow B | 10% | 87.5% | ✗ Poor → motivated heavy variant |
| IP2P Snow (light) | 8% | 87.5% | ✗ Poor → motivated heavy variant |
| **IP2P Snow (heavy)** | **~80%** | 87.5% | ✅ Heavy variant **restores acceptance** |

**Per-judge acceptance cho conditions mới added (Qwen2.5, broadened 2026-04-23):**

| Condition | Qwen2.5 accept |
|---|---|
| fog_light | 70.0% |
| fog_medium | 68.0% |
| fog_heavy | 56.0% |
| ip2p_rain | 54.0% |
| ip2p_rain_heavy | 16.0% |
| ip2p_snow_heavy | 6.0% |
| ip2p_snow_light | 4.0% |
| night | 96.0% *(Qwen unexpectedly lenient on night)* |
| night_rain | 32.0% |
| night_snow | 0.0% |

**Observations:**
- `ip2p_rain_heavy` (physics-only variant) **giảm mạnh so với light** (Qwen2.5: 54% → 16%; Qwen3: 68% → 46%). Lý do: physics overlay + blur σ=1.3 làm ảnh "hazy/washed" — VLM diễn giải như "unclear precipitation" hơn là "heavy rain".
- `night_snow` = 0% cho Qwen2.5 và Qwen3 — compound condition (night + snow chain) là hardest case.
- Qwen2.5 **unexpectedly 96% cho night** — contradict known "Qwen strict" pattern. Phải investigate (có thể prompt night không trigger snow-bias).

### 10.5 Kết quả Track B — Final 5-judge panel (2026-04-23 22:18)

**Per-judge accept rate trên DINO-prefiltered samples** (local judges DINO≥0.75, Claude DINO≥0.80; n=50/condition cho synthetic, n=40 cho ACDC baseline):

| Condition | InternVL | Phi-4 | Qwen2.5 | Qwen3 | **Claude** | Panel avg |
|---|---|---|---|---|---|---|
| **ACDC baselines (real weather ceiling)** | | | | | | |
| acdc_fog | — | — | — | 100% | **100%** | 100% |
| acdc_night | — | — | — | 100% | **100%** | 100% |
| acdc_rain | — | — | — | 82.5% | **100%** | 91.2% |
| acdc_snow | — | — | — | 35% | **90%** | 62.5% |
| **Synthetic (main pipelines)** | | | | | | |
| fog_heavy | — | — | — | 92% | **100%** | 96.0% |
| fog_medium | 100% | 100% | 70% | 80% | **98%** | 89.6% |
| fog_light | 100% | 98% | 70% | 84% | **98%** | 90.0% |
| ip2p_rain | — | — | — | 68% | **98%** | 83.0% |
| ip2p_rain_heavy | 100% | 66% | 28% | 46% | **90%** | 66.0% |
| ip2p_snow_heavy | — | — | — | 16% | **90%** | 53.0% |
| ip2p_snow_light | 78% | 34% | 4% | 6% | **88%** | 42.0% |
| night | 22% | 56% | 88% | 40% | **100%** | 61.2% |
| night_rain | 100% | 54% | 54% | 44% | **72%** | 64.8% |
| **night_snow** | 76% | 24% | 4% | 0% | **6%** | **22.0%** |
| st_rain_a | — | — | — | 46% | **76%** | 61.0% |
| st_rain_b | — | — | — | 26% | **68%** | 47.0% |
| st_rain_c | — | — | — | 40% | **58%** | 49.0% |
| st_snow_b | — | — | — | 0% | **48%** | 24.0% |

**Majority vote trên subset 7 conditions có đủ 4 local judges (n=50):**

| Condition | ≥2/3 (I+P+Q2.5) | ≥2/4 (+ Q3) | ≥3/4 (stricter) |
|---|---|---|---|
| fog_light | 98% | 98% | 92% |
| fog_medium | 100% | 100% | 96% |
| ip2p_rain_heavy | **76%** | 88% | **44%** |
| ip2p_snow_light | 32% | 32% | 8% |
| night | 56% | 66% | 36% |
| night_rain | 78% | 88% | 54% |
| night_snow | 24% | 24% | 0% |

**Finding quan trọng — Claude Sonnet lenient hơn tất cả local VLMs:**

| Condition | Claude | Best local | Gap | Diễn giải |
|---|---|---|---|---|
| ip2p_snow_light | 88% | Qwen3 6% | **+82pp** | Claude nhìn light snow là valid; local VLMs đòi heavy coverage |
| night | 100% | Qwen2.5 88% | +12pp | CycleGAN-Turbo night validated universally |
| ip2p_rain_heavy | 90% | InternVL 100% | −10pp | Physics-only rain heavy validated by both |
| ip2p_snow_heavy | 90% | Qwen3 16% | **+74pp** | Parameter g=12 hiệu quả, local VLMs vẫn strict |
| st_snow_b | 48% | Qwen3 0% | +48pp | Style snow_b inconsistent — partial agreement |
| night_snow | 6% | InternVL 76% | **−70pp** | **Outlier** — InternVL over-accept, Claude chính xác hard case |
| acdc_snow | 90% | Qwen3 35% | +55pp | Ngay cả real ACDC snow cũng bị local VLMs reject → bias mạnh |

**Diễn giải tổng quát:**
1. **Claude là judge đáng tin nhất** — 0 parse errors, ACDC baselines gần ceiling (100/100/100/90 fog/night/rain/snow), explanations chi tiết nhất. Khi bất đồng với local VLMs, thường Claude đúng.
2. **`acdc_snow` = 35% cho Qwen3** chứng minh local VLMs có bias strict-on-snow với **real** data, không chỉ synthetic. Claude 90% trên real ACDC snow → benchmark ceiling đáng tin hơn.
3. **Local VLMs systematically underestimate** synthetic quality → majority vote 3-local gives biased-low estimate. Panel 5-judge (hoặc Claude-only) thực tế phù hợp hơn cho reporting.
4. **Hardest case cross-confirmed:** `night_snow` fails cho Claude (6%) + Qwen3 (0%) → **compound condition không rescue được**. Cần chấp nhận limitation trong paper.

### 10.6 Track A vs Track B comparison

| Condition | Track A majority (I+P+Q2.5) | Track B majority (I+P+Q2.5) | Track B Claude | Δ A→B |
|---|---|---|---|---|
| night | 58% | 56% | 100% | −2pp (local) / +42pp (Claude) |
| ip2p_snow_light | 8% | 32% | 88% | +24pp (local) / +80pp (Claude) |
| ip2p_rain_heavy | (N/A) | 76% | 90% | N/A |
| fog_heavy | 98% | (N/A) | 100% | +2pp |

**Finding:** DINO pre-filter **boost acceptance** ở marginal cases (snow_light +24pp local); Claude tolerance **reveal hidden quality** (snow_light 88% thay vì 8–32%). DINO filter giúp "marginal cases" nhưng **không cứu được "fundamental condition failures"** như `night_snow`.

### 10.7 Phát hiện key: DINO threshold KHÔNG predict VLM jury

**Test đặc biệt (2026-04-18):** [`test_dino_threshold_vs_jury.py`](../validation/vlm_jury/test_dino_threshold_vs_jury.py)
- Lấy snow_strong, chọn **10 sample ABOVE** DINO≥0.75 + **10 sample BELOW** DINO<0.75 → send 3 judges.
- Results: [`dino_threshold_jury_{internvl,phi4,qwen}.json`](../validation/results/vlm_jury/) (20 records/judge).

**Kết quả (agreement DINO group vs Jury decision):**

| Judge | ABOVE accept % | BELOW accept % | Pattern |
|---|---|---|---|
| InternVL | ~90% | ~80% | small gap — DINO barely matters |
| Phi-4 | ~80% | ~70% | small gap |
| Qwen2.5 | ~0% | ~0% | Qwen reject everything — DINO group irrelevant |

**Conclusion:** overall agreement **~50–52% (random)**. DINO ≥ 0.75 **không** predict jury accept. Two metrics capture complementary quality aspects:
- **DINO** = structural preservation at feature level (geometry, objects intact).
- **VLM Jury** = perceptual realism (weather effect believable).
- Ảnh có thể có DINO=0.95 (rất giống gốc) nhưng jury reject vì "weather effect không đủ mạnh"; ngược lại DINO=0.50 (structure shifted) nhưng jury accept vì "looks convincingly like rain".

**Hệ quả quan trọng cho paper:**
1. **Không dùng DINO làn filter sole criteria** — phải release cả DINO + jury per-image.
2. Filter recommendation hiện tại (README `docs/methods.md` §1.7): release `dino_sim`, `ssim`, `decision` per image → user chọn.
3. Paper phải frame rõ: "DINO preserves structure ≠ jury accepts realism; use both."

### 10.8 Track C — Full snow_strong sweep (9,012 inferences)

**Protocol:** 3,004 snow_heavy images × 3 judges, không DINO pre-filter → full sweep để có distribution đầy đủ.

**Per-judge acceptance (preliminary):**
- InternVL: ~75–80% (lenient baseline)
- Phi-4: ~60–70%
- Qwen2.5: ~15–25% (known strict-on-snow bias)

**Purpose:** dense validation cho decision "chọn IP2P diffusion g=12 cho snow_heavy" trong `methods.md` §1.2. Kết quả InternVL 89.3%, Phi-4 92.3%, Qwen 8.9% (sampled subset) được verify trên full 3,004 ảnh → kết luận mạnh hơn.

**Results:** [`snow_strong_full_{internvl,phi4,qwen}.json`](../validation/results/vlm_jury/) (per-image `decision` + `explanation` cho all 3,004 ảnh).

### 10.9 Inter-judge agreement & bias (updated với Claude)

**Cohen's κ (Track A, pairwise):** 0.16–0.36 (fair–moderate) across 3 local judges.

**Single-judge strictness (overall accept rate across all conditions):**

| Judge | Overall | Bias pattern | Khi convergent với Claude |
|---|---|---|---|
| Claude Sonnet 4.6 | **~82%** (18 conditions, DINO≥0.80) | Most lenient; ACDC ceiling 100/100/100/90 | **Reference standard** |
| InternVL | ~78% | Lenient — high accept across the board | Thường đồng với Claude cho positives |
| Phi-4 | ~57% | Balanced middle | Đồng với Claude ~60% |
| Qwen2.5 | ~57%, **2–4% on snow** | Strict on snow — giống Gemini bias (Ruck et al. 2026) | Bất đồng snow mạnh |
| Qwen3 | ~44% (18 conditions) | Stricter than Qwen2.5 overall, even reject **real ACDC snow** (35%) | Bất đồng strongest |

**Finding quan trọng về bias:** Qwen3 reject 65% real ACDC snow images → **local-VLM bias không chỉ là "strict on synthetic snow" mà là "strict on snow period"**. Điều này validate lại cần Claude Sonnet để establish reliable ceiling.

**Bias mitigation options:**
- **Majority ≥2/3 (3 locals)** — biased-low for snow conditions.
- **Majority ≥3/5 (5 judges)** — khả thi cho 7 conditions có đủ data; balanced.
- **Claude-only reporting** — cleanest, defensible (Anthropic-grade model, 0 parse errors), sử dụng được với caveat "strongest single judge".
- **Report range** — lean toward [min, max] của panel thay vì single number.

### 10.10 Sample size & cost accounting (final)

**Tổng inferences tính đến 2026-04-23 22:18:**

| Track | Inferences | Resources |
|---|---|---|
| Track A (3 local × 560 → Qwen2.5 extended → 910) | 2,030 | ~6 GPU-hours |
| Track B (3 local × 350 + Qwen3 × 860 + **Claude × 860**) | 2,770 | ~6 GPU-hours + Claude API |
| Track C (3 local × 3,004 snow_strong) | 9,012 | ~30 GPU-hours |
| DINO-threshold probe (3 local × 20) | 60 | <1 GPU-hour |
| Snow strong test probe (3 local × 20) | 60 | <1 GPU-hour |
| **TOTAL** | **~13,932** | **~43 GPU-hours + Claude API** |

Đây là **validation approach đắt nhất** trong ConSynth-X nhưng duy nhất trả lời perceptual realism không bị cross-domain. Claude API giúp establish trustworthy ceiling mà local VLMs không làm được.

### 10.11 TODO còn lại cho VLM Jury

| Priority | Item | Status |
|---|---|---|
| ~~HIGH~~ | ~~Chờ Claude Sonnet 4.6 job xong → rerun table~~ | ✅ **DONE 2026-04-23 22:18** |
| HIGH | Rerun `make_vlm_jury_table.py` với 5-judge panel + Claude | Pending |
| HIGH | Decide paper reporting strategy: (a) majority ≥3/5, (b) Claude-primary + local-secondary, (c) range-reporting | Open decision |
| MEDIUM | Compute Cohen's κ **per condition** — identify conditions có highest disagreement | Pending |
| MEDIUM | Claude chạy thêm cho remaining Track A conditions để full overlap với local panel | Pending |
| MEDIUM | Investigate `night_snow` universal failure (Claude 6%, Qwen3 0%) — is compound condition pipeline fixable? | Open research question |
| LOW | Qualitative audit: sample 10 Claude explanations/condition để audit reasoning | Pending |

### 10.12 Output artifacts (final)

**Files — code:** [`validation/vlm_jury/`](../validation/vlm_jury/) (8 modules), [`validation/make_vlm_jury_table.py`](../validation/make_vlm_jury_table.py) (auto-regenerate tables sau mỗi job).

**Files — jobs:** [`jobs/`](../jobs/) chứa **23 VLM jury SLURM scripts**:
- Initial Track A: `vlm_jury_{qwen,phi4,internvl}.sh` + `vlm_jury_*_missing.sh` (resume)
- Track B DINO075: `vlm_jury_dino075_{qwen,phi4,internvl,qwen3}.sh` + `_rest_*.sh` + `_snowlight_*.sh`
- Track C full: `vlm_jury_snow_full.sh`
- **Claude API: `vlm_jury_claude_sonnet.sh`** (done 2026-04-23 22:18)
- Rain heavy revalidation: `rerun_vlm_jury_rain_heavy.sh`
- Snow strong test: `test_snow_strong_jury.sh`
- DINO threshold probe: `dino_threshold_jury.sh`

**Results:** [`validation/results/vlm_jury/`](../validation/results/vlm_jury/) — 22 JSON files
- Track A main: `{internvl2.5-8b,phi-4-multimodal,qwen2.5-vl-7b}_results.json` + `.backup_20260423.json`
- Track B: `*_dino0.75_results.json` (4 local judges) + **`claude-sonnet-4.6_dino0.8_results.json`** (794KB, 860 rows, 18 conditions, 0 parse errors)
- Track C: `snow_strong_full_{internvl,phi4,qwen}.json` (≈1MB mỗi file)
- Probes: `dino_threshold_jury_*.json`, `snow_strong_test_*.json`
- Summary: `vlm_jury_summary.json` (legacy — **needs regeneration với 5-judge panel**), `vlm_jury_table.{pdf,png,txt}`

**Logs:** [`logs/vlm_jury/`](../logs/vlm_jury/) — SLURM `.out`/`.err` files organized per-judge per-run, bao gồm `jury_claude_login.out` (8.7KB, 0 errors).

---

## 11. Approach A7 — Per-image DINO/SSIM Retention

**Ý tưởng:** Compute DINOv3 cosine + SSIM per ảnh → release per-image CSV để **user chọn filter theo application**, không enforce single threshold.

**Reference:** Extension of Ruck et al. (2026) embedding methodology, adapted for dataset quality reporting.

**Pipeline:**
1. DINOv3 ViT-L/16 CLS embedding cho (original, augmented) pair.
2. Cosine similarity → `dino_sim ∈ [−1, 1]`.
3. SSIM on 256×256 resized → `ssim ∈ [−1, 1]`.
4. Per-image CSV release.

**Kết quả (2026-04-19, 13 conditions):**

| Condition | N | DINO mean | SSIM mean | Retention @ DINO≥0.75 |
|---|---|---|---|---|
| ST Rain A | 2,211 | 0.919 | 0.821 | **98%** |
| ST Rain B/C | 1,243–1,208 | 0.886–0.898 | 0.796–0.809 | 93–95% |
| ST Snow A | 2,159 | 0.898 | 0.775 | 96% |
| ST Snow B/C | 1,649–1,796 | 0.859–0.867 | 0.687–0.703 | 91–92% |
| IP2P Rain | 3,004 | 0.800 | 0.591 | 74% |
| IP2P Snow (light) | 3,004 | 0.774 | 0.590 | 67% |
| IP2P Snow (heavy) | 3,004 | 0.790 | 0.598 | 73% |
| Fog light | 1,002 | **0.938** | **0.788** | **99%** |
| Fog medium | 1,001 | 0.918 | 0.719 | 99% |
| Fog heavy | 1,001 | 0.868 | 0.617 | 93% |
| Night (CycleGAN) | 3,004 | 0.841 | 0.390 | 88% |

**Rain light vs rain heavy (post-hoc 2026-04-22, paired 1:1):**

| Variant | DINO mean | SSIM mean | Pass DINO≥0.75 | Pass DINO≥0.70 |
|---|---|---|---|---|
| `rain` (light, IP2P g=10) | 0.871 | 0.756 | 91.7% | 95.3% |
| `rain_heavy` (physics-only) | 0.754 | 0.522 | 39.8% | 73.8% |

Heavy có DINO/SSIM thấp đúng kỳ vọng (stronger overlay + blur → lower similarity) nhưng không quá thấp. Kaggle v6 sample dùng threshold **không đồng đều**: light DINO≥0.75, heavy DINO≥0.70.

**Key findings:**
1. **Edit magnitude hierarchy:** Fog light < Style transfer < Fog heavy ≈ Night < IP2P diffusion.
2. **DINO robust to texture changes:** Night SSIM 0.39 nhưng DINO 0.84 — global tone change preserves structure.
3. **SSIM degrades faster than DINO** cho IP2P/night — consistent với DINO's invariance to pixel-level changes.
4. **DINO vs SSIM correlation moderate** (r=0.40 for snow_heavy) — complementary, not substitutable.
5. **DINO ≥ 0.75 threshold không correlate với VLM Jury** (~50% agreement, random).

**Filter recommendation (paper stance):**
- Không enforce single threshold (no empirical support).
- Release per-image DINO + SSIM → user selects based on task:
  - Downstream training: permissive (keep more data).
  - Evaluation benchmark: strict (ensure quality).
  - Human-visible validation: VLM Jury (most correlated with perception).

**Files:** [`validation/extract_dino_ssim_all.py`](../validation/extract_dino_ssim_all.py), [`validation/extract_dino_ssim_snow_strong.py`](../validation/extract_dino_ssim_snow_strong.py), [`validation/make_retention_chart.py`](../validation/make_retention_chart.py), [`validation/make_retention_charts_all.py`](../validation/make_retention_charts_all.py), [`validation/make_dino_threshold_grid.py`](../validation/make_dino_threshold_grid.py), [`validation/make_dino_threshold_grid_rain.py`](../validation/make_dino_threshold_grid_rain.py).
**Results:** [`validation/results/dino_ssim/*.csv`](../validation/results/dino_ssim/) (13 conditions).

**Paper figures:**
- Fig 1 (page 5): `fig4_snow_heavy_retention.pdf` — 3-panel (DINO curve, SSIM curve, joint heatmap).
- Fig 2 (page 6): `fig5_retention_curves_all.pdf` — 13-condition comparison.

---

## 11b. Approach A8 — End-to-end QC funnel (paper-grade accounting)

**Mục tiêu.** Cho reviewer Nature Sci Data thấy đầy đủ: với mỗi (source × condition), bao nhiêu ảnh đi vào pipeline → bao nhiêu sống qua SSIM+LPIPS filter → bao nhiêu đạt DINO≥0.75 audit. Đây là tổng hợp paper-grade gộp từ A1+A7, kèm figure funnel để dùng trực tiếp trong methods section.

**Quan trọng — scope của filter.** SSIM+LPIPS chỉ áp cho rain (light & heavy paired). Snow / fog / night / night_weather KHÔNG filter ở gen time → `N_kept < N_input` cho các condition này là do **partial generation** (subset run), không phải filter rejection. Báo cáo tách 2 cột:
- `filter_retention_pct` — chỉ rain, nghĩa thực sự "% qua filter".
- `coverage_pct` — mọi condition khác, "% source được generate".

**Pipeline.**
```
N_input (source dataset)
    │
    ▼
IP2P diffusion + physics overlay
    │
    ▼  (rain only)
SSIM ∈ [0.60, 0.95] AND LPIPS < 0.35
    │
    ▼
N_kept (rows in output Arrow / meta.csv KEEP)
    │
    ▼  (post-hoc audit, doesn't filter released set)
DINOv3 cosine ≥ 0.75
    │
    ▼
N×P(DINO≥0.75)
```

**Filter retention (rain only) — 2026-04-29:**

| Source | rain_light/heavy filter retention |
|---|---:|
| cs_train | 51.75% |
| cs_test  | 54.99% |
| soda_voc | **24.14%** |
| soda_ktsh| **21.15%** |

→ Threshold calibrated trên CS construction; SODA street scenes drop rate cao hơn ~3× → **domain-dependent**. Sensitivity analysis recommended trong paper.

**DINO≥0.75 audit (% of kept):**

| Condition family | Best | Worst |
|---|---|---|
| snow_light × 4 sources | 99.4% (cs_train) | 98.3% (soda_voc) |
| rain_light × 4 sources | 91.7% (cs_test) | 84.7% (soda_voc) |
| rain_heavy × 4 sources | 63.2% (soda_voc) | **46.8% (soda_ktsh)** |
| snow_heavy × 4 sources | 85.9% (soda_voc) | 72.5% (cs_train) |
| fog × 3 (CS test) | 99.3% (medium) | 93.3% (heavy) |
| night (CS test) | — | 87.6% |
| **night_rain (CS test, Order-B)** | — | **10.8%** ← worst-case |
| night_snow (CS test, Order-B) | — | 54.3% |

**Key paper claims:**
1. Rain filter cắt SODA gắt hơn CS ~3× → domain shift visible at filter stage.
2. rain_heavy DINO consistency < rain_light đáng kể (e.g. soda_ktsh 88.1% → 46.8%) → physics overlay heavy đẩy ảnh xa thêm sau IP2P.
3. snow_light là pipeline robust nhất (98-99% DINO≥0.75 across all sources).
4. **Order-B night_rain là worst-case của toàn dataset** (DINO mean 0.447, 10.8% audit pass) — phải explicitly flag trong paper, không hide.

**Files.**
- [`validation/build_qc_funnel.py`](../validation/build_qc_funnel.py) — counter (Arrow rows + meta.csv status + DINO retention).
- [`validation/extract_dino_ssim_extended.py`](../validation/extract_dino_ssim_extended.py) — streaming DINO/SSIM cho CS train + SODA-VOC + SODA-KTSH (12 conditions chưa có trong A7).
- [`validation/plot_qc_funnel.py`](../validation/plot_qc_funnel.py) — 3 figures (exemplar funnel, overview bars, retention curves 4-panel).
- [`validation/refresh_qc_funnel.sh`](../validation/refresh_qc_funnel.sh) — wrapper rebuild sau khi có CSV mới.
- Job templates: [`validation/jobs/extract_dino_ssim_extended_pitzer.sh`](../validation/jobs/extract_dino_ssim_extended_pitzer.sh), `..._cardinal.sh`.

**Results.**
- [`validation/results/qc_funnel/funnel_counts.csv`](../validation/results/qc_funnel/funnel_counts.csv) — 20 rows × (n_input, n_kept, filter_retention_pct, coverage_pct, dino_mean, dino_p25/50/75, dino_ge_{50,60,70,75,80,85,90}_pct, n_after_dino_ge_75).
- [`validation/results/qc_funnel/funnel_report.md`](../validation/results/qc_funnel/funnel_report.md).
- [`validation/results/qc_funnel/figures/`](../validation/results/qc_funnel/figures/): `funnel_exemplar_cs_test_rain_light.pdf`, `funnel_overview_yields.pdf`, `funnel_dino_distributions.pdf`.

**Anomaly cần điều tra.** CS train snow_heavy = 6,009/7,009 (85.7%) — `methods.md` nói `--no-filter` nhưng disk có drop. Likely partial regeneration. Verify trước submit.

---

## 12. Tier 3 — Human Perceptual Study

Flask app tại [`human_validation/app.py`](../human_validation/app.py). Chạy: `cd human_validation && python app.py` → `http://localhost:5000`. Admin login `admin` → Dashboard → Load Images → Export CSV.

**Database schema (SQLite, WAL mode):** `users`, `images`, `turing_responses`, `realism_responses`, `recognition_responses`.

**Conditions covered (11):** original, weather_style_rain_{0,1,2}, weather_style_snow_{0,1,2}, night, night_rain, night_snow, small.

### Task T1 — Turing Test (fooling rate)
- Annotator thấy 1 ảnh (real hoặc synthetic), phân biệt: Real / Synthetic.
- Trộn real ACDC + synthetic ConSynth-X cho mỗi condition.
- Metric: **fooling rate** = % synthetic images bị đánh label "real".
- Reaction time (`response_ms`) logged để detect rushed responses.

### Task T2 — Realism Rating (MOS)
- **ITU-R BT.500** rating scale **1–5**:
  - 1 = Bad (obviously fake)
  - 2 = Poor
  - 3 = Fair
  - 4 = Good
  - 5 = Excellent (realistic)
- Per condition → Mean Opinion Score (MOS).
- Báo cáo: MOS ± 95% CI per condition.

### Task T3 — Condition Recognition (confusion matrix)
- Annotator thấy ảnh, chọn từ 8 options: Clear / Rain / Snow / Night / Night+Rain / Night+Snow / Fog / Small.
- Metric: **recognition accuracy** + confusion matrix per ground-truth condition.
- Kết quả mong đợi validate A3 (weather classifier) — nếu human cũng nhận ra đúng → augmentation recognizable ở perceptual level.

**Sample preparation:** [`human_validation/sample_images.py`](../human_validation/sample_images.py) — random sample N ảnh/condition từ `augmentation_data/` vào `human_validation/static/`.

**Status (2026-04):** Platform đã deploy, recruiting annotators.

---

## 13. Tổng hợp — Convergent Evidence

**Bảng so sánh 7 approaches:**

| Approach | Level | Discrimination | Conditions Validated | Recommended Use |
|---|---|---|---|---|
| A1 UnivFD | Fake detection | ✗ không phân biệt | none | supplementary only |
| A2 FID/KID × 3 datasets | Distribution | ✓ night, snow, fog | 3/4 (rain inconclusive) | co-primary |
| A3 Weather Classifier | Semantic | ✓ phân biệt rõ | rain, snow (not night) | co-primary |
| A4 Texture Fidelity | Micro-level | ✓ diffusion vs style | all | complementary |
| A4b Belief Fusion | Micro + uncertainty | ✓ + conflict detection | all | complementary |
| A5 Relative Mahalanobis | Per-image embedding | ✓ CLIP (night 38%) | 4/4 (small shifts) | supporting |
| **A6 VLM Jury (4–5 judges, 3 tracks)** | Perceptual, zero-shot | ✓ phân biệt rõ, per-image explanation | 13+ conditions | **🎯 PRIMARY** |
| A7 DINO/SSIM Retention | Per-image similarity | ✓ edit magnitude | N/A (descriptive) | filter + figures |

**Convergent validation per condition:**

| Condition | A2 FID | A3 Classifier | A4 Texture | A5 Mahalanobis (CLIP) | A6 VLM Claude | A6 VLM Majority | Verdict tổng hợp |
|---|---|---|---|---|---|---|---|
| **Night (CycleGAN)** | ✅ −34% | — (no class) | ⚠ over-smooth | ✅ 38% gap | ✅ 100% | ⚠ 56–66% (local) | ✅ **distribution + Claude strong**, local strict |
| **Rain (IP2P light)** | ⚠ ≈ baseline | ✓ 63.5% | ✅ faithful | ✅ 25% | ✅ 98% | ✅ 72% | ✅ validated by 4/5 |
| **Rain (IP2P heavy — physics)** | — | — | — | — | ✅ 90% | 76% ≥2/3, 44% ≥3/4 | ✅ Claude confirms; local split |
| **Rain (ST)** | ⚠ ≈ baseline | ✅ 70–89% | moderate | ✅ 25% | ⚠ 58–76% | ⚠ 46–70% | ⚠ ST quality variable |
| **Snow (IP2P light)** | ✅ −5% | ✗ 25.7% | ✅ faithful | ⚠ | ✅ 88% | ✗ 8–32% | ⚠ **Claude+texture accept, local+classifier reject** — disagreement |
| **Snow (IP2P heavy)** | — | (not rerun) | — | — | ✅ 90% | ⚠ 16% (Qwen3) | ✅ **Claude rescues**; local snow-bias persists |
| **Snow (ST)** | ✅ −5 to −9% | ✅ 62–95% | mixed (snow_1 outlier) | ✅ 27% | ⚠ 48% | ✗ 0–10% | ⚠ heterogeneous |
| **Fog heavy** | ✅ −6 to −10% | N/A | expected HF loss | ✅ 18% | ✅ **100%** | ✅ 98% | ✅ **strongest** (exceeds ACDC) |
| **Night_rain** | — | — | — | — | ✅ 72% | ✅ 78–88% | ✅ Compound OK |
| **Night_snow** | — | — | — | — | ✗ **6%** | ✗ **0–24%** | ✗ **Universal failure** — paper limitation |

**Narrative for paper (Technical Validation section) — updated 2026-04-23:**
1. **Primary metric: VLM Jury (A6)** — 4 judges local (InternVL, Phi-4, Qwen2.5, Qwen3) + 1 API (Claude Sonnet 4.6, queued), 3 tracks (paired random, DINO-prefiltered, full snow_strong sweep), ~13,000 inferences. Zero-shot, domain-agnostic, per-image explanations auditable. Key finding: DINO ≥ 0.75 không predict jury → phải release cả hai.
2. **Co-primary (distribution):**
   - FID/KID multi-reference (A2) — night (−34%), snow (3/3 datasets), fog (2/3).
   - Weather Classifier (A3) — rain/snow recognizability ở semantic level.
3. **Complementary:** Texture fidelity + belief fusion (A4+A4b) — validates micro-level artifact absence. Per-image Mahalanobis (A5) — CLIP night 38% gap closed.
4. **Supplementary:** UnivFD (A1) — explain why fake-detection doesn't work on edit-based augmentation.
5. **Per-image quality release:** DINO + SSIM (A7) + jury decision+explanation — users filter by task.
6. **Cross-domain limitation:** all reference datasets are driving/outdoor, not construction — frame as motivation for ConSynth-X rather than failure. **VLM Jury mitigates this** by not requiring reference distribution.

---

## 14. Asymmetries — điều phải giải thích rõ trong paper

### 14.1 Rain heavy = physics-only, Snow heavy = diffusion
Xem chi tiết `methods.md` §1.2 IP2P Rain — 2 Intensity Variants. Tóm tắt:
- Snow heavy = g=12 diffusion (cần thay đổi surface coverage — snow accumulation).
- Rain heavy = physics overlay tăng cường (density + blur σ=1.3) trên output của rain light.
- Lý do: đã test 5 pilot iterations diffusion-based heavy rain (v1–v5), đều hallucinate scene. Physics-only deterministic, reproducible, CPU-only.

### 14.2 LPIPS filter chỉ apply cho rain, không snow
- IP2P snow rất insensitive với SSIM threshold (99% → 90% từ 0.30 → 0.60).
- IP2P rain sensitive → LPIPS < 0.35 thêm filter layer.
- **Chưa justify tại sao threshold 0.35** (literature recommend 0.1–0.15 cho imperceptible). **TODO** trong `checklist.md`.

### 14.3 Sensitivity analysis áp dụng cho IP2P (main), không cho Style Transfer (ablation)
- Scope decision 2026-04-21: NST di dời sang ablation archive (xem `CLAUDE.md`).
- Mâu thuẫn parameter trong NST (style_weight 10k vs 100k, steps 10 vs 50) **không block submission**.
- Kết quả sensitivity sweep áp dụng cho IP2P production; số filter cho ST giữ cho supplementary.

### 14.4 Cross-domain bias của weather classifier
- Original clear images chỉ 6.1% accuracy cho "sun/clear".
- Nguyên nhân: classifier train trên driving, nhầm construction (đất, bụi) thành overcast/rain.
- Phải note trong paper: accuracy của augmented conditions đọc là **relative improvement over 6.1% baseline**, không phải absolute recognition rate.

### 14.5 DINO vs SSIM vs VLM Jury disagreement
- DINO ≥ 0.75 và VLM Jury agreement ~50% (random).
- DINO đo structural preservation ở feature level; VLM đánh giá visual realism; SSIM đo pixel.
- Không dùng single threshold — release all 3 per-image.

---

## 15. Data Integrity — kiểm chứng lại sau rain_heavy revalidation

Sau khi thêm rain_heavy variant (2026-04-21), đã re-run tất cả 5 validation approaches cho rain_heavy:

| Job | Script | Backup preserved? |
|---|---|---|
| FID/KID rerun | [`jobs/rerun_fid_kid_rain_heavy.sh`](../jobs/rerun_fid_kid_rain_heavy.sh) | ✓ `fid_kid_backup_20260421_2329/` |
| Relative Mahalanobis rerun | [`jobs/rerun_mahalanobis_rain_heavy.sh`](../jobs/rerun_mahalanobis_rain_heavy.sh) | ✓ `relative_mahalanobis_backup_20260421_2329/` |
| Texture Fidelity rerun | [`jobs/rerun_texture_fidelity_rain_heavy.sh`](../jobs/rerun_texture_fidelity_rain_heavy.sh) | ✓ `texture_fidelity_backup_20260421_2329/` |
| VLM Jury rerun | [`jobs/rerun_vlm_jury_rain_heavy.sh`](../jobs/rerun_vlm_jury_rain_heavy.sh) | ✓ `vlm_jury_backup_20260421_2329/` |
| Weather Classifier rerun | [`jobs/rerun_weather_cls_rain_heavy.sh`](../jobs/rerun_weather_cls_rain_heavy.sh) | ✓ `weather_cls_backup_20260421_2329/` |
| DINO/SSIM rerun | [`jobs/extract_dino_ssim_rain_heavy.sh`](../jobs/extract_dino_ssim_rain_heavy.sh) | ✓ `dino_ssim_backup_20260421_2329/` |

Runbook chi tiết: [`docs/rain_heavy_revalidation_runbook.md`](rain_heavy_revalidation_runbook.md).

---

## 16. File & Script Map — quick reference

### Automated metric scripts (`validation/`)
| Script | Approach | Output |
|---|---|---|
| `realism_detector.py`, `run_realism_validation.py` | A1 UnivFD | `results/full_univfd/` |
| `compute_fid_kid.py` | A2 FID/KID | `results/fid_kid/` |
| `weather_classifier.py` | A3 Classifier | `results/weather_cls/` |
| `compute_texture_fidelity.py` | A4 Texture | `results/texture_fidelity/` |
| `belief_fusion.py` | A4b Belief | `results/texture_fidelity/belief_*.json` |
| `compute_relative_mahalanobis.py`, `analyze_mahalanobis_vs_original.py` | A5 Mahalanobis | `results/relative_mahalanobis/` |
| `vlm_jury/*.py` (6 modules) | A6 VLM Jury | `results/vlm_jury/` |
| `extract_dino_ssim_all.py`, `make_retention_chart*.py` | A7 Retention | `results/dino_ssim/` |

### Reference data acquisition
| Script | Dataset | Destination |
|---|---|---|
| `download_acdc.py` | ACDC | `reference_data/acdc/` |
| `download_reference.py` | WeatherBench + WeatherNet | `reference_data/weatherbench/`, `reference_data/weathernet/` |
| `collect_construction_weather_flickr.py` | Flickr (construction + weather) | `reference_data/construction_weather/` |
| `collect_construction_weather_openverse.py` | Openverse | `reference_data/construction_weather/` |
| `filter_construction_weather_clip.py` | CLIP filter | `reference_data/construction_weather/filtered/` |

### Human study (`human_validation/`)
| File | Role |
|---|---|
| `app.py` | Flask app (T1/T2/T3 + admin dashboard) |
| `sample_images.py` | Sample N images/condition → `static/` |
| `templates/` | HTML UI |
| `validation.db` | SQLite (WAL), 5 tables |
| `static/` | Served images |

### Job scripts (`jobs/` — SLURM)
- 7 extract DINO/SSIM scripts (per condition).
- 22 VLM jury scripts (judge × condition).
- 5 rerun scripts for rain_heavy revalidation.
- 1 texture fidelity, 1 relative mahalanobis, 1 FID/KID.

### Results (`validation/results/`)
- 24 subdirectories, 6 top-level PDF/PNG figures (retention charts, jury table).
- Backups preserved at `*_backup_20260421_2329/` before rain_heavy rerun.

---

## 17. TODO — hiện chưa fix trong validation

Từ `docs/checklist.md`:

| Priority | Item | Ảnh hưởng |
|---|---|---|
| CRITICAL | Document tại sao LPIPS chỉ apply cho rain, không snow | Methodology gap |
| CRITICAL | Snow không có LPIPS scores (implementation gap) | Asymmetry chưa justify |
| CRITICAL | Sensitivity mAP per threshold — **đang chạy** (6 SLURM jobs) | Chưa chọn được threshold data-driven |
| MEDIUM | Random seeds unify (42/77/99 trong các scripts khác nhau) | Reproducibility claim |
| MEDIUM | Cite Tremblay et al. (2021) cho physics overlay baseline | Lit support |
| MEDIUM | Link meteorological rainfall-rate → particle density formula | Physics justification |
| MEDIUM | Data leakage verification script (train/val/test separation) | Integrity |

---

## 18. Tóm tắt cho reviewer

**ConSynth-X validation là gì?** Multi-tier: **VLM Jury là primary metric** (4 local + 1 API judge đã xong, 3 tracks, **~13,932 inferences**), cộng 6 automated approaches bổ sung + 3 human tasks, cross-validated trên 3 real weather reference datasets (ACDC, WeatherBench, WeatherNet).

**Kết quả cuối (2026-04-23 22:18) — Claude Sonnet 4.6 ceiling per condition:**

| Category | Best | Conditions ≥ 90% | Conditions fail |
|---|---|---|---|
| ACDC baselines (real) | 100/100/100/90 (fog/night/rain/snow) | 3/4 | — (snow 90% ceiling) |
| Synthetic fog | 100% | fog_{light,medium,heavy} | — |
| Synthetic rain (IP2P) | 98% | ip2p_rain, ip2p_rain_heavy (90%) | — |
| Synthetic snow (IP2P) | 90% | ip2p_snow_heavy, ip2p_snow_light (88%) | — |
| Night (CycleGAN) | 100% | night | — |
| Compound (night+weather) | 72% | night_rain | **night_snow (6%)** |
| Style transfer | 76% | — | st_snow_b (48%), others 58–76% |

**Điều gì thuyết phục nhất?**
1. **VLM Jury per-image explanations** — không chỉ score mà còn reasoning, auditable bởi reviewer.
2. **4–5 judge panel** (InternVL + Phi-4 + Qwen2.5 + Qwen3 + Claude Sonnet 4.6) → majority vote mitigates single-judge bias (đặc biệt Qwen strict-on-snow pattern giống Gemini trong Ruck et al. 2026).
3. **3 tracks song song** — random paired, DINO-prefiltered, full snow_strong sweep — cross-check kết luận không depend on sampling choice.
4. **Key finding inspectable:** DINO ≥ 0.75 và jury agreement chỉ ~50% (random) → phải release cả hai.
5. Convergent evidence across 5+ approaches (VLM jury, FID/KID, classifier, Mahalanobis, texture).
6. Cross-reference FID/KID (tránh bias 1 dataset).
7. Per-image release (DINO + SSIM + jury decision + explanation).
8. Asymmetries documented công khai (rain heavy physics vs snow heavy diffusion; IP2P snow light failure → heavy rescue via jury evidence).

**Điều gì là limitation honest?**
1. Không có construction-site real weather reference → motivation paper; **VLM Jury là approach duy nhất không bị ảnh hưởng**.
2. IP2P snow light thất bại ở 25.7% classifier accuracy + 8% jury majority — rescued bằng heavy variant (80% jury).
3. Weather classifier có cross-domain bias (clear original 6.1%) — note explicitly.
4. DINOv3 Mahalanobis gap closed <10% do domain gap driving vs construction.
5. LPIPS filter asymmetry rain vs snow — chưa justify đầy đủ.
6. VLM Jury inter-judge κ=0.16–0.36 (moderate disagreement) → mitigated bằng majority vote.
7. `night_snow` compound condition là hardest case (jury 0–24%) — chưa có solution, có thể phải accept limitation trong paper.

**Data integrity:** Không fabricate, không cherry-pick threshold, report as-is cả khi bất ngờ (IP2P snow 25.7%; DINO vs jury ~50% agreement; Qwen2.5 unexpectedly lenient on night 96%; night_snow = 0% cho Qwen3).
