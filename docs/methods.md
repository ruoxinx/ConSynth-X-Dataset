# Methods & Algorithms

Tài liệu này trình bày các phương pháp và thuật toán do tôi tự đề xuất/triển khai — **không** trích dẫn trực tiếp từ bài báo. Mỗi mục giải thích lý do, ý tưởng, và logic đằng sau quyết định thiết kế.

---

## 1. Data Generation Pipeline

### 1.1 Multi-condition Augmentation Strategy

**Ý tưởng**: Tạo synthetic data bao phủ nhiều điều kiện khắc nghiệt (weather, night, small) thay vì chỉ 1 điều kiện.

**Lý do**: Construction sites thực tế có nhiều điều kiện đồng thời (mưa + tối + vật thể nhỏ). Model cần robust trên tất cả.

**Cách triển khai**: 3 pipeline độc lập (weather, day2night, outpainting), mỗi pipeline bảo toàn annotations → gộp để train.

### 1.2 Weather Augmentation — 2 Methods

#### Method 1: Neural Style Transfer (VGG19 + MiDaS + Physics Particles)

**Pipeline**: Input → VGG19 style transfer (tone/color) → MiDaS depth estimation → Physics-based rain/snow particles → Output

**Parameters cần justify (TODO — CHƯA CÓ JUSTIFICATION):**

| Parameter | Giá trị hiện tại | File | Status justification |
|---|---|---|---|
| style_weight | **10,000 (workers) vs 100,000 (pipelines)** | `arrow_augmentation_worker.py:39`, `snow_pipeline.py:143` | **MÂU THUẪN — cần resolve giá trị canonical** |
| content_weight | 2 | `snow_pipeline.py:145` | Chưa justify. Gatys et al. (2016) dùng giá trị khác |
| steps | **10 (workers) vs 50 (pipelines)** | workers vs pipelines | **MÂU THUẪN — cần resolve** |
| MiDaS baseline | 0.54 | `snow_pipeline.py:67` | **Nguồn gốc không rõ** — có thể là camera param KITTI? Cần verify và cite |
| MiDaS focal | 721.09 | `snow_pipeline.py:68` | **Nguồn gốc không rõ** — giống KITTI focal length? Cần verify |
| visibility_ratio (light) | 0.6-0.9 | `snow_pipeline.py:115` | Chưa justify |
| visibility_ratio (heavy) | 0.1-0.3 | `snow_pipeline.py:122` | Chưa justify |
| darkness multipliers | Xem INTENSITY_CONFIG | `snow_pipeline.py:109-134` | Chưa justify |

**TODO**: Chạy sensitivity analysis cho style_weight ∈ {1000, 10000, 50000, 100000, 500000} và report downstream mAP.

#### Method 2: InstructPix2Pix (IP2P) + Physics Overlay

**Pipeline**: Input → IP2P diffusion (text-guided editing) → Physics overlay (rain streaks/snow flakes) → SSIM+LPIPS filter → Output

**Parameters cần justify (TODO — CHƯA CÓ JUSTIFICATION):**

| Parameter | Giá trị hiện tại | File | Status justification |
|---|---|---|---|
| image_guidance_scale | 1.5 | `batch_worker.py:105,109` | Chưa justify. IP2P paper recommend range [1.0, 2.0] nhưng không nói optimal |
| guidance_scale (rain) | 10.0 | `batch_worker.py:105` | Chưa justify. Tại sao 10.0? |
| guidance_scale (snow) | 8.0 | `batch_worker.py:109` | Chưa justify. Tại sao khác rain? |
| num_inference_steps | 30 | `batch_worker.py:131` | Chưa justify. Tại sao không 20 hoặc 50? |
| rain prompt (v1, deprecated) | "make it a heavy rainy day, dark overcast sky, wet muddy ground" | — | **BỎ** — xem lý do bên dưới |
| rain prompt (v2, current) | "a rainy day with dark overcast sky, rain falling, grey clouds" | `batch_worker.py:104` | Fixed 2026-04-08 |
| snow prompt | "a cold winter day with snow, frost on surfaces, grey sky, snow on the ground" | `batch_worker.py:108` | Giữ nguyên — snow không có vấn đề tương tự |

**PHÁT HIỆN QUAN TRỌNG — Prompt v1 rain gây lỗi (2026-04-08):**

Prompt v1 chứa cụm **"wet muddy ground"** khiến IP2P hiểu lệch: thay vì chỉ tạo atmosphere mưa (sky, clouds, rain), model tập trung biến đổi mặt đất thành bùn/nước. Hậu quả:
- **Hallucinate vũng nước lớn** trên mặt đất, thay đổi cấu trúc scene
- **Che khuất objects** (đặc biệt rebar nằm trên đất)
- **LPIPS tăng cao** do structural deformation → 48% rain images bị filter DROP
- **SSIM giảm** do thay đổi texture đất quá mạnh

**Nguyên nhân gốc**: IP2P là text-guided model — khi prompt yêu cầu "wet muddy ground", model ưu tiên sửa ground texture thay vì tạo atmospheric rain. Physics overlay đã xử lý rain streaks + fog, nên IP2P chỉ cần tạo atmosphere (overcast sky, lighting).

**Fix**: Loại bỏ "wet muddy ground", giữ lại atmospheric description. Prompt v2: `"a rainy day with dark overcast sky, rain falling, grey clouds"`.

**Bài học**: Với IP2P + physics overlay pipeline, prompt nên focus vào **atmospheric effects** (sky, clouds, lighting) và để physics module xử lý **particle effects** (rain streaks, snowflakes). Tránh yêu cầu IP2P thay đổi scene structure (ground, objects).

**Dữ liệu cần regenerate**: Toàn bộ rain diffusion data (7,009 images × 2 shards). Snow giữ nguyên.

#### Method 2b: ControlNet + IP2P (đang test, 2026-04-08)

**Ý tưởng**: Thêm ControlNet vào pipeline IP2P để **khóa cấu trúc ảnh gốc** trong khi IP2P chỉ thay đổi atmosphere. Giải quyết triệt để vấn đề hallucinate ground/objects.

**Hai approach đang test:**

| Approach | Model | Control Signal | Mechanism |
|---|---|---|---|
| **B) ControlNet IP2P** | `lllyasviel/control_v11e_sd15_ip2p` + SD 1.5 | Original image | ControlNet trained on IP2P editing pairs — hiểu "edit this image" |
| **C) Canny CN + img2img** | `lllyasviel/sd-controlnet-canny` + SD 1.5 | Canny edge map | Edge map khóa outlines, img2img strength=0.35 chỉ edit atmosphere |

**Tại sao approach C có thể tốt hơn:**
- `strength=0.35` giới hạn lượng noise → ảnh gốc giữ ~65% content (SDEdit principle, Meng et al. 2022)
- Canny edges khóa outlines của excavator, rebar, workers → model KHÔNG thể thay đổi object shapes
- `controlnet_conditioning_scale=0.8` balance giữa edge preservation và creative freedom

**Literature support:**
- SDEdit (Meng et al., ICLR 2022): noise-then-denoise approach, controllable faithfulness via noise level
- ControlNet (Zhang et al., 2023): additional conditioning preserves spatial structure
- InstructRL4Pix (2024): RL-based training đạt higher SSIM/PSNR than vanilla IP2P

**Test**: Job 4754807 — 5 samples × 3 methods (vanilla IP2P vs ControlNet IP2P vs Canny CN+img2img)

**TODO**: Chạy sensitivity analysis cho guidance_scale ∈ {6, 8, 10, 12, 14} và report SSIM/LPIPS/downstream mAP.

#### So sánh các models đã test (2026-04-08, cập nhật 04-09)

**QUAN TRỌNG — SSIM không phù hợp cho weather augmentation (2026-04-09):**

SSIM đo pixel-level similarity → phạt cả thay đổi mong muốn (darker sky, rain) lẫn không mong muốn (hallucinate). Thay bằng **DINO patch similarity** (DINOv2-ViT-S/14) — đo semantic structure preservation, không phạt atmospheric changes.

Evidence: 50 samples OLD prompt, tìm được:
- 5 cases "SSIM bị lừa": DINO < 0.55 nhưng SSIM > 0.50 (ground bùn, brightness giống)
- 5 cases "SSIM đánh thấp": DINO > 0.70 nhưng SSIM < 0.65 (weather tốt, objects giữ)
- Figures: `paper/figures/dino_eval/disagreement_5plus5.png`

| Method | DINO Patch ↑ | SSIM | Verdict |
|---|---|---|---|
| **C) Canny CN + img2img** | **0.744** | 0.590 | **Winner DINO** — strong weather + objects preserved |
| A) Vanilla IP2P (new prompt) | 0.674 | 0.692 | Winner SSIM — nhưng vì ít thay đổi |
| FLUX.1 Kontext | 0.639 | 0.281 | SSIM cực thấp nhưng DINO global=0.92 |
| IP2P SD1.5 | 0.675 | 0.667 | Balanced |
| SDXL IP2P | 0.435 | 0.443 | Tệ cả hai metric |
| B) ControlNet IP2P | 0.196 | 0.214 | Generate ảnh mới — worst |

**Recommendation**: Dùng DINO patch similarity làm primary metric cho quality evaluation. SSIM vẫn report nhưng không dùng để so sánh methods hoặc set threshold.
| Canny CN + img2img | *đang test* | Expected: highest SSIM nhờ edge + low strength |

#### Method 3: Night Weather — IP2P + Physics on Night Images

**Pipeline**: Night image (CycleGAN-Turbo output) → IP2P diffusion (night-specific prompt) → Physics overlay (rain/snow) → SSIM+LPIPS filter → Output

**Ý tưởng**: Kết hợp 2 điều kiện khắc nghiệt đồng thời (night + weather) tạo compound conditions thực tế hơn. Construction sites thường gặp mưa/tuyết ban đêm, không chỉ riêng lẻ.

**Cách triển khai**: Chain 2 pipelines — night trước (CycleGAN ổn định hơn, 1:1 mapping), weather sau (IP2P có thể filter). IP2P prompts tuned cho night scene (references wet reflections, artificial lights) thay vì day scene (overcast sky, grey clouds).

**Quyết định thiết kế — Night-first order:**
- CycleGAN-Turbo tạo 1:1 mapping (3,004 → 3,004), không mất ảnh nào
- IP2P + filter có thể drop images → đặt sau để chỉ mất ở bước cuối
- Night-specific prompts tận dụng visual context đã có (tối, đèn nhân tạo)

**Quyết định thiết kế — Quality comparison against original:**
- SSIM/LPIPS so sánh với ảnh gốc (day), KHÔNG với ảnh night trung gian
- Lý do: đo tổng khoảng cách visual từ source, đảm bảo compound augmentation không biến đổi quá xa

**Parameters:**

| Parameter | Rain Night | Snow Night | Status justification |
|---|---|---|---|
| IP2P prompt | "a rainy night with rain falling, wet reflections on surfaces, dark overcast sky" | "a cold winter night with snow falling, frost on surfaces, snow on the ground" | Tuned for night context — khác prompt day weather |
| guidance_scale | 10.0 | 8.0 | **Kế thừa từ day weather** — chưa tune riêng cho night. TODO: sweep |
| image_guidance_scale | 1.5 | 1.5 | Kế thừa từ day weather |
| num_inference_steps | 30 | 30 | Kế thừa từ day weather |
| SSIM range | [0.6, 0.95] | [0.5, 0.95] | Kế thừa từ day weather — so sánh vs original (not night) |
| LPIPS threshold | < 0.35 | N/A | Kế thừa từ day weather |

**TODO**: Chạy sensitivity analysis riêng cho night weather — threshold optimal có thể khác day weather vì SSIM baseline thấp hơn (night + weather = 2 lần biến đổi).

### 1.3 Physics-based Particle Overlay

**Cả 2 methods đều dùng physics overlay sau augmentation chính.**

**Parameters cần justify (TODO — CHƯA CÓ JUSTIFICATION):**

| Parameter | Giá trị | File | Status |
|---|---|---|---|
| Rain layer 1 (far) | n=1500-2500, len=10-22px, alpha=0.25 | `physics.py:55` | Empirical, không có photographic/physical basis |
| Rain layer 2 (mid) | n=1000-2000, len=18-35px, alpha=0.35 | `physics.py:57` | Empirical |
| Rain layer 3 (near) | n=300-700, len=25-50px, alpha=0.40 | `physics.py:59` | Empirical |
| Rain angle | 75-87° | `physics.py:50` | Physically reasonable nhưng chưa cite meteorological data |
| Fog strength | 0.15-0.30 | `physics.py:47` | Empirical |
| Snow layer 1 | n=1500-3000, r=1-2px, alpha=0.35 | `physics.py:93` | Empirical |
| Snow layer 2 | n=500-1000, r=2-4px, alpha=0.50 | `physics.py:94` | Empirical |
| Snow layer 3 | n=100-300, r=4-6px, alpha=0.65 | `physics.py:95` | Empirical |

**TODO**: Cite Tremblay et al. (2021) cho physics-based approach, hoặc chạy visual quality study.

### 1.4 SSIM/LPIPS Quality Filtering

**Ý tưởng**: Lọc bỏ ảnh augmented quá giống gốc (không có tác dụng) hoặc quá hỏng (không dùng được).

**Parameters cần justify (TODO — CHƯA CÓ SENSITIVITY ANALYSIS):**

| Parameter | Giá trị | Áp dụng | Status |
|---|---|---|---|
| SSIM lower (rain) | 0.6 | Style Transfer + IP2P | **Chưa justify tại sao 0.6 mà không phải 0.5** |
| SSIM lower (snow) | 0.5 | Style Transfer + IP2P | **Chưa justify tại sao khác rain** |
| SSIM upper | 0.95 | Tất cả | Reasonable nhưng chưa sensitivity analysis |
| LPIPS threshold | 0.35 | **Chỉ rain IP2P** | **Chưa justify**: (1) tại sao 0.35? literature recommend ~0.1-0.15 cho imperceptible, (2) tại sao chỉ rain mà không snow? |

**SENSITIVITY ANALYSIS — ĐANG CHẠY:**
- Script: `generation/sensitivity/ssim_lpips_sweep.py` (sweep đã xong)
- Script: `generation/sensitivity/run_sensitivity.py` (YOLOv8 training đang chạy)
- Data quantity sweep results: `generation/sensitivity/results/ssim_sweep_results.csv`
- 6 configs submitted: loose/moderate/current/strict/tight/no_lpips
- Kết quả downstream mAP: chờ tại `generation/sensitivity/detection_results/metrics_sens_*.json`

**Phát hiện từ sweep (data quantity):**
- LPIPS filters thêm ~10% rain images so với SSIM-only (SSIM=4022 → +LPIPS=3626)
- Snow không có LPIPS scores (chưa compute) → **cần document đây là implementation gap**
- IP2P snow rất insensitive với SSIM threshold (99% → 90% từ 0.30 → 0.60)
- Style Transfer snow_1 rất sensitive (91% → 23% từ 0.30 → 0.60)

**TODO sau khi YOLOv8 xong:**
1. Ghi kết quả mAP per threshold vào bảng bên dưới
2. Chọn threshold dựa trên data (không chọn trước)
3. Document tại sao LPIPS chỉ áp dụng cho rain

### 1.5 Outpainting Bbox Transfer Algorithm

**Ý tưởng**: Khi outpaint mở rộng canvas, bbox phải được scale + offset tương ứng.

**Lý do**: Outpainting thay đổi coordinate system → nếu copy bbox thẳng sẽ sai vị trí.

**Thuật toán**:
```
1. Tính scale_factor từ Gaussian distribution (mean=0.25, std=0.05, range=[0.2, 0.4])
2. Paste ảnh gốc vào canvas mới tại vị trí (paste_x, paste_y)
3. Với mỗi bbox [x1,y1,x2,y2] (normalized):
   a. Convert to pixel: x1_px = x1 * orig_w * resize_factor
   b. Add offset: x1_new = (x1_px + paste_x) / new_w
   c. Normalize to new canvas
4. Skip degenerate boxes (zero-area)
```

**Parameter cần justify:**

| Parameter | Giá trị | Status |
|---|---|---|
| scale_mean | 0.25 | Documented trong AUGMENTATION_REPORT.md, nhưng chưa sensitivity analysis |
| scale_std | 0.05 = (0.4-0.2)/4 | Formula documented nhưng tại sao `/4`? No justification |
| scale_range | [0.2, 0.4] | Documented: "original occupies 40-70% of canvas area" |

---

## 1.6 Realism Validation — 3 Approaches Tested (2026-04-10)

**Vấn đề cần giải quyết**: Các quality metrics hiện tại (SSIM, LPIPS, DINO) chỉ đo similarity với ảnh gốc — không trả lời "ảnh augmented trông có thật không?" hoặc "có trông giống weather thật không?". Giáo viên yêu cầu validation mạnh hơn để chứng minh giá trị dataset.

Đã thử nghiệm **3 approaches**, mỗi approach có ưu/nhược điểm khác nhau:

---

### Approach 1 (ABANDONED): UniversalFakeDetect (UnivFD)

**Ý tưởng**: Dùng AI-generated image detector. Fooling rate cao = augmentation trông thật.

**Setup**:
- Model: `WisconsinAIVision/UniversalFakeDetect` (Ojha et al., CVPR 2023, ArXiv: 2302.10174)
- Frozen CLIP:ViT-L/14 + trained FC classifier
- License: MIT

**Kết quả (full dataset, 22,601 ảnh, 11 conditions)**:

| Condition | Fooling Rate | Mean Score | Gap from original |
|---|---|---|---|
| original | 100.0% | 0.246 | --- |
| diffusion_rain | 100.0% | 0.284 | +0.037 |
| diffusion_snow | 100.0% | 0.274 | +0.027 |
| weather_style_rain_0/1/2 | 99.5-100% | 0.328-0.343 | +0.08-0.10 |
| weather_style_snow_0/1/2 | 99.9-100% | 0.329-0.374 | +0.08-0.13 |
| night | 100.0% | 0.301 | +0.055 |
| small | 100.0% | 0.292 | +0.044 |

**Kết luận: KHÔNG PHÙ HỢP**
- Fooling rate 99.5-100% cho **tất cả** conditions → không phân biệt được quality
- Lý do: UnivFD được train để detect ảnh AI-generated **hoàn toàn** (faces, scenes tạo từ đầu). Ảnh augmented ConSynth-X là **edit trên ảnh thật** — nền vẫn là photograph thật, chỉ thêm weather effects. Detector luôn thấy "real photograph".
- Score gap có differentiate nhẹ nhưng không meaningful cho paper.
- **Files**: `validation/realism_detector.py`, `validation/run_realism_validation.py`
- **Results**: `validation/results/full_univfd/` (giữ lại cho supplementary, không dùng làm primary metric)

---

### Approach 2 (VALIDATED): FID/KID vs 3 Real Weather Reference Datasets

**Ý tưởng**: So sánh distribution ảnh augmented với ảnh weather thật. FID/KID thấp = gần real weather. Multi-reference cross-validation tăng tính thuyết phục.

**Setup**:
- **3 reference datasets** (cross-validation, giảm bias từ 1 dataset):
  - **ACDC** (Sakaridis et al., ICCV 2021): 3,578 real driving images — rain (1K), snow (572), fog (1K), night (1K). CC BY-NC-SA 4.0.
  - **WeatherBench** (Guan et al., arXiv 2509.11642): 1K subset/condition — rain, snow, haze→fog. Real-world paired images.
  - **WeatherNet-05**: snow (1,875), fog (1,261). Apache-2.0.
- Feature extractor: InceptionV3 (pool3, 2048-dim) — standard cho FID
- KID: polynomial kernel `(x·y/d + 1)^3`, 100 subsets × min(500, N) samples
- Condition alias mapping: fog ↔ haze (WeatherBench compatibility)
- Total: **39 pairs** evaluated (16 ACDC + 14 WeatherBench + 9 WeatherNet)

**Kết quả tóm tắt** (best augmentation vs original baseline per condition):

| Condition | Best Method | ACDC Δ FID | WBench Δ FID | WNet Δ FID |
|---|---|---|---|---|
| **Night** | CycleGAN-Turbo | **−91.6 (−34%)** | — | — |
| **Snow** | diffusion / style_2 | −10.7 (−5%) | −1.8 (−1%) | −12.0 (−9%) |
| **Fog** | diffusion_heavy | +9.1 (+4%) | **−20.2 (−10%)** | **−9.0 (−6%)** |
| **Rain** | ≈ baseline | +2.5 (+1%) | −0.2 (0%) | — |

**Phân tích**:
- **Night validation thuyết phục nhất**: FID giảm 34% (178.9 vs 270.5). CycleGAN-Turbo rất hiệu quả.
- **Snow validated trên cả 3 datasets**: augmented snow FID luôn thấp hơn baseline. Nhất quán.
- **Fog validated trên WeatherBench + WeatherNet**: diffusion_heavy tốt nhất (−6% đến −10%). ACDC fog ngược lại (original gần hơn) — có thể do driving fog khác construction fog.
- **Rain inconclusive**: augmentation ≈ baseline trên cả ACDC và WeatherBench. Cross-domain gap (construction vs driving/outdoor) dominates weather effect.
- **Style transfer vs Diffusion**: gần nhau cho snow; diffusion tốt hơn cho fog.

**Kết luận**: 3/4 conditions validated (night, snow, fog). Rain cần cải thiện hoặc giải thích cross-domain gap. Multi-reference approach tăng độ tin cậy so với single-dataset.

- **Files**: `validation/compute_fid_kid.py`, `validation/download_reference.py`
- **Results**: `validation/results/fid_kid/` (JSON + 3 barplots + LaTeX table)

---

### Approach 3 (SUCCESS — Primary Metric): Weather Classifier Recognition

**Ý tưởng**: Dùng pretrained weather classifier để phân loại ảnh augmented. Accuracy cao = classifier nhận ra đúng weather condition → augmentation realistic.

**Setup**:
- **Model**: `prithivMLmods/Weather-Image-Classification`
- Architecture: SigLIP2 (`google/siglip2-base-patch16-224`) fine-tuned
- Training data: WeatherNet-05-18039
- Overall accuracy: 85.89% on held-out test
- License: Apache-2.0
- 5 classes: cloudy/overcast, foggy/hazy, rain/storm, snow/frosty, sun/clear

**Kết quả (full dataset, 3004 ảnh/condition cho CS dataset)**:

| Condition | Expected | Accuracy | Top Predicted | Verdict |
|---|---|---|---|---|
| **original** | sun/clear | 6.1% | rain/storm (57.8%) | cross-domain bias |
| small (outpaint) | sun/clear | 8.4% | rain/storm (37.9%) | cross-domain bias |
| **weather_style_rain_0** | rain/storm | **88.6%** | rain/storm | ✓ excellent |
| weather_style_rain_1 | rain/storm | 71.3% | rain/storm | ✓ good |
| weather_style_rain_2 | rain/storm | 70.8% | rain/storm | ✓ good |
| **weather_style_snow_1** | snow/frosty | **95.0%** | snow/frosty | ✓ excellent |
| weather_style_snow_2 | snow/frosty | 92.7% | snow/frosty | ✓ excellent |
| weather_style_snow_0 | snow/frosty | 61.8% | snow/frosty | ✓ moderate |
| diffusion_rain (IP2P) | rain/storm | 63.5% | rain/storm | ✓ moderate |
| **diffusion_snow (IP2P)** | snow/frosty | **25.7%** | rain/storm (49.1%) | ✗ failed |
| night (CycleGAN) | N/A | — | rain/storm (95.1%) | no night class |

**Key findings**:

1. **Style transfer rain**: 70-89% accuracy (consistent across 3 intensity levels) — validated well
2. **Style transfer snow**: 62-95% accuracy — snow_1/snow_2 rất tốt (>92%), snow_0 có 33% bị classify nhầm thành rain
3. **IP2P diffusion rain**: 63.5% — kém hơn style transfer rain
4. **IP2P diffusion snow**: **chỉ 25.7%** — thất bại rõ rệt, 49% bị classify thành "rain storm". IP2P có thể tạo atmosphere tối giống storm thay vì snow effect rõ ràng.
5. **Construction domain bias**: Original clear images chỉ 6.1% accuracy cho "sun/clear" — classifier (train trên driving scenes) không quen với construction sites (đất, bụi, equipment → trông giống overcast/rain). Đây là **cross-domain artifact**, cần note trong paper.
6. **Night images**: 95.1% classified as rain — classifier không có "night" class, có vẻ treat night images với dark atmosphere giống storm. Không validate được night pipeline qua metric này.

**Kết luận: SUCCESS** — đây là metric **phân biệt rõ nhất** giữa các augmentation methods. Primary metric cho paper.

**Cách interpret cho paper**:
- Style transfer weather augmentation (rain + snow) được classifier nhận ra đúng với accuracy 62-95%
- IP2P diffusion weather có vấn đề với snow (cần investigate)
- Construction domain là challenge — cần mention explicitly
- Night và small augmentations không validate được qua weather classifier (khác nature)

**Files**: `validation/weather_classifier.py`, `validation/jobs/run_weather_cls.sh`
**Results**: `validation/results/weather_cls/` — JSON, CSV, figures, LaTeX table

---

### Tổng hợp 4 Approaches

| Approach | Level | Discrimination | Conditions Validated | Recommended Use |
|---|---|---|---|---|
| UnivFD (fooling rate) | Semantic | ✗ không phân biệt | none | supplementary only |
| **FID/KID × 3 datasets** | Distribution | ✓ night, snow, fog | **3/4** (rain inconclusive) | **co-primary** |
| **Weather Classifier** | Semantic | ✓ phân biệt rõ | rain, snow (not night) | **co-primary** |
| Texture Fidelity + DS Fusion | Micro-level | ✓ diffusion vs style | all | **complementary** |

**Kế hoạch cho paper (Technical Validation section)**:
1. Co-primary: Weather classifier (Approach 3) — validates perceptual weather recognizability
2. Co-primary: FID/KID multi-reference (Approach 2) — validates distributional similarity, especially night (−34%)
3. Complementary: Texture fidelity + belief fusion (Approach 4) — validates micro-level artifact absence
4. Supplementary: UnivFD (giải thích tại sao fake detection không work cho edit-based augmentation)
5. Note: Cross-domain limitation (all reference datasets are driving/outdoor, not construction)

---

### Approach 4 (SUCCESS): Texture-based Fidelity Assessment (GLCM + LBP + DCT + Haralick)

**Ý tưởng**: Dùng 4 kênh texture features truyền thống để phát hiện **micro-level artifacts** mà semantic metrics (FID/KID, weather classifier) bỏ qua: unnatural noise patterns, missing gray-tone continuity, frequency anomalies.

**Reference**: Duminil, Ieng & Gruyer (2025). "Fidelity assessment of synthetic images with multi-criteria combination under adverse weather conditions." Scientific Reports. — **[V] Verified: method section read, adapted for distribution comparison instead of CNN classification.**

**4 kênh features:**

| Feature | Dim | Phát hiện gì | Implementation |
|---|---|---|---|
| **GLCM** (Gray Level Co-occurrence Matrix) | 48 | Thiếu đa dạng gray-tone, discontinuity texture toàn cục | 6 properties × 2 distances × 4 angles, quantize 64 levels |
| **LBP** (Local Binary Pattern) | 26 | Micro-texture artifacts: rain streaks lặp, snow particles đồng đều bất thường | Uniform LBP, radius=3, 24 points → 26-bin histogram |
| **DCT** (Discrete Cosine Transform) | 12 | Thiếu natural noise (over-smooth) hoặc HF artifacts (particle overlay) | HF/LF energy ratio + DC stats + spectral entropy × 3 channels |
| **Haralick** | 16 | Statistical texture anomalies tổng hợp (ASM, contrast, correlation, entropy) | 8 metrics × (mean + std) trên patches 64×64, sampled ≤25 patches |

**So sánh phân phối bằng**: Wasserstein distance (per-dim mean), KS test (significance), mean shift (L2), covariance divergence (Frobenius). Không cần GT — chỉ so distribution original vs augmented vs ACDC real weather.

**Kết quả (2026-04-11, 300 images/condition):**

| Augmentation | GLCM_W | LBP_W | DCT_W | Haralick_W | **Composite** | Đánh giá |
|---|---|---|---|---|---|---|
| **Diffusion snow (IP2P)** | 0.807 | 0.003 | 1.337 | 0.288 | **0.609** | Tốt nhất — texture gần original |
| **Diffusion rain (IP2P)** | 0.823 | 0.004 | 1.411 | 0.273 | **0.628** | Tốt nhất — texture gần original |
| Style snow_0 (light) | 2.129 | 0.009 | 8.928 | 0.780 | 2.961 | Thay đổi vừa |
| Style rain_1 (moderate) | 2.057 | 0.005 | 10.594 | 1.044 | 3.425 | Thay đổi vừa |
| Style rain_2 (heavy) | 2.050 | 0.005 | 11.413 | 0.911 | 3.595 | Thay đổi vừa |
| Style rain_0 (light) | 2.506 | 0.007 | 12.628 | 1.064 | 4.051 | Thay đổi đáng kể |
| Style snow_2 (heavy) | 2.493 | 0.009 | 14.810 | 1.055 | 4.592 | Thay đổi đáng kể |
| Fog light | 3.803 | 0.003 | 12.886 | 1.735 | 4.607 | Expected (fog mờ) |
| Fog medium | 4.420 | 0.004 | 15.814 | 1.986 | 5.556 | Expected |
| **Style snow_1 (moderate)** | **4.711** | 0.009 | **17.188** | **1.931** | **5.960** | **Outlier — artifacts mạnh** |
| Fog heavy | 5.220 | 0.006 | 19.933 | 2.405 | 6.891 | Expected (fog mạnh) |
| **Night (CycleGAN)** | 4.765 | 0.003 | **29.811** | 1.971 | **9.138** | **DCT rất cao — over-smoothed** |

**Key findings:**

1. **Diffusion (IP2P) >> Style Transfer về texture fidelity**: Composite 0.61-0.63 vs 2.96-5.96. IP2P giữ gần như nguyên vẹn micro-texture (KS p-value > 0.05 ở nhiều dimensions → không significant change), trong khi style transfer thay đổi significant 100% dimensions (p≈0.000).
2. **Night (CycleGAN-Turbo) over-smoothed**: DCT_W = 29.81 (gấp 21× diffusion rain). CycleGAN làm mất high-frequency content tự nhiên — có thể cần post-processing thêm noise.
3. **Style snow_1 là outlier**: Composite 5.96 vs snow_0 (2.96) và snow_2 (4.59). Style reference snow_1 có thể tạo artifacts nhiều hơn → cần kiểm tra lại style image.
4. **Fog thay đổi texture nhiều là expected**: Fog physically removes HF detail (atmospheric scattering). Đây không phải artifact mà đặc tính vật lý đúng.
5. **ACDC comparison confirms**: fog/night augmentations gần ACDC real weather hơn original (Wasserstein thấp hơn). Rain/snow augmentations cũng gần ACDC hơn trong nhiều features.

**Insight bổ sung cho paper — complementary với các approaches khác:**
- Weather classifier (Approach 3): **Style transfer thắng** (rain 70-89% vs diffusion 63.5%) — tức style transfer tạo ảnh trông giống rain hơn ở semantic level
- Texture fidelity (Approach 4): **Diffusion thắng** (Composite 0.63 vs 3.42) — tức diffusion giữ texture tự nhiên hơn
- **Kết luận**: Style transfer tạo hiệu ứng weather mạnh hơn (classifier nhận ra dễ) nhưng phải trả giá bằng texture artifacts. Diffusion tinh tế hơn (ít artifacts) nhưng weather effect nhẹ hơn. **Trade-off realism vs recognizability**, giống phát hiện của Ruck et al. (2026).

**Files**: `validation/compute_texture_fidelity.py`, `jobs/texture_fidelity_validation.sh`
**Results**: `validation/results/texture_fidelity/` (JSON, 6 plots, LaTeX table)

#### Approach 4b: Dempster-Shafer Belief Fusion trên Wasserstein Scores

**Ý tưởng**: Thay vì chỉ report Wasserstein distance đơn giản, kết hợp 4 criteria (GLCM, LBP, DCT, Haralick) bằng Dempster-Shafer belief theory để cho ra **fidelity score (H)** kèm **uncertainty (Ω)** và **conflict** giữa các criteria. Đây là contribution chính của Duminil et al. (2025).

**Implementation** (adapted — skip CNN, dùng Wasserstein scores):

| Tầng | Paper gốc | Implementation |
|---|---|---|
| 1. Feature extraction | GLCM, LBP, DCT, Haralick | ✓ Giữ nguyên (Approach 4) |
| 2. Score generation | 3 Xception CNNs + HaMeC linear | **Thay bằng**: Wasserstein → similarity via exp(-λ·W), λ calibrated per feature |
| 3. BBA generation | BBFs Φ₁, Φ₂ (Eq. 14) | ✓ Implement đúng paper: α₀ reliability + τ threshold |
| 4. CRC combination | Conjunctive Rule (Eq. 20-23) | ✓ Implement đúng paper: generalized N-source fusion |

**Parameters** (following paper Section "Implementation on datasets"):

| Parameter | Giá trị | Justification |
|---|---|---|
| τ (threshold) | **0.6** | Paper: "pessimistic, slightly higher than neutral 0.5" — Sc < 0.6 supports H̄ |
| α₀ (GLCM, LBP, DCT) | **0.8** | Paper dùng model accuracy (~80-99%). Vì skip CNN, dùng 0.8 conservative default |
| α₀ (Haralick) | **0.5** | Paper: "S_H criterion set to 0.5 as it is impossible to obtain similar accuracy" |
| λ calibration | **ln(2) / median_W** | Per-feature: median Wasserstein → Sc=0.5 (at τ boundary) |

**Kết quả (2026-04-11):**

| Augmentation | m(H) Fidelity ↑ | m(H̄) Artifact ↓ | m(Ω) Uncertainty | Conflict | Verdict |
|---|---|---|---|---|---|
| **Diffusion snow** | **0.862** | 0.000 | 0.138 | 0.000 | **FAITHFUL** |
| **Diffusion rain** | **0.854** | 0.000 | 0.146 | 0.000 | **FAITHFUL** |
| Fog light | 0.054 | 0.491 | 0.385 | 0.069 | mixed |
| Night (CycleGAN) | 0.018 | 0.734 | 0.170 | **0.078** | ARTIFACT + conflict |
| Style snow_0 | 0.017 | 0.439 | 0.529 | 0.014 | uncertain |
| Style rain_0 | 0.000 | 0.503 | 0.497 | 0.000 | ARTIFACT |
| Style rain_1 | 0.000 | 0.256 | 0.744 | 0.000 | uncertain |
| Style rain_2 | 0.000 | 0.292 | 0.708 | 0.000 | uncertain |
| **Style snow_1** | 0.000 | **0.831** | 0.169 | 0.000 | **ARTIFACT** (strongest) |
| Style snow_2 | 0.000 | 0.623 | 0.377 | 0.000 | ARTIFACT |
| Fog heavy | 0.000 | 0.830 | 0.170 | 0.000 | ARTIFACT (expected) |
| Fog medium | 0.000 | 0.675 | 0.325 | 0.000 | ARTIFACT (expected) |

**Key findings bổ sung so với Wasserstein đơn giản (Approach 4):**

1. **Conflict phát hiện criteria disagreement**: Night (CycleGAN) có conflict=0.078 — LBP nói texture OK (Sc=0.648 > τ) nhưng GLCM/DCT nói hỏng nặng. Wasserstein đơn giản trung bình hoá, belief fusion **phát hiện mâu thuẫn** → insight Night giữ micro-texture (LBP OK) nhưng mất macro-texture/frequency (GLCM/DCT bad).
2. **Uncertainty quantifies confidence**: Style rain_1/2 có Ω > 0.7 = **chưa đủ evidence** để kết luận artifact hay faithful — nằm gần ranh giới τ=0.6. Khác với style snow_1 (H̄=0.831, Ω=0.169) = **chắc chắn artifact**.
3. **Diffusion rain/snow: 4 criteria unanimous** (conflict=0, Ω thấp) — tất cả đồng ý faithful. Kết quả mạnh hơn chỉ nói "Wasserstein thấp".

**Files**: `validation/belief_fusion.py`
**Results**: `validation/results/texture_fidelity/belief_fusion_results.json`, `belief_fusion_summary.png`, `belief_bba_per_condition.png`, `belief_radar_comparison.png`

---

### Approach 5: Relative Mahalanobis Distance (CLIP + DINOv2)

**Ý tưởng**: Per-image metric đo proximity của ảnh augmented tới real adverse-condition distribution trong embedding space, dùng **relative formulation** triệt tiêu shared background features (construction scene giữ nguyên giữa original và augmented).

**Reference**: Ruck, Vautravers, Chalkley & Thomas (2026). "Scalable Evaluation of the Realism of Synthetic Environmental Augmentations in Images." ArXiv: 2603.04325. — **[V] Verified: đọc full paper, implement Eq. 1-2, Section 3.4.**

**Khác gì so với FID/KID (Approach 2)?**

| Khía cạnh | FID/KID | Relative Mahalanobis |
|---|---|---|
| Granularity | Distribution-level (1 score cho cả tập) | **Per-image** |
| Backbone | InceptionV3 | **CLIP ViT-L/14 (768-dim) + DINOv2 ViT-L (1024-dim)** |
| Reference | WeatherNet (generic outdoor) | **ACDC per-condition** Gaussian |
| Background handling | Không tách | **d_rel = d_k - d_0** (subtract background distance) |

**Pipeline (4 steps):**
1. Extract CLIP + DINOv2 features cho ACDC real weather (fog/rain/snow/night), split reference/holdout
2. Fit multivariate Gaussian N(μₖ, Σₖ) per condition + N(μ₀, Σ₀) background (pooled all conditions)
3. Baseline: 100 held-out ACDC images per condition → expected near-zero d_rel (upper bound)
4. Per augmented image: d_rel = d_k(x) - d_0(x), report as -d_rel (higher = closer to real weather)

**Key equations (Ruck et al. 2026):**
- d_k(x) = √((x - μₖ)ᵀ Σₖ⁻¹ (x - μₖ))  — Mahalanobis distance to condition k (Eq. 1)
- d_rel = d_k(x) - d_0(x)  — relative distance, subtracts background (Eq. 2)

**Parameters:**

| Parameter | Giá trị | Justification |
|---|---|---|
| CLIP model | openai/ViT-L-14 (via open_clip) | Paper: "openai/clip-vit-large-patch14" |
| DINOv2 model | facebook/dinov2-large (via transformers) | Paper: "facebook/dinov2-vitl16-pretrain-lvd1689m" |
| Holdout per condition | 100 | Paper: "100 held-out images per condition" |
| Regularization (Σ) | 1e-5 × I | Numerical stability for matrix inversion |

**Kết quả (2026-04-12, 300 images/condition):**

**CLIP -d_rel (higher = closer to real weather):**

| Condition | ACDC Baseline | Original (clear) | Best Augmentation | Best -d_rel | **% Gap Closed** |
|---|---|---|---|---|---|
| **Night** | -2.97 | -39.32 | CycleGAN-Turbo | **-25.60** | **38%** |
| **Snow** | -8.89 | -53.83 | style_snow_2 | **-41.60** | **27%** |
| **Rain** | -3.63 | -22.17 | style_rain_1 | **-17.57** | **25%** |
| **Fog** | -3.65 | -28.07 | fog_heavy | **-23.57** | **18%** |

**DINOv2 -d_rel:**

| Condition | ACDC Baseline | Original | Best Augmentation | Best -d_rel |
|---|---|---|---|---|
| Night | -15.09 | -74.23 | CycleGAN-Turbo | **-70.13** |
| Snow | -40.78 | -119.07 | diffusion_snow | **-115.74** |
| Rain | -17.05 | -60.65 | diffusion_rain | **-59.74** |
| Fog | -15.31 | -68.95 | fog_heavy | **-68.83** |

**Key findings:**

1. **Tất cả augmentations kéo distribution gần ACDC real weather** — -d_rel tốt hơn original ở mọi condition trong CLIP space. Validation thành công.
2. **Night = cải thiện lớn nhất** (38% gap closed trong CLIP) — CycleGAN-Turbo rất hiệu quả ở semantic level. Nhưng texture fidelity (Approach 4) cho thấy night bị over-smoothed (DCT_W=29.81) → **CLIP semantic tốt, texture artifacts** — đúng trade-off Ruck et al. phát hiện.
3. **CLIP vs DINOv2 divergence rõ rệt**:
   - CLIP nhạy với augmentation (shift 4-14 points)
   - DINOv2 gần như không đổi cho style transfer (shift <1 point), nhạy hơn với diffusion
   - Consistent với paper: "CLIP produces tighter condition clusters, DINOv2 shows greater sensitivity to low-level texture"
4. **Rain: style transfer ≈ diffusion** trong CLIP (-17.6 vs -18.1) — tương đương semantic level.
5. **Snow: style transfer > diffusion trong CLIP** (-41.6 vs -44.1), **diffusion > style transfer trong DINOv2** (-115.7 vs -117.4) — trade-off semantic vs texture, consistent với texture fidelity findings.
6. **Fog: monotonic với intensity** — heavy (-23.57) > medium (-24.54) > light (-24.79) > original (-28.07). Physically correct.

**Cross-approach synthesis (Approaches 3-5):**

| Metric | Style Transfer thắng | Diffusion thắng | Night |
|---|---|---|---|
| Weather Classifier (Approach 3) | ✓ Rain 70-89%, Snow 62-95% | ✗ Rain 63.5%, Snow 25.7% | N/A (no class) |
| Texture Fidelity (Approach 4) | ✗ Composite 3-6 | ✓ Composite 0.6 | ✗ Composite 9.1 |
| Belief Fusion (Approach 4b) | ✗ H=0, ARTIFACT | ✓ H=0.85, FAITHFUL | ✗ H=0.02, ARTIFACT |
| Rel. Mahalanobis CLIP (Approach 5) | ≈ tied (rain), ✓ snow | ≈ tied (rain), ✗ snow | ✓ Best improvement |
| Rel. Mahalanobis DINOv2 (Approach 5) | ✗ nearly no shift | ✓ slight improvement | ✓ moderate shift |

**Conclusion**: Style transfer tạo weather effect mạnh hơn (recognizable, semantic shift) nhưng phải trả giá bằng texture artifacts. Diffusion giữ texture tự nhiên hơn nhưng weather effect tinh tế hơn. Night CycleGAN semantic tốt nhưng over-smoothed. **Cả 2 methods có giá trị — recommend dùng cả hai trong training mix.**

**Files**: `validation/compute_relative_mahalanobis.py`, `jobs/relative_mahalanobis.sh`
**Results**: `validation/results/relative_mahalanobis/` (JSON, 3 plots, LaTeX tables)

---

## 2. Benchmarking Design

### 2.1 Per-condition Evaluation Protocol

**Ý tưởng**: Evaluate model performance separately per condition, không chỉ overall.

**Lý do**: Overall mAP có thể mask performance drops trên specific conditions (e.g., rebar drops 70% in weather nhưng excavator chỉ drop 8%).

### 2.2 Multi-label Annotation System

**Ý tưởng**: Mỗi ảnh có thể có nhiều condition labels (rain + low_light + occlusion).

**Lý do**: Thực tế construction sites có nhiều điều kiện đồng thời. Single-label annotation là oversimplification.

### 2.3 VLM Task Design (Description / VQA / Detection)

**Ý tưởng**: 3 tasks đo lường 3 khả năng khác nhau của VLM trên construction images.

**Lý do**: Description đo scene understanding, VQA đo safety reasoning, Detection đo spatial awareness.

---

## 3. Training Strategies

### 3.1 YOLOv8 Training Configurations

**Ý tưởng**: 5 configs với mix augmentation khác nhau để đo impact.

**Lý do**: Isolate contribution của mỗi augmentation type.

**Configs**: original, original+weather, original+weather+night, original+weather+small, original+weather+small+night

### 3.2 Augmentation Mix Ratios

**Ý tưởng**: Gộp tất cả augmentation types vào training set.

**Lý do**: Balanced representation across conditions.

**TODO**: Document tại sao không dùng weighted sampling.

---

## 4. Evaluation & Analysis

### 4.1 Cross-condition Robustness Metric

**Ý tưởng**: So sánh detection performance across original/weather/night/small test sets.

**Lý do**: Measure robustness không chỉ là accuracy trên 1 test set.

### 4.2 Radar/Heatmap Visualization Approach

**Ý tưởng**: Radar chart cho per-model comparison, heatmap cho per-class × per-condition.

**Lý do**: Dễ nhận diện patterns (model nào yếu ở condition nào).
