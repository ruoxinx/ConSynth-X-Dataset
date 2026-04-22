# Background & Summary (Draft)

> Draft cho Nature Scientific Data paper. Tuân thủ Data Descriptor format — mô tả dataset, KHÔNG trình bày research findings.
> 
> Quy tắc: Fact from data → không cần cite. Claim from literature → PHẢI cite. Design decision → ghi rõ.

---

## Background

Computer vision models for construction site monitoring have demonstrated strong performance under controlled conditions, yet their deployment in real-world construction environments remains challenging. Construction sites operate continuously across diverse weather conditions, lighting scenarios, and spatial configurations, exposing vision systems to rain, snow, fog, nighttime darkness, glare, and varying object scales simultaneously. Prior work has shown that object detection accuracy degrades by 40–50% when models trained on clear-weather imagery are evaluated under adverse conditions [cite: SODA cross-condition results — **fact from our data**: mAP50 drops from 0.819 to 0.446 on weather test set, and to 0.454 on night test set].

Several datasets exist for construction site object detection, including the Construction Site dataset (3 classes: excavator, rebar, worker with hard hat) and SODA (15 construction-related classes in Pascal VOC format). However, these datasets predominantly contain images captured under favorable conditions, limiting their utility for training robust models. While weather augmentation techniques have been explored in autonomous driving contexts [cite: Tremblay et al. 2021, Gurbindo et al. 2025 — **cited based on abstract only**], no large-scale, publicly available dataset exists that systematically pairs construction site imagery with controlled synthetic weather, lighting, and scale variations along with multi-label condition annotations.

Synthetic data augmentation offers a scalable alternative to manual data collection under hazardous conditions. Recent advances in text-guided diffusion models [cite: Brooks et al. 2023 — InstructPix2Pix], unpaired image translation [cite: img2img-turbo — **citation needed: find paper**], inpainting architectures [cite: FLUX.1 — **citation needed: find technical report**], and physics-based atmospheric models [cite: Koschmieder 1924; Tremblay et al. 2021] provide the tools to generate realistic weather effects, day-to-night conversions, and scale modifications while preserving object annotations. An earlier neural style transfer approach [cite: Gatys et al. 2016] was evaluated and is retained in this work as an ablation baseline (see §Methods / Ablation), but is not part of the main pipeline.

## Summary

We present ConSynth-X, a multi-condition construction site dataset comprising 85,613 images across two source datasets — Construction Site (3 classes, 17,597 images) and SODA (15 classes, 68,016 images) — spanning four condition categories: original, weather (rain and snow), nighttime, and small-scale (outpainted). Each image carries multi-label condition annotations from a taxonomy of 40+ condition types organized into 6 categories (weather, lighting, scale/size, image quality, scene complexity, domain shift).

The dataset was generated through three synthetic augmentation pipelines applied to 22,850 base images:

- **Weather augmentation (main pipeline)**: InstructPix2Pix (IP2P) text-guided diffusion combined with physics-based rain/snow particle overlays and Koschmieder fog scattering with Depth Anything V2 depth, producing 2 rain intensities, 2 snow intensities, and 3 fog visibility zones per image. Quality filtering uses SSIM + LPIPS thresholds (rain) and DINOv3 feature similarity (post-hoc release metric). A legacy VGG neural-style-transfer variant (3 rain + 3 snow intensities, SSIM-filtered, ~57K images across both datasets) is retained as an ablation baseline under [`experiments/ablation_style_transfer/`](../experiments/ablation_style_transfer/) for reproducibility and for reporting the realism-vs-recognizability trade-off [cite: Ruck et al. 2026].
- **Day-to-night conversion** (3,004 images): CycleGAN-Turbo single-step diffusion for realistic illumination transformation, with 1:1 mapping preserving all bounding box annotations.
- **Scale augmentation via outpainting** (2,324 images): FLUX.1-Fill-dev inpainting extends the image canvas with a truncated Gaussian scale distribution (mean=0.25, range=[0.20, 0.40]), making objects appear proportionally smaller. Bounding boxes are automatically transferred with coordinate offset compensation.

All augmentation pipelines preserve or correctly transfer object annotations. Weather and nighttime transforms are pixel-level operations that do not alter object positions; outpainting applies explicit geometric coordinate transformation with degenerate-box filtering.

Technical validation demonstrates that training a YOLOv8 detector on the augmented dataset improves mean average precision (mAP@0.5) from 0.558 to 0.975 on the Construction Site dataset (+75%). On the SODA dataset, augmented training improves cross-condition robustness: night-condition mAP@0.5 increases from 0.454 to 0.785 (+73%), and weather-condition mAP@0.5 increases from 0.446 to 0.810 (+82%), while maintaining original-condition performance (0.819 → 0.828).

ConSynth-X is publicly available at [TODO: HuggingFace/Zenodo URL] under a CC BY-NC 4.0 license (dataset) and Apache 2.0 license (code), with complete generation scripts enabling reproduction and extension of the augmentation pipeline.

---

## Ghi chú nội bộ (KHÔNG đưa vào paper)

### Citations cần verify/tìm trước khi submit:
- [ ] img2img-turbo paper — hiện chỉ có GitHub link, cần tìm paper chính thức
- [ ] FLUX.1 technical report — cần tìm
- [ ] Construction Site dataset paper gốc (LouisChen15 trên HuggingFace) — cần cite
- [ ] SODA dataset paper gốc — cần cite
- [ ] Tremblay et al. 2021 — cần verify exact numbers (21%, 73%) từ full text
- [ ] Gurbindo et al. 2025 — cần verify claims từ full text

### Fact vs Claim:
- "mAP drops 40-50%" → **fact from our SODA data** (0.819 → 0.446/0.454), không cần cite external
- "57,435 weather images" → **fact from our pipeline output**, verify từ AUGMENTATION_REPORT.md
- "improve mAP from 0.558 to 0.975" → **fact from our detection metrics**, file: metrics_original.json / metrics_original_weather.json

### Design decisions embedded (cần document trong methods.md):
- Tại sao 6 style variants? (3 rain + 3 snow) → design choice, không từ literature
- Tại sao truncated Gaussian cho scale? → design choice, documented trong AUGMENTATION_REPORT.md
- Tại sao SSIM filtering range? → sensitivity analysis đang chạy, sẽ reference supplementary

### Số liệu cần double-check:
- [ ] 85,613 total = 17,597 (CS) + 68,016 (SODA) — **OUTDATED**: con số này dựa trên 6-style NST. Cần recompute với main pipeline IP2P (2 rain + 2 snow) + fog 3 zones. Số ablation ST riêng biệt.
- [ ] 22,850 base = 3,004 (CS test) + 19,846 (SODA) — CS train (7,009) có augmented không? Cần clarify dataset scope
- [ ] ~~57,435 weather = 10,266 (CS) + 47,169 (SODA)~~ — **OUTDATED**: đây là số NST ablation. IP2P main pipeline có row counts riêng — xem `dataset_card.md` version history + `augmentation_data/` ground truth.
- [ ] Augmentation report nói CS original là "3,004" nhưng đó là test set. Train set có 7,009 images (2 Arrow shards). Paper nên clarify dùng split nào.
- [ ] **Action item (mới 2026-04-22)**: Tổng hợp row counts IP2P main (rain light/heavy, snow light/heavy, fog × 3, night, night_rain, night_snow, small) từ `augmentation_data/` → thay thế tất cả con số 57,435 / 85,613 / 10,266 / 47,169 trong Summary.
