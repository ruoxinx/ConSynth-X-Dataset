# Literature & Citations

Tài liệu này tổng hợp các luận điểm (statements), tham số, và phương pháp được trích dẫn từ các bài báo khoa học. Mỗi mục ghi rõ nguồn trích dẫn để đảm bảo tính học thuật.

**Quy tắc**: Chỉ list những gì đã VERIFY từ paper. Nếu chưa đọc full text → ghi "cited based on abstract only."

---

## 1. Augmentation & Robustness

### 1.1 Neural Style Transfer

**Statement**: Neural style transfer sử dụng VGG19 feature extraction để tách và kết hợp content/style của ảnh.
**Source**: Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). "Image Style Transfer Using Convolutional Neural Networks." CVPR, pp. 2414-2423.
**DOI**: 10.1109/CVPR.2016.265
**Parameters từ paper**: style_weight/content_weight ratio, LBFGS optimizer
**Verified**: Yes — đọc paper, method mô tả đúng.

**TODO — CẦN VERIFY**:
- Paper gốc dùng style_weight bao nhiêu? So sánh với giá trị 10,000 / 100,000 của chúng ta.
- Paper recommend bao nhiêu optimization steps? So sánh với 10 / 50 steps.

### 1.2 InstructPix2Pix

**Statement**: IP2P cho phép text-guided image editing bằng Stable Diffusion fine-tuned trên synthetic paired data (450K+ examples).
**Source**: Brooks, T., Holynski, A., & Efros, A. A. (2023). "InstructPix2Pix: Learning to Follow Image Editing Instructions." CVPR.
**ArXiv**: 2211.09800
**Verified**: Yes — đọc paper, method đúng.

**TODO — CẦN VERIFY**:
- Paper recommend image_guidance_scale range nào? Hiện dùng 1.5.
- Paper recommend guidance_scale range nào? Hiện dùng 8.0-10.0.
- Paper recommend num_inference_steps bao nhiêu? Hiện dùng 30.

### 1.3 MiDaS Monocular Depth Estimation

**Statement**: MiDaS DPT_Large ước lượng depth map từ single image, sử dụng mixed-dataset training.
**Source**: Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer." TPAMI.
**ArXiv**: 1907.01341
**Verified**: Yes — model name và approach đúng.

**TODO — CẦN VERIFY**:
- baseline=0.54 và focal=721.09 trong code (`snow_pipeline.py:67-68`) — đây có phải KITTI camera params? Nếu đúng → cite KITTI dataset paper. Nếu không → document nguồn gốc.

### 1.4 Day-to-Night (img2img-turbo)

**Statement**: img2img-turbo integrates three modules from latent diffusion models into a single end-to-end network, sử dụng LoRA adapters, skip connections, và Zero-Convs cho one-step unpaired image translation.
**Source**: Parmar, G., Park, T., Narasimhan, S., & Zhu, J.-Y. (2024). "One-Step Image Translation with Text-to-Image Models." ArXiv: 2403.12036.
**GitHub**: [GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo) (commit `86f5414`, MIT license)
**Authors**: Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, Jun-Yan Zhu (CMU + Adobe)
**Verified**: Yes — verified từ GitHub repo (architecture, license, checkpoint usage).

### 1.5 FLUX.1-Fill-dev (Outpainting)

**Statement**: FLUX Fill là inpainting model dựa trên FLUX.1 architecture.
**Source**: Black Forest Labs. "FLUX.1-Fill-dev." HuggingFace: black-forest-labs/FLUX.1-Fill-dev
**Verified**: Partial — xác nhận từ HuggingFace model card.

**TODO**: Tìm technical paper hoặc blog post chính thức.

### 1.6 IP2P for Weather Augmentation in Object Detection

**Statement**: IP2P weather augmentation cải thiện robustness của Faster R-CNN và YOLOv10 trên BDD100K/ACDC.
**Source**: Gurbindo, U., Brando, A., Abella, J., & König, C. (2025). "Object detection in adverse weather conditions for autonomous vehicles using Instruct Pix2Pix." IJCNN 2025.
**ArXiv**: 2505.08228
**Verified**: **Cited based on abstract only — full text not verified for specific metric values.**

### 1.7 Physics-Based Rain Rendering

**Statement**: Physics-based rain rendering cải thiện object detection 21% trên nuScenes; rain được đánh giá 73% more realistic than SOTA.
**Source**: Tremblay, M., Halder, S. S., de Charette, R., & Lalonde, J.-F. (2021). "Rain Rendering for Evaluating and Improving Robustness to Bad Weather." IJCV.
**ArXiv**: 2009.03683
**Verified**: Partial — con số 21% và 73% lấy từ abstract/web summary, chưa verify baseline/dataset context.

**TODO — CẦN VERIFY**:
- 21% improvement: baseline gì? dataset nào? metric nào (mAP@0.5 hay mAP@0.5:0.95)?
- 73% more realistic: so với method nào? human study methodology?

---

## 2. Quality Metrics

### 2.1 LPIPS (Perceptual Similarity)

**Statement**: Deep features (AlexNet/VGG) outperform SSIM cho perceptual similarity measurement.
**Source**: Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." CVPR.
**ArXiv**: 1801.03924
**Verified**: Yes — paper validates LPIPS > SSIM generally.

**QUAN TRỌNG — DESIGN DECISION, KHÔNG PHẢI TỪ PAPER**:
- Threshold 0.35 là **empirical selection** — paper KHÔNG recommend threshold cụ thể.
- Áp dụng chỉ cho rain (không snow) là **design decision** — paper không nói gì về rain-specific.
- Lý do (chưa documented): rain streaks gây structural deformation mà SSIM bắt không tốt → cần LPIPS. Snow không deform structure → SSIM đủ.
- **TODO**: Chạy sensitivity analysis LPIPS ∈ {0.20, 0.25, 0.30, 0.35, 0.40} để validate threshold.

### 2.2 SSIM (Structural Similarity)

**Statement**: SSIM đo similarity giữa 2 ảnh dựa trên luminance, contrast, structure.
**Source**: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). "Image quality assessment: from error visibility to structural similarity." IEEE TIP.
**Verified**: Classic metric, well-known.

**DESIGN DECISION — KHÔNG TỪ LITERATURE**:
- Threshold range [0.5, 0.95] (snow) và [0.6, 0.95] (rain) là **empirical**.
- **Chưa có sensitivity analysis** ở các thresholds khác.
- **TODO**: Chạy sweep SSIM_lower ∈ {0.3, 0.4, 0.5, 0.6, 0.7} → report ảnh kept + downstream mAP.

### 2.3 FID (Fréchet Inception Distance)

**Statement**: FID đo distribution distance giữa real và generated images.
**Source**: Heusel, M. et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS.
**Verified**: Yes — standard metric.

### 2.4 DINOv2 Structural Similarity

**Statement**: DINOv2 self-supervised features capture semantic structure tốt hơn pixel-level metrics.
**Source**: Oquab, M. et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." ArXiv 2304.07193.
**Verified**: Yes — dùng trong `examples/validation_metrics.ipynb`.

**PHÁT HIỆN MỚI (2026-04-09)**: DINO patch similarity phù hợp hơn SSIM cho weather augmentation evaluation.
- SSIM phạt cả thay đổi mong muốn (sky darkening, rain) và không mong muốn (hallucinate)
- DINO đo semantic structure (objects, layout) — không phạt atmospheric/color changes
- Evidence: 50 samples, 5 cases SSIM bị lừa (DINO bad, SSIM good), 5 cases SSIM đánh thấp (DINO good, SSIM bad)
- Figures: `paper/figures/dino_eval/disagreement_5plus5.png`, `paper/figures/dino_eval/dino_vs_ssim.png`
- **Recommendation**: dùng DINO làm primary quality metric trong paper, SSIM làm secondary/reference

---

## 3. Object Detection

### 3.1 YOLOv8

**Statement**: YOLOv8 là real-time object detection model.
**Source**: Jocher, G. et al. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
**Verified**: Yes — code implementation.

**TODO**: Xác định exact version (hiện tại unpinned trong environment.yml).

### 3.2 Evaluation Metrics (BERTScore, BLEU, ROUGE-L, METEOR)

**TODO**: Cite từng metric paper trong `evaluations/description_eval.py`.

---

## 4. Construction Domain

### 4.1 Construction Site 10k Dataset

**Statement**: ConstructionSite 10k là dataset 10,013 ảnh construction site với annotations cho image captioning, safety rule violation VQA, và visual grounding (3 object classes: excavator, rebar, worker_with_white_hard_hat).
**Source**: Chen, X. & Zou, Z. (2025). "Are Large Pre-trained Vision Language Models Effective Construction Safety Inspectors?" ArXiv: 2508.11011.
**License**: CC-BY-NC-4.0
**HuggingFace**: [`LouisChen15/ConstructionSite`](https://huggingface.co/datasets/LouisChen15/ConstructionSite)
**Verified**: Yes — verified từ HuggingFace dataset card (splits, classes, license, annotations).

### 4.2 SODA Dataset

**Statement**: SODA (Site Object Detection dAtaset) là large-scale dataset với 20,000+ ảnh construction site, 15 object classes (workers/materials/machines/layout), collected từ multiple sites dưới nhiều điều kiện khác nhau. Đạt mAP tối đa 81.47% với deep learning detection models.
**Source**: Duan, R., Deng, H., Tian, M., Deng, Y., & Lin, J. (2022). "SODA: A large-scale open site object detection dataset for deep learning in construction." *Automation in Construction*, 142, 104499.
**DOI**: 10.1016/j.autcon.2022.104499 | **ArXiv**: 2202.09554
**Verified**: Yes — verified từ ScienceDirect abstract + author page (dataset details, download link, class count).

### 4.3 SODA-ktsh (Extended)

**Statement**: SODA-ktsh extends SODA với 16 common construction scene categories và visual-language annotations cho image captioning tasks.
**Source**: Deng, H., Fu, K., Yu, B., Li, H., Duan, R., Deng, Y., & Lin, J.R. (2025). "Enabling High-Level Worker-Centric Semantic Understanding of Onsite Images Using Visual Language Models with Attention Mechanism and Beam Search Strategy." *Buildings*, 15(6), 959.
**DOI**: 10.3390/buildings15060959
**Verified**: Partial — verified từ author page và search results. Full text không accessible (403).

---

## References (Đầy đủ)

[1] Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). "Image Style Transfer Using Convolutional Neural Networks." CVPR, pp. 2414-2423. DOI: 10.1109/CVPR.2016.265

[2] Brooks, T., Holynski, A., & Efros, A. A. (2023). "InstructPix2Pix: Learning to Follow Image Editing Instructions." CVPR. ArXiv: 2211.09800

[3] Ranftl, R. et al. (2020). "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer." TPAMI. ArXiv: 1907.01341

[4] Zhang, R. et al. (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." CVPR. ArXiv: 1801.03924

[5] Gurbindo, U. et al. (2025). "Object detection in adverse weather conditions for autonomous vehicles using Instruct Pix2Pix." IJCNN. ArXiv: 2505.08228 — **abstract only**

[6] Tremblay, M. et al. (2021). "Rain Rendering for Evaluating and Improving Robustness to Bad Weather." IJCV. ArXiv: 2009.03683 — **partial verification**

[7] Wang, Z. et al. (2004). "Image quality assessment: from error visibility to structural similarity." IEEE TIP.

[8] Heusel, M. et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS.

[9] Oquab, M. et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." ArXiv: 2304.07193

[10] Chen, X. & Zou, Z. (2025). "Are Large Pre-trained Vision Language Models Effective Construction Safety Inspectors?" ArXiv: 2508.11011

[11] Duan, R., Deng, H., Tian, M., Deng, Y., & Lin, J. (2022). "SODA: A large-scale open site object detection dataset for deep learning in construction." Automation in Construction, 142, 104499. DOI: 10.1016/j.autcon.2022.104499

[12] Deng, H., Fu, K., Yu, B., Li, H., Duan, R., Deng, Y., & Lin, J.R. (2025). "Enabling High-Level Worker-Centric Semantic Understanding of Onsite Images Using Visual Language Models with Attention Mechanism and Beam Search Strategy." Buildings, 15(6), 959. DOI: 10.3390/buildings15060959

[13] Parmar, G., Park, T., Narasimhan, S., & Zhu, J.-Y. (2024). "One-Step Image Translation with Text-to-Image Models." ArXiv: 2403.12036

[14] Zhang, L., Rao, A., & Agrawala, M. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." ICCV. ArXiv: 2302.05543 — **[V] Verified: ControlNet architecture, canny/depth conditioning**

[15] Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., & Ermon, S. (2022). "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations." ICLR. ArXiv: 2108.01073 — **[V] Verified: noise-then-denoise, controllable faithfulness**

[16] Huang, Y. et al. (2024). "InstructRL4Pix: Training Diffusion for Image Editing by Reinforcement Learning." ArXiv: 2406.09973 — **[A] Abstract only: RL-based editing achieves higher SSIM/PSNR than vanilla IP2P**

[17] Huang, S. et al. (2025). "Diffusion Model-Based Image Editing: A Survey." IEEE TPAMI. ArXiv: 2402.17525 — **[V] Comprehensive survey, classification of editing methods**

### 2.5 UniversalFakeDetect (Realism Validation)

**Statement**: CLIP:ViT-L/14 features + nearest-neighbor / linear classifier generalizes across GAN và diffusion generators cho AI-generated image detection.
**Source**: Ojha, U., Li, Y., & Lee, Y. J. (2023). "Towards Universal Fake Image Detectors that Generalize Across Generative Models." CVPR.
**ArXiv**: 2302.10174
**GitHub**: [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
**License**: MIT
**Verified**: Yes — verified từ paper + GitHub repo. Dùng frozen CLIP features, trained FC on ProGAN, generalizes to LDM/GLIDE/DALL-E.

**Cách dùng trong ConSynth-X**: Đo fooling rate — % ảnh augmented bị classify nhầm thành real. Metric này bổ sung cho DINO/SSIM/LPIPS bằng cách trả lời trực tiếp "ảnh augmented trông có thật không?"

[18] Ojha, U., Li, Y., & Lee, Y. J. (2023). "Towards Universal Fake Image Detectors that Generalize Across Generative Models." CVPR. ArXiv: 2302.10174 — **[V] Verified**

### 2.6 WeatherNet-05 (Reference Dataset cho Realism Validation)

**Statement**: WeatherNet-05-18039 là dataset 18,039 ảnh weather classification với 5 classes: rain/storm (1,927), snow/frosty (1,875), foggy/hazy (1,261), cloudy/overcast (6,702), sun/clear (6,274).
**Source**: `prithivMLmods/WeatherNet-05-18039` (HuggingFace)
**License**: Apache-2.0
**Download**: `datasets.load_dataset("prithivMLmods/WeatherNet-05-18039")`
**Verified**: Yes — downloaded và verified image counts per class, all splits loaded successfully.

**Cách dùng trong ConSynth-X**: Reference distribution cho FID/KID computation (Approach 2 — partial success). Cũng là training data cho weather classifier (Approach 3 — primary metric).

### 2.7 Weather-Image-Classification Model (SigLIP2 — Primary Validation Metric)

**Statement**: Weather classifier fine-tuned từ SigLIP2-base-patch16-224 trên WeatherNet-05-18039. Overall test accuracy 85.89%, classifies ảnh vào 5 classes weather. Primary metric cho ConSynth-X realism validation.
**Source**: `prithivMLmods/Weather-Image-Classification` (HuggingFace)
**Base model**: `google/siglip2-base-patch16-224` (Zhai et al., "SigLIP: Sigmoid Loss for Language Image Pre-Training", ArXiv: 2303.15343)
**License**: Apache-2.0
**Verified**: Yes — loaded và tested full ConSynth-X dataset (3004 images/condition).

**Per-class test metrics (từ model card)**:
- sun/clear: P=0.912, R=0.885, F1=0.898
- cloudy/overcast: P=0.849, R=0.876, F1=0.863
- snow/frosty: P=0.834, R=0.845, F1=0.839
- foggy/hazy: P=0.834, R=0.813, F1=0.823
- rain/storm: P=0.764, R=0.759, F1=0.762

**Cách dùng trong ConSynth-X**: Classify ảnh augmented → check xem classifier có nhận ra đúng weather condition không. Accuracy cao = augmentation realistic và recognizable.

### 2.8 ACDC (KHÔNG DÙNG — wrong HuggingFace dataset)

**Status**: **ABANDONED** — `mathpluscode/ACDC` trên HuggingFace là medical imaging (cardiac MRI), không phải driving weather dataset. Official ACDC driving weather cần download từ acdc.vision.ee.ethz.ch với registration. Đã chuyển sang WeatherNet-05 thay thế.

**Source**: Sakaridis, C., Dai, D., & Van Gool, L. (2021). "ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding." ICCV.
**ArXiv**: 2104.13395
**Note**: Giữ làm backup option nếu WeatherNet không đủ — cần manual download + registration.

[19] Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). "SigLIP: Sigmoid Loss for Language Image Pre-Training." ArXiv: 2303.15343 — **[V] Verified**

[20] Sakaridis, C., Dai, D., & Van Gool, L. (2021). "ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding." ICCV. ArXiv: 2104.13395 — **[A] Abstract only** (not used in validation)

[21] Bińkowski, M., Sutherland, D. J., Arbel, M., & Gretton, A. (2018). "Demystifying MMD GANs." ICLR. ArXiv: 1801.01401 — **KID metric, unbiased alternative to FID for small samples**

[22] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS. — **FID metric**

### 2.9 Texture-based Fidelity Assessment (Duminil et al., 2025)

**Statement**: 4 kênh texture features (GLCM, LBP, DCT, Haralick) kết hợp bằng Dempster-Shafer belief theory phân biệt ảnh real vs synthetic ở micro-texture level. Real images có GLCM diagonal liên tục, synthetic thiếu gray-tone diversity. DCT cho thấy synthetic thiếu natural high-frequency noise.
**Source**: Duminil, E., Ieng, S.-S., & Gruyer, D. (2025). "Fidelity assessment of synthetic images with multi-criteria combination under adverse weather conditions." *Scientific Reports*.
**DOI**: Citation needed — paper read from PDF in `paper/validate/`
**Verified**: **[V] Verified — đọc full paper, method section, results. Adapted approach cho distribution comparison (Wasserstein/KS) thay vì CNN binary classification + belief fusion.**

**Cách dùng trong ConSynth-X**: Extract 4 texture features từ original + augmented images, compare distributions bằng Wasserstein distance + KS test. Phát hiện micro-level artifacts mà FID/KID và weather classifier bỏ qua. Files: `validation/compute_texture_fidelity.py`

[23] Duminil, E., Ieng, S.-S., & Gruyer, D. (2025). "Fidelity assessment of synthetic images with multi-criteria combination under adverse weather conditions." Scientific Reports. — **[V] Verified**

### 2.10 Scalable Realism Evaluation (Ruck et al., 2026)

**Statement**: Framework đánh giá realism của weather augmentation bằng 2 modalities: (1) VLM Jury (GPT-4o, Claude, Gemini — binary accept/reject), (2) Embedding-based distributional analysis (CLIP + DINOv3, Relative Mahalanobis Distance). Generative AI methods (Qwen 0.948, Gemini 0.902) outperform rule-based (imgaug 0.263) ~3.6×. Trade-off: rule-based giữ semantic nhưng trông giả (97.5% realism failure), generative trông thật nhưng thỉnh thoảng thay đổi scene content (74% semantic failure).
**Source**: Ruck, D., Vautravers, P., Chalkley, O., & Thomas, J. (2026). "Scalable Evaluation of the Realism of Synthetic Environmental Augmentations in Images." ArXiv: 2603.04325.
**Verified**: **[V] Verified — đọc full paper. Key insight: trade-off realism vs semantic preservation confirmed by our texture fidelity results (style transfer = weather mạnh + texture artifacts vs diffusion = subtle + ít artifacts).**

**Cách dùng trong ConSynth-X**: (1) Relative Mahalanobis Distance approach có thể áp dụng cho construction domain. (2) VLM Jury approach khả thi cho future work. (3) Trade-off finding corroborates our texture fidelity vs weather classifier results.

[24] Ruck, D., Vautravers, P., Chalkley, O., & Thomas, J. (2026). "Scalable Evaluation of the Realism of Synthetic Environmental Augmentations in Images." ArXiv: 2603.04325. — **[V] Verified, Eq. 1-2 implemented in `validation/compute_relative_mahalanobis.py`**

---

**TODO — CẦN THÊM**:
- [ ] FLUX.1 paper hoặc technical report
- [ ] KITTI paper (nếu baseline/focal params từ KITTI)
- [ ] Step1X-Edit paper (ArXiv: 2504.17761) — nếu dùng trong tương lai
- [ ] Duminil et al. DOI — tìm chính xác từ Scientific Reports
