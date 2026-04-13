# Data Sources

Tài liệu này liệt kê tất cả nguồn dữ liệu được sử dụng trong dự án, bao gồm cách tải, format, và vị trí lưu trữ.

**STATUS**: Nhiều mục chưa hoàn thiện — xem TODO bên dưới.

---

## 1. Construction Site Dataset (Primary) — ConstructionSite 10k

**Nguồn**: HuggingFace
**HuggingFace Repo**: [`LouisChen15/ConstructionSite`](https://huggingface.co/datasets/LouisChen15/ConstructionSite)
**Format**: Parquet (PIL.JpegImageFile for images)
**License**: CC-BY-NC-4.0
**Classes**: excavator, rebar, worker_with_white_hard_hat (3 object classes)
**Splits**: Train = 7,009 images | Test = 3,004 images | **Total = 10,013**
**Vị trí local**: `~/LMUData/`

**Annotations bao gồm**:
- Image captions (detailed descriptions)
- Safety rule violation VQA (4 rules, bounding boxes + reasons)
- Visual grounding (bounding boxes cho 3 object classes)
- Attributes: illumination, camera_distance, view, quality_of_info

**Creators**: Xuezheng Chen (UBC), Zhengbo Zou (Columbia)
- Contact: xuezheng@student.ubc.ca, zhengbo.zou@columbia.edu

**Cách tải**:
```python
from datasets import load_dataset
dataset = load_dataset("LouisChen15/ConstructionSite")
# Requires HuggingFace login + agree to contact information sharing terms
```

**Paper**:
Chen, X. & Zou, Z. (2025). "Are Large Pre-trained Vision Language Models Effective Construction Safety Inspectors?" ArXiv: 2508.11011.

```bibtex
@misc{chen2025largepretrainedvisionlanguage,
      title={Are Large Pre-trained Vision Language Models Effective Construction Safety Inspectors?},
      author={Xuezheng Chen and Zhengbo Zou},
      year={2025},
      eprint={2508.11011},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.11011},
}
```

**TODO — CÒN THIẾU**:
- [ ] Checksum (SHA256) cho mỗi Parquet file
- [ ] Verify terms of use chi tiết (access requires agreement)

---

## 2. SODA Dataset

**Nguồn**: SharePoint (HKUST) + Baidu Cloud mirror
**Download**: [SharePoint link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ycdeng_connect_ust_hk/EiQLht3OhstGnKXrjFXyRZYBIXFjUC43jUUNVBXfM_kkKg?e=jJ2Nhv) | Baidu Cloud (pwd: 9cnh)
**Format**: Pascal VOC (XML annotations)
**Classes**: 15 classes (person, helmet, vest, hook, fence, board, slogan, rebar, handcart, ebox, hopper, wood, scaffold, brick, cutter)
**Categories**: workers, materials, machines, layout
**Images**: ~20,000+ images từ multiple construction sites
**Vị trí local**: `/users/PGS0407/binben14/VietHuy/ConstructionSite/SODA/`

**Paper gốc**:
Duan, R., Deng, H., Tian, M., Deng, Y., & Lin, J. (2022). "SODA: A large-scale open site object detection dataset for deep learning in construction." *Automation in Construction*, 142, 104499.
**DOI**: 10.1016/j.autcon.2022.104499
**ArXiv**: 2202.09554

```bibtex
@article{duan2022soda,
  title={SODA: A large-scale open site object detection dataset for deep learning in construction},
  author={Duan, Rui and Deng, Hui and Tian, Mao and Deng, Yichuan and Lin, Jiarui},
  journal={Automation in Construction},
  volume={142},
  pages={104499},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.autcon.2022.104499}
}
```

**Liên quan — SODA-ktsh (extended dataset)**:
Deng, H., Fu, K., Yu, B., Li, H., Duan, R., Deng, Y., & Lin, J.R. (2025). "Enabling High-Level Worker-Centric Semantic Understanding of Onsite Images Using Visual Language Models with Attention Mechanism and Beam Search Strategy." *Buildings*, 15(6), 959.
**DOI**: 10.3390/buildings15060959
- SODA-ktsh extends SODA với 16 construction scene categories + visual-language annotations

**License**: Không ghi rõ trong paper hoặc download page. Paper trên ArXiv có CC BY 4.0 (cho bài báo), nhưng **dataset license không được nêu rõ ràng**. Tên dataset có "open" → ngụ ý public use, nhưng cần confirm với authors.
- ⚠️ **Action needed**: Contact authors (ctycdeng@scut.edu.cn hoặc lin.jiarui@tsinghua.edu.cn) để xác nhận license cho dataset.

**TODO — CÒN THIẾU**:
- [ ] License chính thức (đã gửi email hỏi authors? → ghi ngày gửi + phản hồi)
- [ ] Số lượng ảnh chính xác per split (train/val/test)
- [ ] Checksum
- [ ] Lý do chọn dataset này (so với alternatives)

---

## 3. Augmented Data

### 3.1 Weather Augmented (Style Transfer)

**Vị trí**: `/users/PGS0407/binben14/VietHuy/ConstructionSite/augmentation_data/weather/`
**Format**: Arrow + filtered images
**Phương pháp sinh**: Neural style transfer (rain_0-2, snow_0-2)
**Statistics**:
- Construction Site: 3,004 original → 10,266 after SSIM filtering
- SODA: 19,846 original → 47,169 after filtering

### 3.2 Weather Augmented (IP2P Diffusion)

**Vị trí**: `augmentation_data/construction_site/rain_snow/diffusion/`
**Format**: Arrow + meta CSV
**Phương pháp sinh**: InstructPix2Pix + physics overlay
**Statistics**: TODO — tổng hợp từ meta CSV files

### 3.3 Day-to-Night

**Vị trí**: `augmentation_data/night/`
**Checkpoint**: `ConstructionSite/day2night/checkpoints/day2night.pkl`
**Phương pháp sinh**: img2img-turbo (One-Step Image Translation)
**Statistics**: 3,004 images (1:1 mapping)
**Repo**: [GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo) (commit `86f54146590ffb4543c8cf85b5a36657da670924`)
**License**: MIT
**Paper**: Parmar, G., Park, T., Narasimhan, S., & Zhu, J.-Y. (2024). "One-Step Image Translation with Text-to-Image Models." ArXiv: 2403.12036.

```bibtex
@article{parmar2024onestep,
  title={One-Step Image Translation with Text-to-Image Models},
  author={Parmar, Gaurav and Park, Taesung and Narasimhan, Srinivasa and Zhu, Jun-Yan},
  journal={arXiv preprint arXiv:2403.12036},
  year={2024}
}
```

**TODO — CÒN THIẾU**:
- [ ] Checksum (SHA256) cho day2night.pkl

### 3.4 Outpainting (Small/Scale)

**Vị trí**: `augmentation_data/small/`
**Phương pháp sinh**: FLUX.1-Fill-dev inpainting
**Statistics**:
- Construction Site: 3,004 → 1,323 (44% success rate)
- SODA: 19,846 → 1,001 (5% success rate — TODO: document nguyên nhân low rate)

---

## 4. Annotations

**Vị trí**: `/users/PGS0407/binben14/VietHuy/ConstructionSite-10k-Implementation/Annotations/`

**TODO — THIẾU**:
- [ ] Format mô tả (YOLO? VOC? COCO?)
- [ ] Schema documentation
- [ ] Sample annotations
- [ ] Annotation guidelines / inter-annotator agreement

---

## 5. Style References

**Rain styles**: `generation/weather/rain_snow/style_transfer/rain_style/` (3 images)
**Snow styles**: `generation/weather/rain_snow/style_transfer/snow_style/` (3 images)

**TODO — THIẾU**:
- [ ] Source/origin của 6 ảnh style reference
- [ ] License / permission to use
- [ ] Criteria để chọn 3 ảnh rain + 3 ảnh snow (tại sao những ảnh này?)

---

## 6. Pre-trained Models / Checkpoints

| Model | Source | License | Version/Revision | Checksum | Status |
|---|---|---|---|---|---|
| InstructPix2Pix | HuggingFace [`timbrooks/instruct-pix2pix`](https://huggingface.co/timbrooks/instruct-pix2pix) | **MIT** | TODO: pin revision hash | TODO | **CHƯA PIN** |
| MiDaS DPT_Large | [`isl-org/MiDaS`](https://github.com/isl-org/MiDaS) | **MIT** | TODO: pin version | TODO | **CHƯA PIN** |
| VGG19 (style transfer) | PyTorch torchvision | **BSD 3-Clause** | TODO: document origin of local `.pth` | TODO | **NGUỒN LOCAL CHƯA RÕ** |
| day2night.pkl | [`GaParmar/img2img-turbo`](https://github.com/GaParmar/img2img-turbo) | **MIT** | commit `86f5414` | TODO | ✅ Pinned |
| FLUX.1-Fill-dev | HuggingFace [`black-forest-labs/FLUX.1-Fill-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) | **FLUX.1 [dev] Non-Commercial** | TODO: pin revision | TODO | **CHƯA PIN** |
| YOLOv8 | [Ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** (hoặc Enterprise) | TODO: pin version | TODO | **CHƯA PIN** |
| Depth Anything V2 | [`DepthAnything/Depth-Anything-V2`](https://github.com/DepthAnything/Depth-Anything-V2) | **Apache-2.0** (Small) / **CC-BY-NC-4.0** (Base/Large/Giant) | TODO: pin revision | TODO | **CHƯA PIN** |

### License Details cho Models

**InstructPix2Pix** — MIT. Permissive, cho phép commercial + research. Không hạn chế.

**MiDaS** — MIT (Copyright © 2019 Intel ISL). Permissive, không hạn chế.

**VGG19 (torchvision)** — BSD 3-Clause (Copyright © Soumith Chintala 2016). Pretrained weights trained on ImageNet.
- ⚠️ **Lưu ý**: Local `.pth` files trong `generation/weather/VGG/` — cần verify đây có phải torchvision weights hay custom weights từ nguồn khác.

**img2img-turbo (day2night)** — MIT. Permissive. Authors: CMU + Adobe.

**FLUX.1-Fill-dev** — FLUX.1 [dev] Non-Commercial License.
- ✅ Research/academic use: **cho phép**
- ✅ Generated outputs (ảnh outpainted): **được dùng cho mọi mục đích kể cả commercial**
- ❌ Model không được dùng commercial/production
- ❌ Không được dùng cho military, surveillance, biometric
- ❌ Output không được dùng để train model cạnh tranh với FLUX

**YOLOv8 (Ultralytics)** — AGPL-3.0.
- ✅ Research/academic use: **cho phép**
- ⚠️ **AGPL-3.0 yêu cầu**: nếu dùng trong sản phẩm, toàn bộ code phải open-source
- Dự án ConSynth-X dùng YOLOv8 cho downstream evaluation (không deploy) → OK cho research paper
- Code đã Apache 2.0 → **không conflict** vì YOLOv8 chỉ dùng làm evaluation tool, không integrate vào codebase

**Depth Anything V2**:
- Small variant: Apache-2.0 (permissive)
- Base/Large/Giant: CC-BY-NC-4.0 (non-commercial only)
- ⚠️ Cần ghi rõ variant nào đang dùng

**CRITICAL**: Vẫn còn 5 models chưa pin version. Khi code chạy lại có thể load version khác → kết quả không reproducible.

---

## 7. Reference Data for Validation

### 7.1 WeatherNet-05-18039 (Primary Reference — FID/KID + Classifier Training)

**Nguồn**: HuggingFace — prithivMLmods
**URL**: [`prithivMLmods/WeatherNet-05-18039`](https://huggingface.co/datasets/prithivMLmods/WeatherNet-05-18039)
**License**: **Apache-2.0** (permissive)
**Format**: Parquet + PIL images
**Total original**: 18,039 images, 5 weather classes
**Local subset**: snow (1,875) + fog (1,261) = 3,136 images (rain/cloudy/clear removed to save quota)
**Vị trí local**: `validation/reference_data/weathernet/` (per-class folders, resized to 640x640)

**Local class distribution**:
| Class | Count |
|---|---|
| snow/frosty | 1,875 |
| foggy/hazy | 1,261 |

**Removed classes** (quota management, 2026-04-11): rain/storm (1,927), cloudy/overcast (6,702), sun/clear (6,274) — not needed since ACDC + WeatherBench now provide rain reference.

**Mục đích**:
1. Reference distribution cho FID/KID computation (Approach 2 — snow + fog)
2. Training data cho Weather-Image-Classification model (Approach 3 — primary metric, original 18K)

**Download**: `python validation/download_reference.py --source weathernet`

### 7.2 Weather-Image-Classification Model (Primary Validation Tool)

**Nguồn**: HuggingFace — prithivMLmods
**URL**: [`prithivMLmods/Weather-Image-Classification`](https://huggingface.co/prithivMLmods/Weather-Image-Classification)
**Architecture**: SigLIP2 (`google/siglip2-base-patch16-224`) fine-tuned
**License**: **Apache-2.0**
**Training data**: WeatherNet-05-18039 (see 7.1)
**Overall accuracy**: 85.89% on held-out test

**Per-class metrics**:
| Class | Precision | Recall | F1 |
|---|---|---|---|
| sun/clear | 0.912 | 0.885 | 0.898 |
| cloudy/overcast | 0.849 | 0.876 | 0.863 |
| snow/frosty | 0.834 | 0.845 | 0.839 |
| foggy/hazy | 0.834 | 0.813 | 0.823 |
| rain/storm | 0.764 | 0.759 | 0.762 |

**Mục đích**: Primary validation metric — classify ConSynth-X augmented images để verify classifier nhận ra đúng weather condition intended.

**Usage**:
```python
from transformers import AutoImageProcessor, SiglipForImageClassification
model = SiglipForImageClassification.from_pretrained("prithivMLmods/Weather-Image-Classification")
processor = AutoImageProcessor.from_pretrained("prithivMLmods/Weather-Image-Classification")
```

**Auto-downloaded** via transformers `from_pretrained()` on first run.

### 7.3 ACDC Dataset (Reference — FID/KID)

**Nguồn**: ETH Zurich
**URL**: [ACDC Dataset](https://acdc.vision.ee.ethz.ch/)
**License**: CC BY-NC-SA 4.0
**Format**: PNG images (driving scenes)
**Total local**: 3,578 images, 4 weather conditions
**Vị trí local**: `validation/reference_data/acdc/rgb_anon/{fog,night,rain,snow}/`

**Condition distribution (local subset)**:
| Condition | Images |
|---|---|
| fog | 1,000 |
| night | 1,006 |
| rain | 1,000 |
| snow | 572 |

**Paper**:
Sakaridis, C., Dai, D., & Van Gool, L. (2021). "ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding." *ICCV 2021*. ArXiv: 2104.13395.

```bibtex
@inproceedings{sakaridis2021acdc,
  title={ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding},
  author={Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
  booktitle={ICCV},
  year={2021}
}
```

**Mục đích**: Gold-standard reference cho FID/KID — real driving scenes dưới 4 adverse conditions. Dùng song song với WeatherBench và WeatherNet.

### 7.4 WeatherBench Dataset (Reference — FID/KID)

**Nguồn**: GitHub / Google Drive
**GitHub**: [guanqiyuan/WeatherBench](https://github.com/guanqiyuan/WeatherBench)
**License**: Chưa ghi rõ trên repo — cần confirm với authors (qyuanguan@gmail.com)
**Format**: JPEG images (real-world paired: input degraded + target clean)
**Total**: 42,002 paired images (train: 41,402 + test: 600)
**Local subset**: 1,000 images/condition (chỉ input/degraded), tổng 3,000
**Vị trí local**: `validation/reference_data/weatherbench/{rain,snow,haze}/`

**Condition distribution (local subset)**:
| Condition | Images (local) | Images (full) |
|---|---|---|
| rain | 1,000 | 14,929 |
| snow | 1,000 | 13,259 |
| haze (→fog) | 1,000 | 13,814 |

**Paper**:
Guan, Q. et al. (2025). "WeatherBench: A Real-World Benchmark for Weather Image Restoration." ArXiv: 2509.11642.

```bibtex
@article{guan2025weatherbench,
  title={WeatherBench: A Real-World Benchmark for Weather Image Restoration},
  author={Guan, Qiyuan and others},
  journal={arXiv preprint arXiv:2509.11642},
  year={2025}
}
```

**Mục đích**: Real-world reference cho FID/KID. Ảnh thực tế (không synthetic). Haze maps tới fog augmentation. Subset 1K/condition đủ cho FID/KID.

**Lưu ý**: WeatherBench dùng "haze" cho condition tương đương "fog" — script `compute_fid_kid.py` có alias mapping tự động.

### 7.5 UniversalFakeDetect (ABANDONED)

**Status**: Tested but not used as primary metric.
**URL**: [`WisconsinAIVision/UniversalFakeDetect`](https://github.com/WisconsinAIVision/UniversalFakeDetect)
**License**: MIT
**Architecture**: Frozen CLIP:ViT-L/14 + trained FC classifier (linear 768→1)
**Weights file**: `fc_weights.pth` (~4KB, Linear layer only) — downloaded from official repo
**Vị trí local**: `validation/weights/fc_weights.pth`
**Reason abandoned**: Fooling rate 99.5-100% cho tất cả conditions → không phân biệt được quality (xem `docs/methods.md` section 1.6).

### 7.4 ACDC (BACKUP OPTION, NOT USED)

**Status**: **Không dùng**. HuggingFace dataset `mathpluscode/ACDC` là medical cardiac MRI, không phải driving weather. Official ACDC driving weather cần download từ acdc.vision.ee.ethz.ch với registration.

**URL**: [acdc.vision.ee.ethz.ch](https://acdc.vision.ee.ethz.ch)
**License**: CC BY-NC-SA 4.0
**Images**: ~4,006 adverse condition images (fog, night, rain, snow ~1,000 each)

**Paper**: Sakaridis, C., Dai, D., & Van Gool, L. (2021). "ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding." ICCV. ArXiv: 2104.13395.

**TODO**: Download manually nếu WeatherNet không đủ cho paper validation.

---

## Notes

- Dữ liệu **không** được copy vào ConSynth-X, chỉ tham chiếu bằng absolute path trong code.
- Bảng data paths chi tiết: xem [`docs/infrastructure.md`](infrastructure.md) mục "Data Locations".
- Xem thêm `data_card.md` và `dataset_card.md` tại root project để biết chi tiết schema.
- **Tất cả absolute paths phải được thay bằng relative paths hoặc environment variables trước khi publish.**
