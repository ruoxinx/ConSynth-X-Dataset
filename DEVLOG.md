# Development Log — ConSynth-X

> Nhật ký theo dõi các thay đổi và cập nhật của dự án theo ngày.

---

## 2026-04-30 — Release v1 packaged + Kaggle mirror live

**Output:** `release_pipeline/` shipped a 35-shard, 122,156-row release.

**Storage**
- Parquet: `/fs/scratch/PGS0407/binben14/ConSynth-X-release-v1/` (50 GB, embedded JPEG bytes)
- COCO mirror: `/fs/scratch/PGS0407/binben14/ConSynth-X-release-v1-coco/` (53 GB, decoded JPEG + annotations.json)

**Public**
- Kaggle: `viethuyduong/construction-site-augmentation-data` — uploaded as 3 zips (cs10k 13 GB, soda_voc 31 GB, soda_ktsh 6.4 GB) + metadata. v5 indexed (totalBytes=53.7 GB) but per-version GCS sync pending.
- HuggingFace `Ben11304/ConSynth-X` — repo created but private storage cap hit on first push; awaiting public/upgrade decision.

**Coverage** (cs10k 44,105 / soda_voc 56,021 / soda_ktsh 22,030)
- 11 conditions × 3 subs in Table 3 (rain_night/snow_night cs10k-only).
- soda_ktsh captions: 21,789 / 22,030 = 98.9% — `fog_*` and `snow_heavy` shards originally `captions=[]`; recovered by joining from `rain_*.arrow` and `snow_light.arrow` via image_id (caption_join_sources in registry; verbatim copy verified by agent: 0 byte drift, 0 overwrite, 5,328 rows filled).
- soda_ktsh night ships 1,135 / 1,500 — 365 rows had empty/non-JPEG bytes from a failed upstream day2night batch; skipped at repack with explicit warnings.
- DINO scores: 8/13 v1.x conditions done (cs_small × 2, soda_voc × 6); 5 soda_ktsh conditions still computing on pitzer gpu-exp.

**Schema** — see `docs/data_sources.md` §7. Key choices:
- bbox: normalised xyxy in Parquet, COCO mirror in absolute pixel `[x,y,w,h]`.
- `quality_alert = (dino_sim < 0.75)`, nullable when DINO not yet computed.
- Fixed-seed pipelines per condition (IP2P weather: `seed = 42 + image_index`; fog: `42 + batch_start`; day2night and FLUX outpaint: fixed 42).

**Repack runner discoveries (in audit JSONs)**
- 36 cs10k rows: `rule_violations[].bbox = null` (rule fired with reason text but no bbox at source) — preserved.
- 3 soda_voc rows (zl220–222): irreconcilable VOC `<size>` vs JPEG dims — skipped.
- 1 shard missed during initial registry: cs10k `test/snow_heavy/snow_heavy.arrow` (3,004 rows). Fixed; bumped cs10k total 41,101 → 44,105.
- Train/test path collision (`<condition>/<basename>.parquet` overwriting) caught + fixed by split-prefixed basename (`train__<basename>.parquet`).

**Documentation**
- `paper/dataset_description.{tex,pdf}` updated for post-augmentation truth (MiDaS DPT_Large not Depth Anything V2; FLUX.1-Fill-dev not FLUX.1-dev; per-shard row counts; caption coverage).
- HF-style README at `paper/DSA/{readme.md,README.md}` and mirrored at release root.
- Schema dump at `<release>/schema.json`; checksums (83 entries) at `<release>/checksums.sha256`.

---

## 2026-04-29

### Image-quality-control funnel — end-to-end accounting (N_input → SSIM+LPIPS → DINO audit)

**Mục tiêu.** Cho paper section "Image Quality Control": ghi rõ **mỗi (source × condition) có bao nhiêu ảnh đi vào pipeline, bao nhiêu sống qua filter, bao nhiêu đạt audit DINO≥0.75**, để reviewer Nature Sci Data thấy toàn bộ pipeline minh bạch.

**Scope filter (quan trọng — không nhầm lẫn).** SSIM+LPIPS filter **chỉ áp cho rain** (light & heavy paired):
- Rain SSIM ∈ [0.60, 0.95], LPIPS < 0.35 (`generation/weather/rain_snow/diffusion/batch_worker.py:25-27,168`).
- Snow / fog / night / night_weather → **không filter** ở generation time. `N_kept < N_input` cho các condition này phản ánh **partial generation** (chạy subset, ví dụ snow_heavy = b2 1k, fog = subset 1k của CS test), KHÔNG phải filter rejection.

**Code mới.**
- [`validation/build_qc_funnel.py`](validation/build_qc_funnel.py) — đếm N_input/N_kept/DINO retention per (source × condition); tách `filter_retention_pct` (chỉ rain) khỏi `coverage_pct` (mọi condition khác).
- [`validation/extract_dino_ssim_extended.py`](validation/extract_dino_ssim_extended.py) — streaming chunk=256, lazy decode JPEG bytes → tính DINOv3 + SSIM cho 12 (source, condition) chưa có CSV: CS train + SODA-VOC + SODA-KTSH × {rain_light, rain_heavy, snow_light, snow_heavy}. Cần streaming vì soda_voc/snow_light = 19,623 ảnh — preload toàn bộ PIL OOM với 64G.
- [`validation/plot_qc_funnel.py`](validation/plot_qc_funnel.py) — 3 figures: exemplar (CS test rain_light, 4-stage funnel), overview bar chart (rain solid bars vs non-rain dotted; DINO≥0.75 hatched overlay), DINO retention curves 4-panel per source.
- [`validation/refresh_qc_funnel.sh`](validation/refresh_qc_funnel.sh) — wrapper rebuild funnel CSV + figures sau khi DINO compute landing.
- SLURM job scripts: [`validation/jobs/extract_dino_ssim_extended_pitzer.sh`](validation/jobs/extract_dino_ssim_extended_pitzer.sh) (V100-32g, gpu-exp, 96G mem) và `..._cardinal.sh` (H100, gpu, 96G mem) làm race backup.

**Compute.** 12 (source, condition) DINO/SSIM CSVs sinh trên Pitzer V100-32g node `p0339` (3 jobs parallel: cs_train + soda_voc + soda_ktsh). Wall ≈ 75 phút từ start tới CSV cuối (soda_voc snow_light 19,623 rows là bottleneck — ~30 phút riêng nó). OOM lần đầu (`--mem=64G`) đã fix bằng streaming + bump 96G. Cardinal duplicates hủy sau khi Pitzer xong.

**Findings cho paper.**

| Source | rain filter retention |
|---|---:|
| cs_train | 51.75% |
| cs_test | 54.99% |
| soda_voc | **24.14%** |
| soda_ktsh | **21.15%** |

→ Rain SSIM/LPIPS thresholds (calibrated trên CS) cắt **~75-79% ảnh trên SODA** — domain shift rõ rệt: SODA street scenes có cấu trúc khác CS construction sites, nên IP2P rain output rời xa original hơn → bị filter. Ghi vào paper như sensitivity caveat.

**DINO≥0.75 audit (post-hoc) cho mọi condition × source:**
- **night_rain** worst-case: DINO mean 0.447, chỉ **10.8% pass** DINO≥0.75 — Order-B (CycleGAN day2night → IP2P weather) cộng dồn semantic drift. Sẽ flag rõ trong paper.
- rain_heavy DINO consistency thấp hơn rain_light đáng kể: cs_train 90.4 → 58.3%, soda_ktsh 88.1 → 46.8% — physics overlay heavy đẩy ảnh xa thêm sau IP2P.
- snow_light gần như perfect: 96-99% retention (gen) + 98-99% DINO≥0.75 → pipeline ổn định nhất.
- fog × 3 và night (CS test): pass-through không filter, DINO≥0.75 đạt 87-99%.

**Outputs.**
- [`validation/results/qc_funnel/funnel_counts.csv`](validation/results/qc_funnel/funnel_counts.csv) (20 rows, full DINO stats + filter/coverage tách bạch).
- [`validation/results/qc_funnel/funnel_report.md`](validation/results/qc_funnel/funnel_report.md).
- [`validation/results/qc_funnel/figures/funnel_exemplar_cs_test_rain_light.pdf`](validation/results/qc_funnel/figures/funnel_exemplar_cs_test_rain_light.pdf), `funnel_overview_yields.pdf`, `funnel_dino_distributions.pdf`.

**Anomalies pending.**
- **CS train snow_heavy = 6,009/7,009** (85.7%) trong khi `docs/methods.md` nói `--no-filter`. Có thể là partial regeneration. Cần verify trước submit.
- Discrepancies với paper Table 1 (snow/small counts) — đã tracked trong checklist.

---

## 2026-04-28

### Zero-shot detection robustness benchmark — unified 100-img matched-pair set across 11 conditions

**Mục tiêu.** Trả lời GAP-1 trong `docs/checklist.md`: "augmented conditions có thực sự stress detection không?" Trước đây các sanity bench dùng *subset khác nhau* per-condition (rain_light có 4,790 ids, fog test có 1,000 ids, intersect = 0) → ∆mAP không cross-comparable.

**Thiết kế.** 100 SODA-VOC ids deterministic (seed=42) có ≥1 person GT (460 boxes), generate qua mọi condition cùng lúc — **matched-pair**. Source materialised tại [`detection_validation/source_100/`](detection_validation/source_100/) với 3 artifact: `clear/` (jpgs), `clear.arrow` (4-col SODA-VOC schema), `manifest.json` (COCO GT person-only). Builder: [`bench/build_unified_100.py`](bench/build_unified_100.py).

**Generation pipeline (Pitzer V100, total ≈40 min wall-clock).** 6 SLURM jobs trong [`jobs/det_val/`](jobs/det_val/):

| Job | Pipeline | Output | Walltime |
|---|---|---|---|
| `j1_weather_ip2p` | IP2P rain (light prompt + default physics) + IP2P snow + IP2P snow_heavy (g=12, igs=1.2) | `rain_light/`, `snow_light/`, `snow_heavy/` (JPGs) + `rain_light.arrow`, `snow_light.arrow` | 21:07 |
| `j2_rain_heavy_cpu` | `bench/heavy_rain_jpgs.py`: `add_natural_rain(intensity='heavy_fog')` post-process trên `rain_light/` JPGs | `rain_heavy/` (JPGs) | 1:19 |
| `j3_fog` | `arrow_fog_worker_soda.py` × 3 intensities (Depth-Anything-V2-Small + Perlin fog) | `fog_light.arrow`, `fog_medium.arrow`, `fog_heavy.arrow` | 9:44 |
| `j4_night` | `day2night_soda_worker.py` (CycleGAN-Turbo `day_to_night`) | `night/images/` (JPGs) | 2:12 |
| `j5_night_weather` | `voc_b2_night_worker.py` × 2 (rain_light → CycleGAN night → rain physics; snow_light → CycleGAN night → snow physics) | `night_rain/batch_0-100.arrow`, `night_snow/batch_0-100.arrow` | 5:00 |
| `j6_small` | `flux_pipeline_worker_voc.py` array 10×10 (FLUX.1-Fill outpainting với resize 0.5, scale 0.25–0.4, 28 steps) | `small/{JPEGImages,Annotations}/` — **bbox được transfer sang canvas mới** | ~25 min/task parallel |

Lưu ý kỹ thuật: `j5` dùng VOC-B2 flow (weather → night) thay vì night → weather, để khớp với existing `submit_voc_b2_night.py` convention. `j6` dùng `flux_pipeline_worker_voc.py` (vs `flux_pipeline_worker.py` arrow flow) để có annotation transfer XML output.

**Detection eval.** [`bench/zero_shot_unified.py`](bench/zero_shot_unified.py) chạy 3 detector COCO-pretrained zero-shot, person-only IoU=0.5 via pycocotools:
- YOLOv8m (`ultralytics`)
- Faster R-CNN R50-FPN-v2 (`torchvision.models.detection`, `box_score_thresh=0.0`)
- DETR R50 (`facebook/detr-resnet-50` no_timm, `threshold=0.0`)

Submit: `jobs/det_val/run_unified_eval.sh` (3-task array, V100). Walltime mỗi task: 1:41 / 3:13 / 2:56.

**Kết quả** (`bench/sanity_results/unified_{model}.csv`):

| condition | YOLOv8m mAP@0.5 | Faster R-CNN mAP@0.5 | DETR mAP@0.5 |
|---|---:|---:|---:|
| **clear (baseline)** | **0.509** | **0.627** | **0.529** |
| rain_light  | 0.203 (∆+0.31) | 0.218 (∆+0.41) | 0.201 (∆+0.33) |
| rain_heavy  | 0.140 (∆+0.37) | 0.149 (∆+0.48) | 0.110 (∆+0.42) |
| snow_light  | 0.404 (∆+0.11) | 0.381 (∆+0.25) | 0.390 (∆+0.14) |
| snow_heavy  | 0.259 (∆+0.25) | 0.249 (∆+0.38) | 0.252 (∆+0.28) |
| fog_light   | 0.443 (∆+0.07) | 0.518 (∆+0.11) | 0.451 (∆+0.08) |
| fog_medium  | 0.437 (∆+0.07) | 0.509 (∆+0.12) | 0.445 (∆+0.08) |
| fog_heavy   | 0.426 (∆+0.08) | 0.477 (∆+0.15) | 0.417 (∆+0.11) |
| night       | 0.355 (∆+0.15) | 0.384 (∆+0.24) | 0.300 (∆+0.23) |
| **night_rain** | **0.069 (∆+0.44)** | **0.092 (∆+0.53)** | **0.067 (∆+0.46)** |
| night_snow  | 0.256 (∆+0.25) | 0.256 (∆+0.37) | 0.233 (∆+0.30) |
| small       | 0.279 (∆+0.23) | 0.340 (∆+0.29) | 0.274 (∆+0.25) |

**Quan sát chính** (consistent across 3 detectors):
1. **Robust band — fog × 3** (∆ 0.07–0.15): Fog chỉ giảm contrast/visibility, không thay đổi texture frequencies → person silhouette vẫn detectable.
2. **Mid band — snow_light, night, small** (∆ 0.10–0.30): Single-axis distribution shift, detectors degrade gracefully.
3. **Heavy band — rain × 2, snow_heavy, night_snow** (∆ 0.25–0.48): Cao-frequency physics overlay (rain streaks, snow flakes) phá feature maps; heavy_fog rain post-process gây drop nhiều nhất trong rain group.
4. **Worst case — `night_rain`** (∆ 0.44–0.53, ~85% mAP loss): Combinational stress (low-light + rain) cho thấy đây là condition khó nhất.
5. **Faster R-CNN** có clear baseline cao nhất (0.627) nhưng cũng drop tuyệt đối lớn nhất → có dấu hiệu over-fit COCO; YOLOv8m và DETR degrade uniform hơn.
6. `small` (∆ 0.23–0.29) đánh giá *task difficulty* khi objects nhỏ + canvas mở rộng — đây là test case riêng vì GT bbox bị transform; mAP đo trên GT mới của chính nó (FLUX preserve count = 460 boxes).

**Implications cho paper.** Closes GAP-1 trong `docs/checklist.md`: synthesized conditions tạo *task-relevant* distribution shift đủ mạnh để stress 3 representative detection paradigms (single-stage CNN, two-stage CNN, transformer). Bảng này có thể đi vào Technical Validation section như evidence rằng dataset có *practical utility* cho robustness research, không chỉ "không gây hại". Limitation: zero-shot COCO-pretrained, chưa benchmark training-on-aug; có thể follow up trong supplementary.

**Files**:
- Generation: [`bench/build_unified_100.py`](bench/build_unified_100.py), [`bench/heavy_rain_jpgs.py`](bench/heavy_rain_jpgs.py), [`jobs/det_val/`](jobs/det_val/)
- Eval: [`bench/zero_shot_unified.py`](bench/zero_shot_unified.py), [`jobs/det_val/run_unified_eval.sh`](jobs/det_val/run_unified_eval.sh)
- Data: [`detection_validation/`](detection_validation/) — 11 condition outputs + `source_100/`
- Results: [`bench/sanity_results/unified_{yolov8m,fasterrcnn,detr}.{csv,json}`](bench/sanity_results/)

---

## 2026-04-27

### Dataset overview figures (DAWN-style) + CS snow_heavy train file appearance + Table 1 reconciliation

**Figures created.** Four new generation scripts under `paper/`, all reading directly from `augmentation_data/` Arrow tables:

| Script | Output | Purpose |
|---|---|---|
| [`paper/generate_class_statistics.py`](paper/generate_class_statistics.py) | `fig_class_distribution.{pdf,png}` (4-panel bars) + `class_distribution_counts.csv` | DAWN-style per-class bbox counts × 5 conditions for CS (3 classes) and SODA-VOC (top-5 / mid-5 / bottom-5 by frequency) |
| [`paper/generate_condition_pies.py`](paper/generate_condition_pies.py) | `fig_condition_share.{pdf,png}` (3 pies) + `condition_share_counts.csv` | Per-dataset pie of condition share, intensities merged, CS train+test merged |
| [`paper/generate_condition_pies_dino_filtered.py`](paper/generate_condition_pies_dino_filtered.py) | `fig_condition_share_dino_filtered.{pdf,png}` (loose) + `fig_condition_share_dino_strict.{pdf,png}` (5-tier sweep) + sweep CSV | Effect of DINOv3 filter `light≥{0.70,0.80,0.85,0.90}` / `heavy≥{0.60,0.70,0.75,0.80}` on the pie. Pass rate measured on CS test split (only split with DINO scores), extrapolated to other splits — caveat in the suptitle |
| [`paper/generate_dataset_overview_figure.py`](paper/generate_dataset_overview_figure.py) | `fig_dataset_overview.{pdf,png}` | Single combined figure (3 pies + 4 bar panels) ready for paper inclusion via `\includegraphics` |

**Row-count cache.** [`paper/figures/arrow_row_counts.json`](paper/figures/arrow_row_counts.json) caches `num_rows` per arrow file (~50 GB combined). The pie scripts read cache first → skip the full disk scan on re-runs. To force re-count: delete the entry (or the whole file).

**SODA-KTSH "Original" bug fix.** First pie pass mistakenly used `SODA_KTSH_ORIGINAL_TOTAL = 1500` (the row count of `soda_ktsh/night/soda_ktsh_day2night.arrow`, which is only a subset). Per `data_card.md` 2026-04-25 entry, the canonical SODA-KTSH source corpus = **9,988**. Verified by inspecting unique `image_id`s in `soda_ktsh/rain_snow/diffusion/snow_light.arrow` (9,671 unique ids ranging up to `ktsh13999`, all ≤ 9,988). After fix, KTSH pie: Original 30.8% / Rain 13.0% / Snow 33.0% / Fog 13.9% / Night 4.6% / Small 4.6%.

**Disk vs paper Table 1 — three discrepancies discovered.** Direct row-count of the Arrow tables under `augmentation_data/` no longer agrees with [`paper/main.tex`](paper/main.tex) Table 1 (`tab:dataset-overview`):

| Condition | Paper Table 1 | Disk now (row-count) | Cause |
|---|---:|---:|---|
| CS Snow (light+heavy) | 12,699 | **15,704** | `construction_site/rain_snow/diffusion/train/snow_heavy/snow_heavy.arrow` (rows=6,009) was generated **2026-04-27 14:04** — appeared after the 2026-04-25 verification entry (which explicitly noted "snow_heavy train deferred"). With train_snow_heavy now on disk, total = 9,695 light + 6,009 heavy = 15,704. The DINO csv `ip2p_snow_heavy.csv` still has 3,004 entries that map to a planned test_snow_heavy (no test arrow exists on disk). |
| CS Small | 1,323 (test only) | **2,823** | `small/train/small_constructionsite_train.arrow` (rows=1,500) is on disk and was not counted in Table 1. |
| SODA-VOC Snow (light+heavy) | 19,623 | **20,623** | Paper counts only snow_light (19,623); snow_heavy 1,000-row subset was either missed or treated as a paired sub-sample. |

**Action items:**
- [ ] Reconcile [`paper/main.tex`](paper/main.tex) Table 1 line ≈234–238 with the disk numbers. Decide whether to (a) update the table, or (b) state explicitly which rows are excluded ("snow_heavy train deferred", "small only test", "snow_heavy paired subset of light").
- [ ] If the figure is added to the paper, update Section 3 (Data Records) caption to reference `fig_dataset_overview` and re-state totals to match the disk.

**SODA-VOC Small × 3 projection.** User flagged that 2,000 additional `small` SODA-VOC samples are in flight (from 1,000 → 3,000 target). The pie + class-distribution scripts apply a uniform ×3 multiplier on the SODA-VOC Small bucket (constant `SODA_PROJECT['Small'] = 3.0` in [`paper/generate_class_statistics.py`](paper/generate_class_statistics.py); cache override `soda_voc/small/soda_small.arrow: 3000` in `arrow_row_counts.json`). When the augmentation job completes, **revert both overrides** so the scripts re-measure from the actual arrow.

**DINO filter pass rates (CS test split, used for the threshold-sweep extrapolation):**

| Tier | thr light/heavy | rain_l | rain_h | snow_l | snow_h |
|---|---|---:|---:|---:|---:|
| LOOSE   | 0.70/0.60 | 94.3% | 88.4% | 99.6% | 92.1% |
| MID     | 0.80/0.70 | 85.4% | 73.8% | 98.1% | 82.4% |
| STRICT  | 0.85/0.75 | 74.3% | 62.3% | 94.4% | 73.4% |
| V_STRICT| 0.90/0.80 | 50.5% | 44.6% | 81.7% | 59.5% |

STRICT (0.85/0.75) matches the threshold pair used in the Kaggle public release per `dataset_card.md` v6. Extrapolating those rates to CS train + SODA-VOC + SODA-KTSH (no DINO scores there yet) is a simulation, not a measurement — caveat is rendered in the suptitle of `fig_condition_share_dino_filtered.png`.

---

## 2026-04-25

### Dataset statistics — full direct verification + paper Table 1 + .md sync

Resolved §3.3 Dataset Statistics in `paper/main.tex` (3 `\verify{...}` placeholders) by direct row-count of every Arrow file under `augmentation_data/` plus the upstream sources. Method: pyarrow `open_file`/`open_stream` over each `.arrow`, plus `find … | wc -l` over SODA `JPEGImages/`/`Annotations/`.

**Verified-from-source numbers** (no longer transitive through docs):

| Quantity | Source | Count |
|---|---|---:|
| CS10k train | upstream HF `construction_site-train-{00000,00001}-of-00002.arrow` (3,500 + 3,509) | 7,009 |
| CS10k test | upstream HF `construction_site-test.arrow` | 3,004 |
| CS10k total | sum | 10,013 |
| SODA-VOC | `SODA VOCdevkit/.../JPEGImages/*.jpg` (matched by 19,846 XML annotations) | 19,846 |
| SODA-KTSH | `soda-ktsh/images/*.jpg` | 9,988 |

**Augmented row counts** (all counted from `augmentation_data/`):
- CS rain: test 1,652 + train 3,627 = 5,279; rain_heavy paired = 5,279 → rain total 10,558
- CS snow: snow_light test 2,940 + snow_heavy test 3,004 + snow_light train 6,755 = 12,699 (snow_heavy train pending)
- CS fog (heavy/medium/light): 1,001 + 1,001 + 1,002 = 3,004
- CS night: 3,004 (test); CS night_weather: rain_night 3,004 + snow_night 3,004 = 6,008
- CS small: 1,323
- SODA-VOC rain 4,790 + rain_heavy 4,790 = 9,580; snow 19,623; fog 3 × 1,000 = 3,000; night 19,846; small 1,000
- SODA-KTSH: rain 2,112 + rain_heavy 2,112 + snow 9,671 = 13,895

**Totals:** CS = 46,609 (10,013 baseline + 36,596 augmented). SODA-VOC = 72,895 (19,846 + 53,049). Combined main dataset = **119,504**. SODA-KTSH (13,895) released alongside but excluded from combined total to avoid double-counting source images.

**Files updated:**
- [`paper/main.tex`](paper/main.tex) §3.3: replaced `\verify{actual count}`, `\verify{recompute}` × 2 with verified numbers; rewrote Table 1 with all 7 condition rows; added two prose paragraphs covering scoping conventions, light/heavy pairing, SODA 3,000-subset note, NST exclusion. Caption simplified, removed test/train breakdown in cells per author preference.
- [`generation/AUGMENTATION_REPORT.md`](generation/AUGMENTATION_REPORT.md): resolved "Pending" entries — §3.4 SODA night = 19,846; §4.5 night_rain/snow = 3,004 each on CS test (SODA-VOC night_weather still Pending); §1 base-datasets table now shows full CS train+test split and adds SODA-KTSH; §8 Construction Site Main Pipeline table replaced with verified per-split counts (subtotal 36,596 augmented + 46,609 incl. baseline); §8 SODA section split into SODA-VOC table (53,049 / 72,895) and SODA-KTSH table (13,895).
- [`data_card.md`](data_card.md): split provenance footnote on Train/test splits to cite direct verification method; added "Full released-dataset row counts" table mirroring paper Table 1, with combined-total caveat about SODA-KTSH and NST archive exclusions.

**Outstanding numbers still marked pending in code/docs (do NOT report as final):**
- CS snow_heavy train split (only test 3,004 generated; train deferred per `data_card.md` 2026-04-21 entry)
- SODA-VOC night_weather (rain_night, snow_night) — pipeline not yet run on SODA
- Abstract `\verify{post-NST count, $\sim$68k}` in [`paper/main.tex`](paper/main.tex) line 72 — actual is 119,504 (incl. baseline) or 89,645 (augmented only); awaiting author decision on which framing to use in abstract before resolving.

---

## 2026-04-22

### Style-transfer ablation status — .md system consolidation

User đã quyết định dứt khoát (confirmed today): **VGG Neural Style Transfer CHỈ dùng cho ablation study, KHÔNG còn là pipeline chính thức.** Quyết định gốc ghi trong entry 2026-04-21 ("Decision: Move VGG neural style transfer rain/snow to ablation archive") nhưng một số .md vẫn present NST song song với IP2P như peer methods → gây nhầm lẫn. Hôm nay consolidation:

**Files updated**:
- [`CLAUDE.md`](CLAUDE.md): Thêm section "Scope decision — Main pipeline vs Ablation" ngay sau mục tiêu nghiên cứu. Liệt kê hệ quả thực tế cho agents khi đọc repo.
- [`docs/methods.md`](docs/methods.md) §1.2: Đổi tiêu đề "Weather Augmentation — 2 Methods" → "Main Pipeline (IP2P) + Ablation Archive (Legacy NST)". Method 1 NST được gắn nhãn `(ABLATION-ONLY)`, Method 2 IP2P gắn nhãn `(MAIN PIPELINE)`. TODO items sensitivity analysis cho style_weight → marked deferred/ablation-only.
- [`docs/methods.md`](docs/methods.md) §1.4 SSIM/LPIPS filter table: làm rõ threshold áp dụng cho main IP2P, ST dùng cùng threshold nhưng là ablation.
- [`docs/plan.md`](docs/plan.md): Phase 0.1, 0.2, 0.3 (style_weight, steps, MiDaS params) marked DEFERRED ablation-only. Phase 4.3 (Gatys verification) marked deferred. Phase 5.3/5.5 reframed: ST vs IP2P comparison là ablation/supplementary, không dùng để chọn method (đã chốt IP2P).
- [`docs/paper_background_summary.md`](docs/paper_background_summary.md): Rewrite Background paragraph về synthetic augmentation (NST đứng sau diffusion+inpainting, gắn nhãn "retained as ablation baseline"). Summary bullet "Weather augmentation 57,435 images NST" → "Main pipeline IP2P + physics + fog; NST retained as ablation (~57K archived)". Đánh dấu tất cả row counts cũ (85,613 / 10,266 / 47,169) là OUTDATED, cần recompute từ `augmentation_data/` ground truth.
- [`docs/checklist.md`](docs/checklist.md) CRITICAL section: style_weight/steps mismatch → marked resolved (ablation-only). MiDaS params verification → deferred. IP2P guidance params remain as main TODO.
- [`generation/AUGMENTATION_REPORT.md`](generation/AUGMENTATION_REPORT.md): Preamble scope note. §2 title "Weather Augmentation (Neural Style Transfer)" → "Weather Augmentation — Ablation Baseline (Legacy Neural Style Transfer)", với status banner. §8 Summary split thành "Main Pipeline" tables (CS + SODA) vs "Ablation Archive" tables, tránh gộp nhầm row counts. §9 Design Decision #1 "Style transfer over GAN" → "IP2P diffusion over style transfer (2026-04-21)" với justification (texture fidelity 0.86 vs 0.00, ít hallucinate hơn).

**Không update**: `README.md`, `dataset_card.md`, `data_card.md`, `docs/architecture.md`, `experiments/ablation_style_transfer/README.md` — đã phản ánh đúng trạng thái từ 2026-04-21.

**Non-goal**: Không xoá code `generation/weather/rain_snow/style_transfer/` và không xoá data `experiments/ablation_style_transfer/` — cần giữ để regenerate được ablation results khi reviewer yêu cầu.

### .md system audit + Batch A fixes (+ Batch A' follow-up gaps)

Comprehensive audit of 28 .md files vs paper/main.tex + code + `augmentation_data/` ground truth. Full findings in [`docs/md_audit_2026-04-21.md`](docs/md_audit_2026-04-21.md) — 28 findings (5 CRITICAL, 8 MAJOR, 8 MEDIUM, 5 MINOR, 2 STYLE).

**Batch A (11 findings fixed, safe, no narrative decision needed):**
- Deleted obsolete simulator artifacts (`generation/simulation_engine.md`, `scenario_templates/`, `parameter_ranges.json`) — no code references found
- Fixed §5 Outpainting sub-numbering `4.x → 5.x` in `AUGMENTATION_REPORT.md`
- Renamed duplicate `Approach 6: Human Perceptual Validation` → `Approach 8` in `docs/checklist.md`
- Renamed duplicate `§4B.5 Files Created` → `§4B.7` in `docs/plan.md`
- Fixed `taxonomy/extreme_conditions_definition.md`: `ConstructionCV-ExtremeConditions` → `ConSynth-X`
- Updated `CLAUDE.md` paper pipeline list to include `fog`
- Added vendored-copy banner to `generation/weather/libs/Weather_Effect_Generator/README.md`
- Resolved ACDC self-contradiction in `docs/data_sources.md` (§7.4 deleted) and `docs/literature.md` §2.8 (ABANDONED → VERIFIED primary reference)
- Updated `docs/data_sources.md` §3.1 path for style transfer to `experiments/ablation_style_transfer/`
- Removed `parameter_ranges.json` reference from `AUGMENTATION_REPORT.md:408`

**C1 decision — Option B chosen by user**: Rewrite ST as ablation-only in main paper. DEVLOG 2026-04-21 warned this requires either regenerating IP2P at multiple intensity levels or justifying methodology asymmetry. User generated `rain_heavy` on 2026-04-21 — but as physics-only overlay (not diffusion-heavy). Asymmetry with snow (which has 2 diffusion-based levels) needs paper framing when Batch B narrative rewrite happens.

**Batch A' (9 gaps) — triggered by rain_heavy + Kaggle update not propagated to docs:**
- `dataset_card.md`: Rewrote Kaggle sample table with accurate row counts (300 not 100 for DINO-filtered rain files); added v3 + v6 version history entries
- `README.md`: Replaced "Three pipelines" line with detailed per-condition list showing rain has 2 intensities (IP2P light + physics heavy)
- `augmentation_data/README.md`: Updated layout tree with `rain_heavy/`, added row-count table entries, rewrote Provenance section with full rain_heavy physics spec
- `generation/AUGMENTATION_REPORT.md` §1: Rewrote overview condition table — rain light/heavy, snow light/heavy, fog 3 zones, night, compound night_rain/night_snow, small, and style_transfer as ablation. Still needs §2 rewrite in Batch B.
- `docs/architecture.md`: Updated Augmentation Conditions table with rain 2-level split
- `docs/methods.md`: Added new subsection "IP2P Rain — 2 Intensity Variants (2026-04-21)" with full technical spec: pilot history v1-v5, physics config (fog 0.30-0.45, 3 streak layers with n/length/alpha per layer, Gaussian blur σ=1.3), production worker details (deterministic seed, CPU-only, metadata preservation), row counts, post-hoc DINO/SSIM metrics, asymmetry-with-snow justification
- `docs/plan.md`: Added Phase 5B "Rain_heavy Re-validation" with 11 subtasks (4 done, 7 pending SLURM submission)
- `docs/literature.md` §1.7: Linked Tremblay et al. (2021) to rain_heavy physics-overlay design decision; added Gupta et al. Weather_Effect_Generator citation

### SLURM runbook + validation script patches for rain_heavy

Patched 7 validation script registries to add `ip2p_rain_heavy` / `diffusion_rain_heavy` entries pointing to `augmentation_data/construction_site/rain_snow/diffusion/test/rain_heavy/`:
- `validation/extract_dino_ssim_all.py`
- `validation/compute_relative_mahalanobis.py`
- `validation/weather_classifier.py`
- `validation/compute_fid_kid.py`
- `validation/compute_texture_fidelity.py`
- `validation/vlm_jury/data_loader.py`
- `validation/make_retention_charts_all.py` (also relabeled existing `ip2p_rain` → "IP2P Rain (light)")

Created 6 SLURM scripts in `jobs/`:
- `extract_dino_ssim_rain_heavy.sh` (~25 min, `vlm-new` env for DINOv3)
- `rerun_weather_cls_rain_heavy.sh`, `rerun_fid_kid_rain_heavy.sh`, `rerun_mahalanobis_rain_heavy.sh`, `rerun_texture_fidelity_rain_heavy.sh`
- `rerun_vlm_jury_rain_heavy.sh` (wrapper submitting 3 judge jobs)

Plus master runbook [`docs/rain_heavy_revalidation_runbook.md`](docs/rain_heavy_revalidation_runbook.md) with submission order, expected metric outcomes, and stale-path flag for 4 scripts whose `diffusion_rain` entries still point to pre-2026-04-20 `$CONSYNTH_DATA_ROOT/output/construction_site_test/diffusion_rain_heavy/` (separate maintenance issue, not addressed).

### Rain_heavy DINO extraction + Kaggle v6 upload

**Chained SLURM job** `jobs/rain_heavy_dino_and_filter.sh` (job 5025824, completed in 5 min; first attempt 5023889 failed because `CONSYNTH_DATA_ROOT` env var not inherited in Bash-tool-spawned sbatch — script now exports it explicitly):

- Step 1 — DINO/SSIM extraction:
  - `validation/results/dino_ssim/ip2p_rain_heavy.csv` (1,652 rows, DINO mean **0.754**, SSIM mean **0.522**, min/max DINO 0.103/0.953)
  - Extract skipped existing CSVs (logic in `extract_dino_ssim_all.py:147`) → only computed rain_heavy
- Step 2 — Filter 300 samples per user thresholds:
  - `cs_diff_rain_test.arrow`: 300 rows, 110 MB (from 1,515 eligible at DINO≥0.75, 91.7% of light)
  - `cs_diff_rain_heavy_test.arrow`: 300 rows, 81 MB (from 1,219 eligible at DINO≥0.70, 73.8% of heavy)
  - Script: [`scripts/update_kaggle_rain_dino_filtered.py`](scripts/update_kaggle_rain_dino_filtered.py) (deterministic seed 42)
- Step 3 — Kaggle upload v6: `kaggle datasets version` uploaded all 23 Arrow files (CLI snapshots the whole directory; unchanged files re-uploaded). Status: `ready` as of 17:07 UTC 2026-04-22.

**Observation**: Rain_heavy DINO mean 0.754 vs rain light 0.871 (drop ~13%). SSIM drops more sharply (0.522 vs 0.756, drop ~31%). Consistent with physics overlay + Gaussian blur affecting pixel-level structure more than DINO's semantic features. Empirical confirmation that DINO threshold should be looser for heavy (0.70) than light (0.75).

**Backups**: All validation result dirs snapshot-copied before re-validation runs:
`validation/results/{weather_cls,fid_kid,relative_mahalanobis,texture_fidelity,vlm_jury,dino_ssim}_backup_20260421_2329/`.

---

## 2026-04-21

### Decision: Rain 2-level scheme — heavy = physics-only overlay on IP2P light (no diffusion)

**Quyết định**: Rain có 2 intensity levels. **light = existing IP2P output (unchanged)**;
**heavy = apply `add_natural_rain(intensity='heavy_fog')` on top of light** — pure
CPU physics overlay, no additional diffusion pass.

**Lý do**:
- Pilot v1 (g=8 vs g=12 same prompt): SSIM giữa 2 level chỉ khác ~9% → không phân biệt được bằng mắt.
- Pilot v2 (3-level prompt-driven theo test_snow_stronger.py: light/medium/heavy với prompt leo thang): "torrential downpour" prompt ở heavy bị hallucination cao, fail nhiều.
- Pilot v3 (2-level g=8 light + g=10 heavy với physics overlay parametrized): span SSIM 0.17, nhưng heavy nhìn chưa đủ mạnh.
- Pilot v4 (g=11 + fog 0.30-0.45 + Gaussian blur σ=1.3): span 0.30, tốt hơn nhưng grain chưa rõ.
- Pilot v5 (thêm density 1.7× + alpha 0.35-0.55): span 0.42, quá nhiều thay đổi từ diffusion khi combine.
- Chốt: **bỏ diffusion cho heavy**, chỉ dùng physics-only trên light → deterministic, không fail, không tốn GPU, dễ reproduce.

**Physics config (`heavy_fog` intensity)**:
- Fog haze strength: uniform(0.30, 0.45)
- 3 streak layers: n∈[3000,4500]/[2200,3600]/[900,1600], length 12-60px, thick 1-3, alpha 0.35-0.55
- Post-processing: Gaussian blur kernel 5×5, σ=1.3 (reduced-visibility effect)
- Backward compatible: `add_natural_rain(image)` hoặc `intensity='heavy'` giữ nguyên behavior production v4.

**Code**:
- [generation/weather/rain_snow/diffusion/physics.py](generation/weather/rain_snow/diffusion/physics.py) — thêm `intensity: str = 'heavy'` param với 3 mode (`heavy` default, `light`, `heavy_fog`).
- [generation/weather/rain_snow/diffusion/apply_heavy_physics_to_light.py](generation/weather/rain_snow/diffusion/apply_heavy_physics_to_light.py) — production worker CPU-only.
- Pilot scripts: [test_rain_intensity.py](generation/weather/rain_snow/diffusion/test_rain_intensity.py) (v2-v5, 4-col grids dưới `validation/results/rain_intensity_test_v{2..5}/`); [test_rain_heavy_physics_only.py](generation/weather/rain_snow/diffusion/test_rain_heavy_physics_only.py) (pilot cuối).

**Production outputs** (4 targets, CPU only, ~50 min elapsed):
| Target | Rows | Size | Path |
|---|---|---|---|
| CS test | 1,652 (7 shards) | 424 MB | `augmentation_data/construction_site/rain_snow/diffusion/test/rain_heavy/` |
| CS train | 3,627 | 937 MB | `augmentation_data/construction_site/rain_snow/diffusion/train/rain_heavy/train_rain_heavy.arrow` |
| SODA VOC | 4,790 | 2.6 GB | `augmentation_data/soda_voc/rain_snow/diffusion/rain_heavy.arrow` |
| SODA KTSH | 2,112 | 608 MB | `augmentation_data/soda_ktsh/rain_snow/diffusion/rain_heavy.arrow` |

**Schema intensity hiện tại toàn dataset**:
- Rain: light (IP2P g=10 mặc định) + heavy (light + heavy_fog physics) — 2 levels.
- Snow: light (IP2P g=8) + heavy (IP2P g=12, Apr 18 via `generate_snow_strong.sh`) — 2 levels, giữ nguyên.
- Fog: 3 zones heavy/medium/light via Koschmieder visibility — giữ nguyên.
- Night: single level — giữ nguyên.

**Kaggle update**: bổ sung 5 sample files (100 rows/file) vào [augmentation_data_sample/](augmentation_data_sample/),
upload version mới lên `viethuyduong/consynth-x-augmentation-sample`:
- `cs_diff_rain_heavy_{test,train}.arrow` (paired image_ids với existing light samples)
- `soda_voc_diff_rain.arrow` (NEW light baseline, chọn ids ∈ `soda_voc_original.arrow` để không cần repack)
- `soda_voc_diff_rain_heavy.arrow`
- `soda_ktsh_diff_rain_heavy.arrow`

Script: [scripts/update_kaggle_rain_heavy.py](scripts/update_kaggle_rain_heavy.py).

---

### Decision: Move VGG neural style transfer rain/snow to ablation archive

**Quyết định**: Loại bỏ VGG neural style transfer khỏi main weather augmentation
pipeline của ConSynth-X; IP2P diffusion trở thành phương pháp weather duy nhất
cho rain/snow trong dataset chính. Style transfer output được **giữ lại** như
ablation archive để so sánh baseline và đảm bảo reproducibility.

**Lý do**: IP2P đạt texture fidelity và realism cao hơn (xem `paper/main.tex` §4:
unanimous belief fusion $H{=}0.86$, FID/KID thấp hơn). Tinh giản dataset thành
một pipeline weather nhất quán, giảm dung lượng main release ~43 GB.

**Thao tác filesystem** (mv trên cùng NFS mount — rename atomic, non-destructive):
- `augmentation_data/soda_voc/rain_snow/style_transfer/` (37 GB)
  → `experiments/ablation_style_transfer/soda_voc/rain_snow/style_transfer/`
- `augmentation_data/construction_site/rain_snow/style_transfer/` (5.9 GB)
  → `experiments/ablation_style_transfer/construction_site/rain_snow/style_transfer/`
- `soda_ktsh` không có style_transfer → không đổi.
- Generation code tại `generation/weather/rain_snow/style_transfer/` **không**
  di chuyển (giữ để regenerate khi cần).

**Code/doc đã cập nhật**:
- `experiments/ablation_style_transfer/README.md` — mới, giải thích lý do và
  layout archive.
- `slide/gen_all_conditions_grid.py` — path `CS_AUG/rain_snow/style_transfer/...`
  → biến mới `CS_ST_ABLATION` trỏ vào archive.
- `generation/utils/pack_soda_voc_to_arrow.py` — docstring usage example trỏ
  vào archive path.
- `README.md` — "Four synthetic pipelines" → "Three" + thêm dòng về ablation
  archive; bảng repository layout thêm `experiments/`.
- `data_card.md` — thêm dòng giải thích main dataset dùng IP2P cho rain/snow,
  style transfer nằm ở ablation archive.

**Chưa cập nhật (flag tới maintainer)**:
- `paper/main.tex` — vẫn trình bày Neural Style Transfer là **Method 1**
  (§3.1.1), kèm bảng "Weather (style): 47,169 images", và framing
  "complementary pipelines" ở §5. Thay đổi paper narrative vượt quá scope
  của operation này; cần quyết định research-integrity:
  1. Giữ paper như cũ, trình bày cả hai pipeline, chỉ main release dùng IP2P.
  2. Viết lại §3.1 + §4 để định vị style transfer là ablation từ đầu
     → nếu chọn hướng này phải báo cáo lại DINO/SSIM/FID có-và-không-style-transfer
     để tránh "justify ngược" (vi phạm research integrity rule #2 trong CLAUDE.md).

**Cảnh báo methodology**: IP2P hiện chỉ có **1 mức cường độ** (`rain.arrow`,
`snow.arrow`), trong khi style transfer có 3 mức (`rain_{0,1,2}.arrow`). Nếu
paper báo cáo intensity-based exposure cho rain/snow, cần regenerate IP2P ở
nhiều mức guidance_scale trước khi coi vấn đề này là closed (consistent
methodology across hazards — per memory `feedback_research_integrity.md`).

---

## 2026-04-20

### Fix: DINO/SSIM for IP2P conditions was pointed at pre-v4 data

**Phát hiện**: CSV `validation/results/dino_ssim/ip2p_{rain,snow_light,snow_heavy}.csv`
và chart `dino_ssim_retention_all.{png,pdf}` + grid `rain_heavy_dino075_grid.jpg`
được tính trên output IP2P cũ (30/3/2026, pre-v4, IP2P-only không physics overlay,
không LPIPS filter) tại `ConstructionSite/output/construction_site_test/diffusion_{rain,snow}_*/`.

Kết quả: rain grid không thấy rain streaks, mean DINO rain = 0.82 (quá cao,
do IP2P-only chỉ edit "wet ambient" rất nhẹ) → nhầm tưởng data load sai.

**Thật ra** v4 best config (IP2P + physics fog + rain streaks + LPIPS<0.35 filter,
config trong `ConstructionSite/weather_aug/rain_snow/test_ip2p_v3.py` +
`batch_worker_v4.py`) đã chạy xong 31/3/2026 và output ở
`augmentation_data/construction_site/rain_snow/diffusion/test/{rain,snow_heavy,snow_light}/`
(7 batch files, naming `batch_0-500.arrow` etc. khớp SLURM job v4_rain_*_0331_1107).

**Fix**:
- `validation/extract_dino_ssim_all.py`: đổi `IP2P_DATA` sang
  `CONSYNTHX/augmentation_data/construction_site/rain_snow/diffusion/test/`;
  `ip2p_rain` → `rain/`, `ip2p_snow_*` → `snow_{heavy,light}/`.
- `validation/make_dino_threshold_grid_rain.py`: cập nhật `AUG_DIR` cùng path.
- Xóa 3 CSV cũ + rain grid cũ + ip2p_rain_20samples.jpg.
- Submit SLURM `validation/jobs/rerun_ip2p_v4_dino_ssim.sh` (job 5009442,
  FAILED — env `VLM` có transformers 4.49.0 không nhận DINOv3) → re-submit
  job 5009583 với env `vlm-new` (transformers 5.3.0). Cập nhật
  `docs/infrastructure.md` ghi chú yêu cầu env.
- Job 5009583: step 1 (extract) thành công. Step 2 (charts) fail vì
  `vlm-new` thiếu matplotlib → chạy lại step 2+3 bằng env `VLM` trên node
  login. 3 CSV + 2 chart + 1 grid đã cập nhật với v4 data.

**Kết quả v4 (IP2P + physics + LPIPS filter):**

| Condition | N pairs | DINO mean | SSIM mean |
|---|---|---|---|
| IP2P Rain v4 | 1652 (từ 3004, ~45% drop) | 0.871 | 0.756 |
| IP2P Snow light v4 | 2940 | 0.927 | 0.723 |
| IP2P Snow heavy v4 | 3004 | 0.790 | 0.598 |

Rain grid v4: Above DINO≥0.75 = 1515 (91.7%), Below = 137 (8.3%) → rain
streaks vật lý hiển thị rõ trong cả 2 nhóm sample.

**Paper update (`paper/main.tex`, `paper/figures/`):**
- Copy v4 retention chart + table → `fig5_retention_curves_all.pdf`,
  `fig6_retention_table.pdf`.
- Sửa đoạn "Retention characteristics across all augmentation conditions"
  (Section 3): pattern cũ nói IP2P diffusion nói chung `≤ 77% at DINO ≥ 0.80`
  không còn đúng với v4 (IP2P rain 85%, snow light 98%, do LPIPS filter giữ
  lại phần semantic-preserving). Rewrite để phân biệt IP2P rain/snow-light
  (filter giữ `>85%`) vs snow-heavy/night (aggressive edits, `59%`/`77%`).
- Sửa SSIM mean snow_light: `0.76 → 0.72` (line 138).
- Các số khác ở fig4 caption vẫn đúng (73% DINO≥0.75, 84% SSIM≥0.5 cho
  snow_heavy N=3004) vì v4 snow_heavy không có LPIPS filter → số liệu giữ nguyên.
- Rebuild `main.pdf` (25 pages, 10.6MB).

**Quan sát**: DINO mean v4 rain (0.871) không thấp hơn pre-v4 (0.82) dù
có rain streaks, vì (a) LPIPS filter đã loại 45% ảnh "too different", (b)
DINOv3 encode semantic content → thin rain streaks overlay ít ảnh hưởng embedding.
SSIM (pixel-level) phản ánh tốt hơn: v4 rain SSIM 0.756 (tight vì filter giữ
ảnh semantic-preserved). IP2P Snow heavy có DINO/SSIM thấp nhất trong nhóm
IP2P (0.790/0.598) — biến đổi mạnh nhất, snow phủ white overlay diện rộng.

**Bài học**: khi registry trong extraction script trỏ đến thư mục có tên
trông-hợp-lý (`diffusion_rain_heavy`) mà KHÔNG phải output pipeline mới nhất →
kết quả evaluation sai lệch mà không có warning. Cần
sanity-check mean DINO sau mỗi lần gen (v4 rain expected ~0.5-0.7 do physics
overlay, pre-v4 ~0.82).

### Self-contained refactor for DSA Best Data Award submission

Restructured repository so that reviewers can clone ConSynth-X and run the
generation + evaluation pipelines without depending on the adjacent
`ConstructionSite/` working tree.

Changes:
- **Vendored** `Weather_Effect_Generator` into `generation/weather/libs/Weather_Effect_Generator/`
  (upstream `hgupta01/Weather_Effect_Generator@7d62b67`, Apache-2.0) with
  first-party additions: `rain_pipeline.py`, `snow_pipeline.py`,
  `weather_pipeline.py`, `style_transfer.py`, modified `lib/style_transfer_utils.py`
  and `Snow_Effect_Generator.py`. Attribution + provenance: `libs/Weather_Effect_Generator/NOTICE.md`.
  Removed duplicated copies previously sitting next to the worker scripts in
  `rain_snow/style_transfer/`.
- **Added** `img2img-turbo` as a pinned git submodule at
  `generation/day2night/img2img-turbo/` (`GaParmar/img2img-turbo@86f5414`, MIT).
- **Centralised** path resolution via `generation/_paths.py`, `.env.example` and
  environment variables `CONSYNTH_REPO_ROOT`, `CONSYNTH_DATA_ROOT`,
  `CONSYNTH_CONDA_ENV`, `CONSYNTH_WEIGHTS_URL` / `CONSYNTH_WEIGHTS_SRC`.
  Replaced hardcoded `/users/PGS0407/...` references in `generation/`,
  `benchmarks/`, `validation/`, `experiments/`, `paper/`, `slide/`,
  `human_validation/` and `benchmarks/vlm/configs/base.yaml`.
- **Weights manifest** at `weights/` with `checksums.sha256` (`day2night.pkl`,
  `rain_vgg_512.pth`, `snow_vgg_512.pth`). `download.sh` pulls each file
  directly from its **original upstream host** (CMU for `day2night.pkl`;
  Weather_Effect_Generator Google Drive for the VGG `.pth` files — we are
  not the model authors and do not re-host). It then verifies SHA256 and
  symlinks the files to the paths each pipeline expects. A local mirror
  override (`CONSYNTH_WEIGHTS_SRC`) is supported for offline clusters.
- **Production IP2P worker**: verified `generation/weather/rain_snow/diffusion/batch_worker.py`
  is the canonical version (newer than `ConstructionSite/weather_aug/rain_snow/batch_worker_v4.py`:
  imports `physics` instead of `test_ip2p_v3`, has CLI overrides, `--no-filter`,
  and the v2 prompt without "wet muddy ground"). No copy needed.

---

## 2026-04-20

### Kaggle Sample Dataset v2 — Add IP2P Snow Heavy

**Action**: Updated public sample dataset at https://www.kaggle.com/datasets/viethuyduong/consynth-x-augmentation-sample with new IP2P snow heavy variant.

**Changes**:
- Added: `cs_diff_snow_heavy_test.arrow` — 100 samples from new `diffusion_snow_heavy` set (gs=12, 2026-04-18, no-filter)
- Renamed: `cs_diff_snow_test.arrow` → `cs_diff_snow_light_test.arrow` to disambiguate from heavy variant
- Upload size delta: +28 MB (heavy) vs −38 MB (light renamed)

**Upload script**: `scripts/update_kaggle_sample.py` (reads schema from existing sample, samples 100 deterministic via seed=42, writes with matching columns)

**Command used**:
```bash
cd augmentation_data_sample
kaggle datasets version -m "Add IP2P snow heavy variant (gs=12); rename old snow test to snow_light"
```

**Why this matters**: Reviewers can now compare both snow intensity variants side-by-side without downloading the full dataset. Aligns public sample with paper's IP2P snow_heavy introduction (Section 2.2.2).

---

## 2026-04-19

### Full Retention Analysis: DINO + SSIM Across 13 Augmentation Conditions

**Mục tiêu**: Đánh giá retention rate của tất cả augmentation variants dưới DINO similarity và SSIM thresholds, để inform filter selection.

#### Pipeline:
1. Extract DINOv3 ViT-L/16 CLS embeddings cho 3004 ảnh gốc + tất cả augmented conditions
2. Compute cosine similarity (original vs augmented)
3. Compute SSIM (256x256 resized) per-pair
4. Generate retention curves + heatmap

#### Files:
- Script: `validation/extract_dino_ssim_all.py`, `validation/make_retention_charts_all.py`
- Data: `validation/results/dino_ssim/*.csv` (13 conditions)
- Charts: `validation/results/dino_ssim_retention_all.{pdf,png}`, `dino_ssim_retention_table.{pdf,png}`
- Paper: `paper/figures/fig4_snow_heavy_retention.{pdf,png}`, `fig5_retention_curves_all.pdf`

#### Summary (DINO mean / SSIM mean per condition):

| Condition | N | DINO mean | SSIM mean | Retention@DINO≥0.75 |
|---|---|---|---|---|
| ST Rain A | 2211 | 0.919 | 0.821 | **98%** |
| ST Rain B | 1243 | 0.886 | 0.796 | 93% |
| ST Rain C | 1208 | 0.898 | 0.809 | 95% |
| ST Snow A | 2159 | 0.898 | 0.775 | 96% |
| ST Snow B | 1649 | 0.859 | 0.687 | 91% |
| ST Snow C | 1796 | 0.867 | 0.703 | 92% |
| IP2P Rain | 3004 | 0.800 | 0.591 | 74% |
| IP2P Snow (light) | 3004 | 0.774 | 0.590 | 67% |
| IP2P Snow (heavy) | 3004 | 0.790 | 0.598 | 73% |
| Fog (light) | 1002 | **0.938** | **0.788** | **99%** |
| Fog (medium) | 1001 | 0.918 | 0.719 | 99% |
| Fog (heavy) | 1001 | 0.868 | 0.617 | 93% |
| Night (CycleGAN) | 3004 | 0.841 | 0.390 | 88% |

#### Key findings:

1. **Style transfer + light fog preserve DINO structure rất tốt** (>90% retention ở DINO ≥ 0.80). Minimal semantic shift.
2. **IP2P diffusion + night aggressive global edits** → larger DINO distances (retention ≤77% at DINO ≥ 0.80).
3. **SSIM degrades faster than DINO** cho IP2P và night — pixel-level changes > structural feature changes. Consistent với DINO robustness to texture.
4. **DINO vs SSIM correlation moderate** (r=0.40 cho snow_heavy). Không thay thế được nhau → joint filtering more restrictive.

#### Implication for filter selection:

- **Mild edits** (fog_light, ST rain): SSIM filtering đủ
- **Aggressive edits** (IP2P, night): DINO-based filtering để tránh discard quá nhiều data
- **Không có threshold tối ưu chung** — phụ thuộc condition và downstream task

---

## 2026-04-18

### IP2P Snow Heavy Variant + VLM Jury Validation

**Motivation**: VLM Jury evaluation (2026-04-15) phát hiện IP2P snow light (gs=8) có acceptance rate rất thấp (InternVL 30%, Phi-4 8%, Qwen 4%). Cần variant mạnh hơn để snow effect visible.

#### Changes:
- **Heavy variant**: `guidance_scale=12.0`, `image_guidance_scale=1.2`
- **New prompt**: "a cold winter day with heavy snow, thick snow covering the ground and surfaces, grey overcast sky, snowfall"
- **No pre-filter** (`--no-filter`) — keep all 3004 images, user designs filter post-hoc
- Added `--guidance-scale`, `--image-guidance-scale`, `--prompt`, `--no-filter` CLI args to `batch_worker.py`

#### Structure changes:
- `ConstructionSite/output/.../diffusion_snow_heavy` (old) → renamed to `diffusion_snow_light`
- New `diffusion_snow_heavy` = strong config output (3004 unfiltered)
- `ConSynth-X/augmentation_data/.../snow/` → `snow_light/` + symlink `snow_heavy → ConstructionSite/...`
- Updated code references: `data_loader.py`, `compute_relative_mahalanobis.py`

#### VLM Jury on heavy variant (3 judges × 3004 images):

| Judge | Acceptance | vs light variant |
|---|---|---|
| InternVL | **89.3%** | +59% (from 30%) |
| Phi-4 | **92.3%** | +84% (from 8%) |
| Qwen | 8.9% | +5% (still strict) |

**Pattern**: Heavy variant dramatically cải thiện InternVL + Phi-4 acceptance. Qwen vẫn strict trên mọi snow variant (known bias, tương tự Gemini trong Ruck et al.).

#### Empirical observation: DINO threshold ≠ VLM Jury agreement

Tested: DINO ≥ 0.75 threshold on snow_heavy. VLM Jury agreement chỉ ~50-52% với DINO split (random level).

**Interpretation**: DINO đo structural preservation ở feature level, VLM Jury đánh giá visual realism + semantic coherence. Complementary metrics, không thay thế được.

**Conclusion (per research integrity rules)**: Không có empirical justification để dùng DINO ≥ 0.75 làm filter. Release both variants với per-image DINO + SSIM scores; user chọn threshold theo application.

---

## 2026-04-17

### VLM Jury Evaluation — Full Dataset

**Method**: 3 local VLM judges (Qwen2.5-VL-7B, InternVL2.5-8B, Phi-4-multimodal) đánh giá pair (original | augmented) → binary accept/reject. Following Ruck et al. (2026) methodology.

#### Results per condition (majority vote 2/3):

| Condition | Majority | ACDC baseline |
|---|---|---|
| Fog heavy | **98%** | 97.5% |
| IP2P Rain | 72% | 87.5% |
| ST Rain (A/B/C) | 70% | 87.5% |
| Night | 58% | 90.0% |
| ST Snow B | 10% | 87.5% |
| IP2P Snow light | 8% | 87.5% |

**Pattern**: Fog augmentation vượt cả real ACDC fog. Rain good. Night moderate. Snow weak (led to heavy variant introduction).

**Inter-judge agreement (Cohen's κ)**: 0.16-0.36 — moderate. InternVL lenient (78%), Qwen strict (57%), Phi-4 middle (57%).

---

## 2026-04-15

### DINOv2 → DINOv3 Migration

**Motivation**: Ruck et al. (2026) paper gốc dùng DINOv3, nhưng implementation ban đầu dùng DINOv2 (do transformers 4.49 không support DINOv3). Nâng cấp transformers → 5.5.4 để match paper methodology.

#### Changes:
- `validation/compute_relative_mahalanobis.py`: DINOv2Embedder → DINOv3Embedder
- `paper/main.tex`: All DINOv2 references → DINOv3
- `paper/references.bib`: Added Siméoni et al. 2025 DINOv3 citation
- `slide/slide_simple.tex`: Updated labels
- `jobs/relative_mahalanobis.sh`: Updated description

#### Results comparison (CLIP unchanged; DINOv3 new):

| Condition | DINOv2 gap closed | DINOv3 gap closed | Change |
|---|---|---|---|
| Night | 8.0% | 9.0% | +1% |
| Snow (ST) | 2.1% | 2.8% | +0.7% |
| Rain (ST) | 1.1% | 1.2% | +0.1% |
| Fog | 0.2% | 0.9% | +0.7% |

DINOv3 slightly higher than DINOv2 but **pattern identical** — cross-domain limitation (ACDC driving ≠ construction) persists regardless of embedding choice.

---

## 2026-04-12

### Cross-Validation Synthesis — Tổng hợp 5 Approaches

**Mục tiêu**: Nhìn xuyên suốt kết quả từ 5 validation approaches (UnivFD, FID/KID, Weather Classifier, Texture Fidelity, Relative Mahalanobis) để rút ra kết luận tổng thể về chất lượng bộ dữ liệu.

---

#### Finding 1: Hai pipeline augmentation bổ sung cho nhau, không thay thế

Đây là những gì data cho thấy — nhìn xuyên suốt tất cả metrics:

| Metric | IP2P Diffusion | Style Transfer |
|---|---|---|
| Texture fidelity (belief H) | **0.85-0.86** (faithful) | 0.00 (artifact/uncertain) |
| Weather classifier accuracy | 26-64% (yếu-trung bình) | 62-95% (trung bình-tốt) |
| UnivFD score gap | **0.027-0.037** (nhỏ nhất) | 0.081-0.128 |
| FID vs ACDC | 194-205 | 192-228 |

**Pattern**: IP2P Diffusion giữ texture gốc tốt (trông tự nhiên, detector không phân biệt được) nhưng hiệu ứng thời tiết nhẹ (classifier khó nhận ra). Style Transfer tạo hiệu ứng thời tiết rõ hơn (classifier nhận ra) nhưng texture bị thay đổi nhiều hơn.

**Interpretation**: Đây là trade-off **realism vs recognizability** — corroborates Ruck et al. (2026) observation. Hai pipeline bổ sung cho nhau: diffusion cho "subtle weather" và style transfer cho "obvious weather". Paper nên present cả hai và discuss trade-off thay vì chọn một.

---

#### Finding 2: Night augmentation (CycleGAN-Turbo) — strongest semantic shift, weakest texture

Đây là những gì data cho thấy:

| Metric | Night | Nhận xét |
|---|---|---|
| FID vs ACDC night | **178.9** (best, −34% vs baseline 270.5) | Semantic gần ảnh đêm thật nhất |
| Texture composite | **9.138** (worst, DCT_W=29.81) | Over-smoothed, mất HF content |
| Belief fusion conflict | **0.078** (highest) | LBP vs GLCM/DCT disagree |
| Mahalanobis gap closed | **38%** (best in CLIP) | Feature-space improvement lớn nhất |

**Pattern**: Night augmentation thành công ở mức semantic (FID, Mahalanobis) nhưng có texture artifact rõ. Belief fusion conflict cao nhất (0.078) cho thấy LBP (micro-texture local) vẫn OK nhưng GLCM/DCT (global texture + frequency) bị thay đổi mạnh.

**Interpretation**: CycleGAN-Turbo giỏi biến đổi tông màu, ánh sáng toàn cục nhưng "over-smooth" → mất natural noise/grain. Đây là hạn chế known của CycleGAN-based methods. Paper nên report as-is, không cần fix.

---

#### Finding 3: Diffusion Snow — realistic nhưng không "trông như tuyết"

Đây là những gì data cho thấy:

| Metric | Diffusion Snow | Style Snow (best) |
|---|---|---|
| Weather classifier | **25.7%** (49% → rain) | 95.0% |
| Texture fidelity (H) | **0.862** (faithful) | 0.00 |
| FID vs ACDC snow | **205.3** (< baseline 216.0) | 206.6-228.2 |
| UnivFD fooling rate | **100.0%** | 99.9% |

**Pattern**: Diffusion snow đạt điểm cao nhất ở texture fidelity VÀ FID gần real snow hơn baseline, nhưng weather classifier chỉ nhận ra 26%. Lưu ý: original construction images cũng chỉ 6% sun/clear → classifier có bias hệ thống trên domain construction.

**Interpretation (2 giả thuyết, chưa verify)**:
1. IP2P tạo atmospheric effect (trời xám, light diffuse) mà không tạo visible snow particles → classifier không thấy "snow" nhưng FID thấy domain shift phù hợp
2. Weather classifier bias trên construction images — cần visual inspection để phân biệt

**Action needed**: Visual inspection diffusion snow vs style snow để xác nhận giả thuyết nào đúng. KHÔNG nên dismiss 26% accuracy mà cũng KHÔNG nên dismiss 0.862 texture fidelity — report cả hai.

---

#### Finding 4: Augmentation giữ detection performance — nhưng chưa chứng minh improvement

Đây là những gì data cho thấy:

- Baseline (no aug): mAP50=0.562, mAP50-95=0.338
- Current (10K aug images): mAP50=0.547, mAP50-95=0.339
- Excavator: +1.0% AP50, Rebar: −3.6% AP50, Worker: −2.0% AP50

**Pattern**: mAP50-95 gần như không đổi (+0.001). mAP50 giảm nhẹ (−1.5%). Per-class cho thấy object lớn (excavator) hưởng lợi, object nhỏ/dày đặc (rebar, worker) giảm nhẹ.

**Critical caveat**: Val set chỉ có clean (clear-weather) images. Test này chỉ chứng minh augmentation **không gây hại** trên clean conditions, CHƯA chứng minh augmentation **cải thiện robustness** trên weather conditions. Để chứng minh giá trị thực sự của dataset, cần:
- Test detection trên ảnh weather/night (real hoặc augmented test set)
- Cross-condition evaluation matrix

---

#### Finding 5: VLM yếu trên construction domain — validate motivation cho Paper 2 & 3

Đây là những gì data cho thấy:

- Best VLM overall F1: InternVL2.5-26B = 0.124 (original), Qwen3-VL-8B IoU=0.298
- LLaVA-1.5-7B: 100% error rate (500/500 fail) trên mọi condition
- Extreme conditions giảm thêm: InternVL2.5 night F1=0.094 (−24% vs original)
- gpt-4o-mini: 45-54% error rate (API failures hoặc format errors)

**Pattern**: Tất cả VLM đều rất yếu (F1 < 0.13) ngay cả trên ảnh gốc. Extreme conditions làm giảm thêm. Model nhỏ (LLaVA 7B) hoàn toàn fail.

**Interpretation**: Construction site là domain niche chưa được VLM training tốt. Đây là motivation mạnh cho Paper 2 (benchmark) và Paper 3 (improvement).

---

#### Gaps còn thiếu cho Paper 1

Dựa trên kết quả hiện tại, 3 gaps chính:

1. **Detection under weather conditions** — CRITICAL cho paper
   - Hiện chỉ test trên clean val set → chỉ chứng minh "không gây hại"
   - Cần cross-condition detection (train on clean+aug → test on weather) để chứng minh dataset value
   
2. **SODA dataset validation** — chưa có
   - Tất cả validation results chỉ cho Construction Site dataset (3,004 images)
   - SODA (19,846 images) chưa có validation results → missing 2/3 dataset

3. **Visual inspection diffusion snow** — cần confirm
   - 26% classifier accuracy vs 86% texture fidelity → contradiction cần giải thích bằng visual evidence
   - Nên có figure so sánh diffusion snow vs style snow vs real snow trong paper

---

## 2026-04-11

### FID/KID Multi-Reference Validation — 3 Datasets, 39 Pairs

**Mở rộng FID/KID validation** từ 1 dataset (WeatherNet snow-only) lên 3 reference datasets, 4 conditions, 39 pairs.

**Reference datasets**:
| Dataset | Conditions | N images | Source |
|---|---|---|---|
| ACDC (ICCV 2021) | rain, snow, fog, night | 3,578 | Real driving scenes, ETH Zurich |
| WeatherBench (arXiv 2509.11642) | rain, snow, haze→fog | 3,000 (1K/cond subset) | Real-world paired, 42K total |
| WeatherNet-05 | snow, fog | 3,136 | HuggingFace outdoor scenes |

**Kết quả tóm tắt** (FID ↓ = better, so với original baseline):

| Condition | Best Method | Best Δ FID | Validated on |
|---|---|---|---|
| **Night** | CycleGAN-Turbo | **−34%** (178.9 vs 270.5) | ACDC |
| **Snow** | style_snow_2 / diffusion | −5% to −9% | All 3 datasets |
| **Fog** | diffusion_fog_heavy | −6% to −10% | WeatherBench + WeatherNet |
| **Rain** | ≈ baseline | inconclusive | ACDC + WeatherBench |

**Bug fixes**:
- `cond_dir.glob("*.jpg")` → `cond_dir.rglob("*.jpg")` — ACDC nested dirs weren't scanned
- Added `PYTHONUNBUFFERED=1` to SLURM job for realtime log output
- Added condition alias mapping (`fog` ↔ `haze`) for WeatherBench compatibility

**Fog augmentation added**: 3 intensity levels (heavy/medium/light) from `augmentation_data/construction_site/fog/diffusion/test/`

**Files modified**: `validation/compute_fid_kid.py` (FOG_DATA path, CONDITION_PAIRS, REFERENCE_DATASETS, alias mapping), `jobs/fid_kid_validation.sh` (PYTHONUNBUFFERED)

---

### Texture-based Fidelity Assessment — Approach 4 cho Realism Validation

**Vấn đề**: FID/KID và weather classifier đo macro-level (semantic distribution, weather recognition) nhưng bỏ qua micro-level artifacts: unnatural noise patterns, frequency anomalies, missing gray-tone diversity trong ảnh augmented.

**Giải pháp**: Implement 4 kênh texture features theo Duminil et al. (2025) — adapted cho distribution comparison thay vì CNN classification:

| Feature | Phát hiện | Dim |
|---|---|---|
| GLCM | Global texture discontinuity | 48 |
| LBP | Micro-texture pattern artifacts | 26 |
| DCT | Frequency energy anomalies (thiếu/thừa HF noise) | 12 |
| Haralick | Statistical texture metrics tổng hợp | 16 |

**Setup**: 300 images/condition, 12 augmentation conditions + 4 ACDC real weather references. So sánh bằng Wasserstein distance + KS test. Job 4896342, ~19 phút trên 8-core CPU (không cần GPU).

**Kết quả — Ranking theo Composite Texture Artifact Score (lower = more natural):**

| Rank | Augmentation | Composite | Key Feature |
|---|---|---|---|
| 1 | Diffusion snow | **0.609** | DCT_W=1.34 (gần original) |
| 2 | Diffusion rain | **0.628** | KS p>0.05 nhiều dims (not significant) |
| 3-8 | Style transfer (các loại) | 2.96-4.59 | 100% dims significant (p≈0.000) |
| 9-11 | Fog (3 mức) | 4.61-6.89 | Expected — fog removes HF |
| 12 | **Night (CycleGAN)** | **9.138** | **DCT_W=29.81 — over-smoothed** |

**Phát hiện quan trọng:**

1. **Diffusion (IP2P) giữ texture tốt nhất** — gap 5× vs style transfer. IP2P chỉ thay đổi atmosphere, giữ nguyên micro-texture.
2. **Trade-off realism vs recognizability** (corroborates Ruck et al. 2026):
   - Style transfer: weather classifier accuracy 70-95% (recognizable) + texture artifacts cao (Composite 3-6)
   - Diffusion: weather classifier accuracy 63.5% (ít recognizable hơn) + texture artifacts thấp (Composite 0.6)
   - → Paper nên present cả hai metrics và discuss trade-off
3. **Night CycleGAN over-smoothed** — DCT_W=29.81 gấp 21× diffusion. Mất natural HF noise. Cần investigate: post-processing noise? khác model?
4. **Style snow_1 outlier** — Composite 5.96 vs snow_0 (2.96), snow_2 (4.59). Style reference image snow_1 có thể gây artifacts bất thường.
5. **ACDC comparison**: fog/night augmentations gần ACDC real weather hơn original → augmentation hiệu quả ở texture level.

**Files tạo**:
- `validation/compute_texture_fidelity.py` — script chính (GLCM + LBP + DCT + Haralick + Wasserstein/KS)
- `jobs/texture_fidelity_validation.sh` — SLURM job (CPU-only, 8 cores, 32GB, 4h)
- Output: `validation/results/texture_fidelity/` (JSON + 6 plots + LaTeX table)

**Docs cập nhật**: `docs/methods.md` (Approach 4), `docs/checklist.md` (results + TODOs), `docs/literature.md` ([23] Duminil + [24] Ruck)

### Dempster-Shafer Belief Fusion (Approach 4b)

**Implement**: Dempster-Shafer belief theory fusion trên 4 Wasserstein scores, skip CNN training. Theo Eq. 14-23 từ Duminil et al. (2025).

**Pipeline**: Wasserstein → exp(-λ·W) similarity [0,1] → BBFs (Φ₁, Φ₂ with α₀, τ) → BBAs per criterion → CRC combination → {H, H̄, Ω, Conflict}

**Parameters**: τ=0.6 (pessimistic), α₀=0.8 (GLCM/LBP/DCT), α₀=0.5 (Haralick), λ calibrated per-feature (median_W → Sc=0.5)

**Kết quả quan trọng:**
- **Diffusion rain/snow = FAITHFUL** (H=0.854/0.862) — 4 criteria unanimous, conflict=0, uncertainty thấp
- **Style transfer = ARTIFACT hoặc uncertain** — H=0, m(H̄) varies 0.26-0.83
- **Night CycleGAN = ARTIFACT + conflict cao nhất (0.078)** — LBP disagrees: micro-texture OK (Sc=0.648 > τ) nhưng GLCM/DCT bad → insight mới: Night giữ local patterns nhưng mất global texture + frequency
- **Style snow_1 = strongest artifact** (H̄=0.831) — confirms outlier from Approach 4

**Phát hiện mới nhờ belief fusion (không thấy từ Wasserstein đơn)**:
1. **Conflict** phân biệt "tất cả criteria đồng ý bad" vs "criteria không đồng ý" — Night có profile artifact khác fog/style
2. **Uncertainty** phân biệt "chắc chắn artifact" (snow_1: Ω=0.17) vs "chưa rõ" (rain_1: Ω=0.74)
3. **Unanimous agreement** cho diffusion → evidence mạnh hơn "average Wasserstein thấp"

**Files**: `validation/belief_fusion.py` (chạy trên CPU, vài giây, đọc JSON kết quả từ Approach 4)
**Output**: `belief_fusion_results.json`, `belief_fusion_summary.png`, `belief_bba_per_condition.png`, `belief_radar_comparison.png`

**Also researched**: 2 validation papers từ `paper/validate/`:
- Duminil et al. (2025) — texture features + belief theory → adapted GLCM/LBP/DCT/Haralick
- Ruck et al. (2026) — VLM Jury + Relative Mahalanobis Distance → **implemented (Approach 5)**

---

### Relative Mahalanobis Distance — Approach 5 (Ruck et al. 2026)

**Implement**: Per-image relative Mahalanobis distance trong CLIP + DINOv2 embedding spaces vs ACDC real weather.

**Bug fix**: DINOv2 via `torch.hub` crash do xformers CUDA build issue → chuyển sang `transformers.AutoModel("facebook/dinov2-large")`. Job 4897210 fail → resubmit 4897885 → thành công (01:41 AM).

**Kết quả CLIP (% gap closed toward ACDC baseline):**

| Condition | Best Augmentation | % Gap Closed |
|---|---|---|
| Night | CycleGAN-Turbo | **38%** |
| Snow | style_snow_2 | **27%** |
| Rain | style_rain_1 | **25%** |
| Fog | fog_heavy | **18%** |

**Key findings:**
1. All augmentations closer to ACDC than original — validated
2. Night largest improvement (38%) in CLIP, but over-smoothed in texture (Approach 4)
3. CLIP sensitive to style transfer, DINOv2 nearly unchanged (<1 point shift)
4. Snow: style > diffusion (CLIP), diffusion > style (DINOv2) — semantic vs texture trade-off
5. Cross-approach synthesis: style transfer = strong weather + artifacts, diffusion = subtle + natural

**Files**: `validation/compute_relative_mahalanobis.py`, `jobs/relative_mahalanobis.sh`
**Output**: `validation/results/relative_mahalanobis/` (JSON + 3 plots + LaTeX)

---

## 2026-04-10

### Realism Validation Pipeline — 3 Approaches Tested

**Vấn đề**: Giáo viên góp ý cần validate bộ dữ liệu có giá trị. Quality metrics hiện tại (SSIM, LPIPS, DINO) chỉ đo similarity với ảnh gốc, không trả lời "ảnh augmented có trông thật không?".

**Approach 1: UniversalFakeDetect (UnivFD) — ABANDONED**

- Model: `WisconsinAIVision/UniversalFakeDetect` (CVPR 2023, MIT)
- Setup: CLIP ViT-L/14 + trained FC classifier, weights từ official GitHub repo
- Test scale: Full dataset, 22,601 images, 11 conditions
- **Kết quả: fooling rate 99.5-100% cho TẤT CẢ conditions** — metric không phân biệt được
- **Lý do thất bại**: UnivFD train để detect ảnh AI-generated hoàn toàn. ConSynth-X augmentation là edit trên ảnh thật → detector luôn thấy "real photograph"
- Files: `validation/realism_detector.py`, `validation/run_realism_validation.py`
- Results: `validation/results/full_univfd/` (supplementary only)

**Approach 2: FID/KID vs Real Weather Reference — PARTIAL**

- Attempt 1: ACDC (`mathpluscode/ACDC` trên HF) → **sai dataset** (medical cardiac MRI, không phải weather)
- Attempt 2: `dgural/bdd100k` → **rate limit 429** (dataset 7GB, HF throttle)
- Attempt 3: `34data/bdd100k-weather-classification` → không có images (text only)
- **Solution**: Dùng **WeatherNet-05-18039** (`prithivMLmods/WeatherNet-05-18039`)
  - 18,039 images, 5 weather classes, Apache-2.0
  - Download thành công toàn bộ
- Compute: InceptionV3 pool3 features + FID + KID (polynomial kernel)
- **Kết quả**:
  - **Snow validated**: augmented snow FID 116-125 < original→snow 128.8 (augmentation đưa distribution gần real snow)
  - **Rain failed**: original→rain FID 133.7 < augmented rain 135-145 (cross-domain gap construction vs outdoor dominates)
- Files: `validation/compute_fid_kid.py`, `validation/download_reference.py`
- Results: `validation/results/fid_kid/`

**Approach 3: Weather Classifier (SigLIP2) — SUCCESS, PRIMARY METRIC**

- Model: `prithivMLmods/Weather-Image-Classification` (SigLIP2 fine-tuned, Apache-2.0)
- Base: `google/siglip2-base-patch16-224`, 5 classes, 85.89% test accuracy
- Training data: WeatherNet-05-18039 (same as Approach 2 reference)
- **Ý tưởng**: Thay vì đo distribution, dùng classifier trực tiếp. Feed augmented rain → expect classifier say "rain".
- Test scale: Full dataset, 3,004 images/condition, 11 conditions

**Kết quả chính**:

| Condition | Accuracy | Verdict |
|---|---|---|
| original (construction clear) | 6.1% → sun/clear | cross-domain bias (classifier expects driving) |
| small (outpainting) | 8.4% | cross-domain bias |
| weather_style_rain_0 | **88.6%** | ✓ validated |
| weather_style_rain_1 | 71.3% | ✓ validated |
| weather_style_rain_2 | 70.8% | ✓ validated |
| weather_style_snow_0 | 61.8% | moderate |
| **weather_style_snow_1** | **95.0%** | ✓ excellent |
| weather_style_snow_2 | 92.7% | ✓ excellent |
| diffusion_rain (IP2P) | 63.5% | moderate, kém hơn style transfer |
| **diffusion_snow (IP2P)** | **25.7%** | ✗ FAILED, 49% classify as rain |
| night (CycleGAN) | N/A | no night class, 95% classify as rain |

**Key findings cho paper**:

1. **Weather classifier phân biệt rõ** giữa các methods (unlike UnivFD)
2. **Style transfer rain/snow**: validated tốt (70-95% recognition)
3. **IP2P diffusion**: rain OK (63.5%), **snow FAILED (25.7%)** — cần investigate
4. **Cross-domain bias**: construction images (đất/bụi/equipment) không khớp distribution driving scenes. Original clear images chỉ 6.1% "sun/clear", 57.8% bị classify thành rain/storm
5. **Night pipeline**: không validate được qua weather classifier (cần separate metric)

**Files**: `validation/weather_classifier.py`, `validation/jobs/run_weather_cls.sh`
**Results**: `validation/results/weather_cls/` (JSON, CSV, accuracy barplot, confusion heatmap, LaTeX table)

**Quyết định**: Dùng Approach 3 (weather classifier) làm primary validation metric cho paper. FID/KID snow (Approach 2) làm secondary. UnivFD (Approach 1) chỉ supplementary (để giải thích tại sao không work cho edit-based augmentation).

**TODO tiếp theo**:
- Investigate tại sao IP2P diffusion_snow fail (25.7% vs style transfer 92.7%)
- Document cross-domain bias explicitly trong paper
- Tìm separate validation metric cho night pipeline

---

## 2026-04-09

### DINO vs SSIM — Phát hiện SSIM không phù hợp cho weather augmentation

**Phát hiện quan trọng**: SSIM đo pixel-level similarity, phạt cả thay đổi mong muốn (darkened sky, rain streaks, color shift) lẫn không mong muốn (hallucinate objects, structural damage). Không phân biệt được "good edit" vs "bad edit".

**DINO patch similarity** (DINOv2-ViT-S/14) đo semantic structure — objects, layout, spatial composition — không phạt atmospheric changes. Phù hợp hơn cho weather augmentation.

**Kết quả ranking bằng DINO (50 samples):**

| Method | DINO Patch | SSIM | Insight |
|---|---|---|---|
| Canny CN + img2img | **0.744** | 0.590 | Winner DINO — strong weather + preserved objects |
| Vanilla IP2P | 0.674 | **0.692** | Winner SSIM — nhưng chỉ vì ít thay đổi |
| FLUX Kontext | 0.639 | 0.281 | SSIM cực thấp nhưng DINO khá (global=0.92) |
| SDXL IP2P | 0.435 | 0.443 | Tệ cả hai |
| ControlNet IP2P | 0.196 | 0.214 | Generate ảnh mới |

**Disagreement cases (50 samples OLD prompt):**
- 5 cases "SSIM bị lừa": DINO < 0.55 nhưng SSIM > 0.50 — ground biến bùn nhưng brightness giống
- 5 cases "SSIM đánh thấp": DINO > 0.70 nhưng SSIM < 0.65 — weather edit tốt, objects giữ, pixel khác
- DINO-SSIM correlation = 0.907 — cao nhưng disagreement cases quan trọng nhất

**Bài học cho paper**: Dùng DINO thay SSIM làm primary quality metric. SSIM vẫn report nhưng không dùng để rank methods.

**Figures tạo**: `dino_eval/disagreement_5plus5.png`, `dino_eval/dino_vs_ssim.png`, `dino_eval/old_prompt_20samples_dino.png`

---

## 2026-04-08

### Model Comparison — Tìm model tốt nhất cho weather editing

Đã test 4 models cho rain augmentation (5 samples, same prompt):

| Model | Mean SSIM | Verdict |
|---|---|---|
| **IP2P SD1.5 + new prompt** | **0.676** | Best — subtle atmospheric edit |
| SDXL IP2P | 0.442 | Over-edits, phá scene structure |
| FLUX.1 Kontext | 0.281 | Tạo ảnh mới hoàn toàn — model quá mạnh |

**Kết luận**: Model mạnh hơn ≠ tốt hơn cho task này. Weather augmentation cần **subtle edit**, không cần creative generation. IP2P SD1.5 (860M params) outperform cả SDXL (2.6B) và FLUX (12B) vì nó edit vừa phải.

### ControlNet + IP2P — Structure Preservation (đang test)

Thay vì đổi model, thêm **ControlNet** để khóa structure:
- **Approach B**: `control_v11e_sd15_ip2p` — ControlNet trained on IP2P pairs
- **Approach C**: Canny ControlNet + img2img (strength=0.35) — edge map + low noise
- Job 4754807 submitted, đang chạy

Literature support: SDEdit (Meng et al. ICLR 2022), ControlNet (Zhang et al. ICCV 2023), InstructRL4Pix (2024)

### IP2P Rain Prompt Bug — "wet muddy ground" gây hallucinate

**Phát hiện:** Prompt v1 `"make it a heavy rainy day, dark overcast sky, wet muddy ground"` khiến IP2P tập trung biến đổi mặt đất thành bùn/nước thay vì tạo atmospheric rain. Kết quả: 48% rain images bị DROP (SSIM thấp, LPIPS cao).

**Root cause:** Cụm "wet muddy ground" — IP2P diễn giải thành "replace ground texture with mud/water", gây structural deformation lớn. Physics overlay đã xử lý rain streaks, nên IP2P không cần modify ground.

**Fix:**
- Prompt v2: `"a rainy day with dark overcast sky, rain falling, grey clouds"` (chỉ atmospheric)
- Files changed: `batch_worker.py`, `batch_worker_soda.py`, `physics.py`
- Test: 20 samples submitted (job 4753144)
- Full regen job: `jobs/regen_rain_v5.sh` (chờ review test samples trước)

**Bài học:** Với IP2P + physics pipeline, prompt nên focus atmosphere, để physics module xử lý particles. Tránh yêu cầu IP2P thay đổi scene structure.

### Sensitivity Analysis — Kết quả và Hạn chế

- 7 configs YOLOv8n đã train xong (baseline + 6 threshold configs)
- **Phát hiện:** loose/moderate/no_lpips cho kết quả IDENTICAL — Arrow files đã pre-filtered nên threshold lỏng hơn không thể recover ảnh đã DROP
- Chỉ strict/tight thực sự filter khác nhau
- Snow không có LPIPS scores → no_lpips config vô nghĩa cho snow
- Val set chỉ có clean images → baseline thắng (expected, uninformative)

### Paper Template

- Tạo `paper/main.tex` — Nature Scientific Data Data Descriptor format
- Tạo `paper/references.bib` — 13 references với verification status
- Sinh 3 figures: comparison grid, quality spectrum, filter distributions
- 15 trang, compile OK

---

## 2026-04-06

- Khởi tạo file `DEVLOG.md` để theo dõi tiến trình phát triển dự án.
- Phân tích so sánh 2 phương pháp weather augmentation:
  - **Method 1:** Neural Style Transfer (VGG19) + MiDaS depth + physics particles — đã có downstream detection (mAP +75%)
  - **Method 2:** InstructPix2Pix + physics overlay + LPIPS/SSIM filter — chưa đánh giá downstream
  - Tạo notebook phân tích: `examples/compare_weather_augmentation.ipynb`
  - Tạo tài liệu tham khảo: `related_paper/weather_augmentation_references.md` (7 bài báo)
  - **Kết luận tạm:** Chưa thể khẳng định method nào tốt hơn — cần chạy YOLOv8 với data IP2P để so sánh downstream
- Xác định publication roadmap: 3 bài báo theo thứ tự
  1. **Scientific Data** (ưu tiên hiện tại) — dataset paper
  2. **VLM Benchmark** — đánh giá 11+ VLMs trên construction images
  3. **VLM Object Detection Improvement** — cải thiện detection cho VLM
- Cập nhật `CLAUDE.md` với roadmap chi tiết
- **Research Integrity Audit** — kiểm tra toàn diện dự án trước Scientific Data submission:
  - **CRITICAL**: 20+ parameters không có justification (SSIM thresholds, LPIPS, style weights, IP2P guidance, physics overlay params, MiDaS depth constants)
  - **CRITICAL**: Mâu thuẫn style_weight (10k vs 100k) và steps (10 vs 50) giữa workers và pipelines
  - **CRITICAL**: Không model nào có pinned version (IP2P, MiDaS, FLUX, VGG19, YOLOv8)
  - **CRITICAL**: Data provenance thiếu (URLs, licenses, checksums cho Construction Site + SODA datasets)
  - **PASSED**: Data integrity, annotation preservation, provenance tracking, citation disclaimers, JPEG quality
  - Cập nhật `docs/methods.md` — ghi rõ từng parameter cần justify + TODO sensitivity analysis
  - Cập nhật `docs/literature.md` — ghi rõ từng citation verified/unverified + TODO items
  - Cập nhật `docs/data_sources.md` — ghi rõ tất cả thông tin thiếu cho từng dataset/model
  - Thêm Pre-Publication Checklist vào `CLAUDE.md`
- Tạo `plan.md` — phân phối công việc thành 7 phases, dependency graph, ước tính thời gian
- Restructure documentation: `CLAUDE.md` → hub, tách nội dung vào `docs/` (checklist, infrastructure, architecture)
- **Sensitivity Analysis (Phase 1):**
  - Viết `generation/sensitivity/ssim_lpips_sweep.py` — sweep SSIM/LPIPS trên CSV metrics (xong)
  - Viết `generation/sensitivity/run_sensitivity.py` — all-in-one filter+export+train+eval (xong)
  - Viết `generation/sensitivity/submit_all.py` — submit 6 SLURM jobs (xong)
  - Chạy sweep analysis: 10 groups × 9 thresholds → `results/ssim_sweep_results.csv`
  - Submit 6 YOLOv8 training jobs (loose/moderate/current/strict/tight/no_lpips) — **đang chạy**
  - Key finding: LPIPS lọc thêm ~10% rain images, snow không có LPIPS data (gap)
- Khảo sát 5 papers tương tự trên Scientific Data (2024-2025) → xác định validation requirements
  - Cần ≥3 detection models (hiện chỉ có YOLOv8n)
  - Cần public data deposit (HuggingFace/Zenodo) trước review
  - Cần annotation quality verification
- Viết draft Background & Summary: `docs/paper_background_summary.md`
  - Ghi rõ fact/claim/citation status cho mỗi statement
  - Flag 6 citations cần verify, 4 số liệu cần double-check
