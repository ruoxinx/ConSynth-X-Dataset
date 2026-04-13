# Strict Scientific Review — ConSynth-X (Nature Scientific Data)

**Paper:** "ConSynth-X: A Multi-Condition Synthetic Dataset for Construction Site Computer Vision Under Extreme Field Conditions"  
**Target venue:** Nature Scientific Data (Data Descriptor)  
**Review standard:** Q1 journal / top-tier conference (per reviewer guidelines v2.0)

---

## Revision History

| Review Round | Date | Score | Recommendation |
|---|---|---|---|
| **Round 1** | 2026-04-12 | 2.5 / 5.0 | Reject |
| **Round 2** | 2026-04-13 | **3.1 / 5.0** | **Major Revision** |

### Changes Resolved Since Round 1

| Round 1 Issue | Status | Details |
|---|---|---|
| W1: 11+ \todo{} placeholders | **Partially fixed** | Reduced from 11+ to ~7. Resolved: style_weight/steps (line 120), guidance_scale justification (line 136), SODA outpainting documented (line 203), model versions pinned (Table 10), Acknowledgements filled (line 668), equation label fixed (line 197). |
| W2: Table 7 empty | **NOT fixed** | Still entirely \todo{fill}. Remains CRITICAL. |
| W3: Fog pipeline undocumented | **FIXED** | New Section 2.3 "Fog Augmentation" with Koschmieder equation, Depth Anything V2 citation, beta parameters, and methodology. Well done. |
| W4: Citation issues | **Partially fixed** | Depth Anything V2 now properly cited (references.bib lines 293-299). Remaining: Tremblay [P], Gurbindo [P], WeatherBench [A] unverified arXiv, Ruck/Duminil no DOI. |
| W5: Weather classifier bias | **NOT fixed** | Unchanged. Still acknowledged but no mitigation (delta-probability, human study, etc.). |
| W6: Snow LPIPS gap | **Partially fixed** | \note{} removed. Now justified with visual inspection of 100 random samples (line 157). Not fully resolved but substantially better — an honest empirical justification. |
| Table 6 "moderate" values | **FIXED** | DINOv2 column now reports exact numbers: 8%, 0.4%, 0.8%, 5% (lines 430-433). |
| Dataset/Code URLs | **Addressed** | Reworded to "upon acceptance" (lines 221, 628) — acceptable for submission. |
| SODA results unverified | **Partially fixed** | Now explicitly flagged as "Preliminary results" (line 506) — more honest framing. |
| Broken eq reference | **FIXED** | Eq.\ref{eq:bbox-transfer} now works with proper \label (line 197, referenced line 211). |

---

## 1. Summary

This paper presents ConSynth-X, a multi-condition construction site dataset comprising over 125,000 images generated from two source datasets (Construction Site 10k and SODA). Four augmentation pipelines are employed: (1) neural style transfer with depth-aware physics overlay + InstructPix2Pix diffusion for weather effects, (2) Koschmieder atmospheric scattering with Depth Anything V2 for fog, (3) CycleGAN-Turbo for day-to-night conversion, and (4) FLUX.1-Fill-dev outpainting for scale variation. Dataset quality is validated through five complementary approaches spanning realism, distributional fidelity, texture preservation, and downstream utility. The paper identifies a complementary trade-off between diffusion (high texture fidelity, subtle weather) and style transfer (strong weather effects, texture artifacts).

---

## 2. Overall Assessment (Round 2)

| Criterion | R1 Score | **R2 Score** | Change | Justification |
|---|---|---|---|---|
| **Novelty & Significance** | 3.0 | **3.0** | — | Unchanged. Systematic multi-pipeline approach is useful. Individual components are off-the-shelf; novelty lies in the combination and validation framework. |
| **Technical Soundness & Reproducibility** | 2.0 | **3.0** | +1.0 | Major improvement: parameters resolved, fog pipeline documented, model versions pinned, equation label fixed. Still held back by remaining TODOs and Table 7. |
| **Experimental Validation & Rigor** | 2.0 | **2.5** | +0.5 | SODA results properly flagged as "preliminary." Mahalanobis table now quantitative. But Table 7 (cross-condition detection) still empty — this alone caps the score. |
| **Feasibility & Practical Impact** | 3.0 | **3.5** | +0.5 | Outpainting low success rate now documented with explanation (scene diversity causing FLUX.1 failures). Snow LPIPS justified via visual inspection. |
| **Citation Accuracy & Related Work Fairness** | 2.5 | **3.0** | +0.5 | Depth Anything V2 properly cited. Still 2 [P] partial and 1 unverified arXiv ID. |
| **Clarity & Presentation** | 2.5 | **3.5** | +1.0 | Fog section adds clarity. Guidance scale justified. "Moderate" replaced with numbers. "Upon acceptance" framing is appropriate. Fewer placeholders. |

**Average score: 3.1 / 5.0** (up from 2.5)

**Recommendation: Major Revision**

> The paper has improved substantially from a clearly unfinished draft toward a reviewable manuscript. The fog pipeline documentation, parameter justifications, and model version pins address critical reproducibility gaps. However, **Table 7 (cross-condition detection) remains entirely empty**, which means the central dataset utility claim is still unsupported. This single issue prevents advancement to Minor Revision.

---

## 3. Major Strengths

1. **Comprehensive multi-metric validation framework** (Sections 4.1–4.5). The five-dimensional validation (realism, weather recognizability, distributional fidelity, texture fidelity, downstream detection) is more rigorous than most synthetic data papers and reveals non-obvious trade-offs. The cross-validation summary (Table 9) is particularly insightful.

2. **Honest reporting of trade-offs and limitations.** The paper does not hide that diffusion snow is only 25.7% recognizable, that CycleGAN over-smooths DCT content, or that the weather classifier has cross-domain bias. SODA results are now appropriately flagged as "preliminary." This transparency is commendable.

3. **Practical pipeline design insights.** The prompt design lesson (Section 2.2, "wet muddy ground" causing hallucinations), the ControlNet + SDEdit structure preservation approach, and the guidance scale rationale are valuable contributions that aid reproducibility.

4. **Fair comparison with prior work.** The comparison with Ding et al. (ExtCon/UIA-YOLOv5) is specific and fair, correctly identifying its limitations without misrepresenting its contributions.

5. **(New) Well-documented fog pipeline.** The Koschmieder model with three beta levels and Depth Anything V2 depth estimation is clearly described with reproducible parameters.

---

## 4. Major Weaknesses (Remaining)

### W1. [CRITICAL — UNCHANGED] Cross-Condition Detection Results Missing (Table 7)

Table 7 — "Cross-condition object detection performance (mAP@0.5)" — remains **entirely filled with \todo{fill}** (lines 493–497). This is the single most important table in a dataset paper: it demonstrates that the dataset actually improves model robustness under adverse conditions. The accompanying TODO note (line 502) explicitly acknowledges: "CRITICAL: current evaluation uses clean val set only — need weather/night test set evaluation."

Without this evidence:
- The central claim that ConSynth-X improves robustness is **unsupported**.
- The sensitivity analysis (Table 8) only evaluates on clean images, showing mAP@0.5 *lower* than baseline (0.562 → 0.547/0.538/0.542) — which is the expected but *less interesting* result.
- The "preliminary" SODA results (+73% night, +82% weather) cannot substitute because they are unverified and from style transfer only.
- Table 9 (Cross-validation summary) still has 2 \todo{fill} entries for Night and Outpainting detection mAP (line 583).

**This single issue is the primary blocker for acceptance.**

### W2. [MAJOR — UNCHANGED] Weather Classifier Validation Confounded by Cross-Domain Bias

The weather classifier validation (Section 4.1.2, Table 4) remains confounded:
- Original construction site images are classified as "sun/clear" only **6.1%** of the time, with **57.8% classified as "rain/storm"** — before any augmentation.
- The classifier's 63.5–88.6% rain accuracy for augmented images cannot be meaningfully separated from the baseline misclassification tendency.
- The authors acknowledge this but present no mitigation.
- **Recommendation:** Report the **change in class probability** (Δ probability pre/post augmentation) as a fairer metric, or add a brief human evaluation on a random subset.

### W3. [MAJOR — PARTIALLY IMPROVED] Citation Integrity Issues

Progress: Depth Anything V2 now properly cited. But remaining issues:

- **Tremblay et al. (2021)**: Still [P]. "Up to 21% improvement" and "up to 73% more realistic" — exact metric, detector, and dataset still unverified from full text.
- **Gurbindo et al. (2025)**: Still [P]. Specific mAP values unverified.
- **WeatherBench (2025)**: Still [A] with arXiv ID 2509.11642 flagged as "may not resolve." One of three FID/KID reference datasets relies on an unverifiable citation.
- **Ruck et al. (2026)** and **Duminil et al. (2025)**: No DOI, arXiv preprint only. The entire Sections 4.2.2 and 4.3 depend on these two works.

### W4. [MINOR — NEW] Remaining \todo{} Placeholders

Reduced from 11+ to ~7, but still present:

| Location | Content |
|---|---|
| Table 7 (lines 493–497) | 20x \todo{fill} — ALL cross-condition results |
| Line 502 | \todo{} note about cross-condition evaluation |
| Table 9 (line 583) | 2x \todo{fill} — Night and Outpainting detection mAP |

Additionally, the \todo{}, \verify{}, and \note{} macro definitions remain in the preamble (lines 31–33) — these should be removed or commented out before submission.

### W5. [MINOR — PARTIALLY IMPROVED] Snow LPIPS Quality Filtering

The raw \note{} marker is gone and replaced with a justified explanation: "Visual inspection of 100 random snow samples confirmed no systematic structural artifacts" (line 157). This is an improvement, but:
- 100 out of 6,755 is a 1.5% sample — consider reporting this sampling rate.
- Without LPIPS scores computed, the claim cannot be verified by reviewers or readers.
- Consider computing LPIPS for snow even if only for reporting purposes (not necessarily filtering), to demonstrate the assertion quantitatively.

---

## 5. Detailed Comments by Section

### Abstract & Introduction
- The abstract claims "three complementary synthetic augmentation pipelines" but the paper now describes **four** (weather ST/IP2P, fog, night, outpainting). Consider updating the abstract to mention fog separately, or clarify the counting.
- The Background cross-condition mAP drop (0.819 → 0.446/0.454) is still presented without experimental details. Add a forward reference to the section where this evaluation is described.

### Methods
- **Section 2.2.1 (line 120-121):** Style weight and steps are now documented (100,000 weight, 50 steps) with reference to Gatys et al. Good. The mention of a lower-stylization test (10,000/10 steps) producing insufficient effects is a useful ablation note.
- **Section 2.2.2 (line 136):** Guidance scale now justified via Brooks et al. default range. Acceptable.
- **Section 2.3 (Fog):** Well-written new section. The Koschmieder equation, three beta levels, and Depth Anything V2 are clear and reproducible. Minor: cite Koschmieder (1924) or a modern reference for the atmospheric scattering model — currently only the depth estimation is cited.
- **Section 2.5 (line 203):** Outpainting success rate now documented with explanation (44% Construction Site, ~5% SODA due to scene diversity). Good improvement, but consider reporting the total compute time/cost for SODA outpainting to contextualize the feasibility.
- **Annotation Preservation (line 211):** Eq. reference now works. Clean.

### Experiments & Results
- **Table 4 (Weather classifier):** Night row still reports "95% → rain/storm" — this confusing notation should be clarified. Does this mean 95% of night images are misclassified as rain/storm? If so, this is a classifier failure, not a validation metric.
- **Table 5 (FID/KID):** Rain FID (-1%) is still within noise. The text correctly says "inconclusive" — consider not bolding any value in the rain row.
- **Table 6 (Mahalanobis):** Now quantitative with exact DINOv2 percentages (8%, 0.4%, 0.8%, 5%). Good fix.
- **Table 8 (Sensitivity):** Still only evaluates on clean test images. The text now references Table 7 for cross-condition results (line 534), but Table 7 is empty.
- **Table 9 (Cross-validation summary):** Style transfer texture H reported as 0.00 but Table 8 shows 0.017 for ST Snow light. Minor inconsistency (likely rounding) — clarify.

### Validation
- The Dempster-Shafer calibration parameters λ_c are still not reported. How were they calibrated? This affects reproducibility.
- The conflict value K=0.078 for night is noted but not contextualized — what threshold constitutes "high" conflict in the Dempster-Shafer framework?

### Reproducibility
- Model versions now pinned in Table 10 (e.g., `timbrooks/instruct-pix2pix`, `DPT_Large`, commit `86f5414`, `Depth-Anything-V2-Large`). Good improvement.
- YOLOv8 version is `ultralytics==8.1.x` — pin to exact minor version (e.g., 8.1.47).
- SHA-256 checksums promised "in the final release" (line 638) — acceptable for submission but must be delivered.
- SODA license concern remains: "no formal license specified by the authors" — discuss implications for ConSynth-X redistribution.

---

## 6. Recommendation

**Major Revision**

The paper has made significant progress from Round 1:
- **Resolved:** Fog pipeline documentation, parameter justifications, model version pins, equation labels, acknowledgements, Mahalanobis exact values, SODA results framing, outpainting documentation.
- **Partially resolved:** Placeholder count reduced, snow LPIPS justified, Depth Anything V2 citation added.
- **Still blocking:**
  1. **Table 7 cross-condition detection results** — must be completed.
  2. **2 partial + 1 unverifiable citation** — must be resolved.
  3. **Weather classifier bias** — needs mitigation or delta-probability reporting.
  4. **Remaining ~7 \todo{} markers** — must be removed.

Once Table 7 is filled with cross-condition detection results showing that augmented training improves weather/night robustness, and the remaining citations are verified, this paper should reach **Minor Revision** or **Accept** quality. The validation framework, honest reporting, and practical pipeline insights are strong contributions.

---

## 7. Questions for Authors (Updated)

1. **Table 7 (Cross-condition detection):** This remains the primary blocker. When will these results be available? Can you provide at least a preliminary result (e.g., Original-only vs. +All on weather test set) to demonstrate the direction of the robustness improvement?

2. **Weather classifier bias mitigation:** Given the 57.8% baseline rain misclassification, would you consider reporting Δ(class probability) as a supplementary metric? For example: mean P(rain|augmented) - mean P(rain|original) would isolate the augmentation effect from the domain bias.

3. **Koschmieder citation:** The Koschmieder atmospheric scattering model is used but not directly cited. Consider citing the original (Koschmieder, 1924) or a modern formulation (e.g., Narasimhan & Nayar, 2002 "Vision and the Atmosphere").

4. **Tremblay et al. full-text verification:** Can you confirm from the full text which specific detector and dataset correspond to the "up to 21% improvement" claim? The DOI is listed in the .bib notes (10.1007/s11263-020-01366-3) — full-text verification should be straightforward.

5. **WeatherBench arXiv ID:** Has 2509.11642 been verified to resolve? If not, the .bib note already suggests citing the GitHub repo instead — please make this decision before submission.

6. **Dempster-Shafer λ_c calibration:** How were the per-feature calibration parameters determined? Were they taken from Duminil et al. directly, or calibrated on your data? This is important for reproducibility.

---

*Review Round 2 — This review follows the Strict Scientific Reviewer Guidelines v2.0. The reviewer acknowledges substantial improvement from Round 1 and encourages the authors to prioritize completing Table 7 (cross-condition detection) as the single highest-impact action for the next revision.*
