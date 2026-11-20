# Stain-distribution research extension

Status, 12 November 2026: preprocessing pilot completed. Classifier training and
performance comparisons have not been run. Original project files are unchanged.

The next candidate, decomposition-aware strength control, is implemented in
controller.py. Read CONTROLLER_STUDY.md for its hypothesis, matched controls,
limitations and run instructions. Its stability pilot is separate from the
target-distribution preview below; neither establishes classification benefit.

## Research basis and novelty status

Kim et al., *Stain SAN: simultaneous augmentation and normalization for
histopathology images*, Journal of Medical Imaging 11(4), 044006 (2024).
DOI: https://doi.org/10.1117/1.JMI.11.4.044006
Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11342968/
Author code: https://github.com/taebinkim7/stain-san
Reviewed commit: f158d7956c32b85910afbdeb2714ccf402a64c26

The paper explicitly proposes elliptical Gaussian and uniform target distributions
as future work. Those ideas belong to the authors. Our candidate contribution is
their controlled implementation and evaluation for eight-class gastric tissue
classification, including performance, stability, and preprocessing cost.

Focused searches for "Stain SAN" with "elliptical", "uniform distribution", and
"covariance" did not establish a direct published follow-up implementing this
exact gastric comparison. This is limited search evidence, not proof of novelty.
Broader covariance-based stain augmentation and later stain sampling methods
remain related prior art. Do not claim a first-ever algorithm or guaranteed
publication. The paper's suggestion alone is not our invention.

## What is implemented

- NumPy/Pillow SVD stain extraction and reconstruction corresponding to the
  authors' SVD path, with numerical compatibility checks.
- Original-style spherical Gaussian target: median 6-component stain vector,
  variance equal to the total training variance divided by six.
- Experimental elliptical Gaussian target: full 6x6 training covariance with
  10% shrinkage toward the spherical covariance. Shrinkage preserves the trace.
- Experimental uniform target: independent components centered on the same
  median with half-width sqrt(3 * spherical_variance).
- All targets have equal total variance BEFORE nonnegative clipping. Clipping
  can change both the mean and covariance; the pilot reports its effects.
- Shared concentration perturbation U(0.8, 1.2) during training.
- Deterministic median-matrix normalization during validation/inference for
  every SAN variant, matching the original default script's intended protocol.
- Explicit fallback to unchanged images when extraction is unsuitable.

Unlike the upstream package, this port does not import spams or TensorFlow, and
uses modern NumPy types. It adds checks for too few tissue pixels and rank
deficiency. Only the SVD path and scalar-intensity SAN branch are covered; this
is not a reproduction of every published experiment or the NMF implementation.

## Run on this PC

From the parent of stain_research:

```powershell
& 'C:\Users\Amlan\Desktop\gastric detection\.venv\Scripts\python.exe' stain_research\check.py --project 'C:\Users\Amlan\Desktop\gastric detection'
& 'C:\Users\Amlan\Desktop\gastric detection\.venv\Scripts\python.exe' stain_research\pilot.py --project 'C:\Users\Amlan\Desktop\gastric detection' --output stain_research\runs\preprocessing_repeat
```

Use a fresh output folder on every run. On another PC, copy this research folder,
update project/interpreter paths, and ensure the original manifest resolves there.
Requirements: NumPy, pandas, Pillow; already installed in the project environment.
The optional check.py needs the included upstream checkout. Do not install its
old optional dependencies into the project's environment.

## Completed pilot

160 training images (20 per class), seed 42: 160 successful extractions, zero
failures. Statistics were fitted on training samples only. Eight validation
images were used solely for an illustrative augmentation preview; none were
used to estimate the distribution. Test images were not read.

Outputs in runs/preprocessing_v1:

- training_parameters.json: sampled-training median and covariance parameters.
- fit_manifest.csv: image identity, split, class, and extraction status.
- augmentation_preview.png: original, example color adjustment, spherical,
  elliptical, and uniform variants for all eight classes.
- preview_qc.csv: pixel change and fallback diagnostics.
- summary.json: sampling QC and exact pilot counts.

Sample covariance traces after clipping were approximately 0.02955 (spherical),
0.02943 (elliptical), and 0.02946 (uniform), against target trace 0.02963.
Zero-component fractions were 0.128%, 0.620%, and 0.000%, respectively.
These measure numerical behavior, not classification or clinical effectiveness.
The preview has no obvious geometric displacement, but no expert morphology
review has been performed. Pixel similarity cannot establish diagnostic fidelity.
The color_jitter preview is a fixed illustration, not a complete training policy.

Checks passed: eight-class extraction and reconstruction match against the
upstream mathematical functions; blank-image fallback; deterministic evaluation;
and covariance-trace preservation. Upstream mathematical functions were loaded
selectively for the check without importing optional packages.

## Next experiment on the stronger PC

1. Audit duplicate images and recover patient/slide identifiers before freezing
   partitions. The existing manifest is image-level and cannot support a claim
   of patient-independent performance. Checkpoint selection previously used val.
2. Fit final stain parameters on the training partition, with a recorded image
   list. The 160-image fit above is only for development, not a full-data fit.
3. Train four configurations: standard geometric/RGB augmentation, spherical
   SAN, elliptical SAN, and uniform SAN. Keep model, optimizer, seeds, split and
   training budget identical. Use the existing CNN-BiLSTM-attention initially.
4. For SAN arms apply the transform before division into 16 subpatches. Use
   training=False for deterministic validation. Refit nothing on val/test.
5. Use geometric augmentation consistently. For SAN variants keep intensity
   perturbation identical; record the sampling distribution and shrinkage.
6. Reload each best validation checkpoint, then assess macro-F1, accuracy,
   per-class recall, calibration and latency. Repeat three seeds if practical.
7. Compare clean and predefined independent stain perturbations. Avoid assessing
   robustness only with perturbations identical to the proposed training sampler.
   Synthetic robustness is not cross-hospital generalization.
8. Add a normalization-only control if resources allow: it separates the effects
   of common inference normalization from the benefits of training augmentation.
9. Report failures and class-specific fallback rates across the complete dataset.
   Check that sampling does not destroy class-defining morphology.
10. Select choices using validation data; use untouched test data for final
    confirmation, with uncertainty intervals and appropriate grouping.

No long training job is currently running. There is no evidence yet that the
elliptical or uniform variants outperform the original spherical method.

Suggested working title: "Effect of Target Stain Distributions on Gastric
Histopathology Tissue Classification". Final claims depend on measured results.
