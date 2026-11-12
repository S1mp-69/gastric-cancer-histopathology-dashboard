# Decomposition-aware augmentation: exploratory study

## Hypothesis

Do errors and numerical sensitivity in stain decomposition predict sensitivity
of the resulting augmented image to small input disturbances? If so, does
reducing stain augmentation on those images improve stability beyond uniformly
reducing augmentation?

This is an unconfirmed candidate method. Reconstruction fit and condition number
are numerical diagnostics, not medical image-quality or morphology labels.
High condition numbers here need not indicate clinically unsuitable images.

## Implemented controller

For a source image I, compute optical density V and the SVD stain matrix W using
the compatible Stain SAN extraction. Solve H = least_squares(W, V). On tissue
pixels compute e = ||V - WH||_F / ||V||_F, and k = log(cond(W)).

Fit 10th/90th percentile bounds for e and k using training images only. Scale
each feature to [0, 1], clip, and average to obtain risk q. Set augmentation
strength alpha = 1 - 0.9*q. These choices are fixed exploratory design choices;
they are not optimality guarantees or learned clinical thresholds.

Sample a target stain matrix Wt from the spherical SAN distribution and an
intensity factor s from U(0.8, 1.2). Change optical density using:

    V_new = V + alpha * (Wt*s - W) * H

Then reconstruct RGB, retaining background pixels. This retains the source
decomposition residual. It differs from original SAN reconstruction. All four
pilot arms use this same reconstruction, so residual retention cannot masquerade
as an effect of adaptive control. A later classifier study must separately
compare original SAN versus fixed residual-preserving SAN.

An alpha of zero returns the source image exactly. Failed decompositions return
the image unchanged; none are removed. Geometric augmentation would still be
applied in classifier training, which has not been implemented in this pilot.

## Experiment

Training: 20 images/class, seed 42, for controller bounds and stain distribution.
Validation: 20 separate images/class, seed 17. Test images are not accessed.
For six independent disturbances per image, add RGB Gaussian noise with sigma
one intensity level, round and clip to uint8, then re-estimate decomposition.
Use the same random target and intensity factor for clean/noisy input pairs.

Primary outcome: sensitivity gain = RMS(augmented_noisy - augmented_clean) /
RMS(noisy_input - clean_input). This measures response to synthetic pixel noise,
not classifier robustness or preservation of cancer-related morphology.

Arms:
- Fixed residual-preserving stain augmentation, alpha=1.
- Uniformly weaker, alpha=mean training alpha.
- Adaptive controller.
- Shuffled strength, permuting the adaptive validation alpha values among images.
  This offline control exactly matches their empirical strength distribution;
  it is not a deployable policy.

Also report augmented-versus-source pixel RMS, so an apparent benefit caused by
near-identity augmentation is visible. Uniform training-mean alpha need not equal
the validation mean; the shuffled arm is the stronger matched-strength control.

Assess the association between q and fixed-arm sensitivity using Spearman rho.
A permutation test shuffles q within tissue classes (one-sided positive hypothesis).
Image bootstrap intervals for differences are descriptive, not patient-grouped;
patient identifiers are not available in the manifest. With a single training fit,
they also omit uncertainty in fitting the controller.

## Decision rule

Do not start long classifier runs solely because the code executes. Look for a
positive within-tissue association and consistent improvement against both weaker
and shuffled controls while retaining useful augmentation magnitude. A lower
gain alone is insufficient: identity mapping always has gain near one.

If supported, next test multiple training seeds, a second model/backbone, and
frozen evaluation splits. Required classifier ablations: standard RGB/geometric
augmentation, original SAN, residual-preserving fixed SAN, uniformly weaker SAN,
shuffled controller and adaptive controller. Keep model and training budgets fair.
Measure macro-F1, per-class recall, calibration, timing and augmentation failure
rates. Compare raw-image evaluation consistently before choosing normalization.
Do not claim clinical, cross-hospital or patient-level validation from this pilot.

## Prior art boundaries

- Stain SAN (2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11342968/
  Provides stain extraction and target-distribution sampling. Retain attribution.
- Data-driven color augmentation (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC9852546/
  Admits or discards generated styles based on neighbors in a reference stain
  database. Source decomposition diagnostics are our proposed different signal.
- CASA (2026 preprint): https://arxiv.org/html/2605.13889v1
  Adversarial stain perturbations with budgets calibrated from multicenter data.
  It already covers calibrated perturbation budgets; we do not claim that broadly.
- Understanding Stain Separation Improves Cross-Scanner Adenocarcinoma Segmentation
  with Joint Multi-Task Learning (2025): https://doi.org/10.3233/SHTI250272
  Joint reconstruction and segmentation objectives relate decomposition to
  downstream performance. This pilot uses explicit input diagnostics without
  training a decomposition network. The full method must be compared before
  any novelty claim.

Focused searches for condition-number/reconstruction-residual-controlled stain
augmentation did not identify an exact equivalent. That does not certify novelty.
The module and its mechanism require full related-work review and experimental
evidence before publication claims. Do not advertise a verified new algorithm.

## Run

From the parent of stain_research, using the project Python environment:

```powershell
& 'C:\Users\Amlan\Desktop\gastric detection\.venv\Scripts\python.exe' stain_research\controller_pilot.py --project 'C:\Users\Amlan\Desktop\gastric detection' --output stain_research\runs\controller_repeat
```

Use a fresh output path. Outputs preserve image identities, per-repeat metrics,
summary and preview. The main controller API is controller.apply(image,
stain_parameters, controller_parameters, rng, fixed_strength=None).
