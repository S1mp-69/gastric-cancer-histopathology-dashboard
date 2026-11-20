# Publication direction: 17 November 2026

Recommended working title: **Does Image-Specific Stain Augmentation Help? A Matched-Strength Study in Gastric Histopathology**.

This is a candidate empirical contribution, not certified novelty, an established beneficial algorithm, or a promise of acceptance.

## Exact research question

Does assigning augmentation strength using source-image decomposition diagnostics improve classification beyond (a) changing the overall strength distribution and (b) changing tissue-class-specific strength distributions?

The proposed contribution is a controlled evaluation separating these mechanisms, together with numerical-stability measurements, classifier performance, patient-aware analysis where mapping exists, and reproducible code. Permutation controls themselves are established experimental tools; do not claim their invention.

## Why this direction fits the actual evidence

Completed HiESD three-seed means: standard macro-F1 0.7270; fixed residual-preserving 0.7260; uniform weaker 0.7220; global shuffled 0.7350; adaptive 0.7131. Adaptive-minus-global-shuffled mean is approximately -0.0219. This is descriptive validation evidence, not a significance claim. HMU's separate three-seed adaptive-minus-shuffled mean was -0.00544. Do not pool the cohorts as comparable tasks or describe HMU as patient-independent.

Preprocessing sensitivity results and downstream classification are distinct endpoints. Initial numerical stability findings do not establish diagnostic relevance or clinical image quality. Current classifier evidence does not support the controller's intended improvement.

HiESD validation was used for stopping and its cohort selected using annotations. It has small patient counts and only seven QC-passing lymphoid validation patches. Three training seeds are not three independent patient cohorts. HMU lacks verified patient mapping and has exact-duplicate controls only. These limitations constrain the manuscript's strength and scope.

## Prior-art boundaries checked in this focused review

- Tellez et al. (2019), *Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology*: broad comparisons already exist. A generic claim that augmentation can outperform normalization is not new. https://arxiv.org/abs/1902.06543
- Marini et al. (2023), *Data-driven color augmentation for H&E stained images in computational pathology*: data-driven admissibility of augmented styles is existing work. Broad quality/data-aware stain augmentation is not a new category. https://pmc.ncbi.nlm.nih.gov/articles/PMC9852546/
- Kim et al. (2024), *Stain SAN*: already samples stain matrices from structured target distributions and evaluates downstream classification. Our implementation builds on it and retains attribution. https://pmc.ncbi.nlm.nih.gov/articles/PMC11342968/
- Hong (2026), *Physics-Grounded Adversarial Stain Augmentation with Calibrated Coverage Guarantees*: calibrated augmentation budgets and physically grounded/adversarial stain augmentation are existing claims in this manuscript. Do not claim those broadly. https://arxiv.org/abs/2605.13889
- Schoenpflug et al. (2026), *A protocol for evaluating robustness to H&E staining variation in computational pathology models*: controlled reference-based stain robustness evaluation already exists. Its decomposition/recomposition framework also adjusts the residual component. Robustness evaluation and residual handling alone are not sufficient novelty. https://doi.org/10.1016/j.jpi.2026.100702 ; https://arxiv.org/abs/2603.12886

The focused search did not identify the exact gastric diagnostic-assignment/strength-distribution/class-distribution comparison. This is a tentative gap in the reviewed literature, not proof that no prior work exists. Check citation networks and full texts before any priority statement. Application to gastric tissue alone does not establish novelty.

## Finite work needed to make this a stronger submission

1. Audit the completed matched runs and metrics before further fitting. Verify common configuration, starting weights, image order and preservation. Recompute patient/class summaries and uncertainty appropriate to the available patient counts; do not bootstrap patches as independent patients.
2. Freeze a within-class shuffled control, preserving each class's valid-image strength multiset. Run the existing three matched seeds on HiESD. This asks whether individual image assignment matters after tissue-class effects are held fixed. The labels are used only for an experimental falsification control, not proposed as a deployment policy. No score-dependent stopping or additional policy search.
3. Specify a fixed stain-shift stress test before running saved models. Reuse published reference conditions where feasible; cite their origin. Report synthetic robustness separately from real cross-center generalization. Keep the reserved test out of all selection and exploratory stress testing.
4. Freeze the final analysis and limited confirmatory question before a one-time reserved-test evaluation. Acquire and audit reserved data only at that stage. Restrict confirmation to methods and contrasts chosen in advance; disclose all prior validation use.
5. Integrate positive and negative findings into one manuscript with artifacts and explicit task differences. Seek mentor feedback on contribution and venue scope before submission. Acceptance is uncertain; reviewer expectations may require broader patient coverage or an independent cohort.

## Claims suitable only if the completed analyses support them

- We evaluated whether numerical decomposition diagnostics add information for per-image augmentation strength assignment, using strength-distribution-matched and class-conditional controls.
- Under the specified datasets, splits, models and budgets, the tested adaptive rule did not improve validation macro-F1 over its matched random-assignment control.
- Numerical stability under small input disturbances did not translate into an observed classification advantage for the tested rule.

Do not generalize this to every adaptive augmentation method, every gastric cohort, all stain shifts, or clinically validated safety. Do not claim morphology preservation without appropriate evidence. A standard ResNet model, an attention dashboard, a cache, or a combination of existing components is not by itself a defensible algorithmic novelty.

If a venue requires a beneficial new algorithm, the current evidence does not satisfy that requirement. A substantially different method would require new hypotheses, development data, and independent validation; renaming the current controller or adding architectural modules cannot supply that evidence.

No new training was started for this literature review. No GPU checks were run.
