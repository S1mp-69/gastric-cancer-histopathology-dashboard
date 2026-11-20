# Broader publication-gap review — 17 November 2026

## Scope and conclusion

Twenty-four targeted search queries screened gastric classification, stain augmentation, decomposition uncertainty, selective classification, boundary/annotation uncertainty, soft labels, and HiESD-specific follow-ups. Searches surfaced publisher articles, PubMed/PMC, arXiv, CVF/ECVA/OpenReview proceedings and official dataset/code pages. Selected closest primary sources were opened. This is a focused research landscape search, not an exhaustive systematic review, citation-network audit, patent search, or certification of priority.

The search does not justify a claim that a beneficial, novel algorithm has already been found. It does identify a dataset-specific question worth a low-cost feasibility audit: **Are gastric stain-augmentation rankings robust to the annotation-to-patch labeling and selection process?**

Working title: **From Coarse Annotations to Confident Predictions: Evaluating Label Stability and Stain Augmentation in Gastric Histopathology**.

This is a new candidate direction alongside—not a replacement for—the evidence-backed matched-strength study. The existing controller's negative results remain reportable and must not be hidden by changing the topic.

## Evidence for the problem

The HiESD authors explicitly describe their annotations as spatially imprecise and unsuitable for highly precise boundary segmentation. Their task labels arise by projecting coarse masks onto patches. This motivates investigating labeling/selection sensitivity; it does not demonstrate that current labels are wrong.

Source: Wang et al. (2025), DOI 10.1038/s41597-025-05679-1. https://pmc.ncbi.nlm.nih.gov/articles/PMC12311038/
Official data/code: https://github.com/JSGe-AI/HiESD

Current work has a fixed >=90% patch purity rule, automated QC, patient caps and substantial validation class/patient imbalance. Those decisions define which tissue is evaluated. A study can ask whether model/augmentation conclusions survive plausible alternative annotation-to-patch decisions, with patient-aware aggregation and a fixed image universe.

## Closest competing work and boundaries

1. **MixPatch (2022)** already uses proportion-based soft labels for mixed histopathology regions. Plain tissue-proportion soft labels are not new. It creates mixed patches from pathologist-confirmed clean patches, unlike estimating label stability from existing coarse region annotations. https://www.mdpi.com/2075-4418/12/6/1493
2. **Learning to segment fetal brain tissue from noisy annotations (2023)** introduces spatially varying boundary-aware label smoothing. Shift-averaged labels can overlap mathematically with spatial smoothing; do not claim a new smoothing algorithm without deriving the distinction. https://pmc.ncbi.nlm.nih.gov/articles/PMC9974964/
3. **Impact of Annotation Noise on Histopathology Nucleus Segmentation (2022)** studies synthetic annotation corruptions, including contour, over/undersizing and jitter. Annotation perturbation itself is established. https://publikationen.bibliothek.kit.edu/1000150405/149254045
4. **Histotyping prostate study (2021)** explicitly perturbs/erodes tumor annotations and assesses downstream risk stability. Generic annotation robustness is established. https://www.nature.com/articles/s41698-021-00174-3
5. **Benchmarking common uncertainty estimation methods with histopathological images under domain shift and label noise (2023)** evaluates uncertainty/rejection, ensembles and test-time augmentation. Combining these standard elements does not establish novelty. https://arxiv.org/abs/2301.01054
6. **SAFE-MedPath (October 2026 preprint, not peer-reviewed)** combines stain augmentation, a ResNet18 ensemble, temperature scaling and conformal prediction. This is a particularly close collision for a generic 'safe calibrated histopathology' proposal. https://www.preprints.org/manuscript/202610.0616
7. **Histopath-C (WACV 2026)** benchmarks realistic histopathology corruptions including controlled stain shifts. A generic corruption benchmark would overlap. https://openaccess.thecvf.com/content/WACV2026/papers/Noori_Histopath-C_Towards_Realistic_Domain_Shifts_for_Histopathology_Vision-Language_Adaptation_WACV_2026_paper.pdf
8. **Stain Consistency Learning (2026 journal article; earlier preprint)** already combines stain-specific augmentation with consistency learning. This is not an unoccupied method category. https://pmc.ncbi.nlm.nih.gov/articles/PMC13278751/
9. **Test-Time Stain Adaptation with Diffusion Models (ECCV 2024)** already addresses test-time stain adaptation. Diffusion adaptation is also less appropriate for the current small GPU and short project horizon. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05175.pdf
10. **Bayesian K-SVD (2022)** already applies Bayesian stain decomposition to normalization, augmentation and cancer classification. Bayesian decomposition alone is not a new contribution. https://doi.org/10.1016/j.compmedimag.2022.102048

The exact proposed interaction study on coarse gastric mask-to-patch selection and matched stain augmentation was not identified in this search. That absence supports further investigation only, not a 'first' claim. Annotation uncertainty and label-noise studies must still be compared in full before finalizing a novelty statement.

## Feasibility-first plan; no runs launched for this review

1. Freeze a sensitivity protocol before inspecting new performance results. Use training masks first to measure sensitivity of class proportions and eligibility under explicitly bounded coordinate shifts/boundary changes. Specify scale in mask, level-zero and output pixels. These are synthetic scenarios, not measured inter-pathologist uncertainty. Do not assume a particular displacement is clinically plausible.
2. Keep RGB images fixed when analyzing label-only perturbations. Distinguish label stability from purity-threshold selection, artifact filtering, class mix and patient composition. Do not change all of these and attribute an effect to one variable.
3. Quantify whether a material fraction of patches or eligibility decisions actually changes. Our current >=90%-pure sample may be too stable; if so, do not force larger perturbations until a desired finding appears. Boundary patches from existing training slides could form a separately frozen development cohort, identically included for every training arm.
4. If a methodological experiment is warranted, compare hard labels, generic label smoothing, original area-proportion targets, and an explicitly defined annotation-stability target/weight. All arms need the same RGB cohort, backbone, augmentation, initial weights, sampling and budget. Demonstrate a contribution beyond known smoothing rather than renaming it.
5. Evaluate unchanged reference validation first. If additional boundary validation is created, freeze it without selecting based on failures and report agreement with coarse masks rather than claiming new expert ground truth. Synthetic label variants do not reveal which diagnosis is correct.
6. For interaction analysis, compare saved stain-model predictions under pre-specified evaluation selections, on common cases where appropriate. Report patient/class composition, retained fraction, per-class metrics, patient-aware uncertainty and ranking stability. Avoid declaring the most favorable purity threshold the final benchmark.
7. Preserve the reserved test for a protocol frozen after development. Do not use it to choose perturbations, thresholds, losses, or a publication story.

## Decision

Fastest route from current evidence: matched-strength empirical paper, with within-class control and a fixed robustness analysis as described in PUBLICATION_DIRECTION_20261117.md.

New method candidate: annotation-stability-aware patch learning, conditional on material measurable instability and a mathematical/empirical distinction from existing spatial smoothing and proportion targets. Novelty and positive benefit remain unconfirmed; expert annotation evidence or a second coarse-annotation dataset would strengthen it substantially.

No new research model was trained, no GPU checks were run, no reserved-test data were accessed, and no existing experiment was overwritten for this search. Neither path guarantees publication or a particular venue tier.
