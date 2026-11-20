# Literature comparison and defensible claims

Checked 17 November 2026 against primary publication records.

| Prior work | Established contribution | Implication for this analysis |
|---|---|---|
| Brabec, Komarek, Franc and Machlica, *On Model Evaluation under Non-constant Class Imbalance*, ICCS 2020 | Class imbalance can change scores and classifier ordering; evaluation can be adjusted without subsampling. | A ranking reversal caused by reweighting is not a new general discovery. https://arxiv.org/abs/2001.05571 |
| Opitz, *A Closer Look at Classification Evaluation Metrics and a Critical Reflection of Common Evaluation Practice*, TACL 12:820–836, 2024 | Analyzes prevalence invariance and other metric properties; ordinary macro-F1 is not prevalence invariant. | Equal averaging over class F1 values does not remove prevalence dependence of each precision term. https://aclanthology.org/2024.tacl-1.46/ ; DOI 10.1162/tacl_a_00675 |
| Opitz and Burst, *Macro F1 and Macro F1*, 2019, revised 2021 | Different formulas called macro-F1 can disagree and rank systems differently. | State that this study uses the arithmetic mean of five class-wise F1 scores. Our observed reversal is not a switch between formulas. https://arxiv.org/abs/1911.03347 |

## What the present data add

An independently reproduced gastric-histopathology case study connecting a matched stain-augmentation experiment, a separately measured training-selection sensitivity, and a class-specific explanation of evaluation-weight sensitivity. All fifteen runs and twenty-four predefined reweightings are retained, including results unfavorable to the adaptive controller.

This is a possible empirical component of the paper, not certified originality or a complete new method. The scoped literature comparison does not prove that this exact application has never been published. Its value depends on the full experiment, transparency, external validity and fit to the intended venue.

## Approved factual statements

- All360 weighted confusion matrices and macro-F1/accuracy values were reproduced independently using sklearn from saved per-image probabilities. Source prediction hashes were unchanged.
- The leading mean macro-F1 method changes between the observed validation distribution and hypothetical equal-class weighting.
- The adaptive-minus-shuffled contrast changes sign under some predefined hypothetical mixtures; neither prediction correctness nor class-conditional recall changes under pure class-mass reweighting.
- Lymphoid precision accounts for most of the observed-to-equal-class contrast shift. Both methods classify the same seven lymphoid patches correctly, but differ in false positives.
- Patient-only balancing preserves the original ordering and increases the adaptive-minus-shuffled deficit. The positive equal-class contrast must be reported alongside this unfavorable sensitivity result.

## Statements not supported

- We discovered a new property of macro-F1, a new metric, or a new reweighting method.
- Adaptive augmentation is now superior, robust across populations, or clinically effective.
- Twenty-four correlated weightings are independent replications or yield a statistical significance test.
- Changing training purity/QC caused the ranking reversal. That would require an intervention involving retraining and/or genuinely different evaluation cases.
- Equal class proportions reflect clinical gastric histotype prevalence or provide new independent test validation.
- The case study establishes publication novelty or guarantees acceptance.

## Recommended manuscript role

Keep the original matched stain-assignment question and all mixed/negative results visible. Use this analysis as a sensitivity subsection explaining the scope of ranking claims. Do not rebuild the paper around the claim that metric prevalence sensitivity itself is novel. No further training was needed for verification.

An appropriate claim is: "In this selected gastric validation cohort, the apparent ordering of stain-augmentation methods depended on the specified evaluation distribution; a class-wise decomposition identified rare-class precision as a major source of the change."

Still outstanding for a submission: full campaign source/checkpoint audit beyond this prediction analysis, a coherent final manuscript/report/deck refresh, mentor/author review, a fixed reserved-test protocol before any test access, and venue-specific preparation. This package does not mark those requirements complete.
