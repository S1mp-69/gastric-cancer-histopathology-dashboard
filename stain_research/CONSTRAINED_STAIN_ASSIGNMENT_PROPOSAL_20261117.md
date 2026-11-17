# Concrete algorithm proposal: class-conditional distribution-preserving stain assignment

Status: proposed here; unimplemented and untested. Originality is not certified. This is not the controller evaluated in the completed HMU/HiESD runs, and none of their scores may be attributed to this proposal.

## Search record and limits

This follow-up issued twenty targeted web queries covering sample-adaptive augmentation, fixed distributions, class-conditional budgets, optimal transport, strength assignment and permutation-based policies. Two Google Scholar search URLs were also attempted directly; both were inaccessible through the available browsing tool. A Scholar-domain indexed query did not supply usable Scholar results. Do not say that Scholar was exhaustively screened or that all publications were searched.

Closest primary sources reviewed:

- SapAugment: Learning A Sample Adaptive Policy for Data Augmentation (2021 revision). Adapts augmentation using sample training loss; sample-dependent magnitudes are already prior art. https://arxiv.org/abs/2011.01156
- MetaAugment: Sample-Aware Data Augmentation Policy Learning (AAAI 2021). Learns sample-aware weighting of augmented examples through meta-learning. https://arxiv.org/abs/2012.12076
- When to Learn What: Model-Adaptive Data Augmentation Curriculum (ICCV 2023). Learns augmentation probabilities and magnitudes conditioned on the image/model state. https://openaccess.thecvf.com/content/ICCV2023/papers/Hou_When_to_Learn_What_Model-Adaptive_Data_Augmentation_Curriculum_ICCV_2023_paper.pdf
- CASA: Physics-Grounded Adversarial Stain Augmentation with Calibrated Coverage Guarantees (2026 manuscript). Adversarial Macenko-space perturbations with calibrated budgets are prior art. https://arxiv.org/abs/2605.13889
- Domain Adaptation using Optimal Transport for Invariant Learning in Histopathology Datasets (MIDL 2023 poster). Uses feature-space transport regularization across domains, including target-domain data; does not by itself establish the proposed within-class strength permutation. https://2023.midl.io/virtual/poster/P215.pdf
- FreeAugment (ECCV 2024). Learns augmentation type/order and magnitude ranges, including permutation relaxations for operation selection/order. A generic use of permutations in augmentation is not new. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03848.pdf

The exact class-conditional fixed-strength-multiset assignment mechanism below was not identified in the retrieved primary sources. This supports treating it as a candidate specific contribution, not claiming it is the first. Full citation and prior-art review remains necessary. No guarantee of positive results or acceptance follows from this search.

## Mechanism

For each training batch and class c, collect valid source images B_c and the existing train-fitted diagnostic controller strengths A_c={alpha_i:i in B_c}. Freeze these values for that batch: the proposed policy cannot create larger/smaller strengths, change their counts, or move strengths across tissue classes. Source RGB, basis, concentrations, target stain draw and intensity draw remain attached to their image.

Measure model response to a fixed small stain probe using TRAIN images only:

    g_i = [CE(f(T_epsilon(x_i)), y_i) - CE(f(x_i), y_i)] / epsilon.

Use the same target stain draw for the probe and final transform. Freeze BatchNorm statistics and disable stochastic model inference while estimating g_i; those passes do not update weights. The geometric/RGB randomness and image preprocessing for scoring must be specified and shared across comparator arms.

Choose a within-class strength permutation:

    pi_c* = argmax_pi sum_{i in B_c} g_i * alpha_{pi(i)}.

For this first-order surrogate, matching sorted sensitivities and sorted strengths solves the assignment (the rearrangement inequality is established mathematics, not a new theorem). It is a surrogate, not a proof of maximizing the full nonlinear augmented loss. Train on the selected stain transforms using the ordinary classifier loss. Groups with fewer than two valid images retain their original assignment; record how often reassignment is possible. Invalid decompositions remain unchanged-image fallbacks.

Exact invariant: each class's empirical strength multiset is unchanged. Thus its mean, variance and quantiles are preserved. This does not preserve the distribution of actual image/OD changes, does not prove label/morphology preservation, and does not guarantee robustness or clinical safety.

## Specific proposed difference

The learnable decision is the coupling between images and a fixed class-conditional strength bank, informed by current classifier stain sensitivity. The policy is not allowed to change the marginal bank. Unlike the existing numerical-diagnostic controller, it uses task-model response; unlike unconstrained adaptive magnitude policies, it explicitly freezes class-conditional empirical magnitudes.

Potential manuscript title: **Learning Stain Augmentation Assignment Under Fixed Class-Conditional Strength Distributions**.

Candidate contribution statement, pending prior-art clearance and results:

“We formulate classifier-informed stain-strength allocation as a class-conditional permutation problem, preserving empirical augmentation-strength distributions to isolate assignment effects, and evaluate a first-order allocation rule against diagnostic and randomized assignments.”

Do not replace 'candidate' with 'first', 'proven novel', 'safe' or 'superior' without evidence. Adding optimal-transport terminology to an ordinary assignment problem is not additional novelty.

## Minimal falsifiable experiment

First verify that the finite-difference sensitivity predicts held-out augmentation loss changes on TRAIN-only development images. Measure singleton class groups, finite-difference instability, clipped RGB/OD effects, extra inference time and assignment reproducibility. Stop the proposal if the surrogate is uninformative; do not tune on reserved test data.

If feasible, freeze a small comparison: standard augmentation; original diagnostic assignment; within-class random permutation; and the proposed sensitivity-ranked permutation. Consider reverse-ranked assignment as a mechanistic control. All stain arms use the same per-class bank, source images, stain draws, RGB/geometry, initial weights, optimizer, stopping and budgets. For compute comparisons, include the same scoring passes in decision-control runs or provide both epoch-matched and compute-matched results. Any warmup/probe epsilon must be fixed from development data before final comparison, not chosen from test outcomes.

Use the patient-grouped HiESD split, all relevant tissue classes, three matched seeds, macro-F1, class recall, calibration and a fixed stain-shift stress test. The existing HiESD cohort remains small and annotation-selected; the old HMU cohort lacks verified patient grouping. A stronger submission may need more patient coverage or an independently collected task-compatible cohort. Reserved test data remains excluded until the final protocol and claims are frozen.

Positive algorithmic results would require improvement over random assignment at the same marginal bank, not merely over an excessively strong fixed arm. A null result would not establish that no model-dependent assignment can work; it would concern this specific surrogate, bank and protocol.

No source code was changed, training launched, GPU checks run, or test data accessed for this proposal. Existing numerical and classifier results are preserved and should remain in the research record.
