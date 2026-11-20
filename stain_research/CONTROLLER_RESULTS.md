# Controller pilot results

Completed 12 November 2026. This is an exploratory numerical preprocessing study,
not an evaluation of classifier accuracy, medical quality, or publication novelty.

## Data and execution

160 training images fitted stain statistics and controller bounds. 160 separate
validation images (20/class) were evaluated using six noise realizations per image.
No test images were read, no classifiers were trained and no original files changed.
All image/repeat metrics and sample identities are in runs/controller_v1/.

## Findings

Risk score versus fixed-arm sensitivity: Spearman rho 0.5570. A within-tissue-class
permutation test gave one-sided p=0.001 with 1,000 permutations. This is exploratory
and does not account for unverified patient clusters or selection of the hypothesis.

| Arm | Mean noise-amplification gain (lower is more stable) | Mean augmentation pixel RMS |
| --- | --- | --- |
| Fixed residual-preserving augmentation | 1.3830 | 10.5186 |
| Uniformly weaker | 1.2409 | 6.9091 |
| Shuffled strength | 1.2299 | 6.8336 |
| Adaptive | 1.1843 | 6.6063 |

Adaptive gain was 14.37% lower than fixed and 4.56% lower than uniformly weaker.
Adaptive minus uniform mean difference: -0.0565; descriptive paired image
bootstrap 95% interval [-0.1118, -0.0084]. Patient grouping and training-fit
uncertainty are not reflected by this interval.

Adaptive minus shuffled difference: -0.0456; a separate descriptive paired image
bootstrap (2,000 replicates, seed 42) gave [-0.1012, 0.0002]. This interval includes
zero. Therefore superiority over random strength assignment is NOT established.
Only one shuffled assignment was evaluated. The scripts preserve the full data
needed for further controls; an eventual paper needs repeated assignments.

The adaptive arm's augmentation RMS was also lower than both controls. Strength
matching is not the same as matching actual image change, so some benefit may
still be due to reduced perturbation. A pixel/OD-change-matched control is needed.

There were no fallback transformations in this sample. Adaptive strengths differed
by class: mean 0.4735 for TUM versus 0.7775 for MUS. Class-specific effects must
be measured in a subsequent classifier study; these differences are not inherently
good and may reduce useful tumor augmentation.

## Interpretation and next decision

The premise has preliminary support: numerical decomposition diagnostics correlate
with sensitivity to small pixel disturbances. The controller works and shows a
numerical stability improvement, but the random-assignment control is inconclusive.
Do not claim a proven beneficial classifier or a confirmed novel algorithm.

Before expensive training, repeat on a second validation sample/noise setting,
isolate the condition-number and reconstruction-error components, repeat shuffled
assignments and match actual augmentation change. If supported, run a small
classifier ablation using equal training budgets. A full three-seed comparison
is justified only if those checks remain favorable.

## Verification

check.py passes the existing upstream-compatible SVD checks plus controller checks:
exact zero-strength identity, repeatability, monotonic strength bounds, and blank
fallback. The preview was visually inspected for gross geometric displacement.
That inspection is not expert confirmation of preserved diagnostic features.

Dependencies: existing project NumPy, pandas, Pillow, and SciPy (a dependency of
scikit-learn). No additional packages were installed for this pilot.
