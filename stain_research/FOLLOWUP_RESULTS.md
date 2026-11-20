# Follow-up decision: proceed to a limited classifier pilot

The additional checks support further testing of the mechanism, but do not
establish improved classification, independent statistical confirmation or novelty.
Do not begin a full multi-seed training campaign on the strength of this alone.

## What was tested

80 new validation images, 10 per class, explicitly excluding all 160 images from
the initial controller validation pilot. Training-fitted controller and stain
parameters were reused unchanged. Two noise levels, sigma 0.5 and 2.0 RGB units,
were tested with three realizations per image. The test partition was untouched.

Controls matched actual average augmentation magnitude, measured as the mean
per-image RGB RMS change. They used no labels or classifier outputs. This is
an offline experimental control, not a policy to be fitted on deployment data.

## Numerical results

| Noise sigma | Fixed augmentation gain | Change-matched uniform gain | Adaptive gain |
| --- | --- | --- | --- |
| 0.5 RGB units | 1.7247 | 1.4249 | 1.3049 |
| 2.0 RGB units | 1.4716 | 1.2532 | 1.1646 |

Lower noise-amplification gain means greater numerical stability in this test.
The average augmentation RMS was matched at approximately 6.3479 RGB units.
The adaptive gain was lower by about 8.4% and 7.1% versus matched uniform control.
These are not classifier accuracy improvements.

However, descriptive stratified image-bootstrap intervals for adaptive-minus-
matched-uniform differences included zero at both noise levels:

- sigma 0.5: difference -0.1201, interval [-0.2972, 0.0197].
- sigma 2.0: difference -0.0886, interval [-0.2112, 0.0073].

500 random strength assignments were tested per noise level, each rescaled to
match the same average image change. Four of 500 were at least as stable as the
adaptive policy at sigma 0.5; zero of 500 at sigma 2.0. These frequencies describe
assignments on this particular sample, not uncertainty across patients or datasets.
Zero of 500 does not mean zero probability. This comparison does not supersede
the image-bootstrap uncertainty above.

## Component checks

The condition-number-only controller reduced mean gain at both noise levels
versus its own change-matched uniform control. Its intervals also included zero.
The residual-only controller showed a smaller reduction at sigma 0.5 and no
reduction at sigma 2.0. Thus the current evidence suggests that condition number
may be the more useful component; it does not justify dropping other components
and claiming a validated improved policy on the same validation data.

Condition-number versus fixed-arm sensitivity Spearman correlations were 0.3250
and 0.4358; reconstruction-error correlations were 0.5067 and 0.4316.

## Important bounds on this evidence

- This is sensitivity to tiny synthetic RGB pixel disturbances, not stain-shift
  generalization, preserved morphology or clinical effectiveness.
- Strength controls interpolate 21-point curves. A separate direct primary
  uniform-control check is recorded in direct_control_check.json.
  It reproduced the uniform gain within 0.00015 and matched mean RGB RMS
  within 0.00037 units. Its uncertainty intervals still included zero.
- Matching average RGB RMS does not match per-image or optical-density changes.
- Patient/slide identifiers are not verified; image bootstrap ignores clustering.
- These follow-ups were motivated by the initial exploratory findings; they are
  not a preregistered independent confirmatory study.
- No classifier training has been performed, and novelty remains unconfirmed.

## Recommended next action on the GPU PC

Run environment checks and short smoke tests, then one limited classifier ablation
before expanding. Compare standard augmentation, original SAN, residual-preserving
fixed SAN, uniformly weaker SAN, and the full controller using identical splits,
initialization and budgets. Record class-specific effects and reload the best
checkpoint for evaluation. Expand to multiple seeds only if downstream results
show useful benefit beyond the weaker control. Keep final test data untouched
until choices are fixed. See STRONG_PC_HANDOFF.md for the pasteable prompt.

Exact measurements and seeds are saved under runs/strength_controls_v1/; code is
strength_controls.py. All controller and upstream compatibility checks passed.
