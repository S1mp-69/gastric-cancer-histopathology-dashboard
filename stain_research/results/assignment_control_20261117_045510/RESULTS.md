# Fixed-strength assignment control

Higher CE indicates a harder adversarial challenge on this frozen model, not improved classifier accuracy or clinical benefit. Rank assignment is exact for its linear surrogate, not guaranteed optimal for nonlinear CE.

| Stain direction | Ranked CE | Random mean CE | Difference | Reverse-ranked CE |
|---|---:|---:|---:|---:|
|Same as probe|0.04506|0.04476|+0.00030|0.04453|
|Independent|0.04718|0.04721|-0.00003|0.04695|

Exact class/batch strength multiset checks passed for all23 policies. Reassignable images: 96.9%. Primary engineering gate passed: True.

All160 images are reused training-only development patches (32/class); there are20 random assignments, not20 independent datasets. The source checkpoint historically used validation early stopping. The small balanced diagnostic sample and synthetic batches differ from natural imbalanced training batches. No geometric/RGB jitter was applied. Paired image losses share patients; cluster intervals are descriptive and condition on this model, sample and fixed groups.

The exact candidate table amortizes repeated policy scoring; its timing does not measure an online training policy. Extra probe passes, CPU reconstruction and assignment overhead must be measured during a subsequent controlled pilot. No retraining or validation/test image access occurred. This check cannot establish novelty or downstream benefit.
