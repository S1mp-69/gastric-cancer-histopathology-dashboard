# HiESD selected-subset class-weighted baseline

Single-seed exploratory feasibility result. This is not an unbiased held-out evaluation, a clinical validation, or evidence of controller novelty.

Completed epochs: 10; selected epoch: 3; stop: early_stopping.
Validation accuracy: 0.3589; macro-F1: 0.2736.

| Group | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
|tumor|83|0.2727|0.8675|0.4150|
|mucosa_inflammation|592|0.5800|0.4713|0.5200|
|intestinal_metaplasia|256|0.0385|0.0273|0.0320|
|lymphoid|12|0.2069|0.5000|0.2927|
|other|99|0.1163|0.1010|0.1081|

Patient split is preserved. Validation annotations informed subset selection; only twelve validation lymphoid patches are present. Patches are clustered within a small number of patients. No patient-level confidence interval is claimed. Reserved test data was not used.

The broad five-group semantics were checked against HiESD Figure 1/Table 2. This model, patch extraction, subset selection, and budget differ from the published baseline protocol, so numerical scores are not directly comparable.

Best checkpoint SHA256: `e97a45ad9a6f154f1770c7dc4855c33ed090a55859d60981bc92cba635717ed4`. Reloaded validation probabilities matched saved outputs.

Train weighted smoothed cross-entropy and validation unweighted unsmoothed NLL are distinct losses; their curves and training losses between arms are not directly comparable.

Additional limitations: coarse mask boundaries, no expert patch-label re-review, no GrandQC artifact mask filtering, strong class imbalance, and only one training seed.

Before interpreting augmentation comparisons, use identical split, initialization, geometry, normalization, and budgets. Do not choose methods using reserved test data.
