# HiESD selected-subset baseline

Single-seed exploratory feasibility result. This is not an unbiased held-out evaluation, a clinical validation, or evidence of controller novelty.

Completed epochs: 23; selected epoch: 16; stop: early_stopping.
Validation accuracy: 0.4856; macro-F1: 0.2954.

| Group | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
|tumor|83|0.1534|0.3253|0.2085|
|mucosa_inflammation|592|0.6575|0.7264|0.6902|
|intestinal_metaplasia|256|0.2234|0.1641|0.1892|
|lymphoid|12|0.2917|0.5833|0.3889|
|other|99|0.0000|0.0000|0.0000|

Patient split is preserved. Validation annotations informed subset selection; only twelve validation lymphoid patches are present. Patches are clustered within a small number of patients. No patient-level confidence interval is claimed. Reserved test data was not used.

The broad five-group semantics were checked against HiESD Figure 1/Table 2. This model, patch extraction, subset selection, and budget differ from the published baseline protocol, so numerical scores are not directly comparable.

Best checkpoint SHA256: `55736fcb63b6b68401aba815e6b33856ed6d9be1898599d1bf0f2bad9973b7e6`. Reloaded validation probabilities matched saved outputs.

Train smoothed cross-entropy and validation unsmoothed NLL are distinct losses; their curves are not directly equivalent.

Additional limitations: coarse mask boundaries, no expert patch-label re-review, no GrandQC artifact mask filtering, strong class imbalance, and only one training seed.

Before interpreting augmentation comparisons, use identical split, initialization, geometry, normalization, and budgets. Do not choose methods using reserved test data.
