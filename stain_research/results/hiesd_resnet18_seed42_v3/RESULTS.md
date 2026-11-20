# HiESD selected-subset ResNet18 baseline after consistent QC

Single-seed exploratory feasibility result. This is not an unbiased held-out evaluation, a clinical validation, or evidence of controller novelty.

Completed epochs: 22; selected epoch: 15; stop: early_stopping.
Validation accuracy: 0.8066; macro-F1: 0.7324.

| Group | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
|tumor|99|0.9625|0.7778|0.8603|
|mucosa_inflammation|823|0.8092|0.9380|0.8689|
|intestinal_metaplasia|166|0.8968|0.8373|0.8660|
|lymphoid|7|0.8750|1.0000|0.9333|
|other|156|0.2593|0.0897|0.1333|

Patient split is preserved. Validation annotations informed subset selection; 7 validation lymphoid patches are present. Patches are clustered within a limited number of patients. No patient-level confidence interval is claimed. Reserved test data was not used.

The broad five-group semantics were checked against HiESD Figure 1/Table 2. This model, patch extraction, subset selection, and budget differ from the published baseline protocol, so numerical scores are not directly comparable.

Best checkpoint SHA256: `e060317f2611c418e676f5c5f6482cf0dec95ce05dc333c09c085619a24ad293`. Reloaded validation probabilities matched saved outputs.

Train smoothed cross-entropy and validation unsmoothed NLL are distinct losses; their curves are not directly equivalent.

Automated released GrandQC masks were applied consistently under a frozen QC protocol; this does not establish expert quality validation or landmark registration. Additional limitations: coarse annotation boundaries, no expert patch-label re-review, class imbalance and only one training seed. The cohort, normalization and backbone differ from prior pilot runs: this is a new reference baseline, not a controlled attribution of score changes.

Before interpreting augmentation comparisons, use identical split, initialization, geometry, normalization, and budgets. Do not choose methods using reserved test data.
