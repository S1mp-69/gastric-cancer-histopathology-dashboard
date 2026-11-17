# Fixed-prediction benchmark sensitivity

All15 original validation accuracy/macro-F1 values recomputed from saved probabilities. Same1251 cases and ordered identities throughout; all predictions preserved.

| Scheme | Mean-F1 winner | Adaptive minus shuffled |
|---|---|---:|
|observed_validation|shuffled|-0.02187|
|uniform_patients|shuffled|-0.05754|
|uniform_classes|residual_fixed|+0.01043|
|uniform_classes_and_patients|residual_fixed|+0.01691|
|trainmix_reference__patch|residual_fixed|+0.00267|
|trainmix_reference__patient_class|residual_fixed|+0.01111|
|trainmix_purity_0.6__patch|residual_fixed|+0.00637|
|trainmix_purity_0.6__patient_class|residual_fixed|+0.01557|
|trainmix_purity_0.75__patch|residual_fixed|+0.00522|
|trainmix_purity_0.75__patient_class|residual_fixed|+0.01398|
|trainmix_purity_0.95__patch|residual_fixed|+0.00056|
|trainmix_purity_0.95__patient_class|residual_fixed|+0.00890|
|trainmix_artifact_0.0__patch|residual_fixed|+0.00254|
|trainmix_artifact_0.0__patient_class|residual_fixed|+0.01099|
|trainmix_artifact_0.1__patch|residual_fixed|+0.00268|
|trainmix_artifact_0.1__patient_class|residual_fixed|+0.01112|
|trainmix_artifact_0.2__patch|residual_fixed|+0.00271|
|trainmix_artifact_0.2__patient_class|residual_fixed|+0.01118|
|trainmix_quality_block_off__patch|residual_fixed|+0.00273|
|trainmix_quality_block_off__patient_class|residual_fixed|+0.01065|
|trainmix_cap64__patch|residual_fixed|+0.00545|
|trainmix_cap64__patient_class|residual_fixed|+0.01393|
|trainmix_uncapped__patch|residual_fixed|-0.00401|
|trainmix_uncapped__patient_class|residual_fixed|+0.00562|

Winner changes relative to observed validation: 22/23.
Adaptive/shuffled mean sign reversals: 21/23.

Training-derived class masses are hypothetical evaluation distributions, not validation purity selections. Within-class patient balancing gives equal mass to each represented patient within a class. Global patient balancing is a separate scenario. Class-mixture changes can change precision/F1 without changing class-conditional recall; this is a metric/population effect, not evidence of a better classifier.

No new validation/test images or masks were opened and no model was trained or run. Original validation was already used for early stopping and the cohort was annotation-selected. Only seven lymphoid cases are present, clustered within very few patients. Heavy weights may concentrate on a small number of patches: see weight_details in RESULTS.json. No independent patient confidence interval or clinical benefit is claimed.

Actual effects of retraining on alternative patch policies and performance on newly retained boundary cases remain untested. Do not describe this analysis as proving those effects, correcting annotation errors, or validating a novel method.
