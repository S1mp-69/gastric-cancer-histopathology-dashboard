# Training-only benchmark construction audit

The original 2543-patch training selection was reconstructed exactly from masks and fixed sampling rules.

| Policy | Patches | Patients | Added | Removed | Membership churn | Class-mix TV |
|---|---:|---:|---:|---:|---:|---:|
|reference|2543|16|0|0|0.0%|0.0000|
|purity_0.6|3594|16|1469|418|47.0%|0.0966|
|purity_0.75|3125|16|806|224|30.8%|0.0613|
|purity_0.95|2266|16|91|368|17.4%|0.0412|
|artifact_0.0|2500|16|31|74|4.1%|0.0038|
|artifact_0.1|2552|16|13|4|0.7%|0.0007|
|artifact_0.2|2560|16|26|9|1.4%|0.0011|
|quality_block_off|2762|16|323|104|14.9%|0.0162|
|cap64|1789|16|0|754|29.7%|0.0760|
|uncapped|3996|16|1453|0|36.4%|0.1454|

Predefined selection-sensitivity gate passed: True.

Descriptive change in the selected training cohort; not proof of mislabeled tissue, clinical relevance, changed model rankings, or novelty.

The slide universe was previously annotation-selected. Changes in purity can change class and patient composition; changes after capping may replace rather than simply add patches. Artifact-threshold alternatives retain the original tissue/unknown tests. The quality-block-off alternative removes those three tests as one block. Original alignment exclusions remain fixed. None of these alternatives is established as clinically correct.

No model outputs, validation/test masks, new RGB images or GPU checks were used. Patient IDs/split metadata were read only to restrict the audit to training. No existing manifests, patches or checkpoints were overwritten.

Next, if worthwhile: freeze a common evaluation universe and patient/class-standardized analysis before inspecting saved model predictions. Study ranking stability separately from sample-selection sensitivity. Do not choose a threshold based on the most favorable accuracy or relabel tissue without expert evidence.
