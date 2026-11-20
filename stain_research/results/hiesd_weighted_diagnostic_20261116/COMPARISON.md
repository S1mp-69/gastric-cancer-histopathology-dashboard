# HiESD class-weighted diagnostic follow-up

One post-hoc comparison, seed 42. Same initial weights, split, geometry, normalization, optimizer, scheduler rule and maximum budget. Only the training loss receives inverse-frequency class weights. The validation NLL and evaluation rules remain unweighted.

| Metric | Standard | Class-weighted |
|---|---:|---:|
|Validation accuracy|0.4856|0.3589|
|Validation macro-F1|0.2954|0.2736|
|Selected epoch|16|3|
|Completed epochs|23|10|

| Class | Validation support | Standard recall | Weighted recall | Standard F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
|tumor|83|0.3253|0.8675|0.2085|0.4150|
|mucosa_inflammation|592|0.7264|0.4713|0.6902|0.5200|
|intestinal_metaplasia|256|0.1641|0.0273|0.1892|0.0320|
|lymphoid|12|0.5833|0.5000|0.3889|0.2927|
|other|99|0.0000|0.1010|0.0000|0.1081|

Macro-F1 difference: -0.0218; accuracy difference: -0.1267. These are descriptive differences, not significance or generalization claims.

Weights use N_train/(5*n_class) and were fixed from training counts without tuning: tumor=1.922500, mucosa_inflammation=0.362736, intestinal_metaplasia=1.522772, lymphoid=1.398182, other=2.848148.

Audit: all 1811 coarse annotation regions recomputed; 71 stratified source RGB reconstructions matched exactly. No manifest label-index inconsistency, exact train/validation patch overlap or patient overlap was detected. This is not expert validation of pathology labels or landmark registration. Some reviewed validation "other" patches appeared blurred; they were retained without post-hoc relabeling or filtering.

The subset is small, annotation-selected and strongly clustered: 82/83 validation tumor patches and 242/256 validation metaplasia patches belong to Patient30; 104/110 training lymphoid patches belong to Patient13. Only twelve validation lymphoid patches are present. Neither arm is a clinical model or an unbiased benchmark result.

Class weighting is an imbalance control, not a novel method and not evidence for the stain controller. Do not automatically search more weight formulas, change the validation split, or access reserved test data in response to this comparison.

Original baseline checkpoint, configuration, history and predictions hashes were checked unchanged after the follow-up. Latest/best weighted checkpoints stay isolated from the dashboard.
