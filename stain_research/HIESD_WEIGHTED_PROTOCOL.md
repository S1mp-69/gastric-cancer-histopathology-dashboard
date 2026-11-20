# HiESD class-weighted diagnostic follow-up

This single-seed follow-up is motivated by the unweighted baseline's failures.
It is post hoc, exploratory, and not a confirmatory novelty experiment.

Data audit: all 1811 retained patch regions were checked against coarse released
annotation colors, and 71 stratified RGB patches regenerated exactly from their
source slides. No label-index or extraction inconsistency was found on these
checks. Expert tissue-label review and artifact filtering remain outstanding.
Some reviewed validation 'other' patches appear blurred; no post-hoc filtering,
relabeling, or split change is applied in this comparison.

Freeze the same 769 training and 1042 validation records, seed 42, original
initial.pt bytes, train-only normalization, standard stochastic augmentation,
batch size 32, optimizer and scheduler rules, selection metric, early stopping
patience seven, and maximum forty epochs. Only the training loss changes:
weighted label-smoothed cross entropy with w_c = N_train/(5*n_c).
Train counts in fixed class order tumor/mucosa/metaplasia/lymphoid/other are
80, 424, 101, 110, 54. Weights are 1.9225, 0.3627358491, 1.5227722772,
1.3981818182, 2.8481481481. No weighted sampling or additional policy change.
Validation NLL is unweighted, matching the original scheduler's monitored metric.
The scheduler rule matches, but its realized learning rates and stopping epoch may
differ as a consequence of changed fitting. Weighted and unweighted training loss
magnitudes are not directly comparable.

The isolated trainer is hiesd_weighted_baseline.py; the original trainer remains
byte-identical to the baseline's recorded source hash. The narrow source diff is
stored in results/hiesd_weighted_diagnostic_20261116/trainer_changes.diff.
Common configuration fields and identical initial-checkpoint SHA256 are asserted
before fitting. A disposable preflight measured about 120 images/s and estimated
5.08 minutes for forty epochs, excluding checkpoint/report overhead. The actual
run starts fresh from the original initial checkpoint, not the dry-run state.

Latest/best full checkpoints are saved each epoch under
stain_research/runs/hiesd_weighted_seed42_20261116. No dashboard activation occurs.
The local progress display uses no ChatGPT calls. The supervisor generates plots
and a comparison report under results/hiesd_weighted_diagnostic_20261116 after
successful completion, checking that original baseline artifacts remain unchanged.
No additional experiments start automatically. Interrupted training can resume
with hiesd_weighted_baseline.py --run stain_research/runs/hiesd_weighted_seed42_20261116 --resume.

Interpret per-class recall, F1 and accuracy tradeoffs descriptively. The selected
cohort is small and imbalanced within patients: Patient30 contributes 82/83
validation tumor and 242/256 validation metaplasia patches. Only twelve validation
lymphoid patches are present. No patient-level uncertainty or statistical
superiority is established by this single-seed comparison. Reserved-test images,
annotations and predictions remain excluded.
