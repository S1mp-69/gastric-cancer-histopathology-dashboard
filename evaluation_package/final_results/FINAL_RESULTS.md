# Completed campaign analysis

All 12 runs completed. Three matched seeds per arm. Validation only.

| Arm | Accuracy mean ± SD | Macro F1 mean ± SD | TUM recall mean |
|---|---:|---:|---:|
| standard | 67.70 ± 1.15% | 0.6776 ± 0.0106 | 79.97% |
| residual_fixed | 68.30 ± 1.38% | 0.6822 ± 0.0141 | 74.18% |
| uniform_weaker | 68.22 ± 0.63% | 0.6804 ± 0.0061 | 75.56% |
| adaptive | 68.41 ± 0.59% | 0.6836 ± 0.0047 | 75.56% |

Adaptive minus control, paired macro-F1 differences:
- standard: mean +0.0060; seeds [0.0021, 0.0023, 0.0137]; exploratory 95% t interval [-0.0105, 0.0225]. n=3 and strong assumptions; not patient-level uncertainty.
- residual_fixed: mean +0.0014; seeds [0.0227, -0.0101, -0.0084]; exploratory 95% t interval [-0.0445, 0.0473]. n=3 and strong assumptions; not patient-level uncertainty.
- uniform_weaker: mean +0.0032; seeds [0.0128, 0.0044, -0.0076]; exploratory 95% t interval [-0.0223, 0.0286]. n=3 and strong assumptions; not patient-level uncertainty.

All completed-run best hashes, initialization hashes, shared configurations, ordered validation identities, finite best model tensors, prediction metric recomputation and resumable checkpoint keys passed audit. These checks do not independently re-run all inference or prove bit-identical interrupted training.

Inspect class recall and learning curves; epoch ceilings do not prove convergence. Historical numerical preprocessing results and the earlier negative three-epoch pilot remain separate. The test set remains unexamined. Patient/slide mapping and near-duplicate control are unresolved. Shuffled diagnostics were only evaluated in the short pilot, so causal benefit from diagnostic assignment is unconfirmed.

Recovery logs retain AMP-handler and Windows file-lock failures. No new training was launched for this analysis.
