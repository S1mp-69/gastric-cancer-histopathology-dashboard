# Frozen reserved-patient test

The frozen final test completed on 1613 patches from 7 reserved patients. The prespecified adaptive-minus-within-class-shuffled mean macro-F1 difference was -0.0147, with descriptive patient-cluster interval [-0.0376, +0.0069]. Of 1,000 cluster draws, 31 lacked at least one class; fixed five-class zero-division-zero scoring was retained. Only 7 usable patient clusters limit precision. All models were frozen before acquisition; no tuning follows test access. This is held-out public-release testing, not independent clinical external validation.

| Arm | Mean accuracy | Mean macro-F1 | Mean balanced accuracy | ECE (15 bins) | Brier |
|---|---:|---:|---:|---:|---:|
| Standard | 0.6886 | 0.6600 | 0.7003 | 0.1186 | 0.4671 |
| Residual fixed | 0.6939 | 0.6696 | 0.6959 | 0.1233 | 0.4630 |
| Uniform weaker | 0.6822 | 0.6495 | 0.6802 | 0.1346 | 0.4787 |
| Global shuffled | 0.6694 | 0.6350 | 0.6605 | 0.1471 | 0.5117 |
| Adaptive | 0.6939 | 0.6602 | 0.7052 | 0.1148 | 0.4611 |
| Within-class shuffled | 0.7057 | 0.6749 | 0.7087 | 0.1093 | 0.4473 |

All 18 seed results:

| Seed | Arm | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|
| 42 | Standard | 0.7260 | 0.6878 | 0.7362 |
| 42 | Residual fixed | 0.7179 | 0.6750 | 0.7131 |
| 42 | Uniform weaker | 0.6981 | 0.6633 | 0.6960 |
| 42 | Global shuffled | 0.7024 | 0.6811 | 0.7184 |
| 42 | Adaptive | 0.6950 | 0.6636 | 0.7032 |
| 42 | Within-class shuffled | 0.7074 | 0.6859 | 0.7134 |
| 43 | Standard | 0.6826 | 0.6638 | 0.6992 |
| 43 | Residual fixed | 0.6683 | 0.6545 | 0.6724 |
| 43 | Uniform weaker | 0.6999 | 0.6549 | 0.6920 |
| 43 | Global shuffled | 0.6057 | 0.5416 | 0.5627 |
| 43 | Adaptive | 0.6931 | 0.6577 | 0.7094 |
| 43 | Within-class shuffled | 0.7105 | 0.6583 | 0.7039 |
| 44 | Standard | 0.6572 | 0.6283 | 0.6654 |
| 44 | Residual fixed | 0.6956 | 0.6793 | 0.7022 |
| 44 | Uniform weaker | 0.6485 | 0.6304 | 0.6525 |
| 44 | Global shuffled | 0.6999 | 0.6822 | 0.7005 |
| 44 | Adaptive | 0.6937 | 0.6594 | 0.7030 |
| 44 | Within-class shuffled | 0.6993 | 0.6805 | 0.7089 |

No model selection or tuning follows these results. The same patients/cases recur across seeds; do not multiply the patient count by 18. See PER_CLASS_RESULTS.md for all class failures and independent verification receipts under publication_results.
