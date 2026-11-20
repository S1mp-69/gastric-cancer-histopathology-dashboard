# HiESD data flow

The public release contains 104 slides from 44 patients. Allocation is 30 training, seven validation and seven reserved-test patients. Selected/QC-passing sampled coverage is smaller than allocation and must not be presented as the whole release.

| Cohort | Sampled patches | Patients | Slides represented |
|---|---:|---:|---:|
| train | 2543 | 16 | 20 |
| validation | 1251 | 7 | 14 |
| reserved_test | 1613 | 7 | 13 |

Class supports:

| Group | Training | Validation | Reserved test |
|---|---:|---:|---:|
| tumor | 253 | 99 | 316 |
| mucosa_inflammation | 1484 | 823 | 786 |
| intestinal_metaplasia | 373 | 166 | 203 |
| lymphoid | 206 | 7 | 9 |
| other | 227 | 156 | 299 |

Annotation/QC-selected subsets; allocated patients differ from sampled coverage. Coarse masks and small rare-class supports; public-release holdout, not independent cross-hospital validation. QC receipts identify each source alignment check, exclusion and artifact-screening result. No test cases were reassigned to improve these counts.
