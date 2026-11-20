# Per-class results

Validation uses the same 1,251 cases across seeds; support counts are not multiplied by the number of models. Means below describe training-seed variability, not new patient cohorts. Partial arms are labeled by their completed seed count.

| Arm | Class | Seeds | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Standard | tumor | 3 | 99 | 0.9048 | 0.8586 | 0.8776 |
| Standard | mucosa_inflammation | 3 | 823 | 0.8109 | 0.9251 | 0.8641 |
| Standard | intestinal_metaplasia | 3 | 166 | 0.8599 | 0.7992 | 0.8278 |
| Standard | lymphoid | 3 | 7 | 0.8750 | 1.0000 | 0.9333 |
| Standard | other | 3 | 156 | 0.2670 | 0.0919 | 0.1323 |
| Residual fixed | tumor | 3 | 99 | 0.8928 | 0.9394 | 0.9139 |
| Residual fixed | mucosa_inflammation | 3 | 823 | 0.8144 | 0.9409 | 0.8730 |
| Residual fixed | intestinal_metaplasia | 3 | 166 | 0.9198 | 0.7932 | 0.8514 |
| Residual fixed | lymphoid | 3 | 7 | 0.8047 | 1.0000 | 0.8843 |
| Residual fixed | other | 3 | 156 | 0.2603 | 0.0684 | 0.1074 |
| Uniform weaker | tumor | 3 | 99 | 0.9056 | 0.8956 | 0.9001 |
| Uniform weaker | mucosa_inflammation | 3 | 823 | 0.8092 | 0.9380 | 0.8689 |
| Uniform weaker | intestinal_metaplasia | 3 | 166 | 0.9050 | 0.7771 | 0.8360 |
| Uniform weaker | lymphoid | 3 | 7 | 0.8426 | 1.0000 | 0.9139 |
| Uniform weaker | other | 3 | 156 | 0.1936 | 0.0598 | 0.0912 |
| Global shuffled | tumor | 3 | 99 | 0.9641 | 0.8283 | 0.8886 |
| Global shuffled | mucosa_inflammation | 3 | 823 | 0.8037 | 0.9526 | 0.8714 |
| Global shuffled | intestinal_metaplasia | 3 | 166 | 0.9172 | 0.7651 | 0.8317 |
| Global shuffled | lymphoid | 3 | 7 | 0.9583 | 1.0000 | 0.9778 |
| Global shuffled | other | 3 | 156 | 0.2498 | 0.0705 | 0.1053 |
| Adaptive | tumor | 3 | 99 | 0.9234 | 0.9057 | 0.9130 |
| Adaptive | mucosa_inflammation | 3 | 823 | 0.8091 | 0.9303 | 0.8655 |
| Adaptive | intestinal_metaplasia | 3 | 166 | 0.8719 | 0.8052 | 0.8370 |
| Adaptive | lymphoid | 3 | 7 | 0.7583 | 1.0000 | 0.8601 |
| Adaptive | other | 3 | 156 | 0.2033 | 0.0577 | 0.0899 |
| Within-class shuffled | tumor | 3 | 99 | 0.9239 | 0.9259 | 0.9243 |
| Within-class shuffled | mucosa_inflammation | 3 | 823 | 0.8159 | 0.9316 | 0.8699 |
| Within-class shuffled | intestinal_metaplasia | 3 | 166 | 0.9146 | 0.8112 | 0.8596 |
| Within-class shuffled | lymphoid | 3 | 7 | 0.6399 | 1.0000 | 0.7794 |
| Within-class shuffled | other | 3 | 156 | 0.1904 | 0.0662 | 0.0981 |

Frozen reserved-test results (all three seeds; no tuning):

| Arm | Class | Support | Mean precision | Mean recall | Mean F1 |
|---|---|---:|---:|---:|---:|
| Standard | tumor | 316 | 0.8650 | 0.6698 | 0.7520 |
| Standard | mucosa_inflammation | 786 | 0.7356 | 0.8545 | 0.7903 |
| Standard | intestinal_metaplasia | 203 | 0.5180 | 0.7685 | 0.6173 |
| Standard | lymphoid | 9 | 0.7535 | 1.0000 | 0.8584 |
| Standard | other | 299 | 0.4472 | 0.2085 | 0.2816 |
| Residual fixed | tumor | 316 | 0.8586 | 0.6002 | 0.7061 |
| Residual fixed | mucosa_inflammation | 786 | 0.7267 | 0.8897 | 0.7999 |
| Residual fixed | intestinal_metaplasia | 203 | 0.5550 | 0.7767 | 0.6473 |
| Residual fixed | lymphoid | 9 | 0.8227 | 1.0000 | 0.9015 |
| Residual fixed | other | 299 | 0.4867 | 0.2129 | 0.2932 |
| Uniform weaker | tumor | 316 | 0.8381 | 0.5918 | 0.6928 |
| Uniform weaker | mucosa_inflammation | 786 | 0.7019 | 0.8906 | 0.7849 |
| Uniform weaker | intestinal_metaplasia | 203 | 0.5706 | 0.7323 | 0.6392 |
| Uniform weaker | lymphoid | 9 | 0.7643 | 1.0000 | 0.8624 |
| Uniform weaker | other | 299 | 0.4802 | 0.1862 | 0.2682 |
| Global shuffled | tumor | 316 | 0.8969 | 0.5411 | 0.6662 |
| Global shuffled | mucosa_inflammation | 786 | 0.6789 | 0.8995 | 0.7702 |
| Global shuffled | intestinal_metaplasia | 203 | 0.5821 | 0.6782 | 0.6151 |
| Global shuffled | lymphoid | 9 | 0.7727 | 1.0000 | 0.8714 |
| Global shuffled | other | 299 | 0.4461 | 0.1839 | 0.2519 |
| Adaptive | tumor | 316 | 0.8622 | 0.6224 | 0.7189 |
| Adaptive | mucosa_inflammation | 786 | 0.7439 | 0.8715 | 0.8024 |
| Adaptive | intestinal_metaplasia | 203 | 0.5243 | 0.8325 | 0.6428 |
| Adaptive | lymphoid | 9 | 0.7500 | 1.0000 | 0.8571 |
| Adaptive | other | 299 | 0.4686 | 0.1996 | 0.2799 |
| Within-class shuffled | tumor | 316 | 0.8819 | 0.6414 | 0.7419 |
| Within-class shuffled | mucosa_inflammation | 786 | 0.7299 | 0.8936 | 0.8034 |
| Within-class shuffled | intestinal_metaplasia | 203 | 0.5726 | 0.8046 | 0.6688 |
| Within-class shuffled | lymphoid | 9 | 0.7762 | 1.0000 | 0.8727 |
| Within-class shuffled | other | 299 | 0.4994 | 0.2040 | 0.2876 |
