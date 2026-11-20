# HiESD matched ResNet-18 augmentation comparison

Exploratory selected-cohort experiment. Common split, initialization within each seed, standard RGB/geometric augmentation, normalization and training budgets. Stain arms add residual-preserving augmentation before the shared standard pipeline. Reserved test data excluded.

| Seed | Arm | Accuracy | Macro-F1 | Tumor recall | Other recall | ECE | Epochs |
|---:|---|---:|---:|---:|---:|---:|---:|
|42|standard|0.8066|0.7324|0.7778|0.0897|0.0703|22|
|42|residual_fixed|0.8329|0.7222|0.9495|0.0705|0.0926|32|
|42|uniform_weaker|0.8153|0.7334|0.9293|0.0449|0.0851|25|
|42|shuffled|0.8050|0.7390|0.8384|0.0833|0.0912|22|
|42|adaptive|0.8090|0.7117|0.8889|0.0513|0.0849|26|
|43|standard|0.7914|0.7321|0.9091|0.1346|0.0801|11|
|43|residual_fixed|0.7826|0.7185|0.9697|0.0385|0.0870|11|
|43|uniform_weaker|0.8050|0.7161|0.8687|0.0513|0.1028|18|
|43|shuffled|0.8034|0.7065|0.7475|0.0192|0.1027|8|
|43|adaptive|0.7858|0.7073|0.9192|0.0577|0.0914|11|
|44|standard|0.8010|0.7166|0.8889|0.0513|0.0827|9|
|44|residual_fixed|0.8225|0.7373|0.8990|0.0962|0.0915|19|
|44|uniform_weaker|0.7922|0.7165|0.8889|0.0833|0.0758|9|
|44|shuffled|0.8161|0.7595|0.8990|0.1090|0.0857|22|
|44|adaptive|0.8153|0.7203|0.9091|0.0641|0.0860|25|

Adaptive-minus-shuffled macro-F1 differences: [{"seed": 42, "adaptive_minus_shuffled_macro_f1": -0.02727781601724455}, {"seed": 43, "adaptive_minus_shuffled_macro_f1": 0.0008025165955484193}, {"seed": 44, "adaptive_minus_shuffled_macro_f1": -0.03912902437471222}].

Seed variation does not provide independent patient samples. Calibration values are descriptive; the validation cohort was used for early stopping. The study follows earlier negative/mixed experiments and is not preregistered confirmation. Exact duplicate control and patient grouping do not establish clinical transportability.

All original baseline artifacts remain hash-verified. Latest/best research checkpoints remain isolated from the dashboard. No claims of novelty, clinical benefit or method superiority are implied by execution or individual rankings.
