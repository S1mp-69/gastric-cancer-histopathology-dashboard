# Solo presentation script

Opening: I tested whether numerical stain-decomposition diagnostics help decide augmentation strength for gastric tissue classification. My central result is a controlled negative finding; I am not claiming a new clinically useful classifier.

Data: The first dataset has eight tissue classes. Exact duplicates crossed its historical splits, so I created a provisional exact-file-controlled manifest. Without image-to-patient IDs I cannot claim patient independence. The second dataset, HiESD, supplies patient IDs; I froze patient partitions before acquisition and screened a selected annotated subset using released quality masks.

Method: A stain model separates basis, concentration and reconstruction residual. The controller weakens perturbations when numerical diagnostics look worse. I compare standard augmentation, fixed full strength, one uniform mean strength, a global shuffle, adaptive source assignment and a within-class shuffle. The last control preserves each tissue class strength bank but removes which image gets which strength. It is an evaluation control, not a deployment feature.

Fairness: The same seed shares initial weights, split, geometry, batch order, optimizer and stopping rules. Runs have a 40-epoch ceiling, seven-epoch early stopping and full resumable checkpoints. The original HMU architecture is CNN/BiLSTM/attention; HiESD uses pretrained ResNet18, so I do not attribute cross-study score differences to one change.

Results: HMU adaptive mean validation macro-F1 is 0.6836 against shuffled 0.6890. HiESD adaptive is 0.7131 against shuffled 0.7350 and standard 0.7270. The controller does not win on average. Lower numerical noise amplification is not proof of better classification. I retain these negative results.

Secondary analysis: Rankings change under hypothetical class and patient weighting. This is a known property of F1, independently checked on saved predictions. It is not a measured effect of changing QC and not an assumed clinical prevalence. Seven validation lymphoid patches make that class particularly fragile.

Final evaluation: Read the frozen-test status on the final-results slide. Explain the prespecified adaptive-minus-within-class contrast, all-arm reporting, small patient-cluster uncertainty and prohibition of tuning after test access. Do not quote pending values as completed results.

Demonstration: The application accepts a patch, displays eight probabilities and stores a prediction row in isolated SQLite. It uses the original configured predictor, not the research ResNet18. Prototype fallback is explicitly identified and training-only reference images are used by the demo wrapper. The output is not a patient diagnosis.

Closing: My contribution is a reproducible empirical study with a stronger assignment control and transparent limitations. Algorithmic novelty, clinical value and publication acceptance remain unconfirmed.

Current snapshot: The final-test protocol was frozen before reserved-pixel acquisition. It evaluates all six arms and seeds 42, 43 and 44 using validation-selected best checkpoints, with adaptive-minus-within-class-shuffled patch macro-F1 as the primary contrast. The seven reserved patients cover 13 source slides (about 7.53 GiB). Model hashes are frozen before acquisition. No tuning follows test access. Current stage: waiting for within-class controls. No final-test scores are available in this snapshot.
