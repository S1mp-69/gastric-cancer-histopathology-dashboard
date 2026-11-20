# Continue on the stronger PC

Copy this entire stain_research folder beside app/, training/ and data/ in the
copied gastric detection project. The main application has not been changed.
The research folder contains preprocessing code, upstream attribution, literature
notes, pilot outputs, and implementation checks. No classifier training has run.

## Current question

Does per-image stain-decomposition quality provide a useful signal for controlling
augmentation, beyond uniformly reducing augmentation strength? The controller
uses log condition number and relative reconstruction error, with training-only
percentile bounds. The research mechanism and controls are in CONTROLLER_STUDY.md.

## Files to read first

- CONTROLLER_RESULTS.md: initial pilot and limitations.
- FOLLOWUP_RESULTS.md: stronger controls and the current decision.
- CONTROLLER_STUDY.md: method and prior-art boundaries.
- controller.py: experimental transform, separate from the original dashboard.
- strength_controls.py: second-sample, noise-level and matched-control study.
- THIRD_PARTY_NOTICE.txt: required attribution for adapted extraction logic.

## Paste into Codex on the stronger PC

```text
Set up this copied gastric histopathology project for a small, controlled
classification pilot of decomposition-aware stain augmentation.

Read stain_research/STRONG_PC_HANDOFF.md, FOLLOWUP_RESULTS.md,
CONTROLLER_STUDY.md and CONTROLLER_RESULTS.md, then inspect the original code.

Verify GPU hardware, CUDA-enabled PyTorch, dependencies and dataset paths.
Create this PC's own Python environment; do not reuse a copied .venv.
If old absolute manifest paths fail, rebase them using verified class/filename
matches, preserving split assignments, and save a new manifest separately.

Inspect the follow-up decision before training. If the data do not support the
controller beyond matched weaker/random controls, report that clearly and do
not automatically launch full training. A small falsification experiment may
still be useful, but do not advertise an improved or uniquely novel method.

If a classifier pilot is justified, implement it in an isolated run directory.
Audit duplicate images and patient/slide identities before freezing splits.
Use the existing CNN-BiLSTM-attention architecture initially. No ResNet50.
Compare standard augmentation, original Stain SAN, residual-preserving fixed
augmentation, uniformly weaker augmentation, and the adaptive controller.
The extra residual-preserving fixed control is mandatory to isolate the effect
of retaining the original decomposition residual from the effect of the controller.

Fit stain/controller parameters using training images only. Keep validation
and test transforms deterministic and explicitly document whether inference
uses raw images or a shared normalization policy. Keep seeds, geometry, optimizer,
initialization and training budgets comparable. Inference should not randomly
augment images unless evaluated as a separate test-time strategy.

Run short GPU smoke tests first, including failed-extraction fallback cases.
Verify loading a saved best checkpoint before validation/test evaluation.
Measure actual epoch time and report the estimate before long runs. Preserve
the untouched test split for final confirmation, and preserve all original
checkpoints, app.db, dataset, logs, reports, and dashboard behavior.

Save all configurations, seeds, failures by class, learning curves, per-image
predictions and timing. Report accuracy, macro-F1 and per-class recall. Do not
invent improved accuracy, clinical validity, novelty, or publication acceptance.
```

## Why a full run is not the first step

Numerical stability is a different outcome from classification accuracy. Adaptive
augmentation might remove beneficial variation or have unequal class effects.
Only a fair classifier comparison can establish downstream value. Start with a
measured smoke test and a limited pilot, then expand when the results warrant it.
