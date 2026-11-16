# Shuffled strength classifier follow-up 16 November 2026

This is an exploratory follow-up chosen after seeing the completed twelve-run
comparison. It is not a preregistered independent confirmation or novelty proof.

Question: does assignment of strengths using an image's own diagnostics help,
compared with a global random permutation of those same strengths each epoch?
Primary outcome: paired adaptive-minus-shuffled validation macro-F1 for seeds42,
43,44, using each run's selected best checkpoint. Report all individual
differences, their mean/SD and an explicitly exploratory n=3 t interval. Secondary:
TUM recall, all class recalls, calibration, failure rates, timing and learning curves.
Do not stop runs because a favorable method ranking appears.

The source RGB, stain basis, concentrations, residual, target sampling, intensity,
geometry and batch order remain receiver-image specific and match the original
trainer streams. Only the training diagnostics are permuted across VALID training
images using SeedSequence(seed,epoch,710301). Since the strength is a deterministic
function of these diagnostics, the empirical valid-image strength multiset is
preserved exactly per epoch. Invalid decompositions retain unchanged-image fallback
and strength0. Validation diagnostics/images remain unchanged. Random fixed points
are allowed. Actual pixel-change magnitude is not guaranteed to match because it
also depends on the receiver image; report this limitation.

Use the frozen provisional HMU split and copied original initial weights. Same
architecture, AdamW, LR0.001, weight decay0.0001, label smoothing0.05, scheduler,
FP16 AMP, gradient clip5, batch64 and4 workers. Up to40 epochs, macro-F1 early
stopping patience7, minimum improvement0.0001. Checkpoints every120seconds and
epoch boundaries. No test images or model activation. New output directory:
stain_research/runs/shuffled_20261116. Original runs remain unchanged.

Three runs are queued sequentially with independent local supervision and bounded
retries; no OpenAI calls are made. Prior full-data stain epochs averaged about
119seconds: maximum training estimate about4hours, with startup/recovery overhead
and early stopping affecting duration. Measure the new first epoch before revising
the estimate. A two-epoch SMALL smoke run validates plumbing/resume only, never
convergence. No additional arms or HiESD training are included in this campaign.

Local commands:
`.venv-research/Scripts/python.exe stain_research/shuffled_classifier.py preflight`
then `.venv-research/Scripts/python.exe stain_research/shuffled_supervisor.py`.
Status: runs/shuffled_20261116/FOLLOWUP_STATUS.json.
Final results: results/shuffled_20261116/summary.json and RESULTS.md.
After completion, inspect results before revising the paper; do not invent a win.
