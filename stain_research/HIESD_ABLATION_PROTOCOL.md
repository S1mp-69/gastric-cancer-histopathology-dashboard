# HiESD matched ResNet-18 augmentation protocol

This is a post-hoc exploratory extension of the completed v3 reference baseline.
The user requested matched augmentation comparisons and three seeds where feasible,
and subsequently asked to skip routine GPU checks. No GPU benchmark or smoke test
is performed by this queue. Existing verified CUDA/AMP settings are reused.

Freeze the 3794 QC-screened patches (2543 train, 1251 validation), original patient
assignments, ImageNet normalization, ResNet-18 architecture, common starting weights
within each seed, batch16, CPU threads2/workers0, unweighted label-smoothed CE,
AdamW lr1e-4/wd1e-4, clip5, plateau scheduler rule, max40 epochs, and early stopping
patience7/min-delta.0001. Validation remains deterministic raw RGB and is used for
model selection only. Reserved-test data and masks are excluded. No dashboard
activation or modification of previous checkpoints occurs.

Five arms: standard, residual-preserving fixed alpha1, uniformly weaker at mean
training alpha, global shuffled assignment of the adaptive training alpha values,
and adaptive source-diagnostic assignment. All stain arms ADD their transform
before the SAME standard geometric/RGB pipeline, rather than changing two policies
at once. Stain targets and intensity draws share a separate image/epoch/seed RNG
across stain arms; geometry/RGB RNG matches the reference baseline. Labels are
not used to permute strengths. Invalid decompositions retain unchanged-image
fallbacks; all 3794 current images decomposed successfully.

Deterministic float64 basis/concentration/diagnostic arrays use an existing
versioned cache implementation. The key includes content hashes, manifest,
controller, transform/cache source, NumPy and dtypes. Fit stain distribution and
10th/90th diagnostic bounds using TRAIN only; validation excluded from fitting.
Random targets, intensity and geometry remain uncached. Fitted cache normalization
is not used: retain fixed ImageNet normalization in every classifier arm.

Cache checks passed: forty cached/uncached transform comparisons were exact.
CPU-only preflight verified sixty standard augmented training tensors against the
original dataset at multiple epochs, every one of the1251 validation tensors,
finite outputs from all arms, and exact strength multisets for120 seed/epoch
combinations. Failed setup checks remain in FAILURE_history records; no training
had started when a NumPy integer index in the preflight was corrected.

Seed42 standard reuses the completed v3 run with its actual checkpoint/history;
the remaining four arms use byte-identical original initial.pt. Seeds43 and44
have one fresh pretrained-head initialization each, saved on CPU and copied
identically to all five arms. No fine-tuned seed42 checkpoint transfer.

Finite queue: four missing seed42 arms first, then complete five-arm seed43/44
blocks if the six-hour start budget can accommodate the predicted whole block.
The estimate uses observed seed42 epoch times, not outcome rankings. Each started
run completes its bounded max40/early-stopping schedule. No policy/weight search
or result-dependent hyperparameter changes. Budget deferrals are reported, not
silently described as three completed replicates. Up to fourteen new runs plus
the reused seed42 baseline. Resume retains full epoch states and never discards
completed runs; at most one epoch's training is lost on interruption.

Existing baseline timing is stored as an inherited reference (roughly fifteen
minutes per forty epochs). Stain augmentation may increase CPU time. Actual epoch
throughput, fallback-independent AMP backoffs, training duration and validation
metrics are saved in each history.json. No repeated GPU probes or benchmarks.

The local named-mutex supervisor prevents duplicate queue instances and keeps the
machine awake without adjusting clock rates or thermal protections. It invokes
independent Python processes, preserving logs/checkpoints if Codex disconnects.
The progress window 'HiESD augmentation comparison' reads local JSON only and
makes no ChatGPT calls. Closing it does not stop the queue.

Outputs: stain_research/runs/hiesd_resnet_ablation_v1 contains queue state,
configuration/source hashes, cache provenance, initial weights, per-run configs,
latest/best full checkpoints, predictions and logs. Portable summaries/calibration/
per-class metrics and plots go to stain_research/results/hiesd_resnet_ablation_v1.
Original baseline artifact hashes are checked after every completed run.

Primary descriptive comparison: adaptive-minus-shuffled macro-F1 paired by seed.
Also report accuracy, per-class precision/recall/F1, 15-bin ECE, multiclass Brier,
actual training time, stopping epochs and full learning curves. Seed variation
is not independent patient uncertainty. Strength distribution matching does not
match realized RGB/OD change. This selected cohort has few patients and just
seven validation lymphoid patches. Do not claim clinical benefit, confirmed
novelty, statistical superiority, or SOTA from execution or an arm ranking.
