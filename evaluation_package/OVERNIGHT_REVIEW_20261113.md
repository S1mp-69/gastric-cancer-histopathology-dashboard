# Overnight review — 13 November 2026, 10:12 IST

Training is still active. At review, five of twelve planned runs were complete;
residual-fixed seed 43 was training at epoch 35. Do not launch a duplicate job.
The remaining queue estimate was about nine hours to the maximum epoch budgets;
early stopping can reduce this. The 24-hour campaign budget still applies.

## First complete matched comparison (seed 42)

All four runs used 23,761 training images, 3,059 validation images, identical
initial-weight hashes, shared geometry/order and a common 40-epoch ceiling with
validation macro-F1 early stopping. Values below come from saved metrics.json
after reloading the selected best checkpoint, not the last training epoch.

| Augmentation | Completed epochs | Validation accuracy | Macro-F1 | TUM recall | NOR recall |
|---|---:|---:|---:|---:|---:|
| Standard | 40 | 68.65% | 0.6867 | 79.27% | 46.52% |
| Residual fixed | 34 (early stop) | 66.72% | 0.6661 | 74.35% | 45.99% |
| Uniform weaker | 40 | 67.80% | 0.6760 | 75.13% | 46.52% |
| Adaptive | 40 | 69.01% | 0.6888 | 74.87% | 47.86% |

Adaptive exceeded standard by 0.36 percentage points of accuracy (11 images)
and 0.0021 macro-F1. This small lead is one-seed exploratory evidence, not a
replicated improvement. Adaptive TUM recall was 4.40 percentage points lower
and DEB recall 14.18 points lower; ADI, MUS and STR recall improved. NOR remains
poorly recognized in both arms. Do not describe the controller as uniformly better.

Standard seed 43 also completed (36 epochs, early stop): accuracy 68.03%,
macro-F1 0.6800. Its corresponding adaptive run has not completed. Do not pool
unmatched seed counts to claim a treatment effect. Wait for paired seeds before
reporting mean differences and variability. Reaching the epoch ceiling does not
prove convergence.

## Evidence limits

The provisional split controls exact file duplicates. Patient/slide independence
and complete near-duplicate freedom remain unverified. No final test inference
was performed. Best-checkpoint reload checks passed for all five completed runs.
Training remained on CUDA; the status sample recorded 90% GPU utilization and
about 2.14 GiB used VRAM. CPU loading/augmentation is expected.

The historical three-epoch pilot remains separate: adaptive macro-F1 0.2651
did not beat the uniformly weaker control 0.2802. Numerical preprocessing
stability, classifier validation and historical notebook metrics are distinct.
Research novelty and clinical benefit remain unconfirmed.

## Reliability and deliverables

Several transient Windows progress-file rename locks caused worker restarts.
The supervisor resumed these runs from saved state and all five completed runs
passed checkpoint reload verification. The initial baseline also required an AMP
overflow-handler repair. These interruptions and repairs belong in the experiment
record; they must not be hidden. At review, a progress-file retry was added for
future workers without interrupting the current run or changing model/augmentation
math. The currently active worker retains its already loaded code until it exits.

The requested morning snapshot was saved at 08:59 IST:
MORNING_STATUS_20261113_0900.md. Training continued afterward.
The package last refreshed at 10:06:49 IST: 15-slide editable PowerPoint and PDF,
10-page editable report and PDF, IEEE source ZIP and locally compiled paper PDF,
solo script/viva and demo checklist. Read evaluation_snapshot.json for the exact
snapshot behind the slide/report tables. This review is newer than that snapshot.

The report format is provisional: the prescribed teacher template is unavailable.
Only the supplied short author name Amlan is verified. Overleaf requires sign-in;
local LaTeX compilation passed, but private project import and its real link are
not complete. GitHub publication of this overnight package remains pending;
do not describe it as pushed. Local checkpoints and original application artifacts
remain separate. The last preservation audit checked 69 protected artifacts and
31,096 dataset files with no changes; it was performed before the overnight runs
finished. The demo rehearsal used an isolated database and prototype-baseline
predictions, not a research checkpoint.

## Suggested evaluation statement

“I implemented a decomposition-aware augmentation controller and an isolated
matched classifier study. The first full-budget seed gives a small adaptive
macro-F1 lead over standard augmentation, with worse tumor recall. Replication
is running, so I do not yet claim a reliable accuracy improvement, patient-level
generalization, novelty or clinical benefit.”

Progress evaluation: today, 13 November, 11:00 AM, B-305 (Asia/Calcutta).
Final evaluation: 19 November, 11:00 AM; final venue unverified.
