# Isolated local classifier workflow

Read `runs/classifier_local_20261113/LOCAL_RESULTS.md` for the completed setup,
audit, timings and limited-pilot outcome. This pilot does not establish improved
accuracy, patient-independent performance or novelty.

Use `.venv-research`, created on this PC from its base Python 3.12 interpreter.
The pre-existing `.venv` and `.venv.copied-from-old-pc` were not reused or changed.
Original project requirements were installed with CUDA 12.8 torch 2.7.0 and
torchvision 0.22.0. Exact resolved versions are saved under the run directory.

## Completed run commands

From the project root in PowerShell:

```powershell
$python = '.\.venv-research\Scripts\python.exe'
$run = 'stain_research/runs/classifier_local_20261113'
& $python stain_research/prepare_classifier.py --output $run
& $python stain_research/check.py --project .
& $python stain_research/classifier_pilot.py prepare --run $run --train-per-class 64 --val-per-class 16
& $python stain_research/classifier_pilot.py smoke --run $run --batch-size 16
# Read smoke/timing_estimate.json and report the estimate BEFORE the next command.
& $python stain_research/classifier_pilot.py pilot --run $run --batch-size 16 --epochs 3
& $python stain_research/verify_preservation.py --run $run
& $python stain_research/report_classifier.py --run $run
```

Existing outputs refuse overwrite. These commands document the executed workflow;
they are not instructions to rerun or extend the campaign automatically. Choose a
fresh output for a separately justified future experiment. The isolated runner
requires a successful smoke result before pilot mode and permits at most five
epochs, with at most 128 training and 32 validation images per class at preparation.

`prepare_classifier.py` saves source-artifact hashes, verifies every dataset path,
matches class plus filename without changing split assignments, hashes image file
bytes for duplicate detection, and excludes every exact-duplicate identity from
the pilot training/validation pool. Test files are hashed solely for audit and
preservation; their pixels are never decoded in this workflow. Additional selected
training/validation pixel-identity checks occur before preparation is frozen.

The old manifest already resolves locally, so no path substitutions were needed.
The original dataset contains 932 exact-file duplicate groups, including 334
cross-split groups. Patient/slide identities remain unverified: class-number
filenames do not map to the clinical CSV. The original holdout is not ready for
confirmatory evaluation. Near-duplicates and alternate-encoding duplicates against
the test set remain unverified.

The six arms are standard RGB/geometric augmentation, original-style SAN,
residual-preserving fixed SAN, uniform weaker SAN, shuffled training strengths,
and the full adaptive controller. All parameters are fitted on selected training
images only. Validation uses raw resized images and shared train-fitted channel
standardization, without random augmentation or stain normalization. All arms use
the existing CNN-BiLSTM-attention architecture and identical saved initialization,
geometry draws, optimizer, batch order and budgets.

The SAN arm covers the upstream-compatible SVD/spherical/scalar-intensity branch,
not all SAN experiments or NMF. Attribution remains in THIRD_PARTY_NOTICE.txt and
upstream/LICENSE. Original SAN reconstruction uses truncation; residual-preserving
arms share rounded reconstruction. The fixed residual arm isolates retention of
the residual from adaptive strength control; original-versus-residual differences
also include those quantization details.

No code path in the isolated runner activates a dashboard model or evaluates the
test set. Every arm verifies best-checkpoint reload before final validation.
Configurations, RNG seeds, timing, curves, class-specific fallbacks, per-image
predictions and class probabilities are retained under the isolated run directory.
The preservation verifier rechecks original protected files and all dataset bytes.

`download_torch_wheel.py` provides a direct official-host download with SHA256
verification because the index's alternate download host stalled on this PC.
The downloaded wheel is retained under `wheelhouse/` for reproducibility.
