# Local computer setup and limited classifier pilot — 13 November 2026

This is an exploratory pipeline/falsification pilot. It establishes neither a reliable classification improvement nor research novelty. No full training campaign was launched.

## Environment

Fresh environment: `.venv-research` (Python 3.12.14), created from this PC's base interpreter. Neither existing virtual environment was reused.
GPU: NVIDIA GeForce RTX 3050 Laptop GPU; 4096 MiB VRAM; capability [8, 6]. PyTorch 2.7.0+cu128, bundled CUDA runtime 12.8. CUDA forward/backward and mixed-precision training succeeded.
Host: AMD Ryzen 5 5600H, 6 cores / 12 threads, approximately 15.3 GiB usable RAM. NVIDIA driver 621.01. No standalone CUDA toolkit is needed for these wheel-based experiments.
The installed PyTorch/torchvision pair follows the project pins and the [official previous-version CUDA 12.8 instructions](https://pytorch.org/get-started/previous-versions/). The direct official torch wheel SHA256 was verified. Exact installed dependencies are in `environment-lock.txt`.

## Data audit and scope

Verified 31,096 PNG paths. Source split counts: {'train': 24876, 'val': 3110, 'test': 3110}. No path rebasing was necessary; an isolated local manifest was still saved.
Found 932 exact-file duplicate groups (2,118 images), including 334 cross-split groups. 165 groups span training and test. There were no cross-class exact-duplicate conflicts.
All exact-file duplicate identities were quarantined from pilot training/validation. Original files and split assignments were retained. Selected pilot training/validation images were also checked for identical decoded RGB pixels; none remained. Near-duplicates, slide clustering and alternate-encoding duplicates against test images remain unverified.
UNVERIFIED. Class_number filenames provide no verifiable patient/slide mapping. Clinical CSV has patient identifiers but no image linkage.
The clinical patient table does not establish image linkage. These image splits cannot support patient-independent or clinical claims. A verified group mapping and a separately audited final evaluation design are needed before a confirmatory campaign.
Test files were read as bytes solely to audit duplication and preservation. No test pixels were decoded, no test transform or model inference ran, and no test data fitted parameters. The source test split was preserved, but its existing duplication means it is not yet a clean confirmatory holdout.

## Fixed protocol

One seed (42), 512 training images (64/class), 128 validation images (16/class), 3 epochs per arm, batch size 16.
Existing CNN encoder + two-layer bidirectional LSTM + additive attention; no ResNet50. Every arm loaded the same saved initialization and used the same batch order, geometric random draws and fixed AdamW budget. Training RNGs for stain and geometry were separated. CUDNN deterministic mode was enabled; warn-only deterministic algorithms do not guarantee bitwise repeatability across hardware.
Training-only stain distribution, controller percentiles, uniform mean strength, shuffled-strength pool and channel statistics were refitted on the selected training subset. The shuffled arm uses one fixed training-strength permutation (seed 43); it is an exploratory random-assignment control, not a learned deployment policy.
Validation uses raw resized RGB with shared training-fitted channel standardization, without stain normalization or stochastic augmentation. Validation data informed earlier preprocessing exploration; this is not an independent confirmatory sample.
Standard augmentation uses the original ColorJitter plus shared geometry. Other arms replace ColorJitter with stain augmentation. Original SAN means the upstream-compatible SVD/spherical/scalar-intensity branch, not reproduction of every published SAN experiment. Original SAN uses its upstream-compatible reconstruction/quantization; residual arms share the controller reconstruction and rounding, so controller-versus-residual-fixed isolates control strength. Original-versus-residual differences also include those small reconstruction/quantization details.
White, black, constant-gray and one-tissue-pixel fallbacks passed in all stain arms. A failed-extraction image also passed CUDA forward/backward. Existing upstream numerical checks passed. Every arm saved, reloaded and evaluated its best checkpoint; saved and reloaded validation probabilities matched within tolerance.

## Timing measured before the pilot

| Arm | Smoke epoch, including validation (s) | Pilot mean epoch (s) | Peak allocated VRAM (MiB) |
|---|---:|---:|---:|
| standard | 5.28 | 4.84 | 396 |
| original_san | 9.43 | 10.24 | 396 |
| residual_fixed | 11.24 | 11.70 | 396 |
| uniform_weaker | 11.29 | 11.70 | 396 |
| shuffled | 11.32 | 12.17 | 396 |
| adaptive | 11.37 | 11.69 | 396 |

The pre-pilot estimate was 3.0 minutes of epoch computation for all six arms, or 4.5 minutes allowing 50% overhead. Setup, checkpoint reloads, plotting and first CUDA startup add time.
Actual measured pilot epoch computation, including validation: 3.12 minutes.
A hypothetical full-data, 40-epoch, six-arm, three-seed campaign extrapolated to approximately 97.1 GPU hours. This is a rough scaling estimate, not an approved plan. Thermal throttling, larger validation sets, I/O and changing hardware load can alter it.

## Validation results after reloading the best checkpoint

| Arm | Best epoch | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| standard | 2 | 0.3359 | 0.2438 |
| original_san | 2 | 0.3281 | 0.2177 |
| residual_fixed | 3 | 0.3047 | 0.2323 |
| uniform_weaker | 3 | 0.3359 | 0.2802 |
| shuffled | 3 | 0.3594 | 0.2768 |
| adaptive | 3 | 0.3516 | 0.2651 |

### Per-class validation recall

| Arm | ADI | DEB | LYM | MUC | MUS | NOR | STR | TUM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | 1.000 | 0.000 | 0.688 | 0.188 | 0.625 | 0.000 | 0.188 | 0.000 |
| original_san | 1.000 | 0.000 | 0.812 | 0.062 | 0.625 | 0.062 | 0.062 | 0.000 |
| residual_fixed | 0.875 | 0.062 | 0.875 | 0.125 | 0.375 | 0.000 | 0.125 | 0.000 |
| uniform_weaker | 0.938 | 0.000 | 0.250 | 0.250 | 0.500 | 0.125 | 0.000 | 0.625 |
| shuffled | 1.000 | 0.188 | 0.812 | 0.250 | 0.500 | 0.000 | 0.125 | 0.000 |
| adaptive | 0.938 | 0.062 | 0.938 | 0.250 | 0.438 | 0.000 | 0.188 | 0.000 |

Adaptive minus uniform-weaker macro-F1: -0.0151; adaptive minus standard: +0.0213. These are descriptive results from a tiny single-seed budget, with best-epoch selection on the same validation subset. They establish neither statistical superiority nor useful converged performance.
The adaptive checkpoint has zero TUM and NOR recall on this 16-image-per-class validation subset; uniformly weaker augmentation has TUM recall 0.625. This is a class-specific concern in a very short, undertrained run, not evidence of a stable treatment effect. These checkpoints are experimental and are not suitable replacements for the dashboard model.
Do not expand automatically on the basis of these results. Classification benefit beyond weaker/random controls, generalization and novelty remain unconfirmed. Resolve image-to-patient/slide mapping and holdout leakage before claims or confirmatory training.

## Preservation and saved artifacts

SHA256 rechecks: 69 protected original artifacts and 31,096 dataset files; passed=True, changed=[].
The dashboard, database, active-model pointer, original checkpoints, training source, reports and existing logs were not modified. No pilot checkpoint was activated.
All six first pilot epochs exactly reproduced the corresponding smoke training loss/accuracy and validation loss/accuracy/macro-F1 on this machine. loader_checks.json records random-state-independent validation and rejection of test rows before file access. No classifier process remained running after completion.
Each arm contains best.pt, history.json, metrics.json, augmentation_events.csv and validation_predictions.csv (all eight class probabilities). Metrics include accuracy, macro-F1, per-class recall, confusion matrix, 10-bin ECE and multiclass Brier score. Learning curves are in pilot/learning_curves.png. Configurations, initialization hash, frozen selections, fit parameters, fallback checks and smoke timing are retained.

## Commands

From the project root in PowerShell:

```powershell
$python = '.\.venv-research\Scripts\python.exe'
& $python -m pip check
& $python stain_research/check.py --project .
& $python stain_research/verify_preservation.py --run stain_research/runs/classifier_local_20261113
```

The preparation, smoke and pilot commands are recorded in LOCAL_CLASSIFIER_README.md. Completed run directories refuse overwrite. The original training entry point can activate dashboard models and evaluate test images, so use the isolated runner for this research workflow.
