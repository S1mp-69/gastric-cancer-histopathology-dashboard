"""Summarize completed limited-pilot artifacts without making accuracy claims."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    read = lambda name: json.loads((run / name).read_text())
    audit, config = read('audit.json'), read('pilot/config.json')
    timing, results = read('smoke/timing_estimate.json'), read('pilot/summary.json')
    integrity = read('preservation_after.json')
    groups = read('duplicate_groups.json')
    baseline = next(r for r in results if r['arm'] == 'standard')
    adaptive = next(r for r in results if r['arm'] == 'adaptive')
    uniform = next(r for r in results if r['arm'] == 'uniform_weaker')
    lines = [
        '# Local computer setup and limited classifier pilot — 13 November 2026', '',
        'This is an exploratory pipeline/falsification pilot. It establishes neither a reliable classification improvement nor research novelty. No full training campaign was launched.', '',
        '## Environment', '',
        f"Fresh environment: `.venv-research` (Python {config['python'].split()[0]}), created from this PC's base interpreter. Neither existing virtual environment was reused.",
        f"GPU: {config['gpu']}; {config['gpu_total_mib']:.0f} MiB VRAM; capability {config['gpu_capability']}. PyTorch {config['torch']}, bundled CUDA runtime {config['cuda_runtime']}. CUDA forward/backward and mixed-precision training succeeded.",
        'Host: AMD Ryzen 5 5600H, 6 cores / 12 threads, approximately 15.3 GiB usable RAM. NVIDIA driver 621.01. No standalone CUDA toolkit is needed for these wheel-based experiments.',
        'The installed PyTorch/torchvision pair follows the project pins and the [official previous-version CUDA 12.8 instructions](https://pytorch.org/get-started/previous-versions/). The direct official torch wheel SHA256 was verified. Exact installed dependencies are in `environment-lock.txt`.', '',
        '## Data audit and scope', '',
        f"Verified {audit['images']:,} PNG paths. Source split counts: {audit['splits']}. No path rebasing was necessary; an isolated local manifest was still saved.",
        f"Found {audit['duplicate_groups']} exact-file duplicate groups ({sum(map(len, groups)):,} images), including {audit['cross_split_duplicate_groups']} cross-split groups. {sum({'train','test'}.issubset({r['split'] for r in g}) for g in groups)} groups span training and test. There were no cross-class exact-duplicate conflicts.",
        'All exact-file duplicate identities were quarantined from pilot training/validation. Original files and split assignments were retained. Selected pilot training/validation images were also checked for identical decoded RGB pixels; none remained. Near-duplicates, slide clustering and alternate-encoding duplicates against test images remain unverified.',
        audit['patient_slide_status'],
        'The clinical patient table does not establish image linkage. These image splits cannot support patient-independent or clinical claims. A verified group mapping and a separately audited final evaluation design are needed before a confirmatory campaign.',
        'Test files were read as bytes solely to audit duplication and preservation. No test pixels were decoded, no test transform or model inference ran, and no test data fitted parameters. The source test split was preserved, but its existing duplication means it is not yet a clean confirmatory holdout.', '',
        '## Fixed protocol', '',
        f"One seed ({config['seed']}), {config['train_images']} training images (64/class), {config['validation_images']} validation images (16/class), {config['epochs']} epochs per arm, batch size {config['batch_size']}.",
        'Existing CNN encoder + two-layer bidirectional LSTM + additive attention; no ResNet50. Every arm loaded the same saved initialization and used the same batch order, geometric random draws and fixed AdamW budget. Training RNGs for stain and geometry were separated. CUDNN deterministic mode was enabled; warn-only deterministic algorithms do not guarantee bitwise repeatability across hardware.',
        'Training-only stain distribution, controller percentiles, uniform mean strength, shuffled-strength pool and channel statistics were refitted on the selected training subset. The shuffled arm uses one fixed training-strength permutation (seed 43); it is an exploratory random-assignment control, not a learned deployment policy.',
        'Validation uses raw resized RGB with shared training-fitted channel standardization, without stain normalization or stochastic augmentation. Validation data informed earlier preprocessing exploration; this is not an independent confirmatory sample.',
        'Standard augmentation uses the original ColorJitter plus shared geometry. Other arms replace ColorJitter with stain augmentation. Original SAN means the upstream-compatible SVD/spherical/scalar-intensity branch, not reproduction of every published SAN experiment. Original SAN uses its upstream-compatible reconstruction/quantization; residual arms share the controller reconstruction and rounding, so controller-versus-residual-fixed isolates control strength. Original-versus-residual differences also include those small reconstruction/quantization details.',
        'White, black, constant-gray and one-tissue-pixel fallbacks passed in all stain arms. A failed-extraction image also passed CUDA forward/backward. Existing upstream numerical checks passed. Every arm saved, reloaded and evaluated its best checkpoint; saved and reloaded validation probabilities matched within tolerance.', '',
        '## Timing measured before the pilot', '',
        '| Arm | Smoke epoch, including validation (s) | Pilot mean epoch (s) | Peak allocated VRAM (MiB) |',
        '|---|---:|---:|---:|',
    ]
    for r in results:
        lines.append(f"| {r['arm']} | {timing['measured_complete_epoch_seconds_by_arm'][r['arm']]:.2f} | {r['epoch_seconds_mean']:.2f} | {r['cuda_peak_allocated_mib']:.0f} |")
    lines += ['', f"The pre-pilot estimate was {timing['limited_3_epochs_all_6_arms_minutes']:.1f} minutes of epoch computation for all six arms, or {timing['limited_3_epochs_with_50pct_overhead_minutes']:.1f} minutes allowing 50% overhead. Setup, checkpoint reloads, plotting and first CUDA startup add time.",
              f"Actual measured pilot epoch computation, including validation: {sum(r['epoch_seconds_mean'] * config['epochs'] for r in results) / 60:.2f} minutes.",
              f"A hypothetical full-data, 40-epoch, six-arm, three-seed campaign extrapolated to approximately {timing['hypothetical_full_data_40_epochs_6_arms_3_seeds_hours']:.1f} GPU hours. This is a rough scaling estimate, not an approved plan. Thermal throttling, larger validation sets, I/O and changing hardware load can alter it.", '',
              '## Validation results after reloading the best checkpoint', '',
              '| Arm | Best epoch | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
    for r in results:
        m = r['validation']
        lines.append(f"| {r['arm']} | {r['best_epoch']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
    lines += ['', '### Per-class validation recall', '',
              '| Arm | ADI | DEB | LYM | MUC | MUS | NOR | STR | TUM |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        recalls = [r['validation']['per_class'][c]['recall'] for c in ['ADI','DEB','LYM','MUC','MUS','NOR','STR','TUM']]
        lines.append('| ' + r['arm'] + ' | ' + ' | '.join(f'{x:.3f}' for x in recalls) + ' |')
    lines += ['', f"Adaptive minus uniform-weaker macro-F1: {adaptive['validation']['macro_f1']-uniform['validation']['macro_f1']:+.4f}; adaptive minus standard: {adaptive['validation']['macro_f1']-baseline['validation']['macro_f1']:+.4f}. These are descriptive results from a tiny single-seed budget, with best-epoch selection on the same validation subset. They establish neither statistical superiority nor useful converged performance.",
              'The adaptive checkpoint has zero TUM and NOR recall on this 16-image-per-class validation subset; uniformly weaker augmentation has TUM recall 0.625. This is a class-specific concern in a very short, undertrained run, not evidence of a stable treatment effect. These checkpoints are experimental and are not suitable replacements for the dashboard model.',
              'Do not expand automatically on the basis of these results. Classification benefit beyond weaker/random controls, generalization and novelty remain unconfirmed. Resolve image-to-patient/slide mapping and holdout leakage before claims or confirmatory training.', '',
              '## Preservation and saved artifacts', '',
              f"SHA256 rechecks: {integrity['original_artifacts_checked']} protected original artifacts and {integrity['dataset_files_checked']:,} dataset files; passed={integrity['passed']}, changed={integrity['changed']}.",
              'The dashboard, database, active-model pointer, original checkpoints, training source, reports and existing logs were not modified. No pilot checkpoint was activated.',
              'All six first pilot epochs exactly reproduced the corresponding smoke training loss/accuracy and validation loss/accuracy/macro-F1 on this machine. loader_checks.json records random-state-independent validation and rejection of test rows before file access. No classifier process remained running after completion.',
              'Each arm contains best.pt, history.json, metrics.json, augmentation_events.csv and validation_predictions.csv (all eight class probabilities). Metrics include accuracy, macro-F1, per-class recall, confusion matrix, 10-bin ECE and multiclass Brier score. Learning curves are in pilot/learning_curves.png. Configurations, initialization hash, frozen selections, fit parameters, fallback checks and smoke timing are retained.', '',
              '## Commands', '', 'From the project root in PowerShell:', '', '```powershell',
              "$python = '.\\.venv-research\\Scripts\\python.exe'",
              '& $python -m pip check',
              '& $python stain_research/check.py --project .',
              '& $python stain_research/verify_preservation.py --run stain_research/runs/classifier_local_20261113', '```', '',
              'The preparation, smoke and pilot commands are recorded in LOCAL_CLASSIFIER_README.md. Completed run directories refuse overwrite. The original training entry point can activate dashboard models and evaluate test images, so use the isolated runner for this research workflow.']
    (run / 'LOCAL_RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(run / 'LOCAL_RESULTS.md')


if __name__ == '__main__':
    main()
