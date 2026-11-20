"""Isolated, bounded CNN-BiLSTM-attention comparison. Never evaluates the test set.

SAN SVD/scalar-intensity port: see THIRD_PARTY_NOTICE.txt and upstream/LICENSE.
Uses existing app architecture/preprocessing through read-only imports. No app,
database, active-model pointer or existing checkpoint is written.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tv
from torchvision.transforms import functional as VF, InterpolationMode
from threadpoolctl import threadpool_limits

from app.ml.constants import TISSUE_CLASSES
from app.ml.model import GastricCancerRNN
from app.ml.preprocessing import image_to_patch_sequence
from controller import apply, diagnostics, fit_controller, strength
from transforms import extract, fit, transform

ARMS = ['standard', 'original_san', 'residual_fixed', 'uniform_weaker', 'shuffled', 'adaptive']
SEED = 42


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_image(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB').resize((224, 224)))


def balanced(frame, split, count):
    return pd.concat([frame[(frame.split == split) & (frame.class_code == code)]
                      .sort_values('image_path').sample(n=count, random_state=SEED)
                      for code in TISSUE_CLASSES], ignore_index=True)


def stain_image(image, arm, parameters, rng, alpha=None):
    if arm == 'standard':
        return image.copy(), {'fallback': False, 'strength': 0.0}
    if arm == 'original_san':
        result, failed = transform(image, parameters['stain'], 'spherical', rng)
        return result, {'fallback': bool(failed), 'strength': 0.0 if failed else 1.0}
    fixed = {'residual_fixed': 1.0, 'uniform_weaker': parameters['uniform_strength'],
             'shuffled': alpha, 'adaptive': None}[arm]
    return apply(image, parameters['stain'], parameters['controller'], rng, fixed)


class Images(Dataset):
    def __init__(self, frame, parameters, arm, training, epoch=0):
        self.frame = frame.reset_index(drop=True)
        if not set(self.frame.split).issubset({'train', 'val'}):
            raise ValueError('Test images are forbidden in the pilot loader')
        self.images = [load_image(p) for p in self.frame.image_path]
        self.parameters, self.arm, self.training, self.epoch = parameters, arm, training, epoch
        self.failures, self.attempts = Counter(), Counter()
        self.strengths = []
        self.jitter = tv.ColorJitter(brightness=.10, contrast=.10, saturation=.05, hue=.02)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = self.images[index]
        if self.training:
            local_seed = SEED * 1000000 + self.epoch * 10000 + index
            image, meta = stain_image(image, self.arm, self.parameters,
                                      np.random.default_rng(local_seed),
                                      self.parameters['shuffled_strengths'][index])
            self.attempts[row.class_code] += 1
            self.failures[row.class_code] += int(meta['fallback'])
            self.strengths.append({'epoch': self.epoch + 1, 'image_path': row.image_path,
                                   'class_code': row.class_code, **meta})
            image = Image.fromarray(image)
            # Separate RNG stream: identical geometry in every arm, independent
            # of how many random draws a stain transform consumes or skips.
            rng = np.random.default_rng(local_seed + 900000000)
            if rng.random() < .5:
                image = VF.hflip(image)
            if rng.random() < .5:
                image = VF.vflip(image)
            image = VF.rotate(image, float(rng.uniform(-180, 180)),
                              interpolation=InterpolationMode.BILINEAR)
            if self.arm == 'standard':
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(local_seed + 800000000)
                    image = self.jitter(image)
        else:
            image = Image.fromarray(image)
        n = self.parameters['normalization']
        inputs = image_to_patch_sequence(image, 56, 16, tuple(n['mean']), tuple(n['std']))
        return inputs, TISSUE_CLASSES.index(row.class_code)


def loader(dataset, batch, shuffle=False, epoch=0):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=True, generator=torch.Generator().manual_seed(SEED + epoch))


def evaluate(model, dataset, batch):
    model.eval()
    probabilities, labels = [], []
    with torch.inference_mode():
        for x, y in loader(dataset, batch):
            logits, _ = model(x.cuda())
            probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().tolist())
            labels.extend(y.tolist())
    probs = np.asarray(probabilities)
    predicted = probs.argmax(axis=1)
    report = classification_report(labels, predicted, labels=list(range(8)),
                                   target_names=TISSUE_CLASSES, output_dict=True, zero_division=0)
    confidence = probs.max(axis=1)
    correct = predicted == np.asarray(labels)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    metrics = {'accuracy': float(accuracy_score(labels, predicted)),
               'macro_f1': float(f1_score(labels, predicted, labels=list(range(8)), average='macro', zero_division=0)),
               'loss': float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12)).mean()),
               'per_class': {c: report[c] for c in TISSUE_CLASSES},
               'confusion_matrix': confusion_matrix(labels, predicted, labels=list(range(8))).tolist(),
               'ece_10_bins': float(ece),
               'multiclass_brier': float(np.square(probs - np.eye(8)[labels]).sum(axis=1).mean())}
    predictions = dataset.frame[['image_path', 'class_code']].copy()
    predictions['predicted_class'] = [TISSUE_CLASSES[i] for i in predicted]
    for i, code in enumerate(TISSUE_CLASSES):
        predictions[f'p_{code}'] = probs[:, i]
    return metrics, predictions


def prepare(run, train_count, val_count):
    out = run / 'prepared'
    out.mkdir(exist_ok=False)
    frame = pd.read_csv(run / 'eligible_train_val.csv')
    train = balanced(frame, 'train', train_count)
    val = balanced(frame, 'val', val_count)
    # Pixel-identical duplicates with different PNG encodings in the selected
    # train/val subset must also fail before any pilot split is frozen.
    pixel_groups = {}
    for row in pd.concat([train, val]).itertuples():
        with Image.open(row.image_path) as image:
            rgb = image.convert('RGB')
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if pixel_hash in pixel_groups:
            raise ValueError(f'Pixel-identical selected images: {row.image_path}, {pixel_groups[pixel_hash]}')
        pixel_groups[pixel_hash] = row.image_path
    train.to_csv(out / 'train.csv', index=False)
    val.to_csv(out / 'val.csv', index=False)
    images = [load_image(p) for p in train.image_path]
    controller = fit_controller(images)
    bases, alphas, failures = [], [], Counter()
    for image, row in zip(images, train.itertuples()):
        try:
            values, basis, _, _ = diagnostics(image)
            bases.append(basis)
            alphas.append(strength(values, controller))
        except ValueError:
            alphas.append(0.0)
            failures[row.class_code] += 1
    stain = fit(bases)
    stain = {k: v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v for k, v in stain.items()}
    total = sum(image.shape[0] * image.shape[1] for image in images)
    channel_sum = sum((image.astype(np.float64) / 255).sum(axis=(0, 1)) for image in images)
    channel_sq = sum(np.square(image.astype(np.float64) / 255).sum(axis=(0, 1)) for image in images)
    mean = channel_sum / total
    std = np.sqrt(np.maximum(channel_sq / total - mean ** 2, 1e-8))
    parameters = {'stain': stain, 'controller': controller,
                  'uniform_strength': float(np.mean(alphas)),
                  'training_strengths': alphas,
                  'shuffled_strengths': np.random.default_rng(43).permutation(alphas).tolist(),
                  'normalization': {'mean': mean.tolist(), 'std': std.tolist()},
                  'fit_failures_by_class': dict(failures),
                  'fit_policy': 'Only selected pilot TRAIN images; no validation/test fitting',
                  'evaluation_policy': 'Raw RGB resized 224; shared train-fitted channel standardization; no stain normalization or random augmentation',
                  'duplicate_audit': 'All dataset file hashes; selected train/val decoded pixel hashes. Near-duplicates and alternate-encoding test duplicates unverified.'}
    save(out / 'parameters.json', parameters)
    seed_all(SEED)
    initial = GastricCancerRNN(num_classes=8)
    torch.save(initial.state_dict(), out / 'initial_state.pt')
    save(out / 'frozen.json', {p.name: digest(p) for p in out.iterdir() if p.is_file()})
    snapshots = run / 'source_snapshot'
    snapshots.mkdir(exist_ok=False)
    sources = [Path(__file__), ROOT/'stain_research/controller.py', ROOT/'stain_research/transforms.py',
               ROOT/'stain_research/prepare_classifier.py', ROOT/'stain_research/check.py',
               ROOT/'app/ml/model.py', ROOT/'app/ml/preprocessing.py', ROOT/'app/ml/constants.py',
               ROOT/'stain_research/STRONG_PC_HANDOFF.md', ROOT/'stain_research/FOLLOWUP_RESULTS.md',
               ROOT/'stain_research/CONTROLLER_STUDY.md', ROOT/'stain_research/CONTROLLER_RESULTS.md',
               ROOT/'stain_research/THIRD_PARTY_NOTICE.txt', ROOT/'stain_research/upstream/LICENSE']
    for source in sources:
        target = snapshots / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f'Prepared {len(train)} train / {len(val)} validation; test excluded', flush=True)


def checks(parameters):
    blank = np.full((224, 224, 3), 255, dtype=np.uint8)
    sparse = blank.copy()
    sparse[0, 0] = 0
    cases = {'white': blank, 'black': np.zeros_like(blank),
             'constant_gray': np.full_like(blank, 127), 'one_tissue_pixel': sparse}
    fallback = {}
    for arm in ARMS[1:]:
        fallback[arm] = {}
        for name, image in cases.items():
            result, meta = stain_image(image, arm, parameters, np.random.default_rng(7), .5)
            assert meta['fallback'] and np.array_equal(result, image), (arm, name)
            fallback[arm][name] = 'unchanged extraction fallback passed'
    # Exercise fallback through geometry, patch packing, CUDA forward/backward.
    x = image_to_patch_sequence(Image.fromarray(blank), 56, 16).unsqueeze(0).cuda()
    model = GastricCancerRNN(8).cuda().train()
    logits, _ = model(x)
    logits.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    del model, x, logits
    torch.cuda.empty_cache()
    return fallback


def run_arm(out, arm, train, val, parameters, prepared, batch, epochs):
    destination = out / arm
    destination.mkdir(exist_ok=False)
    seed_all(SEED)
    model = GastricCancerRNN(8).cuda()
    model.load_state_dict(torch.load(prepared / 'initial_state.pt', map_location='cuda', weights_only=True))
    # Reset dropout RNG after initialization: every arm starts identically.
    seed_all(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=.05)
    train_ds = Images(train, parameters, arm, True)
    val_ds = Images(val, parameters, arm, False)
    torch.cuda.reset_peak_memory_stats()
    history, best = [], (-1.0, -1.0)
    for epoch in range(epochs):
        train_ds.epoch = epoch
        model.train()
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss_sum, correct, count = 0.0, 0, 0
        for x, y in loader(train_ds, batch, True, epoch):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits, _ = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            count += len(y)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        validation_start = time.perf_counter()
        metrics, predictions = evaluate(model, val_ds, batch)
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - validation_start
        entry = {'epoch': epoch + 1, 'train': {'loss': loss_sum / count, 'accuracy': correct / count},
                 'val': metrics, 'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                 'epoch_seconds': train_seconds + validation_seconds}
        history.append(entry)
        score = (metrics['macro_f1'], metrics['accuracy'])
        if score > best:
            best = score
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch + 1,
                        'validation': metrics, 'arm': arm, 'parameters': parameters,
                        'class_names': TISSUE_CLASSES}, destination / 'best.pt')
            predictions.to_csv(destination / 'best_predictions_before_reload.csv', index=False)
        save(destination / 'history.json', history)
        print(f'{out.name}/{arm} epoch {epoch+1}: {entry["epoch_seconds"]:.2f}s '
              f'val_acc={metrics["accuracy"]:.4f} macro_f1={metrics["macro_f1"]:.4f}', flush=True)
    del optimizer, model
    torch.cuda.empty_cache()
    state = torch.load(destination / 'best.pt', map_location='cpu', weights_only=True)
    restored = GastricCancerRNN(8).cuda()
    restored.load_state_dict(state['model_state_dict'])
    metrics, predictions = evaluate(restored, val_ds, batch)
    prior = pd.read_csv(destination / 'best_predictions_before_reload.csv')
    columns = [f'p_{code}' for code in TISSUE_CLASSES]
    np.testing.assert_allclose(prior[columns].to_numpy(), predictions[columns].to_numpy(), rtol=1e-5, atol=1e-6)
    predictions.to_csv(destination / 'validation_predictions.csv', index=False)
    pd.DataFrame(train_ds.strengths).to_csv(destination / 'augmentation_events.csv', index=False)
    result = {'arm': arm, 'best_epoch': state['epoch'], 'validation': metrics,
              'checkpoint_reload_verified': True, 'checkpoint_sha256': digest(destination / 'best.pt'),
              'train_seconds_mean': float(np.mean([e['train_seconds'] for e in history])),
              'validation_seconds_mean': float(np.mean([e['validation_seconds'] for e in history])),
              'epoch_seconds_mean': float(np.mean([e['epoch_seconds'] for e in history])),
              'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
              'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
              'failures_by_class': {c: int(train_ds.failures[c]) for c in TISSUE_CLASSES},
              'attempts_by_class': dict(train_ds.attempts)}
    save(destination / 'metrics.json', result)
    del restored, state
    torch.cuda.empty_cache()
    return result


def curves(out, results):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for result in results:
        history = json.loads((out / result['arm'] / 'history.json').read_text())
        epochs = [e['epoch'] for e in history]
        for ax, key in zip(axes, ['loss', 'accuracy', 'macro_f1']):
            ax.plot(epochs, [e['val'][key] for e in history], marker='o', label=result['arm'])
            ax.set(xlabel='Epoch', ylabel='Validation ' + key)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'learning_curves.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'smoke', 'pilot'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--train-per-class', type=int, default=64)
    parser.add_argument('--val-per-class', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch and an available GPU required')
    run = args.run.resolve()
    if args.mode == 'prepare':
        if not (1 <= args.train_per_class <= 128 and 1 <= args.val_per_class <= 32):
            parser.error('Limited pilot maximum is 128 train / 32 validation per class')
        prepare(run, args.train_per_class, args.val_per_class)
        return
    if not 1 <= args.epochs <= 5:
        parser.error('This runner permits at most 5 epochs; full campaign is intentionally unavailable')
    prepared = run / 'prepared'
    for name, expected in json.loads((prepared / 'frozen.json').read_text()).items():
        if digest(prepared / name) != expected:
            raise RuntimeError('Frozen configuration changed: ' + name)
    train = pd.read_csv(prepared / 'train.csv')
    val = pd.read_csv(prepared / 'val.csv')
    if args.mode == 'pilot' and not (run / 'smoke/timing_estimate.json').exists():
        parser.error('Complete smoke tests and report the timing estimate before the pilot')
    if args.mode == 'pilot':
        smoke_config = json.loads((run / 'smoke/config.json').read_text())
        if args.batch_size != smoke_config['batch_size']:
            parser.error('Keep the batch size used by the successful measured smoke tests')
    out = run / args.mode
    out.mkdir(exist_ok=False)
    parameters = json.loads((prepared / 'parameters.json').read_text())
    if args.mode == 'smoke':
        # One measured complete epoch on the intended limited-pilot subset.
        epochs = 1
    else:
        epochs = args.epochs
    config = {'mode': args.mode, 'seed': SEED, 'epochs': epochs, 'batch_size': args.batch_size,
              'arms': ARMS, 'train_images': len(train), 'validation_images': len(val),
              'architecture': 'Existing GastricCancerRNN: CNN + 2-layer BiLSTM + additive attention',
              'optimizer': 'AdamW lr=0.001 weight_decay=0.0001; clip=5; label_smoothing=.05',
              'scheduler': 'None; equal fixed budget; no early stopping',
              'augmentation': 'Shared H/V flips and rotation [-180,180] bilinear; standard arm alone uses original ColorJitter',
              'original_san_scope': 'Upstream-compatible SVD + spherical target + scalar U(.8,1.2) concentration branch; not all SAN experiments',
              'shuffled_control': 'Fixed permutation seed=43 of TRAIN strengths, assigned across training images; no validation fitting',
              'evaluation': parameters['evaluation_policy'],
              'dataset_caching': 'Selected resized source RGBs cached in host memory; online stain transforms every epoch',
              'test_access': 'NONE in this runner',
              'patient_independence': 'Unverified; exploratory image-split experiment',
              'determinism': 'CUDNN deterministic, benchmark off; deterministic algorithms warn_only; numerical nondeterminism remains possible',
              'amp': True, 'python': sys.version, 'platform': platform.platform(),
              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'gpu_capability': torch.cuda.get_device_capability(0),
              'gpu_total_mib': torch.cuda.get_device_properties(0).total_memory / 1024**2,
              'initial_state_sha256': digest(prepared / 'initial_state.pt'),
              'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT/'app/ml/model.py', ROOT/'app/ml/preprocessing.py', ROOT/'stain_research/controller.py', ROOT/'stain_research/transforms.py']}}
    save(out / 'config.json', config)
    save(out / 'fallback_checks.json', checks(parameters))
    results = []
    for arm in ARMS:
        result = run_arm(out, arm, train, val, parameters, prepared, args.batch_size, epochs)
        results.append(result)
        save(out / 'summary.json', results)
    curves(out, results)
    if args.mode == 'smoke':
        seconds = sum(r['epoch_seconds_mean'] for r in results)
        audit = json.loads((run / 'audit.json').read_text())
        full_scale = audit['splits']['train'] / len(train)
        timing = {'measured_complete_epoch_seconds_by_arm': {r['arm']: r['epoch_seconds_mean'] for r in results},
                  'limited_3_epochs_all_6_arms_minutes': seconds * 3 / 60,
                  'limited_3_epochs_with_50pct_overhead_minutes': seconds * 3 * 1.5 / 60,
                  'hypothetical_full_data_40_epochs_6_arms_3_seeds_hours': seconds * full_scale * 40 * 3 / 3600,
                  'full_estimate_caveat': 'Rough linear extrapolation only; larger validation, I/O, thermal changes and checkpoint overhead can alter it. NOT authorized or launched.'}
        save(out / 'timing_estimate.json', timing)
        print(json.dumps(timing, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
