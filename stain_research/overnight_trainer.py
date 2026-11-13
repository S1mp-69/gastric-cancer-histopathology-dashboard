"""Bounded, resumable four-arm image-level research campaign. No test inference.

All new model artifacts stay under the campaign directory. This module imports
the original architecture read-only and never imports the application/database.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as tv
from torchvision.transforms import functional as VF, InterpolationMode
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ml.model import GastricCancerRNN
from app.ml.constants import TISSUE_CLASSES
from app.ml.preprocessing import image_to_patch_sequence
from efficient_cache import atomic_json as cache_atomic_json, cached_apply, serial, sha

ARMS = ['standard', 'residual_fixed', 'uniform_weaker', 'adaptive']
TZ = ZoneInfo('Asia/Calcutta')
_thread_controller = None


def atomic_json(path, value):
    # Windows readers/OneDrive can briefly deny rename. Retry without changing
    # training, augmentation, cached numerical values or checkpoint semantics.
    for attempt in range(20):
        try:
            return cache_atomic_json(path, value)
        except PermissionError:
            if attempt == 19:
                if Path(path).name == 'progress.json':
                    print('Progress update skipped after transient Windows file lock', flush=True)
                    return
                raise
            time.sleep(.05)


def now():
    return datetime.now(TZ).isoformat(timespec='seconds')


def setup(seed):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(_):
    global _thread_controller
    torch.set_num_threads(1)
    _thread_controller = threadpool_limits(limits=1)


class CachedImages(Dataset):
    def __init__(self, directory, split, arm, seed, epoch, subset=None):
        if split not in ('train', 'val'):
            raise ValueError('Final test images are forbidden in this dataset')
        self.directory = str(directory)
        index = pd.read_csv(Path(directory) / 'index.csv')
        self.indices = np.flatnonzero(index.split.eq(split).to_numpy())
        if subset is not None:
            selected = index.iloc[self.indices].groupby('class_code', group_keys=False).head(max(1, subset//8))
            self.indices = selected.index.to_numpy()[:subset]
        self.frame = index.iloc[self.indices].reset_index(drop=True)
        assert set(self.frame.split) == {split}
        self.labels = np.array([TISSUE_CLASSES.index(c) for c in self.frame.class_code], dtype=np.int64)
        self.params = json.loads((Path(directory) / 'train_parameters.json').read_text())
        self.split, self.arm, self.seed, self.epoch = split, arm, seed, epoch
        self.maps = None
        self.jitter = tv.ColorJitter(brightness=.1, contrast=.1, saturation=.05, hue=.02)

    def __getstate__(self):
        result = self.__dict__.copy(); result['maps'] = None
        return result

    def open(self):
        if self.maps is None:
            names = ['rgb', 'basis', 'concentrations', 'diagnostics', 'valid']
            self.maps = {name: np.load(Path(self.directory)/(name+'.npy'), mmap_mode='r') for name in names}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        self.open()
        row = self.indices[index]
        image = np.array(self.maps['rgb'][row])
        fallback, alpha = False, 0.0
        if self.split == 'train':
            # Stream identities do not depend on worker count, batch size or arm.
            stream = int(self.seed) * 1000000000 + int(self.epoch.value) * 1000000 + int(row)
            if self.arm != 'standard':
                fixed = {'residual_fixed': 1.0, 'uniform_weaker': self.params['uniform_strength'],
                         'adaptive': None}[self.arm]
                image, fallback, alpha = cached_apply(image, self.maps['basis'][row], self.maps['concentrations'][row],
                                                       self.maps['diagnostics'][row], self.maps['valid'][row],
                                                       self.params, np.random.default_rng(stream), fixed)
            image = Image.fromarray(image)
            geometry = np.random.default_rng(stream + 900000000)
            if geometry.random() < .5:
                image = VF.hflip(image)
            if geometry.random() < .5:
                image = VF.vflip(image)
            image = VF.rotate(image, float(geometry.uniform(-180, 180)), interpolation=InterpolationMode.BILINEAR)
            if self.arm == 'standard':
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(stream + 800000000)
                    image = self.jitter(image)
        else:
            image = Image.fromarray(image)
        n = self.params['normalization']
        tensor = image_to_patch_sequence(image, 56, 16, tuple(n['mean']), tuple(n['std']))
        return tensor, int(self.labels[index]), int(fallback), float(alpha), int(index)


class EpochSampler(Sampler):
    def __init__(self, count, seed):
        self.count, self.seed, self.epoch, self.offset = count, seed, 0, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 100003)
        return iter(torch.randperm(self.count, generator=generator).tolist()[self.offset:])

    def __len__(self):
        return self.count - self.offset


def make_loader(dataset, batch, workers, sampler=None):
    kwargs = dict(batch_size=batch, num_workers=workers, pin_memory=True,
                  worker_init_fn=worker_init, generator=torch.Generator().manual_seed(dataset.seed))
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, sampler=sampler, shuffle=False, **kwargs)


def evaluate(model, loader, dataset, amp):
    model.eval()
    probs, labels, indices = [], [], []
    with torch.inference_mode():
        for x, y, _, _, i in loader:
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                logits, _ = model(x.cuda(non_blocking=True))
            probs.extend(torch.softmax(logits.float(), 1).cpu().tolist())
            labels.extend(y.tolist()); indices.extend(i.tolist())
    p, y = np.asarray(probs), np.asarray(labels)
    pred = p.argmax(1)
    report = classification_report(y, pred, labels=list(range(8)), target_names=TISSUE_CLASSES,
                                   output_dict=True, zero_division=0)
    confidence = p.max(1); correct = pred == y
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    metrics = {'accuracy': float(correct.mean()),
               'macro_f1': float(f1_score(y, pred, labels=list(range(8)), average='macro', zero_division=0)),
               'loss': float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
               'per_class': {c: report[c] for c in TISSUE_CLASSES},
               'confusion_matrix': confusion_matrix(y, pred, labels=list(range(8))).tolist(),
               'ece_10_bins': ece, 'multiclass_brier': float(np.square(p-np.eye(8)[y]).sum(1).mean())}
    predictions = dataset.frame.iloc[indices][['image_path', 'class_code']].reset_index(drop=True)
    predictions['predicted_class'] = [TISSUE_CLASSES[int(v)] for v in pred]
    for i, c in enumerate(TISSUE_CLASSES):
        predictions['p_' + c] = p[:, i]
    return metrics, predictions


def atomic_checkpoint(path, state):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def telemetry():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return {'gpu_csv': r.stdout.strip(), 'recorded_at': now()}
    except Exception as exc:
        return {'telemetry_error': str(exc), 'recorded_at': now()}


def benchmark(campaign, directory):
    output = campaign / 'throughput_benchmark.json'
    if output.exists():
        return json.loads(output.read_text())
    setup(42)
    results = []
    # Benchmark actual cached residual augmentation, geometry, transfer and AMP
    # updates. All trials are discarded and use training images only.
    for batch, workers in [(16, 0), (32, 0), (64, 0), (64, 2), (64, 4), (96, 2), (96, 4)]:
        epoch = mp.Value('i', 0, lock=False)
        dataset = CachedImages(directory, 'train', 'residual_fixed', 42, epoch, subset=1536)
        loader = make_loader(dataset, batch, workers)
        setup(42)
        model = GastricCancerRNN(8).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        scaler = torch.amp.GradScaler('cuda', init_scale=4096)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=.05)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        iterator = iter(loader)
        steps, items, loss_finite, skipped = 0, 0, True, 0
        warmup_seconds = None
        begin = time.perf_counter()
        try:
            for step in range(12):
                x, y, _, _, _ = next(iterator)
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits, _ = model(x)
                    loss = criterion(logits, y)
                loss_finite &= bool(torch.isfinite(loss))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                loss_finite &= bool(torch.isfinite(norm))
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                skipped += int(scaler.get_scale() < old_scale)
                torch.cuda.synchronize()
                if step == 2:
                    warmup_seconds = time.perf_counter() - begin
                    begin = time.perf_counter()
                elif step > 2:
                    items += len(y); steps += 1
            seconds = time.perf_counter() - begin
            free, total = torch.cuda.mem_get_info()
            trial = {'batch_size': batch, 'workers': workers, 'warmup_including_worker_start_seconds': warmup_seconds,
                     'timed_steps': steps, 'timed_images': items, 'seconds': seconds,
                     'images_per_second': items / seconds, 'finite_loss_and_gradients': loss_finite,
                     'skipped_amp_steps': skipped, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                     'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
                     'cuda_free_mib': free/1024**2, 'telemetry': telemetry()}
            trial['eligible'] = loss_finite and skipped == 0 and trial['peak_reserved_mib'] <= 3072 and free >= 768*1024**2
        except (RuntimeError, StopIteration) as exc:
            trial = {'batch_size': batch, 'workers': workers, 'eligible': False, 'error': str(exc)}
        results.append(trial)
        atomic_json(campaign / 'throughput_benchmark_partial.json', {'trials': results})
        print('BENCHMARK ' + json.dumps(trial), flush=True)
        del iterator, loader, dataset, model, optimizer, scaler
        torch.cuda.empty_cache()
    eligible = [r for r in results if r['eligible']]
    if not eligible:
        raise RuntimeError('No stable throughput setting with adequate VRAM headroom')
    # Prefer fewer workers when throughput is within five percent of the best.
    fastest = max(r['images_per_second'] for r in eligible)
    selected = min([r for r in eligible if r['images_per_second'] >= .95*fastest],
                   key=lambda r: (r['workers'], -r['images_per_second']))
    result = {'trials': results, 'selected': selected,
              'protocol': 'TRAIN-only warm-cache residual augmentation with real GPU forward/backward, AMP fp16, optimizer updates; benchmark weights discarded',
              'cpu_threads_main': 2, 'cpu_threads_per_worker': 1, 'maximum_workers': 4,
              'headroom_rule': 'CUDA reserved <=3072 MiB and free >=768 MiB; no overclocking or thermal changes'}
    atomic_json(output, result)
    return result


def run_training(campaign, arm, seed):
    config = json.loads((campaign / 'campaign_config.json').read_text())
    directory = Path(json.loads((campaign / 'cache_reference.json').read_text())['directory'])
    out = campaign / f'seed{seed}_{arm}'
    out.mkdir(exist_ok=True)
    if (out / 'metrics.json').exists():
        return
    setup(seed)
    signature = hashlib.sha256(json.dumps({'arm': arm, 'seed': seed, 'config': config}, sort_keys=True).encode()).hexdigest()
    atomic_json(out / 'process.json', {'pid': os.getpid(), 'started_at': now(), 'arm': arm, 'seed': seed})
    atomic_json(out / 'config.json', {'arm': arm, 'seed': seed, 'signature': signature, **config})
    epoch_shared = mp.Value('i', 0, lock=False)
    train_ds = CachedImages(directory, 'train', arm, seed, epoch_shared)
    val_ds = CachedImages(directory, 'val', arm, seed, epoch_shared)
    sampler = EpochSampler(len(train_ds), seed)
    train_loader = make_loader(train_ds, config['batch_size'], config['workers'], sampler)
    val_loader = make_loader(val_ds, config['batch_size'], min(2, config['workers']))
    initial_path = campaign / f'initial_seed{seed}.pt'
    if not initial_path.exists():
        initial_model = GastricCancerRNN(8)
        atomic_checkpoint(initial_path, initial_model.state_dict()); del initial_model
    model = GastricCancerRNN(8).cuda()
    model.load_state_dict(torch.load(initial_path, map_location='cuda', weights_only=True))
    setup(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=.5)
    scaler = torch.amp.GradScaler('cuda', init_scale=4096, enabled=config['amp'])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=.05)
    history, epoch, offset, best, stale = [], 0, 0, (-1.0, -1.0), 0
    cumulative = {'loss_sum': 0.0, 'correct': 0, 'items': 0, 'seconds': 0.0,
                  'failures_by_class': [0]*8, 'attempts_by_class': [0]*8, 'strength_sum_by_class': [0.0]*8, 'amp_skipped_steps': 0}
    empty_cumulative = copy.deepcopy(cumulative)
    if (out / 'last.pt').exists():
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if state['signature'] != signature:
            raise RuntimeError('Resume protocol/configuration signature mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        history, epoch, offset = state['history'], state['epoch'], state['offset']
        best, stale, cumulative = tuple(state['best_score']), state['stale'], state['cumulative']
        restore_rng(state['rng'])
        print(f'Resuming {arm} seed {seed} epoch {epoch+1}, completed items {offset}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    checkpoint_time = time.perf_counter()
    def checkpoint():
        atomic_checkpoint(out / 'last.pt', {'signature': signature, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                          'scaler': scaler.state_dict(), 'history': history, 'epoch': epoch, 'offset': offset,
                          'best_score': best, 'stale': stale, 'cumulative': cumulative,
                          'rng': rng_state(), 'saved_at': now(), 'arm': arm, 'seed': seed,
                          'initial_sha256': sha(initial_path)})
    while epoch < config['max_epochs'] and stale < config['patience']:
        epoch_shared.value = epoch; sampler.epoch = epoch; sampler.offset = offset
        model.train(); session_started = time.perf_counter(); last_progress = 0.0
        for x, y, failed, alpha, _ in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                logits, _ = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            # GradScaler.unscale_ has recorded inf/NaN gradients. Let its
            # documented step-skip/backoff behavior handle fp16 overflow instead
            # of raising before scaler.step can act. Never accept nonfinite loss.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=False)
            old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            skipped = int(scaler.get_scale() < old_scale)
            cumulative['amp_skipped_steps'] += skipped
            if not torch.isfinite(norm):
                if not config['amp'] or not skipped:
                    raise RuntimeError('Nonfinite gradient without a handled AMP overflow')
                print(f'{now()} AMP overflow handled: optimizer step skipped; scale {old_scale} -> {scaler.get_scale()}',flush=True)
            count = len(y); offset += count
            cumulative['items'] += count; cumulative['loss_sum'] += loss.item()*count
            cumulative['correct'] += int((logits.argmax(1) == y).sum())
            labels = y.cpu().numpy()
            for c in range(8):
                mask = labels == c
                cumulative['attempts_by_class'][c] += int(mask.sum())
                cumulative['failures_by_class'][c] += int(failed.numpy()[mask].sum())
                cumulative['strength_sum_by_class'][c] += float(alpha.numpy()[mask].sum())
            if time.perf_counter() - last_progress >= 10:
                elapsed = cumulative['seconds'] + time.perf_counter() - session_started
                atomic_json(out / 'progress.json', {'status': 'training', 'arm': arm, 'seed': seed,
                    'epoch': epoch+1, 'max_epochs': config['max_epochs'], 'items_in_epoch': offset,
                    'total_train_images': len(train_ds), 'epoch_elapsed_seconds': elapsed,
                    'images_per_second': cumulative['items']/max(elapsed,.001),
                    'train_loss_so_far': cumulative['loss_sum']/max(cumulative['items'],1),
                    'last_validation': history[-1]['val'] if history else None,
                    'best_macro_f1': max(0.0,best[0]), 'epochs_without_improvement': stale,
                    'failures_by_class': dict(zip(TISSUE_CLASSES,cumulative['failures_by_class'])),
                    'updated_at': now()})
                last_progress = time.perf_counter()
            if time.perf_counter() - checkpoint_time >= config['checkpoint_seconds']:
                cumulative['seconds'] += time.perf_counter() - session_started
                session_started = time.perf_counter(); checkpoint(); checkpoint_time = time.perf_counter()
        torch.cuda.synchronize()
        train_seconds = cumulative['seconds'] + time.perf_counter() - session_started
        validation_started = time.perf_counter()
        metrics, predictions = evaluate(model, val_loader, val_ds, config['amp'])
        torch.cuda.synchronize(); validation_seconds = time.perf_counter()-validation_started
        scheduler.step(metrics['loss'])
        entry = {'epoch': epoch+1, 'train': {'loss': cumulative['loss_sum']/max(cumulative['items'],1),
                                           'accuracy': cumulative['correct']/max(cumulative['items'],1)},
                 'val': metrics, 'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                 'epoch_seconds': train_seconds+validation_seconds,
                 'learning_rate': optimizer.param_groups[0]['lr'],
                 'images_per_second': len(train_ds)/max(train_seconds,.001),
                 'failures_by_class': dict(zip(TISSUE_CLASSES,cumulative['failures_by_class'])),
                 'attempts_by_class': dict(zip(TISSUE_CLASSES,cumulative['attempts_by_class'])),
                 'mean_strength_by_class': {c: cumulative['strength_sum_by_class'][i]/max(cumulative['attempts_by_class'][i],1) for i,c in enumerate(TISSUE_CLASSES)},
                 'amp_skipped_steps': cumulative['amp_skipped_steps'], 'finished_at': now(),
                 'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
        history.append(entry); atomic_json(out / 'history.json', history)
        score = (metrics['macro_f1'], metrics['accuracy'])
        if metrics['macro_f1'] > best[0] + config['min_delta']:
            best, stale = score, 0
            atomic_checkpoint(out / 'best.pt', {'model_state_dict': model.state_dict(), 'epoch': epoch+1,
                              'validation': metrics, 'arm': arm, 'seed': seed, 'parameters': train_ds.params,
                              'config': config, 'initial_sha256': sha(initial_path), 'saved_at': now()})
            predictions.to_csv(out / 'best_validation_before_reload.csv', index=False)
        else:
            stale += 1
        predictions.to_csv(out / 'latest_validation_predictions.csv', index=False)
        print(f'{now()} {arm} seed={seed} epoch={epoch+1} seconds={entry["epoch_seconds"]:.1f} '
              f'val_acc={metrics["accuracy"]:.4f} macro_f1={metrics["macro_f1"]:.4f} stale={stale}', flush=True)
        epoch += 1; offset = 0; cumulative = copy.deepcopy(empty_cumulative)
        checkpoint(); checkpoint_time = time.perf_counter()
        atomic_json(out / 'progress.json', {'status':'epoch_complete','arm':arm,'seed':seed,'epoch':epoch,
                    'max_epochs':config['max_epochs'],'items_in_epoch':len(train_ds),'total_train_images':len(train_ds),
                    'last_validation':metrics,'best_macro_f1':best[0], 'epochs_without_improvement':stale,
                    'failures_by_class':entry['failures_by_class'], 'updated_at':now()})
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    state = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    restored = GastricCancerRNN(8).cuda(); restored.load_state_dict(state['model_state_dict'])
    metrics, predictions = evaluate(restored, val_loader, val_ds, config['amp'])
    before = pd.read_csv(out / 'best_validation_before_reload.csv')
    columns = ['p_'+c for c in TISSUE_CLASSES]
    np.testing.assert_allclose(before[columns], predictions[columns], rtol=1e-4, atol=1e-6)
    predictions.to_csv(out / 'validation_predictions.csv', index=False)
    result = {'arm':arm,'seed':seed,'best_epoch':state['epoch'],'completed_epochs':len(history),
              'stop_reason':'early_stopping' if stale>=config['patience'] else 'maximum_epoch_budget',
              'validation':metrics,'checkpoint_reload_verified':True,'best_checkpoint_sha256':sha(out/'best.pt'),
              'mean_epoch_seconds':float(np.mean([e['epoch_seconds'] for e in history])),
              'total_epoch_seconds':sum(e['epoch_seconds'] for e in history),
              'failures_by_class':{c:sum(e['failures_by_class'][c] for e in history) for c in TISSUE_CLASSES},
              'initial_sha256':sha(initial_path),'finished_at':now(),'test_evaluated':False,
              'patient_independence':'UNVERIFIED; provisional exact-file-controlled image split'}
    atomic_json(out/'metrics.json',result)
    atomic_json(out/'progress.json',{'status':'complete','updated_at':now(),**result})
    print('RUN COMPLETE '+json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['benchmark','train'])
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    campaign=args.campaign.resolve()
    directory=Path(json.loads((campaign/'cache_reference.json').read_text())['directory'])
    if not json.loads((directory/'progress.json').read_text()).get('finished'):
        raise RuntimeError('Deterministic cache is not complete')
    with threadpool_limits(limits=2):
        if args.mode=='benchmark':
            benchmark(campaign,directory)
        else:
            if args.arm is None: parser.error('--arm required')
            run_training(campaign,args.arm,args.seed)


if __name__=='__main__':
    mp.freeze_support()
    main()
