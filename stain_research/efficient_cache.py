"""Versioned, restartable float64 stain cache for TRAIN and VAL only.

Adapted stain extraction is attributed in THIRD_PARTY_NOTICE.txt. Cached values
are deterministic; targets, intensities and geometry remain random each epoch.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits
from controller import diagnostics, strength, apply
from transforms import fit, rgb2od, sample

ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 'float64-source-decomposition-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(serial(value), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def read_rgb(row):
    if row['split'] not in ('train', 'val'):
        raise ValueError('Test rows forbidden in the deterministic cache')
    data = Path(row['image_path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('Input bytes changed: ' + row['image_path'])
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image.convert('RGB').resize((224, 224)))


def decompose(row):
    image = read_rgb(row)
    try:
        features, basis, concentrations, _ = diagnostics(image)
        if not (np.isfinite(features).all() and np.isfinite(concentrations).all()):
            raise ValueError('Nonfinite decomposition')
        return image, basis, concentrations, features, ''
    except (ValueError, np.linalg.LinAlgError) as exc:
        return image, np.zeros((3, 2)), np.zeros((2, 224 * 224)), np.zeros(2), str(exc)


def cached_apply(image, basis, concentrations, features, valid, parameters, rng, fixed=None):
    if not valid:
        return image.copy(), True, 0.0
    alpha = strength(features, parameters['controller']) if fixed is None else float(fixed)
    if not 0 <= alpha <= 1:
        raise ValueError('Invalid strength')
    if alpha == 0:
        return image.copy(), False, 0.0
    target = sample(parameters['stain'], 'spherical', rng)
    scale = 1 + rng.uniform(-.2, .2)
    od = rgb2od(image).reshape(-1, 3)
    tissue = (od ** 2).sum(axis=1) > .09
    delta = ((target * scale - basis) @ concentrations).T
    updated = od.copy()
    updated[tissue] += alpha * delta[tissue]
    output = np.rint(np.clip(255 * np.exp(-np.maximum(updated, 0)), 0, 255)).reshape(image.shape).astype(np.uint8)
    output.reshape(-1, 3)[~tissue] = image.reshape(-1, 3)[~tissue]
    return output, False, alpha


def cpu_benchmark(rows, output):
    chosen = rows.groupby('class_code', group_keys=False).head(16).to_dict('records')
    started = time.perf_counter()
    images = [read_rgb(r) for r in chosen]
    load_seconds = time.perf_counter() - started
    timings = []
    for workers in (1, 2, 4):
        def operation(image):
            try:
                diagnostics(image)
                return True
            except ValueError:
                return False
        started = time.perf_counter()
        with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
            successful = sum(pool.map(operation, images))
        elapsed = time.perf_counter() - started
        timings.append({'threads': workers, 'blas_threads_per_task': 1, 'seconds': elapsed,
                        'images_per_second': len(images) / elapsed, 'extraction_successes': successful})
    best = max(timings, key=lambda r: r['images_per_second'])['threads']
    atomic_json(output, {'sample_images': len(images), 'input_policy': 'TRAIN only, warm filesystem cache',
                         'read_verify_decode_resize_seconds': load_seconds,
                         'read_verify_decode_resize_images_per_second': len(images) / load_seconds,
                         'decomposition_cpu_trials': timings, 'selected_cache_threads': best})
    return best


def build(manifest, campaign):
    campaign.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest)
    frame = frame[frame.split.isin(['train', 'val'])].sort_values(['split', 'class_code', 'image_path']).reset_index(drop=True)
    if frame.empty or set(frame.split) != {'train', 'val'}:
        raise ValueError('Training and validation rows required')
    assert len(set(frame.sha256)) == len(frame)
    source = {'version': CACHE_VERSION, 'manifest_sha256': sha(manifest),
              'controller_sha256': sha(ROOT / 'stain_research/controller.py'),
              'transform_sha256': sha(ROOT / 'stain_research/transforms.py'),
              'cache_code_sha256': sha(Path(__file__)), 'numpy': np.__version__,
              'resize': 'PIL RGB resize 224x224 default bicubic; no geometry before decomposition',
              'concentrations_dtype': 'float64', 'diagnostics_dtype': 'float64',
              'image_hashes': hashlib.sha256('\n'.join(frame.sha256).encode()).hexdigest(),
              'excluded_partition': 'test'}
    key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    directory = ROOT / 'stain_research/cache' / (CACHE_VERSION + '_' + key[:16])
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(campaign / 'cache_reference.json', {'directory': str(directory), 'key': key, 'source': source})
    progress_path = directory / 'progress.json'
    n = len(frame)
    shapes = {'rgb': ((n, 224, 224, 3), np.uint8), 'basis': ((n, 3, 2), np.float64),
              'concentrations': ((n, 2, 224 * 224), np.float64),
              'diagnostics': ((n, 2), np.float64), 'valid': ((n,), np.bool_)}
    if not progress_path.exists():
        required = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in shapes.values())
        if shutil.disk_usage(directory).free < required + 10 * 1024**3:
            raise RuntimeError('Insufficient disk for the exact float64 cache plus 10 GiB headroom')
        for name, (shape, dtype) in shapes.items():
            np.lib.format.open_memmap(directory / (name + '.npy'), mode='w+', dtype=dtype, shape=shape).flush()
        frame.to_csv(directory / 'index.csv', index=False)
        atomic_json(directory / 'source.json', source)
        atomic_json(progress_path, {'completed': 0, 'total': n, 'key': key, 'finished': False})
    progress = json.loads(progress_path.read_text())
    if progress['key'] != key:
        raise RuntimeError('Cache provenance mismatch')
    if progress.get('finished'):
        print('Reusing complete versioned deterministic cache: ' + str(directory), flush=True)
        return directory
    workers = cpu_benchmark(frame[frame.split == 'train'], campaign / 'cpu_benchmark.json')
    arrays = {name: np.load(directory / (name + '.npy'), mmap_mode='r+') for name in shapes}
    failures_path = directory / 'failures.json'
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    started = time.perf_counter()
    start_index = progress['completed']
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(start_index, n, 64):
            batch = frame.iloc[start:start + 64].to_dict('records')
            for i, result in enumerate(pool.map(decompose, batch), start=start):
                image, basis, concentrations, features, reason = result
                arrays['rgb'][i] = image
                arrays['basis'][i] = basis
                arrays['concentrations'][i] = concentrations
                arrays['diagnostics'][i] = features
                arrays['valid'][i] = not bool(reason)
                if reason:
                    failures.append({'index': i, 'image_path': frame.iloc[i].image_path,
                                     'class_code': frame.iloc[i].class_code, 'split': frame.iloc[i].split, 'reason': reason})
            end = start + len(batch)
            for array in arrays.values():
                array.flush()
            atomic_json(failures_path, failures)
            elapsed = time.perf_counter() - started
            atomic_json(progress_path, {'completed': end, 'total': n, 'key': key, 'finished': False,
                                       'last_session_seconds': elapsed,
                                       'last_session_images_per_second': (end - start_index) / max(elapsed, .001),
                                       'fallback_images': len(failures)})
            if end % 512 < 64 or end == n:
                print(f'Cache {end}/{n}, {(end-start_index)/max(elapsed,.001):.1f} images/s, failures={len(failures)}', flush=True)
    train = frame.split.eq('train').to_numpy()
    valid = np.asarray(arrays['valid'])
    features = np.asarray(arrays['diagnostics'])[train & valid]
    low, high = np.quantile(features, [.1, .9], axis=0)
    controller = {'low': low.tolist(), 'high': high.tolist(), 'minimum_strength': .1,
                  'fit_successes': int((train & valid).sum()), 'fit_failures': int((train & ~valid).sum())}
    stain = fit(np.asarray(arrays['basis'])[train & valid])
    alphas = [strength(v, controller) if ok else 0.0
              for v, ok in zip(np.asarray(arrays['diagnostics'])[train], valid[train])]
    sums, squares, pixels = np.zeros(3), np.zeros(3), 0
    indices = np.flatnonzero(train)
    for start in range(0, len(indices), 64):
        image = np.asarray(arrays['rgb'][indices[start:start + 64]], dtype=np.float64) / 255
        sums += image.sum(axis=(0, 1, 2)); squares += (image ** 2).sum(axis=(0, 1, 2))
        pixels += int(np.prod(image.shape[:3]))
    mean = sums / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean ** 2, 1e-8))
    parameters = serial({'controller': controller, 'stain': stain,
                         'uniform_strength': float(np.mean(alphas)),
                         'normalization': {'mean': mean.tolist(), 'std': std.tolist()},
                         'fit_images': int(train.sum()), 'fit_policy': 'All candidate TRAIN only; validation and test excluded',
                         'fit_fallbacks_by_class': dict(Counter(frame.loc[train & ~valid, 'class_code']))})
    atomic_json(directory / 'train_parameters.json', parameters)
    equivalence = []
    for code, group in frame[frame.split == 'train'].groupby('class_code'):
        for i in group.index[:2]:
            image = np.array(arrays['rgb'][i])
            for fixed in (None, 1.0, parameters['uniform_strength'], 0.0):
                expected, meta = apply(image, parameters['stain'], controller, np.random.default_rng(17), fixed)
                actual, fallback, alpha = cached_apply(image, arrays['basis'][i], arrays['concentrations'][i],
                                                     arrays['diagnostics'][i], arrays['valid'][i], parameters,
                                                     np.random.default_rng(17), fixed)
                if not np.array_equal(expected, actual) or fallback != meta['fallback']:
                    raise AssertionError(f'Cache differs from uncached controller: {code}, {fixed}')
                equivalence.append({'class': code, 'index': int(i), 'strength': fixed, 'exact_pixel_match': True})
    atomic_json(campaign / 'cache_equivalence.json', {'cases': len(equivalence), 'all_exact_pixel_matches': True,
                                                     'details': equivalence})
    progress.update({'completed': n, 'total': n, 'key': key, 'finished': True,
                     'fallback_images': len(failures), 'cache_bytes': sum((directory / (name+'.npy')).stat().st_size for name in shapes),
                     'build_session_seconds': time.perf_counter() - started})
    atomic_json(progress_path, progress)
    print('Cache complete: ' + str(directory), flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest, args.campaign)


if __name__ == '__main__':
    main()
