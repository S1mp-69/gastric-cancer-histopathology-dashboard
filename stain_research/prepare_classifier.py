"""Read-only source audit and isolated manifest preparation; no test inference.

SHA256 includes test FILE BYTES solely for integrity/duplicate detection. No test
images are decoded, transformed, used to fit parameters, or evaluated here.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    protected = [root / 'app.db', root / 'requirements.txt', root / 'README.md']
    for directory in ('app', 'training', 'models', 'model_train', 'data/support', 'data/uploads'):
        protected += [p for p in (root / directory).rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        fingerprints = dict(zip((str(p.relative_to(root)) for p in protected), pool.map(sha, protected)))
    (args.output / 'protected_before.json').write_text(json.dumps(fingerprints, indent=2))
    source = root / 'model_train/runs/dataset_manifest_20260806_local.csv'
    rows = list(csv.DictReader(source.open(newline='', encoding='utf-8')))
    dataset = root / 'data/raw/HMU-GC-HE-30K/all_image'
    images = sorted(dataset.glob('*/*.png'))
    index = defaultdict(list)
    for p in images:
        index[(p.parent.name, p.name)].append(p)
    changes = 0
    for row in rows:
        old = row['image_path']
        name = re.split(r'[\\/]', old)[-1]
        matches = index[(row['class_code'], name)]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous/missing class-filename match: {old}')
        row['image_path'] = str(matches[0].resolve())
        changes += row['image_path'] != old
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
    if len({r['image_path'] for r in rows}) != len(rows) or len(rows) != len(images):
        raise ValueError('Manifest repeated identities or dataset coverage mismatch')
    write_csv(args.output / 'manifest_local.csv', rows, ['image_path', 'class_code', 'split'])
    print(f'Verified {len(rows)} image paths; hashing file bytes for duplicate audit', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(sha, (r['image_path'] for r in rows)))
    groups = defaultdict(list)
    for row, digest in zip(rows, hashes):
        row['sha256'] = digest
        groups[digest].append(row)
    duplicates = [group for group in groups.values() if len(group) > 1]
    write_csv(args.output / 'file_integrity.csv', rows, ['image_path', 'class_code', 'split', 'sha256'])
    (args.output / 'duplicate_groups.json').write_text(json.dumps(duplicates, indent=2))
    cross = [g for g in duplicates if len({r['split'] for r in g}) > 1]
    # Preserve every source assignment; quarantine all duplicate identities from
    # train/validation rather than moving any image or reading test pixels.
    duplicate_paths = {r['image_path'] for g in duplicates for r in g}
    eligible = [r for r in rows if r['split'] != 'test' and r['image_path'] not in duplicate_paths]
    write_csv(args.output / 'eligible_train_val.csv', eligible, ['image_path', 'class_code', 'split', 'sha256'])
    report = {
        'source_manifest': str(source), 'source_manifest_sha256': sha(source),
        'images': len(rows), 'rebased_paths': changes,
        'splits': dict(Counter(r['split'] for r in rows)),
        'class_split_counts': dict(Counter(f"{r['split']}/{r['class_code']}" for r in rows)),
        'duplicate_groups': len(duplicates), 'cross_split_duplicate_groups': len(cross),
        'duplicate_policy': 'All exact-file duplicate identities excluded from pilot train/val; original splits preserved',
        'test_policy': 'File-byte hashes only for audit; no test pixel decoding, fitting or evaluation',
        'patient_slide_status': 'UNVERIFIED. Class_number filenames provide no verifiable patient/slide mapping. Clinical CSV has patient identifiers but no image linkage.',
        'scope': 'Exploratory image-split pilot only; no patient-independent or clinical claims',
        'filename_pattern_class_number': all(re.fullmatch(r'[A-Z]+_\d+\.png', p.name) for p in images),
        'protected_files': len(protected),
    }
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
