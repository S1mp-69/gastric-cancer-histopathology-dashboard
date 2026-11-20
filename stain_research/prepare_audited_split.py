"""Preserve historical assignments while selecting one exact-file copy per group.

This creates an image-level candidate, never a patient-independent split. It
does not decode pixels, fit a model, read test outcomes, or overwrite old runs.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def select_unique(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['sha256']].append(row)
    kept, removed = [], []
    priority = {'test': 0, 'val': 1, 'train': 2}
    for checksum, members in sorted(groups.items()):
        if len({r['class_code'] for r in members}) != 1:
            for row in members:
                removed.append({**row, 'reason': 'conflicting_class_labels', 'retained_image_path': ''})
            continue
        # Never move a historical test image into development. Retain a test
        # representative whenever a duplicate group intersects the old test set.
        members.sort(key=lambda r: (priority[r['split']], r['image_path']))
        kept.append(members[0])
        for row in members[1:]:
            removed.append({**row, 'reason': 'exact_file_duplicate',
                            'retained_image_path': members[0]['image_path']})
    return kept, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.audit / 'file_integrity.csv'
    with source.open(newline='', encoding='utf-8') as stream:
        original = list(csv.DictReader(stream))
    old_assignments = {r['image_path']: r['split'] for r in original}
    if len(old_assignments) != len(original):
        raise ValueError('Repeated source image identities')
    # Verify source image bytes again: the frozen audit must describe today's files.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(sha, (r['image_path'] for r in original)))
    if any(checksum != row['sha256'] for row, checksum in zip(original, actual)):
        raise ValueError('Dataset bytes changed since the source audit')
    selected, excluded = select_unique(original)
    selected.sort(key=lambda r: (r['split'], r['class_code'], r['image_path']))
    assert len({r['sha256'] for r in selected}) == len(selected)
    assert all(old_assignments[r['image_path']] == r['split'] for r in selected)
    assert {r['image_path'] for r in selected}.isdisjoint(r['image_path'] for r in excluded)
    assert len(selected) + len(excluded) == len(original)
    fields = ['image_path', 'class_code', 'split', 'sha256']
    write(args.output / 'manifest_exact_dedup.csv', selected, fields)
    write(args.output / 'excluded_duplicates.csv', excluded, fields + ['reason', 'retained_image_path'])
    portable = [{**r, 'image_path': str(Path(r['image_path']).relative_to(ROOT)).replace('\\', '/')} for r in selected]
    write(args.output / 'manifest_portable.csv', portable, fields)
    # A blank template is a request for verified metadata, not fabricated IDs.
    mapping = [{'class_code': r['class_code'], 'image_name': Path(r['image_path']).name,
                'patient_id': '', 'slide_id': '', 'mapping_source': ''} for r in original]
    write(args.output / 'patient_slide_mapping_to_complete.csv', mapping,
          ['class_code', 'image_name', 'patient_id', 'slide_id', 'mapping_source'])
    counts = Counter(r['split'] for r in selected)
    report = {
        'status': 'PROVISIONAL exact-file-deduplicated image split; not patient independent',
        'source_images': len(original), 'retained_unique_file_images': len(selected),
        'excluded_redundant_or_conflicting_images': len(excluded),
        'retained_counts': dict(counts),
        'retained_class_split_counts': dict(Counter(f"{r['split']}/{r['class_code']}" for r in selected)),
        'excluded_counts': dict(Counter(r['split'] for r in excluded)),
        'policy': 'Retain one lexicographically first representative per exact-file hash; prefer original test, then val, then train; conflicting-label groups excluded entirely',
        'source_assignment_changes': 0, 'remaining_exact_hash_duplicates': 0,
        'test_access': 'Byte hashing only; test pixels not decoded and no test inference performed',
        'remaining_limits': ['Patient and slide mapping unavailable',
                             'Near-duplicate and alternate-encoding duplicate detection incomplete',
                             'Existing exploratory use of this dataset prevents claims of an independent replication'],
        'patient_independent_training_ready': False,
        'source_audit_sha256': sha(source),
        'manifest_sha256': sha(args.output / 'manifest_exact_dedup.csv'),
        'portable_manifest_sha256': sha(args.output / 'manifest_portable.csv'),
    }
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plan = {
        'status': 'NOT LAUNCHED: verified patient/slide linkage is missing',
        'training_ready': False,
        'architecture': 'Existing CNN-BiLSTM-attention', 'seed': 42,
        'candidate_image_split': 'manifest_exact_dedup.csv',
        'future_split_requirement': 'Build a new frozen patient-grouped development/evaluation design from verified linkage; do not infer patient IDs from image numbers',
        'proposed_baseline_max_epochs': 40, 'early_stopping_patience': 7,
        'batch_size': 16, 'optimizer': 'AdamW lr=0.001 weight_decay=0.0001',
        'selection_metric': 'validation macro-F1; deterministic raw RGB evaluation with training-only channel moments',
        'comparison_after_baseline': ['standard', 'uniform_weaker', 'adaptive'],
        'constraints': ['isolated outputs', 'best checkpoint reload', 'no dashboard activation',
                        'test evaluation deferred until choices are fixed', 'equal budgets and initialization for comparisons'],
        'runner_note': 'The existing limited-pilot runner deliberately caps epochs at five. Implement and smoke-test a separate convergence runner only after resolving split readiness.'}
    (args.output / 'baseline_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
