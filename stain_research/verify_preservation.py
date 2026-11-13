"""Verify protected original artifacts and all dataset file bytes after experiments."""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from prepare_classifier import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    original = json.loads((args.run / 'protected_before.json').read_text())
    integrity = list(csv.DictReader((args.run / 'file_integrity.csv').open()))
    expected = [(root / p, h) for p, h in original.items()]
    expected += [(Path(row['image_path']), row['sha256']) for row in integrity]
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(sha, (p for p, _ in expected)))
    changed = [str(p) for (p, before), after in zip(expected, actual) if before != after]
    report = {'original_artifacts_checked': len(original), 'dataset_files_checked': len(integrity),
              'changed': changed, 'passed': not changed}
    (args.run / 'preservation_after.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if changed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
