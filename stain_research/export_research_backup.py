"""Prepare a source-and-evidence GitHub backup without datasets or checkpoints.

Produces a local payload for reviewed GitHub API publication. Does not contact
GitHub, use credentials, alter the dashboard, or activate/evaluate a model.
"""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        for prefix in [str(ROOT), 'C:\\Users\\Amlan\\Desktop\\gastric detection']:
            value = value.replace(prefix + '\\', '').replace(prefix + '/', '').replace(prefix, '.')
        return value.replace('C:/Users/gxhyp/OneDrive/Desktop/gastric detection/', '')
    return value


def export_results():
    source = ROOT / 'stain_research/runs'
    target = ROOT / 'stain_research/results'
    target.mkdir(exist_ok=True)
    for study in ['controller_v1', 'strength_controls_v1', 'decomposition_audit', 'preprocessing_v1']:
        for name in ['summary.json', 'direct_control_check.json', 'stain_parameters.json', 'training_parameters.json']:
            path = source / study / name
            if path.is_file():
                out = target / study / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(portable(json.loads(path.read_text(encoding='utf-8'))), indent=2) + '\n', encoding='utf-8')
    for study, names in {'provenance_audit_20261113': ['local_provenance.json'],
                         'split_audit_20261113': ['summary.json', 'baseline_plan.json', 'verification.json']}.items():
        for name in names:
            path = source / study / name
            if path.is_file():
                out = target / study / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(portable(json.loads(path.read_text(encoding='utf-8'))), indent=2) + '\n', encoding='utf-8')
    pilot = source / 'classifier_local_20261113'
    patterns = ['LOCAL_RESULTS.md', 'audit.json', 'gpu_verification.json', 'loader_checks.json',
                'smoke_pilot_reproducibility.json', 'preservation_after.json',
                'prepared/parameters.json', 'pilot/config.json', 'pilot/summary.json',
                'smoke/timing_estimate.json', 'smoke/fallback_checks.json', 'pilot/fallback_checks.json',
                'pilot/*/history.json', 'pilot/*/metrics.json',
                'prepared/train.csv', 'prepared/val.csv', 'pilot/*/validation_predictions.csv']
    for pattern in patterns:
        for path in pilot.glob(pattern):
            out = target / 'classifier_pilot_20261113' / path.relative_to(pilot)
            out.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == '.json':
                text = json.dumps(portable(json.loads(path.read_text(encoding='utf-8'))), indent=2) + '\n'
            elif path.suffix == '.csv':
                with path.open(newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    fields = reader.fieldnames
                    rows = [{k: portable(v) for k, v in row.items()} for row in reader]
                stream = io.StringIO(newline='')
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
                writer.writeheader(); writer.writerows(rows)
                text = stream.getvalue()
            else:
                text = portable(path.read_text(encoding='utf-8'))
            out.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_results()
    paths = [ROOT / n for n in ['.gitignore', 'README.md', 'requirements.txt', 'database_schema.sql']]
    for directory in ['app', 'training']:
        paths += [p for p in (ROOT / directory).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts
                  and p.suffix in {'.py', '.html', '.js', '.css', '.svg'}]
    paths += [p for p in (ROOT / 'stain_research').glob('*') if p.is_file()
              and p.suffix in {'.py', '.md', '.txt'}]
    for directory in ['stain_research/upstream', 'stain_research/results']:
        paths += [p for p in (ROOT / directory).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and not any(x.endswith('.egg-info') for x in p.parts)
                  and (p.suffix in {'.py', '.md', '.txt', '.json', '.csv'} or p.name == 'LICENSE')]
    forbidden = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}')
    elements, hashes = [], {}
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_text(encoding='utf-8')
        if forbidden.search(content):
            # Ignore this script's own detection regex; never print matched text.
            if path.resolve() != Path(__file__).resolve():
                raise ValueError('Possible credential in backup candidate: ' + relative)
        elements.append({'path': relative, 'mode': '100644', 'type': 'blob', 'content': content})
        hashes[relative] = hashlib.sha256(content.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'tree_elements': elements}, ensure_ascii=False), encoding='utf-8')
    (args.output.parent / 'backup_file_manifest.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(elements), 'total_utf8_bytes': sum(len(e['content'].encode()) for e in elements),
                      'paths': [e['path'] for e in elements]}, indent=2))


if __name__ == '__main__':
    main()
