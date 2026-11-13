"""Check dataset provenance and PNG headers without decoding test pixels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def header(path):
    with Image.open(path) as image:
        # Header/text inspection only: intentionally no load(), convert(),
        # getdata(), tobytes() or array conversion for the untouched test set.
        info = image.info
        candidates = [key for key in info if re.search('patient|slide|specimen|case|description|comment', key, re.I)]
        return image.size, sorted(info), candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'local_provenance.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    images = sorted((ROOT / 'data/raw/HMU-GC-HE-30K/all_image').glob('*/*.png'))
    keys, sizes, candidates = Counter(), Counter(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path, (size, info_keys, possible) in zip(images, pool.map(header, images)):
            keys.update(info_keys); sizes[str(size)] += 1
            if possible:
                candidates.append({'image': str(path.relative_to(ROOT)), 'keys': possible})
    archive = ROOT / 'data/raw/HMU-GC-HE-30K.zip'
    with zipfile.ZipFile(archive) as z:
        members = z.namelist()
    with archive.open('rb') as f:
        checksum = hashlib.file_digest(f, 'md5').hexdigest()
    public = json.loads((args.output / 'figshare_article.json').read_text(encoding='utf-8'))
    zip_file = next(f for f in public['files'] if f['name'] == archive.name)
    with (ROOT / 'data/support/HMU-GC-Clinical.csv').open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f); columns = reader.fieldnames; clinical = list(reader)
    clinical_ids = {row['Patient'].strip() for row in clinical if row.get('Patient', '').strip()}
    # Store counts/schema only; no patient identifiers are exported in this report.
    report = {
        'dataset_images': len(images), 'png_dimensions': dict(sizes), 'png_header_key_counts': dict(keys),
        'png_header_mapping_candidates': candidates,
        'archive_size': archive.stat().st_size, 'archive_md5': checksum,
        'official_archive_md5': zip_file['computed_md5'],
        'archive_matches_official_figshare': checksum == zip_file['computed_md5'],
        'archive_files': len([n for n in members if not n.endswith('/')]),
        'archive_non_image_members': [n for n in members if not n.endswith('/') and not n.lower().endswith('.png')],
        'clinical_rows': len(clinical), 'clinical_distinct_patient_values': len(clinical_ids),
        'clinical_columns': columns,
        'clinical_has_image_or_slide_column': any(re.search('image|slide|patch|file', c or '', re.I) for c in columns),
        'class_number_filenames_only': all(re.fullmatch(r'[A-Z]+_\d+\.png', p.name) for p in images),
        'mapping_verified': False,
        'conclusion': 'No verified patch-to-patient/slide linkage found. Numeric suffixes are not patient IDs. Patient-grouped splitting and confirmatory training remain unavailable.',
        'test_access': 'PNG headers and archive file bytes only; no test pixel decoding, model fitting or inference',
        'sources': ['https://api.figshare.com/v2/articles/25954813',
                    'https://pmc.ncbi.nlm.nih.gov/articles/PMC11754904/',
                    'https://github.com/TomHardy1997/GCTC']}
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
