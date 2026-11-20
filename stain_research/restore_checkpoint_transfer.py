"""Verify downloaded transfer chunks and rebuild ZIPs into a staging directory.

Uses only the Python standard library; never extracts into the project or loads weights.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('download_folder', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--available-only', action='store_true',
                        help='Explicitly verify/rebuild only archives with every chunk present')
    args = parser.parse_args()
    source = args.download_folder.resolve()
    manifest = json.loads((source / 'CHUNK_MANIFEST.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('Transfer manifest is incomplete')
    chunks = {item['file_name']: item for item in manifest['chunks']}
    missing = [a['file_name'] for a in manifest['archives']
               if any(not (source / name).is_file() for name in a['chunks'])]
    if missing and not args.available_only:
        raise FileNotFoundError('Incomplete download. Missing archive chunks: ' + ', '.join(missing)
                                + '. Use --available-only to explicitly process just complete archives.')
    output = args.output.resolve() if args.output else source / 'reassembled'
    if not args.verify_only:
        output.mkdir(parents=True, exist_ok=True)
    for archive in manifest['archives']:
        if archive['file_name'] in missing:
            print('SKIPPED INCOMPLETE', archive['file_name'], flush=True)
            continue
        name = archive['file_name']
        if Path(name).name != name:
            raise ValueError('Unsafe archive filename')
        combined = hashlib.sha256()
        length = 0
        destination = output / name
        writer = None
        if not args.verify_only:
            writer = destination.open('xb')
        try:
            for chunk_name in archive['chunks']:
                if Path(chunk_name).name != chunk_name:
                    raise ValueError('Unsafe chunk filename')
                item = chunks[chunk_name]
                block_hash = hashlib.sha256()
                block_length = 0
                with (source / chunk_name).open('rb') as reader:
                    while block := reader.read(4 * 1024 * 1024):
                        combined.update(block)
                        block_hash.update(block)
                        block_length += len(block)
                        length += len(block)
                        if writer:
                            writer.write(block)
                if block_length != item['bytes'] or block_hash.hexdigest() != item['sha256']:
                    raise ValueError('Chunk integrity failed: ' + chunk_name)
            if length != archive['bytes'] or combined.hexdigest() != archive['sha256']:
                raise ValueError('Archive integrity failed: ' + name)
        finally:
            if writer:
                writer.close()
        print('VERIFIED', name, length, flush=True)
    print('SHA-256 hashes match for the processed archives. Missing archives:', len(missing))
    print('Extract verified ZIPs into a separate staging folder,')
    print('then compare destination files before copying any checkpoints into a project.')
    print('A failed partial output is retained for inspection; existing ZIPs are never overwritten.')


if __name__ == '__main__':
    main()
