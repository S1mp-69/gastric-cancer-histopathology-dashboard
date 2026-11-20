"""Fetch the pinned official wheel directly when the index CDN stalls."""
import hashlib
import time
import urllib.request
from pathlib import Path

url = 'https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp312-cp312-win_amd64.whl'
expected = '1704e5dd66c9221e4e8b6ae2d80cbf54e129571e643f5fa9ca78cc6d2096403a'
out = Path(__file__).parent / 'wheelhouse' / 'torch-2.7.0+cu128-cp312-cp312-win_amd64.whl'
out.parent.mkdir(exist_ok=True)
if out.exists():
    raise FileExistsError(out)
started = time.perf_counter()
total = 0
checksum = hashlib.sha256()
with urllib.request.urlopen(url, timeout=45) as response, out.open('xb') as stream:
    while chunk := response.read(8 * 1024 * 1024):
        stream.write(chunk)
        checksum.update(chunk)
        total += len(chunk)
        if total % (128 * 1024 * 1024) == 0:
            print(f'{total / 1024**3:.2f} GiB downloaded in {time.perf_counter()-started:.0f}s', flush=True)
if checksum.hexdigest() != expected:
    raise RuntimeError(f'Official wheel checksum mismatch: {checksum.hexdigest()}')
print(f'Official SHA256 verified: {out}, {total} bytes', flush=True)
