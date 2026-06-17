from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description='Verify public expected-result files and checksums.')
    parser.add_argument('--expected_results', default='expected_results')
    args = parser.parse_args()
    root = Path(args.expected_results).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'Expected-results directory not found: {root}')
    failures = []
    warnings = []
    for required in ['paper_results_manifest.csv','sha256_manifest.csv']:
        if not (root / required).exists():
            failures.append(f'Missing required file: {required}')
    manifest = root / 'paper_results_manifest.csv'
    if manifest.exists():
        rows = list(csv.DictReader(manifest.open('r', encoding='utf-8-sig', newline='')))
        if not rows:
            failures.append('paper_results_manifest.csv is empty')
        for row in rows:
            pf = row.get('public_file','')
            status = row.get('status','')
            if status == 'COPIED' and pf and not (root / pf).exists():
                failures.append(f'Manifest points to missing file: {pf}')
            elif status != 'COPIED':
                warnings.append(f'Optional result not copied: {row.get("paper_item","")} -> {pf}')
    sha_manifest = root / 'sha256_manifest.csv'
    if sha_manifest.exists():
        rows = list(csv.DictReader(sha_manifest.open('r', encoding='utf-8-sig', newline='')))
        for row in rows:
            p = root / row['path']
            if not p.exists():
                failures.append(f'SHA manifest points to missing file: {row["path"]}')
                continue
            if sha256_file(p) != row['sha256']:
                failures.append(f'SHA mismatch: {row["path"]}')
    if failures:
        print('PAPER RESULT VERIFICATION: FAIL')
        for item in failures:
            print('- ' + item)
        return 1
    print('PAPER RESULT VERIFICATION: PASS')
    for item in warnings:
        print('WARNING: ' + item)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

