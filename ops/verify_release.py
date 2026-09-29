"""Verify release file identities, or explicitly refresh them after reviewed edits."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'build', 'dist', '.cache', '.venv', 'outputs'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included(path):
    return path.name != 'RELEASE_MANIFEST.json' and not any(
        part in EXCLUDED or part.endswith('.egg-info')
        for part in path.relative_to(ROOT).parts
    )


def refresh():
    path = ROOT / 'PROVENANCE.json'
    provenance = json.loads(path.read_text())
    for record in provenance['files']:
        source = ROOT / record['path']
        if source.is_file():
            record.setdefault('extracted_sha256', record['sha256'])
            record['sha256'] = digest(source)
    path.write_text(json.dumps(provenance, indent=2) + '\n')
    files = [
        {'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'bytes': path.stat().st_size}
        for path in sorted(ROOT.rglob('*')) if path.is_file() and included(path)
    ]
    (ROOT / 'RELEASE_MANIFEST.json').write_text(json.dumps({'schema': 'local-release-files-v1', 'files': files}, indent=2) + '\n')


def verify():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    expected = set()
    for record in manifest['files']:
        path = ROOT / record['path']
        if record['path'] in expected:
            raise ValueError(f"duplicate release entry: {record['path']}")
        expected.add(record['path'])
        if not path.is_file() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f"release file missing or changed: {record['path']}")
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and included(p)}
    if actual != expected:
        raise ValueError(f'release file inventory differs: {sorted(actual ^ expected)}')
    return {'verified_release_files': len(expected)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='rewrite hashes after deliberate, reviewed changes')
    args = parser.parse_args()
    if args.refresh:
        refresh()
    print(json.dumps(verify()))
