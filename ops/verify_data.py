"""Verify every bundled dataset's frozen byte identity without loading model code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def verify():
    splits = json.loads((ROOT/'data/manifest.json').read_text())['splits']
    identities = set()
    for split in splits:
        key = (split['config_name'], split['split'])
        if key in identities:
            raise AssertionError(f'duplicate split: {key}')
        identities.add(key)
        path = ROOT / split['data_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != split['parquet_sha256']:
            raise AssertionError(f'dataset hash mismatch: {key}')
    return {'verified_splits':len(splits), 'rows':sum(s['rows'] for s in splits)}

if __name__ == '__main__':
    print(json.dumps(verify()))
