"""Load hash-verified, bundled frozen Parquet splits with optional datasets support."""
from pathlib import Path
import hashlib
import json


def load_split(root, config, split):
    """Load a named split; root is the repository's data directory."""
    from datasets import Dataset
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    matches = [s for s in manifest['splits'] if s['config_name'] == config and s['split'] == split]
    if len(matches) != 1:
        raise ValueError(f'unknown or ambiguous split: {config}/{split}')
    record = matches[0]
    path = root.parent / record['data_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['parquet_sha256']:
        raise ValueError(f'dataset hash mismatch: {config}/{split}')
    dataset = Dataset.from_parquet(str(path))
    if len(dataset) != record['rows']:
        raise ValueError(f'dataset row count mismatch: {config}/{split}')
    return dataset
