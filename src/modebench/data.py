"""Load frozen Parquet splits and bind prompt identities to packaged hashes."""
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import re

from .validation import InputError, loads, positive_integer


def split_record(root, config, split, *, frozen=False):
    """Resolve a manifest entry; frozen=True also checks the packaged registry."""
    root = Path(root).resolve()
    manifest = loads((root / 'manifest.json').read_bytes())
    if not isinstance(manifest, dict) or manifest.get('schema') != 'modebench-portable-data-v1':
        raise InputError('unsupported dataset manifest schema')
    entries = manifest.get('splits')
    if not isinstance(entries, list) or not all(isinstance(s, dict) for s in entries):
        raise InputError('dataset manifest requires a list of split objects')
    matches = [s for s in entries if s.get('config_name') == config and s.get('split') == split]
    if len(matches) != 1:
        raise InputError(f'unknown or ambiguous split: {config}/{split}')
    record = matches[0]
    positive_integer(record.get('rows'), 'dataset rows')
    if not isinstance(record.get('parquet_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', record['parquet_sha256']):
        raise InputError('dataset requires a SHA-256 identity')
    if not isinstance(record.get('data_file'), str):
        raise InputError('dataset requires a data_file path')
    relative = Path(record['data_file'])
    path = (root.parent / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise InputError('dataset file must be inside the data directory')
    if frozen:
        registry = loads(files('modebench').joinpath('frozen_splits.json').read_bytes())
        expected = [s for s in registry['splits'] if s['config_name'] == config and s['split'] == split]
        def identity(item):
            return json.dumps({k: v for k, v in item.items() if k != 'data_file'}, sort_keys=True)
        if len(expected) != 1 or identity(record) != identity(expected[0]):
            raise InputError('dataset manifest does not match the packaged frozen registry')
    return record


def load_split(root, config, split, *, frozen=False):
    """Load a hash-verified split; root is the repository's data directory."""
    try:
        from datasets import Dataset
    except ImportError as error:
        raise InputError("dataset loading requires the 'data' extra: pip install 'modebench[data]'") from error
    root = Path(root).resolve()
    record = split_record(root, config, split, frozen=frozen)
    path = root.parent / record['data_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['parquet_sha256']:
        raise InputError(f'dataset hash mismatch: {config}/{split}')
    dataset = Dataset.from_parquet(str(path))
    if len(dataset) != record['rows']:
        raise InputError(f'dataset row count mismatch: {config}/{split}')
    return dataset


def prompt_id(config, split, index):
    """Stable zero-based row identity within an immutable configuration/split."""
    if type(index) is not int or index < 0:
        raise InputError('prompt row index must be a nonnegative integer')
    return f'{config}/{split}/{index}'


def prepare_requests(root, config, split='eval'):
    """Render frozen requests with IDs and hashes, never reference answers."""
    from .prompts import make_messages, prompt_sha256
    record = split_record(root, config, split, frozen=True)
    dataset = load_split(root, config, split, frozen=True)
    for index, row in enumerate(dataset):
        yield {
            'id': prompt_id(config, split, index),
            'level': record['level'], 'domain': record['domain'],
            'messages': make_messages(record['level'], record['domain'], row),
            'prompt_sha256': prompt_sha256(record['level'], record['domain'], row),
            'dataset_sha256': record['parquet_sha256'],
        }
