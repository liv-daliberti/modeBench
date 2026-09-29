"""Discover frozen splits and download hash-verified bytes into a versioned cache."""
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time
from urllib.request import urlopen

from .validation import InputError

DATA_REVISION = '94c45d1f66d3eb181fac2225dc04d62e08568b8e'
DATA_ORIGIN = f'https://raw.githubusercontent.com/liv-daliberti/modeBench/{DATA_REVISION}/'
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30


def registry_bytes():
    return files('modebench').joinpath('frozen_splits.json').read_bytes()


def discover(config=None, split=None):
    records = json.loads(registry_bytes())['splits']
    selected = [r for r in records if (config is None or r['config_name'] == config)
                and (split is None or r['split'] == split)]
    if not selected:
        raise InputError(f'unknown dataset selection: {config or "*"}/{split or "*"}; use datasets list')
    return selected


def cache_data_root(cache_dir=None):
    if cache_dir is None:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'modebench'
    else:
        base = Path(cache_dir)
    identity = hashlib.sha256(registry_bytes()).hexdigest()
    return base.expanduser().resolve() / identity / 'data'


def _verified(path, expected):
    if not path.is_file() or path.stat().st_size > MAX_DOWNLOAD_BYTES:
        return False
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(65536), b''):
            digest.update(block)
    return digest.hexdigest() == expected


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.modebench-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch(config, split='eval', *, cache_dir=None, offline=False):
    """Return a data root containing the selected immutable split.

    Every reuse verifies SHA-256. Corrupt cache entries fail with their path;
    delete that entry explicitly to download again. No network in offline mode.
    """
    record, = discover(config, split)
    root = cache_data_root(cache_dir)
    path = root.parent / record['data_file']
    if path.exists():
        if not _verified(path, record['parquet_sha256']):
            raise InputError(f'cached dataset hash mismatch: {path}; remove this file and fetch again')
    elif offline:
        raise InputError(f'dataset is not cached: {config}/{split}; run datasets fetch without --offline')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
            with urlopen(DATA_ORIGIN + record['data_file'], timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.modebench-', delete=False) as handle:
                    temporary = Path(handle.name)
                    total = 0
                    while True:
                        block = response.read1(65536)
                        if time.monotonic() > deadline:
                            raise InputError('dataset download deadline exceeded')
                        if not block:
                            break
                        total += len(block)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise InputError('dataset download exceeds byte limit')
                        handle.write(block)
            if not _verified(temporary, record['parquet_sha256']):
                raise InputError(f'download hash mismatch: {config}/{split}; cache not modified')
            os.replace(temporary, path)
        except OSError as error:
            raise InputError(f'dataset download failed: {error}; retry fetch or use a verified offline cache') from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    # Registry bytes are shipped with the installed package, never trusted from the server.
    _atomic_write(root / 'manifest.json', registry_bytes())
    return root
