"""Portable code and runtime identities for evaluation receipts."""
from importlib import metadata
from pathlib import Path
import hashlib
import json
import platform
import subprocess


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def software_identity():
    from . import __version__
    from .diagnostics import (DOMAIN_DEADLINE_SECONDS, PYTHON_DEADLINE_SECONDS,
                              WORKER_MEMORY_BYTES, MAX_REQUEST_BYTES, MAX_REFERENCE_BYTES,
                              MAX_REPLY_BYTES, MAX_RESPONSE_CHARS)
    package = Path(__file__).resolve().parent
    hashes = {
        p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob('*')) if p.suffix in {'.py', '.json'}
    }
    root = package.parent.parent
    git = {'commit': None, 'dirty': None}
    if (root / '.git').exists():
        try:
            commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True, check=True, timeout=3)
            status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                                    capture_output=True, text=True, check=True, timeout=3)
            git = {'commit': commit.stdout.strip(), 'dirty': bool(status.stdout.strip())}
        except (OSError, subprocess.SubprocessError):
            pass
    dependencies = {}
    for name in ('sympy', 'datasets', 'pyarrow'):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        'modebench_version': __version__,
        'verifier_contract': 'modebench-verifiers-v1',
        'diagnostic_contract': 'modebench-status-v1',
        'resource_policy': {
            'python_execution_seconds': PYTHON_DEADLINE_SECONDS,
            'other_execution_seconds': DOMAIN_DEADLINE_SECONDS,
            'worker_address_space_bytes': WORKER_MEMORY_BYTES,
            'request_bytes': MAX_REQUEST_BYTES, 'reply_bytes': MAX_REPLY_BYTES,
            'reference_bytes': MAX_REFERENCE_BYTES, 'response_chars': MAX_RESPONSE_CHARS,
            'parent_request_seconds': 5.0,
        },
        'package_source_sha256': digest_json(hashes),
        'source_files': hashes,
        'git': git,
        'python': platform.python_version(),
        'platform': platform.system(),
        'dependencies': dependencies,
    }
