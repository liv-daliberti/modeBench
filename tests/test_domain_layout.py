"""Refactoring must preserve public imports and fingerprint every domain."""
import hashlib
from importlib import import_module
from pathlib import Path

import pytest
from modebench.identity import software_identity


@pytest.mark.parametrize('old,new', [
    ('mathir','mathir.verifier'),
    ('pantry_plan','pantry_plan.verifier'),
    ('pantry_support_action','pantry_plan.support'),
    ('python_modebench','python_factors.verifier'),
    ('python_modebench_process','python_factors.process'),
])
def test_legacy_imports_resolve_to_the_same_module(old, new):
    assert import_module('modebench.' + old) is import_module('modebench.domains.' + new)


def test_software_fingerprint_covers_nested_domain_implementations():
    import modebench
    root = Path(modebench.__file__).parent
    hashes = software_identity()['source_files']
    for domain in ('countdown','graph_coloring','python_factors','mathir','pantry_plan'):
        for module in ('verifier','grading'):
            path = f'domains/{domain}/{module}.py'
            assert hashes[path] == hashlib.sha256((root/path).read_bytes()).hexdigest()
