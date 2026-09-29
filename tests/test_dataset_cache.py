import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
from modebench import dataset_cache as cache
from modebench.validation import InputError
from modebench.walkthrough import export, summarize

CONFIG = 'level1_countdown'
ROOT = Path(__file__).resolve().parents[1]


def payload():
    return (ROOT / cache.discover(CONFIG, 'eval')[0]['data_file']).read_bytes()


def test_discovery_finds_all_cells_and_splits():
    records = cache.discover()
    assert len(records) == 72
    assert len({(r['level'], r['domain']) for r in records}) == 25
    with pytest.raises(InputError, match='unknown dataset'):
        cache.discover('../../escape', 'eval')


def test_fetch_verifies_and_reuses_without_network(tmp_path, monkeypatch):
    calls = []
    def download(url, timeout):
        calls.append(url)
        assert cache.DATA_REVISION in url and timeout == 30
        return io.BytesIO(payload())
    monkeypatch.setattr(cache, 'urlopen', download)
    root = cache.fetch(CONFIG, cache_dir=tmp_path)
    assert (root / 'manifest.json').read_bytes() == cache.registry_bytes()
    assert cache.fetch(CONFIG, cache_dir=tmp_path, offline=True) == root
    assert len(calls) == 1
    path = root.parent / cache.discover(CONFIG, 'eval')[0]['data_file']
    path.write_bytes(b'corrupt')
    with pytest.raises(InputError, match='cached dataset hash mismatch'):
        cache.fetch(CONFIG, cache_dir=tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize('failure', ['hash', 'size', 'network', 'deadline'])
def test_failed_download_is_never_published(tmp_path, monkeypatch, failure):
    if failure == 'size': monkeypatch.setattr(cache, 'MAX_DOWNLOAD_BYTES', 10)
    if failure == 'deadline': monkeypatch.setattr(cache, 'DOWNLOAD_TIMEOUT_SECONDS', -1)
    def download(*args, **kwargs):
        if failure == 'network': raise OSError('disconnected')
        return io.BytesIO(b'bad' if failure == 'hash' else payload())
    monkeypatch.setattr(cache, 'urlopen', download)
    with pytest.raises(InputError):
        cache.fetch(CONFIG, cache_dir=tmp_path)
    assert not list(tmp_path.rglob('*.parquet'))
    assert not list(tmp_path.rglob('.modebench-*'))
    assert not list(tmp_path.rglob('manifest.json'))


def test_offline_missing_never_uses_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('network used')
    monkeypatch.setattr(cache, 'urlopen', forbidden)
    with pytest.raises(InputError, match='not cached'):
        cache.fetch(CONFIG, cache_dir=tmp_path, offline=True)


def test_concurrent_fetches_publish_only_verified_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'urlopen', lambda *a, **kw: io.BytesIO(payload()))
    with ThreadPoolExecutor(max_workers=3) as executor:
        roots = list(executor.map(lambda _: cache.fetch(CONFIG, cache_dir=tmp_path), range(3)))
    assert len(set(roots)) == 1
    assert cache.fetch(CONFIG, cache_dir=tmp_path, offline=True) == roots[0]
    assert not list(tmp_path.rglob('.modebench-*'))


def test_walkthrough_is_complete_and_tasks_do_not_leak_answers(tmp_path):
    directory = tmp_path / 'demo'
    export(directory)
    tasks = [json.loads(line) for line in (directory/'tasks.jsonl').read_text().splitlines()]
    assert len({r['domain'] for r in tasks}) == 5
    assert all('answer' not in r and 'responses' not in r for r in tasks)
    with pytest.raises(InputError, match='already exists'):
        export(directory)


def test_failed_report_never_prints_numbers_as_scores():
    report = summarize({'schema':'modebench-saved-responses-v3', 'evaluation':{'status':'failed', 'status_counts':{'timeout':1}}, 'cells':[]})
    assert 'FAILED' in report and 'scores are unavailable' in report and 'timeout' in report
