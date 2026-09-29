"""Copy this script outside the checkout and run with a fresh installed environment.

No pytest, source imports, PYTHONPATH, or editable installation is needed.
The only checkout inputs are immutable fixture JSON and optional Parquet files.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--data-source', type=Path)
    args = parser.parse_args()
    import modebench
    site = Path(sysconfig.get_path('purelib')).resolve()
    assert Path(modebench.__file__).resolve().is_relative_to(site), modebench.__file__
    direct = importlib.metadata.distribution('modebench').read_text('direct_url.json')
    assert not direct or not json.loads(direct).get('dir_info', {}).get('editable')
    from modebench.identity import software_identity
    from modebench.domains.mathir import verifier as mathir
    assert importlib.import_module('modebench.mathir') is mathir
    assert 'domains/mathir/verifier.py' in software_identity()['source_files']
    assert 'PYTHONPATH' not in os.environ
    assert not (Path.cwd() / 'pyproject.toml').exists(), 'run outside checkout'
    assert importlib.util.find_spec('torch') is None
    have_data = importlib.util.find_spec('datasets') is not None
    assert have_data == (args.data_source is not None)
    executable = str(Path(sys.executable).parent / 'modebench')

    def cli(*arguments, status=0):
        result = subprocess.run([executable, *map(str, arguments)], text=True, capture_output=True, timeout=90)
        assert result.returncode == status, (arguments, result.returncode, result.stdout, result.stderr)
        return result.stdout

    cli('--help')
    registry = json.loads(cli('datasets', 'list', '--json'))
    assert len(registry) == 72
    cli('walkthrough', '--directory', 'demo')
    cli('walkthrough', '--directory', 'demo', status=2)
    tasks = [json.loads(s) for s in Path('demo/tasks.jsonl').read_text().splitlines()]
    assert len(tasks) == 5 and all('answer' not in r for r in tasks)
    cli('evaluate', 'demo/responses.jsonl', '--run', 'demo/run.json', '--output', 'demo/report.json')
    payload = json.loads(Path('demo/report.json').read_text())
    assert payload['evaluation']['status_counts'] == {'correct':10, 'incorrect':5, 'malformed':5}
    assert len(payload['cells']) == 5
    assert payload['software']['git']['commit'] is None
    for cell in payload['cells']:
        assert (cell['accuracy'], cell['pass_at_k'], cell['distinct_at_k'], cell['pcmd']['d_mode'], cell['pcmd']['reportable']) == (.5, 1., 2., 1., False)
    assert 'Evaluation completed' in cli('report', 'demo/report.json')
    cli('evaluate', 'demo/responses.jsonl', '--output', 'demo/report.json', status=2)

    from modebench.historical_prompts import grade_response
    corpus = json.loads(args.fixtures.read_text())
    matched = 0
    for fixture in corpus['fixtures']:
        for case in fixture['cases']:
            actual = grade_response(fixture['level'], fixture['domain'], fixture['row'], case['response'])
            assert {k:actual[k] for k in case['expected']} == case['expected'], (fixture['config'], case['name'], actual)
            assert actual['status'] in {'correct', 'incorrect', 'malformed'}, actual
            matched += 1
    assert matched == 175
    from modebench.verifier import close_worker
    close_worker()

    if have_data:
        from modebench.dataset_cache import cache_data_root
        root = cache_data_root('cache')
        demo = {r['domain']:r for r in map(json.loads, Path('demo/responses.jsonl').read_text().splitlines())}
        for record in registry:
            if record['level'] != 1 or record['split'] != 'eval' or record['config_name'].endswith('_unique_answer'):
                continue
            target = root.parent / record['data_file']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.data_source / Path(record['data_file']).relative_to('data'), target)
            config = record['config_name']
            common = ['--config', config, '--cache-dir', 'cache', '--offline']
            cli('datasets', 'fetch', *common)
            inspected = json.loads(cli('datasets', 'inspect', *common))
            assert inspected['id'] == f'{config}/eval/0' and 'answer' not in inspected
            requests = Path(f'{config}-requests.jsonl')
            cli('prepare', *common, '--output', requests)
            rows = [json.loads(s) for s in requests.read_text().splitlines()]
            assert len(rows) == 128 and all('answer' not in r for r in rows)
            sample = rows[0]
            sample['responses'] = demo[record['domain']]['responses']
            responses = Path(f'{config}-responses.jsonl')
            responses.write_text(json.dumps(sample) + '\n')
            report = Path(f'{config}-report.json')
            cli('evaluate', responses, *common, '--allow-partial', '--run', 'demo/run.json', '--output', report)
            result = json.loads(report.read_text())
            assert result['dataset']['references_authenticated'] and not result['dataset']['complete']
            assert result['cells'][0]['accuracy'] == .5
            cli('report', report)
    else:
        error = subprocess.run([executable, 'datasets', 'inspect', '--config', 'level1_countdown', '--data-root', 'missing'], capture_output=True, text=True)
        assert error.returncode == 2 and "'data' extra" in error.stderr
    print(json.dumps({'installed_package': str(modebench.__file__), 'data_extra': have_data,
                      'matched_conformance_cases': matched, 'walkthrough_domains': 5}))


if __name__ == '__main__':
    main()
