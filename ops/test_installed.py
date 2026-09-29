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
    assert Path(modebench.__file__).with_name("py.typed").is_file()
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
    card = json.loads(cli('datasets', 'card', '--config', 'level1_countdown'))
    assert card['license'] == 'CC-BY-4.0' and len(card['authors']) == 5
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

    from modebench.api import (
        Task, EvaluationOptions, evaluate_file, grade, make_prompt, read_report, iter_results,
    )
    from importlib.resources import files
    packaged_demo = json.loads(files('modebench').joinpath('walkthrough.json').read_text())
    for record in packaged_demo['records']:
        task = Task(record['id'], record['level'], record['domain'], record['problem'], record['answer'])
        assert make_prompt(task) and grade(task, record['responses'][0])['status'] == 'correct'
    report = read_report('demo/report.json')
    assert report.schema == 'modebench-saved-responses-v4' and report.status == 'completed'
    expected_results = list(iter_results('demo/report.json'))
    assert len(expected_results) == 5

    def interrupt_after_two(progress):
        if progress['phase'] == 'grading' and progress['completed_prompts'] == 2:
            raise KeyboardInterrupt

    options = EvaluationOptions(run=packaged_demo['run'])
    try:
        evaluate_file('demo/responses.jsonl', 'resumed.json', options=options,
                      progress=interrupt_after_two)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('expected interruption')
    assert not Path('resumed.json').exists()
    receipt = evaluate_file('demo/responses.jsonl', 'resumed.json', options=options,
                            resume=True, workers=2)
    assert receipt.prompts == 5 and receipt.status == 'completed'
    assert read_report(receipt.output).cells == report.cells
    assert list(iter_results(receipt.output)) == expected_results
    assert 'Evaluation completed' in cli('report', receipt.output)

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
        from modebench.api import load_tasks
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
            task = next(load_tasks(config, cache_dir='cache', offline=True))
            assert task.id == f'{config}/eval/0' and make_prompt(task)
            assert grade(task, demo[record['domain']]['responses'][0])['status'] == 'correct'
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
