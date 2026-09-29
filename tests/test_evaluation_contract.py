"""Public input failures must not become plausible benchmark scores."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from modebench.cli import evaluate, read_records
from modebench.historical_prompts import grade_response
from modebench.metrics import cell_summary, effective_modes, mode_diversity, rarefied_distinct, verified_mode_counts
from modebench.validation import InputError

ROOT = Path(__file__).resolve().parents[1]
RUN = {'model': {'id': 'example/model', 'revision': 'immutable-revision'},
       'generation': {'temperature': 0.7, 'top_p': 0.95, 'max_tokens': 192, 'seed': 43},
       'prompt_condition': 'registered_hints_v1'}


def record():
    return {'id': 'example', 'level': 1, 'domain': 'countdown',
            'answer': {'verifier': 'countdown', 'numbers': [1, 2, 3], 'target': 6},
            'responses': ['1+2+3', '1*2*3', 'bad', '']}


def test_domain_reference_mismatch_is_not_scored_as_a_different_domain():
    row = record()
    row['domain'] = 'graph_coloring'
    with pytest.raises(InputError, match='does not match domain'):
        evaluate([row])
    with pytest.raises(InputError, match='does not match domain'):
        grade_response(1, 'graph_coloring', row, '1+2+3')


@pytest.mark.parametrize('field,value,message', [
    ('answer', {}, 'verifier'),
    ('answer', '{bad}', 'invalid JSON'),
    ('answer', [], 'answer must be an object'),
    ('responses', [], 'nonempty list'),
    ('responses', [True], 'strings'),
    ('id', True, 'id must'),
    ('id', [], 'id must'),
    ('id', ' ', 'id must'),
    ('level', True, 'level must'),
    ('domain', 'unknown', 'unsupported'),
])
def test_invalid_fields_have_prompt_context(field, value, message):
    row = record()
    row[field] = value
    with pytest.raises(ValueError, match=message) as error:
        evaluate([row], locations=['line 18'])
    assert 'line 18, prompt' in str(error.value)


@pytest.mark.parametrize('field', ['answer', 'responses', 'id', 'level', 'domain'])
def test_missing_fields_are_actionable(field):
    row = record()
    del row[field]
    with pytest.raises(InputError, match=f'missing field: {field}'):
        evaluate([row])


@pytest.mark.parametrize('bad', [None, True, [], 'not an object'])
def test_nonobject_records_are_rejected(bad):
    with pytest.raises(InputError, match='record 1.*object'):
        evaluate([bad])


@pytest.mark.parametrize('answer', [
    {'verifier': 'countdown', 'numbers': [1, 2, 3], 'target': 6.9},
    {'verifier': 'countdown', 'numbers': [True, 2, 3], 'target': 6},
    {'verifier': 'countdown', 'numbers': [], 'target': 6},
    {'verifier': 'countdown', 'numbers': [1, 2, 3]},
])
def test_bad_countdown_spec_is_not_silently_coerced(answer):
    row = record()
    row['answer'] = answer
    with pytest.raises(InputError):
        evaluate([row])


@pytest.mark.parametrize('spec', [
    {'n': 3, 'edges': [[0, 1]]},
    {'n': 3, 'edges': [[1, 4]]},
    {'n': 3, 'edges': [[1, 1]]},
    {'n': 3, 'edges': [[1]]},
    {'n': 3, 'edges': [[True, 2]]},
    {'n': 3, 'edges': [], 'partial_colors': [1, 2]},
    {'n': 3, 'edges': [], 'partial_colors': [1, True, None]},
])
def test_invalid_graph_reference_is_an_input_error(spec):
    row = record()
    row.update(domain='graph_coloring', answer={'verifier': 'graph_coloring', **spec})
    with pytest.raises(InputError):
        evaluate([row])


@pytest.mark.parametrize('counts', [[0.5] * 4, [1.0, 1], [True, 1], [float('nan')],
                                   [float('inf')], [-1, 2], ['2']])
def test_invalid_counts_never_produce_a_metric(counts):
    for metric in (mode_diversity, rarefied_distinct):
        with pytest.raises(ValueError, match='nonnegative integers'):
            metric(counts)


@pytest.mark.parametrize('bad', [0, -1, 0.5, True, float('nan'), float('inf')])
def test_thresholds_and_depth_must_be_positive_integers(bad):
    with pytest.raises(ValueError):
        evaluate([record()], min_defined_prompts=bad)
    with pytest.raises(ValueError):
        cell_summary([], min_defined_prompts=bad)
    with pytest.raises(ValueError):
        rarefied_distinct([1, 1], bad)


@pytest.mark.parametrize('bad', [-0.1, 1.1, True, float('nan'), float('inf')])
def test_invalid_effective_mode_inputs(bad):
    with pytest.raises(ValueError):
        effective_modes(bad)


@pytest.mark.parametrize('attempt', [{'verified': 1, 'canonical_key': 'a'},
                                    {'verified': True, 'canonical_key': None},
                                    {'verified': True, 'canonical_key': ''}])
def test_verified_attempts_require_a_boolean_and_mode_identity(attempt):
    with pytest.raises(ValueError):
        verified_mode_counts([attempt])


@pytest.mark.parametrize('raw', [b'{}\n\n{bad}', b'{}\n\n{"id":"x","id":"y"}',
                              b'{}\n\n{"x":NaN}', b'{}\n\n{"x":1e999}'])
def test_json_errors_report_physical_line_numbers(raw):
    with pytest.raises(InputError, match='line 3:'):
        read_records(raw)


def test_all_records_validated_before_any_grading(monkeypatch):
    def should_not_run(*args):
        pytest.fail('invalid batch reached a verifier')
    monkeypatch.setattr('modebench.evaluation.grade_response', should_not_run)
    bad = record()
    bad['id'] = 'bad'
    bad.pop('answer')
    with pytest.raises(InputError, match='missing field: answer'):
        evaluate([record(), bad])


def test_valid_custom_scores_remain_unchanged_and_identity_is_explicit():
    result = evaluate([record()], min_defined_prompts=1)
    cell = result['cells'][0]
    assert (cell['accuracy'], cell['pass_at_k'], cell['distinct_at_k'], cell['pcmd']['d_mode']) == (0.5, 1, 2, 1)
    assert result['schema'] == 'modebench-saved-responses-v2'
    assert result['dataset'] == {'kind': 'custom', 'references_authenticated': False}
    assert result['run'] is None and result['generation_metadata_status'] == 'unreported'
    assert result['software']['modebench_version'] == '0.2.0'
    assert len(result['software']['package_source_sha256']) == 64
    prompt = cell['prompt_results'][0]
    assert len(prompt['reference_sha256']) == 64
    assert prompt['prompt_sha256'] is None
    assert len(prompt['draws'][0]['attempts']) == 4
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('field', ['model', 'generation', 'prompt_condition'])
def test_mixed_conditions_are_rejected(field):
    a, b = record(), record()
    b['id'] = 'other'
    a['run'], b['run'] = deepcopy(RUN), deepcopy(RUN)
    if field == 'model':
        b['run']['model']['revision'] = 'different'
    elif field == 'generation':
        b['run']['generation']['temperature'] = 0.0
    else:
        b['run']['prompt_condition'] = 'python_level3_neutral_v1'
    with pytest.raises(InputError, match='mixed model or generation'):
        evaluate([a, b])


def test_run_metadata_cannot_silently_mix_with_undeclared_records():
    a, b = record(), record()
    b['id'] = 'other'
    a['run'] = deepcopy(RUN)
    with pytest.raises(InputError, match='mixed declared/undeclared'):
        evaluate([a, b])
    result = evaluate([a, b], run=RUN)
    assert result['run'] == RUN
    assert result['generation_metadata_status'] == 'user_declared'


def test_unrecognized_model_fields_are_not_silently_ignored():
    row = record()
    row['model_id'] = 'would previously be ignored'
    with pytest.raises(InputError, match='put generation metadata in run'):
        evaluate([row])


def test_cli_reports_line_and_prompt_without_traceback_or_output(tmp_path):
    path = tmp_path / 'bad.jsonl'
    row = record()
    del row['answer']
    path.write_text('\n\n' + json.dumps(row) + '\n')
    output = tmp_path / 'result.json'
    run = subprocess.run([sys.executable, '-m', 'modebench.cli', 'evaluate', str(path), '--output', str(output)],
                         cwd=ROOT, text=True, capture_output=True)
    assert run.returncode == 2
    assert "line 3, prompt 'example': missing field: answer" in run.stderr
    assert 'Traceback' not in run.stderr
    assert not output.exists()


@pytest.mark.parametrize('run', [
    {}, {'model': {}, 'generation': {}, 'prompt_condition': 'registered_hints_v1'},
    {**RUN, 'model': {'id': 'x', 'revision': None}},
    {**RUN, 'generation': {}},
    {**RUN, 'generation': {'temperature': float('nan')}},
    {**RUN, 'generation': {'draws_per_prompt': True}},
    {**RUN, 'prompt_condition': 'unknown'},
])
def test_invalid_run_metadata_is_rejected(run):
    with pytest.raises(InputError):
        evaluate([record()], run=run)


def test_declared_draw_budget_must_match_saved_responses():
    run = deepcopy(RUN)
    run['generation']['draws_per_prompt'] = 8
    with pytest.raises(InputError, match='responses count does not match'):
        evaluate([record()], run=run)


def test_report_fingerprint_binds_source_files(monkeypatch):
    from modebench.identity import software_identity
    original = Path.read_bytes
    def altered(path):
        body = original(path)
        return body + b'\n' if path.name == 'grading.py' else body
    before = software_identity()
    monkeypatch.setattr(Path, 'read_bytes', altered)
    after = software_identity()
    assert before['package_source_sha256'] != after['package_source_sha256']
