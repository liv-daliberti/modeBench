"""Frozen evaluation binds references and prompt identities to shipped artifacts."""
from copy import deepcopy
from importlib.resources import files
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from modebench.cli import evaluate
from modebench.data import load_split, prepare_requests, prompt_id, split_record
from modebench.prompts import make_messages, prompt_sha256
from modebench.validation import InputError, validate_reference

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CONFIG = 'level1_countdown'
RUN = {'model': {'id': 'example/model', 'revision': 'fixed-revision'},
       'generation': {'temperature': 0, 'max_tokens': 192},
       'prompt_condition': 'registered_hints_v1'}


def response(index=0):
    return {'id': prompt_id(CONFIG, 'eval', index), 'responses': ['bad', '']}


def score(records, **kwargs):
    return evaluate(records, data_root=DATA, config=CONFIG, run=RUN, **kwargs)


def test_registry_matches_all_published_split_metadata():
    assert json.loads(files('modebench').joinpath('frozen_splits.json').read_text()) == json.loads((DATA / 'manifest.json').read_text())


def test_partial_runs_require_explicit_consent_and_report_missing_ids():
    with pytest.raises(InputError, match='incomplete frozen split: missing 127'):
        score([response()])
    result = score([response()], allow_partial=True)
    identity = result['dataset']
    assert identity['references_authenticated'] is True
    assert identity['complete'] is False
    assert identity['selected_prompts'] == 1 and identity['expected_prompts'] == 128
    assert identity['selected_prompt_ids'] == [prompt_id(CONFIG, 'eval', 0)]
    assert len(identity['missing_prompt_ids']) == 127
    assert result['cells'][0]['accuracy'] == 0
    assert result['evaluation']['groups_per_prompt'] == 1


def test_complete_frozen_split_and_prompt_hashes():
    result = score([response(i) for i in range(128)])
    assert result['dataset']['complete'] is True
    assert result['dataset']['missing_prompt_ids'] == []
    row = load_split(DATA, CONFIG, 'eval')[0]
    p = result['cells'][0]['prompt_results'][0]
    assert p['prompt_sha256'] == prompt_sha256(1, 'countdown', row)
    assert p['prompt_profile']['prompt_condition'] == 'registered_hints_v1'
    assert result['dataset']['split_record']['parquet_sha256'] == split_record(DATA, CONFIG, 'eval')['parquet_sha256']


@pytest.mark.parametrize('change,message', [
    ({'id': 'made-up-id'}, 'unknown frozen prompt'),
    ({'id': 'level1_countdown/train/0'}, 'unknown frozen prompt'),
    ({'level': 2}, 'level/domain'),
    ({'domain': 'graph_coloring'}, 'level/domain'),
    ({'answer': {'verifier': 'countdown'}}, 'omit supplied answer'),
    ({'problem': 'substituted problem'}, 'problem does not match'),
    ({'dataset_sha256': '0' * 64}, 'dataset_sha256 does not match'),
    ({'prompt_sha256': '0' * 64}, 'prompt_sha256 does not match'),
    ({'messages': [{'role': 'user', 'content': 'changed'}]}, 'messages does not match'),
])
def test_forged_bindings_are_rejected(change, message):
    row = {**response(), **change}
    with pytest.raises(InputError, match=message):
        score([row], allow_partial=True)


def test_frozen_requires_declared_registered_prompt_condition():
    with pytest.raises(InputError, match='requires run metadata'):
        evaluate([response()], data_root=DATA, config=CONFIG, allow_partial=True)
    altered = deepcopy(RUN)
    altered['prompt_condition'] = 'python_level3_neutral_v1'
    with pytest.raises(InputError, match='requires registered_hints_v1'):
        evaluate([response()], data_root=DATA, config=CONFIG, run=altered, allow_partial=True)


def test_unknown_config_and_wrong_mode_arguments():
    with pytest.raises(InputError, match='unknown or ambiguous'):
        evaluate([response()], data_root=DATA, config='new_config', run=RUN)
    with pytest.raises(InputError, match='require a frozen config'):
        evaluate([response()], data_root=DATA)


def test_request_export_omits_references_and_is_evaluable():
    requests = list(prepare_requests(DATA, CONFIG))
    assert len(requests) == 128
    for request in requests:
        assert 'answer' not in request and 'answer_mode_count' not in request
        request['responses'] = ['bad']
    result = score(requests)
    assert result['dataset']['complete'] is True


def test_local_manifest_cannot_redefine_a_frozen_identity(tmp_path):
    shutil.copytree(DATA, tmp_path / 'data')
    root = tmp_path / 'data'
    manifest = json.loads((root / 'manifest.json').read_text())
    item = next(s for s in manifest['splits'] if s['config_name'] == CONFIG and s['split'] == 'eval')
    target = tmp_path / item['data_file']
    target.write_bytes(b'replacement dataset')
    item['parquet_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(InputError, match='packaged frozen registry'):
        load_split(root, CONFIG, 'eval', frozen=True)


def test_dataset_path_cannot_escape_data_directory(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    m = json.loads((DATA / 'manifest.json').read_text())
    m['splits'][0]['data_file'] = '../outside.parquet'
    (root / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(InputError, match='inside the data directory'):
        load_split(root, CONFIG, 'eval')


def test_level4_caveats_travel_with_results():
    config = 'level4_mathir'
    result = evaluate([{'id': prompt_id(config, 'eval', 0), 'responses': ['bad']}],
                      data_root=DATA, config=config, run=RUN, allow_partial=True)
    assert result['dataset']['split_record']['difficulty_matched'] is False
    assert result['dataset']['split_record']['level_admitted'] is False
    assert any('not difficulty-matched' in note for note in result['cells'][0]['protocol_notes'])


def test_cli_prepare_and_frozen_evaluate(tmp_path):
    requests = tmp_path / 'requests.jsonl'
    output = tmp_path / 'result.json'
    runfile = tmp_path / 'run.json'
    runfile.write_text(json.dumps(RUN))
    base = [sys.executable, '-m', 'modebench.cli']
    subprocess.run(base + ['prepare', '--config', CONFIG, '--data-root', str(DATA), '--output', str(requests)],
                   cwd=ROOT, check=True, capture_output=True)
    rows = [json.loads(line) for line in requests.read_text().splitlines()]
    for row in rows:
        row['responses'] = ['bad']
    requests.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    command = base + ['evaluate', str(requests), '--config', CONFIG, '--data-root', str(DATA),
                      '--run', str(runfile), '--output', str(output)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    result = json.loads(output.read_text())
    assert result['dataset']['complete'] is True
    assert result['input_sha256'] == hashlib.sha256(requests.read_bytes()).hexdigest()
    original = output.read_bytes()
    repeat = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert repeat.returncode == 2 and 'already exists' in repeat.stderr
    assert output.read_bytes() == original


def test_frozen_cannot_hide_missing_draws_in_separate_k_cells():
    a, b = response(0), response(1)
    b['responses'] = ['bad']
    with pytest.raises(InputError, match='same number of draws'):
        score([a, b], allow_partial=True)


def test_manifest_cannot_change_admission_flags_or_their_types(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    manifest = json.loads((DATA / 'manifest.json').read_text())
    item = next(s for s in manifest['splits'] if s['config_name'] == 'level4_mathir' and s['split'] == 'eval')
    for replacement in (True, 0):
        item['difficulty_matched'] = replacement
        (root / 'manifest.json').write_text(json.dumps(manifest))
        with pytest.raises(InputError, match='packaged frozen registry'):
            split_record(root, 'level4_mathir', 'eval', frozen=True)


@pytest.mark.parametrize('config', ['level1_python_factors', 'level1_mathir', 'level1_pantry_plan'])
def test_other_domain_reference_errors_are_input_errors(config):
    metadata = split_record(DATA, config, 'eval')
    row = dict(load_split(DATA, config, 'eval')[0])
    spec = json.loads(row['answer'])
    field = {'python_factors': 'python_version', 'mathir': 'mathir_version', 'pantry_plan': 'pantry_version'}[metadata['domain']]
    spec[field] = 'unsupported-version'
    with pytest.raises(InputError, match='unsupported'):
        validate_reference(metadata['level'], metadata['domain'], spec)
