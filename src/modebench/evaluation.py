"""Validated saved-response evaluation and self-describing result receipts."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import statistics

from .data import load_split, prompt_id, split_record
from .historical_prompts import _identity, grade_response
from .identity import digest_json, software_identity
from .metrics import POOLED, cell_summary
from .prompts import HISTORICAL, make_messages, profile_metadata, prompt_sha256
from .validation import InputError, json_value, positive_integer, validate_reference, validate_run

RECORD_FIELDS = {
    'id', 'level', 'domain', 'answer', 'responses', 'problem', 'run',
    'messages', 'prompt_sha256', 'dataset_sha256',
}


def _notes(level, domain):
    notes = []
    if level == 4:
        notes.append('Level 4 as a whole is not admitted.')
        if domain == 'mathir':
            notes.append('Level 4 MathIR is not difficulty-matched.')
    return notes


def evaluate(records, *, min_defined_prompts=30, run=None, data_root=None,
             config=None, split='eval', allow_partial=False, locations=None):
    """Evaluate custom references or bind responses to one packaged frozen split.

    Frozen evaluation requires run metadata and complete split coverage unless
    allow_partial=True. All records are validated before grading any response.
    Metadata describe declared generation settings; they do not attest model use.
    """
    min_defined_prompts = positive_integer(min_defined_prompts, 'min_defined_prompts')
    if type(allow_partial) is not bool:
        raise InputError('allow_partial must be a boolean')
    if config is None and (data_root is not None or allow_partial or split != 'eval'):
        raise InputError('data_root, split, and allow_partial require a frozen config')
    records = list(records)
    if not records:
        raise InputError('no prompt records')
    if locations is None:
        locations = [f'record {i}' for i in range(1, len(records) + 1)]
    if len(locations) != len(records):
        raise InputError('locations must match the number of records')
    declared_run = validate_run(run) if run is not None else None
    expected_ids, frozen_rows, metadata = [], {}, None
    if config is not None:
        root = Path(data_root or 'data')
        metadata = split_record(root, config, split, frozen=True)
        dataset = load_split(root, config, split, frozen=True)
        expected_ids = [prompt_id(config, split, i) for i in range(len(dataset))]
        frozen_rows = dict(zip(expected_ids, dataset))

    prepared, seen, inline_runs = [], set(), []
    for location, raw in zip(locations, records):
        label = f'{location}, prompt {str(raw.get("id", "?"))[:100]!r}' if isinstance(raw, dict) else location
        try:
            if not isinstance(raw, dict):
                raise InputError('each record must be an object')
            json_value(raw, 'record')
            unknown = raw.keys() - RECORD_FIELDS
            if unknown:
                raise InputError(f'unknown fields: {", ".join(sorted(unknown))}; put generation metadata in run')
            for field in ('id', 'responses'):
                if field not in raw:
                    raise InputError(f'missing field: {field}')
            rid = raw['id']
            if isinstance(rid, bool) or not isinstance(rid, (str, int)) or not str(rid).strip():
                raise InputError('id must be a nonempty string or integer')
            rid = str(rid)
            responses = raw['responses']
            if not isinstance(responses, list) or not responses or not all(isinstance(s, str) for s in responses):
                raise InputError('responses must be a nonempty list of strings (retain failed draws)')
            if metadata is not None:
                if rid not in frozen_rows:
                    raise InputError(f'unknown frozen prompt ID: {rid}')
                if 'answer' in raw:
                    raise InputError('frozen evaluation resolves answer from the dataset; omit supplied answer')
                level, domain = metadata['level'], metadata['domain']
                supplied = _identity(raw.get('level', level), raw.get('domain', domain))
                if supplied != (level, domain):
                    raise InputError('level/domain does not match the frozen configuration')
                row = dict(frozen_rows[rid])
                if 'problem' in raw and raw['problem'] != row['problem']:
                    raise InputError('problem does not match the frozen row')
                if 'dataset_sha256' in raw and raw['dataset_sha256'] != metadata['parquet_sha256']:
                    raise InputError('dataset_sha256 does not match the frozen split')
            else:
                for field in ('level', 'domain', 'answer'):
                    if field not in raw:
                        raise InputError(f'missing field: {field}')
                level, domain = _identity(raw['level'], raw['domain'])
                row = dict(raw)
                if 'dataset_sha256' in raw:
                    raise InputError('dataset_sha256 requires frozen evaluation with config')
            row['answer'] = validate_reference(level, domain, row['answer'])
            if 'problem' in row and (not isinstance(row['problem'], str) or not row['problem'].strip()):
                raise InputError('problem must be a nonempty string')
            key = level, domain, rid
            if key in seen:
                raise InputError(f'duplicate prompt: {key}')
            seen.add(key)
            inline = validate_run(raw['run']) if 'run' in raw else None
            inline_runs.append(inline)
            if inline is not None:
                if declared_run is None:
                    declared_run = inline
                elif digest_json(inline) != digest_json(declared_run):
                    raise InputError('mixed model or generation conditions: run metadata differ')
            prepared.append((label, raw, rid, level, domain, row))
        except (ValueError, TypeError, KeyError) as error:
            raise InputError(f'{label}: {error}') from error

    if run is None and any(r is not None for r in inline_runs) and any(r is None for r in inline_runs):
        raise InputError('mixed declared/undeclared conditions: supply run metadata for the entire evaluation')
    if metadata is not None:
        if declared_run is None:
            raise InputError('frozen evaluation requires run metadata (model, generation, prompt_condition)')
        if declared_run['prompt_condition'] != HISTORICAL:
            raise InputError('frozen evaluation requires registered_hints_v1; use custom mode for changed conditions')
        missing = [rid for rid in expected_ids if (metadata['level'], metadata['domain'], rid) not in seen]
        if missing and not allow_partial:
            raise InputError(f'incomplete frozen split: missing {len(missing)} of {len(expected_ids)} prompts; explicitly allow_partial to evaluate a subset')
    else:
        missing = None

    draw_counts = {len(raw['responses']) for _, raw, *_ in prepared}
    if metadata is not None and len(draw_counts) != 1:
        raise InputError('frozen evaluation requires the same number of draws per prompt; retain failed draws')
    if declared_run is not None and 'draws_per_prompt' in declared_run['generation']:
        expected_draws = declared_run['generation']['draws_per_prompt']
        for label, raw, *_ in prepared:
            if len(raw['responses']) != expected_draws:
                raise InputError(f'{label}: responses count does not match run.generation.draws_per_prompt')

    # Validate declared prompt identities before making any verifier calls.
    bindings = []
    for label, raw, rid, level, domain, row in prepared:
        try:
            condition = declared_run['prompt_condition'] if declared_run else None
            profile, prompt_hash = None, None
            if condition is not None:
                profile = profile_metadata(level, domain, condition)
                if 'problem' in row:
                    prompt_hash = prompt_sha256(level, domain, row, condition)
            for field in ('prompt_sha256', 'messages'):
                if field in raw:
                    if prompt_hash is None:
                        raise InputError(f'{field} requires problem text and declared run.prompt_condition')
                    expected = prompt_hash if field == 'prompt_sha256' else make_messages(level, domain, row, condition)
                    if raw[field] != expected:
                        raise InputError(f'{field} does not match the registered prompt')
            bindings.append({'reference_sha256': digest_json(row['answer']),
                             'prompt_sha256': prompt_hash, 'prompt_profile': profile})
        except (ValueError, TypeError, KeyError) as error:
            raise InputError(f'{label}: {error}') from error

    cells = defaultdict(list)
    for (_, raw, rid, level, domain, row), binding in zip(prepared, bindings):
        attempts = [grade_response(level, domain, row, text) for text in raw['responses']]
        keys = [a['canonical_key'] for a in attempts if a['verified']]
        cells[(level, domain, len(attempts))].append({
            'id': rid, **binding, 'draws': [{'attempts': attempts}],
            'pass_at_k': float(bool(keys)), 'distinct_at_k': len(set(keys)),
            'accuracy': len(keys) / len(attempts),
        })
    result = []
    for (level, domain, k), prompts in sorted(cells.items()):
        result.append({
            'level': level, 'domain': domain, 'k': k,
            'pass_at_k': statistics.fmean(p['pass_at_k'] for p in prompts),
            'distinct_at_k': statistics.fmean(p['distinct_at_k'] for p in prompts),
            'accuracy': statistics.fmean(p['accuracy'] for p in prompts),
            'pcmd': cell_summary(prompts, POOLED, min_defined_prompts=min_defined_prompts),
            'protocol_notes': _notes(level, domain), 'prompt_results': prompts,
        })
    dataset_identity = {'kind': 'custom', 'references_authenticated': False}
    missing_set = set(missing or [])
    if metadata is not None:
        dataset_identity = {
            'kind': 'frozen', 'references_authenticated': True,
            'registry': 'modebench-portable-data-v1', 'split_record': metadata,
            'manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
            'selected_prompt_ids': [rid for rid in expected_ids if rid not in missing_set],
            'missing_prompt_ids': missing, 'complete': not missing,
            'selected_prompts': len(prepared), 'expected_prompts': len(expected_ids),
            'allow_partial': allow_partial,
            'draws_per_prompt': next(iter(draw_counts)),
        }
        if config.endswith('_unique_answer'):
            dataset_identity['protocol_notes'] = ['Single-answer diagnostic; separate from the main benchmark.']
    return {
        'schema': 'modebench-saved-responses-v2',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'records_sha256': digest_json(records),
        'software': software_identity(),
        'run': declared_run,
        'generation_metadata_status': 'user_declared' if declared_run else 'unreported',
        'run_sha256': digest_json(declared_run) if declared_run else None,
        'dataset': dataset_identity,
        'evaluation': {
            'aggregation': POOLED, 'groups_per_prompt': 1,
            'prompt_weighting': 'equal', 'min_defined_prompts': min_defined_prompts,
            'pass_at_k': 'empirical_saved_group',
            'prompt_hash_semantics': 'expected_registered_messages; does not attest generation',
        },
        'cells': result,
    }
