"""Export an offline five-domain saved-response example and summarize receipts."""
from importlib.resources import files
from pathlib import Path
import json
import shutil
import tempfile

from .validation import InputError


def export(directory):
    directory = Path(directory)
    if directory.exists():
        raise InputError(f'walkthrough destination already exists: {directory}')
    source = json.loads(files('modebench').joinpath('walkthrough.json').read_text())
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.modebench-walkthrough-', dir=directory.parent))
    try:
        from .prompts import make_messages
        tasks = [{k: v for k, v in row.items() if k not in ('answer', 'responses')}
                 for row in source['records']]
        for task, row in zip(tasks, source['records']):
            task['messages'] = make_messages(row['level'], row['domain'], row)
        for name, records in [('tasks.jsonl', tasks), ('responses.jsonl', source['records'])]:
            (temporary / name).write_text(''.join(json.dumps(row) + '\n' for row in records))
        (temporary / 'run.json').write_text(json.dumps(source['run'], indent=2) + '\n')
        (temporary / 'expected.json').write_text(json.dumps(source['expected'], indent=2) + '\n')
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'directory': str(directory), 'domains': 5, 'responses_per_prompt': 4,
            'next': f'modebench evaluate {directory}/responses.jsonl --run {directory}/run.json --output {directory}/report.json'}


def summarize(payload):
    if not isinstance(payload, dict) or payload.get('schema') not in ('modebench-saved-responses-v3', 'modebench-saved-responses-v4'):
        raise InputError('report requires a ModeBench v3 or v4 receipt')
    evaluation = payload.get('evaluation', {})
    status = evaluation.get('status')
    if status not in ('completed', 'failed'):
        raise InputError('report has no valid evaluation status')
    if status == 'failed':
        return 'Evaluation FAILED: scores are unavailable. Inspect failure and attempt diagnostics in the JSON receipt.\n' + json.dumps(payload.get('failure', evaluation.get('status_counts', {})), sort_keys=True)
    lines = ['Evaluation completed', 'level domain k accuracy pass@k distinct@k PCMD reportable']
    for cell in payload['cells']:
        pcmd = cell['pcmd']
        value = lambda x: 'null' if x is None else f'{x:g}'
        lines.append(f"{cell['level']} {cell['domain']} {cell['k']} {value(cell['accuracy'])} "
                     f"{value(cell['pass_at_k'])} {value(cell['distinct_at_k'])} "
                     f"{value(pcmd['d_mode'])} {str(pcmd['reportable']).lower()}")
        for note in cell.get('protocol_notes', []):
            lines.append('  Note: ' + note)
    lines.append('PCMD uses only verified answers; reportable=false means insufficient eligible prompts.')
    return '\n'.join(lines)
