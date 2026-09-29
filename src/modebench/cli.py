"""Grade saved generations without depending on a model provider or trainer."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from .historical_prompts import grade_response, _identity
from .metrics import cell_summary, POOLED


def evaluate(records, *, min_defined_prompts=30):
    cells = defaultdict(list)
    seen = set()
    for record in records:
        level, domain = _identity(record['level'], record['domain'])
        identity = (level, domain, str(record['id']))
        if identity in seen:
            raise ValueError(f'duplicate prompt: {identity}')
        seen.add(identity)
        responses = record['responses']
        if not isinstance(responses, list) or not responses or not all(isinstance(s, str) for s in responses):
            raise ValueError('responses must be a nonempty list of strings (retain failed draws)')
        attempts = [grade_response(level, domain, record, s) for s in responses]
        keys = [a['canonical_key'] for a in attempts if a['verified']]
        cells[(level, domain, len(responses))].append({
            'id': str(record['id']), 'draws': [{'attempts': attempts}],
            'pass_at_k': float(bool(keys)), 'distinct_at_k': len(set(keys)),
            'accuracy': len(keys) / len(responses),
        })
    if not cells:
        raise ValueError('no prompt records')
    result = []
    for (level, domain, k), prompts in sorted(cells.items()):
        result.append({
            'level': level, 'domain': domain, 'k': k,
            'pass_at_k': statistics.fmean(p['pass_at_k'] for p in prompts),
            'distinct_at_k': statistics.fmean(p['distinct_at_k'] for p in prompts),
            'accuracy': statistics.fmean(p['accuracy'] for p in prompts),
            'pcmd': cell_summary(prompts, POOLED, min_defined_prompts=min_defined_prompts),
            'prompt_results': prompts,
        })
    return {'schema': 'modebench-saved-responses-v1', 'cells': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    grade = sub.add_parser('evaluate', help='grade JSONL records with id, level, domain, answer, responses')
    grade.add_argument('input', type=Path)
    grade.add_argument('--output', type=Path, required=True)
    grade.add_argument('--min-defined-prompts', type=int, default=30)
    args = parser.parse_args()
    if args.min_defined_prompts < 1:
        parser.error('--min-defined-prompts must be positive')
    raw = args.input.read_bytes()
    payload = evaluate([json.loads(line) for line in raw.splitlines() if line.strip()], min_defined_prompts=args.min_defined_prompts)
    payload['input_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'cells': len(payload['cells'])}))


if __name__ == '__main__':
    main()
