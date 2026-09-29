"""Grade saved generations without depending on a model provider or trainer."""
import argparse
import hashlib
import json
from pathlib import Path

from .data import prepare_requests
from .evaluation import evaluate  # Retained public import for existing callers.
from .validation import InputError, loads


def read_records(raw):
    records, locations = [], []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(loads(line))
            locations.append(f'line {number}')
        except ValueError as error:
            raise InputError(f'line {number}: {error}') from error
    return records, locations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    grade = sub.add_parser('evaluate', help='grade custom references or a frozen dataset split')
    grade.add_argument('input', type=Path)
    grade.add_argument('--output', type=Path, required=True)
    grade.add_argument('--min-defined-prompts', type=int, default=30)
    grade.add_argument('--run', type=Path, help='JSON with model, generation, and prompt_condition')
    grade.add_argument('--config', help='resolve references from a packaged frozen configuration')
    grade.add_argument('--split', choices=('train', 'dev', 'eval'), default='eval')
    grade.add_argument('--data-root', type=Path)
    grade.add_argument('--allow-partial', action='store_true', help='explicitly allow incomplete frozen split coverage')
    prepare = sub.add_parser('prepare', help='export frozen prompt IDs, hashes, and model messages')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--split', choices=('train', 'dev', 'eval'), default='eval')
    prepare.add_argument('--data-root', type=Path, default=Path('data'))
    prepare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise InputError(f'output already exists: {args.output}')
        if args.command == 'prepare':
            records = list(prepare_requests(args.data_root, args.config, args.split))
            text = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in records)
            summary = {'output': str(args.output), 'prompts': len(records)}
        else:
            raw = args.input.read_bytes()
            records, locations = read_records(raw)
            run = loads(args.run.read_bytes()) if args.run else None
            payload = evaluate(records, min_defined_prompts=args.min_defined_prompts,
                               run=run, data_root=args.data_root, config=args.config,
                               split=args.split, allow_partial=args.allow_partial, locations=locations)
            payload['input_sha256'] = hashlib.sha256(raw).hexdigest()
            text = json.dumps(payload, indent=2, allow_nan=False) + '\n'
            summary = {'output': str(args.output), 'cells': len(payload['cells']),
                       'dataset_kind': payload['dataset']['kind']}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            handle.write(text)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
