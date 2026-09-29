"""Grade saved generations without depending on a model provider or trainer."""
import argparse
import hashlib
import json
from pathlib import Path

from .diagnostics import VerifierExecutionError
from .identity import software_identity
from .data import prepare_requests, load_split, prompt_id
from .dataset_cache import discover, fetch
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


def _dataset_options(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--data-root', type=Path, help='existing local data directory')
    source.add_argument('--cache-dir', type=Path, help='cache base (default: XDG_CACHE_HOME/modebench)')
    parser.add_argument('--offline', action='store_true', help='require verified cached bytes; never download')


def _data_root(args):
    if args.data_root is not None:
        return args.data_root
    if args.cache_dir is None and Path('data/manifest.json').is_file():
        return Path('data')
    return fetch(args.config, args.split, cache_dir=args.cache_dir, offline=args.offline)


def _utility(args):
    if args.command == 'walkthrough':
        from .walkthrough import export
        print(json.dumps(export(args.directory)))
    elif args.command == 'report':
        from .walkthrough import summarize
        payload = loads(args.input.read_bytes())
        print(summarize(payload))
        if payload['evaluation']['status'] == 'failed':
            raise SystemExit(3)
    elif args.dataset_command == 'list':
        records = discover(args.config, args.split)
        if args.json:
            print(json.dumps(records, indent=2))
        else:
            print('level domain config split rows')
            for r in sorted(records, key=lambda r: (r['level'], r['domain'], r['config_name'], r['split'])):
                print(r['level'], r['domain'], r['config_name'], r['split'], r['rows'])
    elif args.dataset_command == 'fetch':
        root = fetch(args.config, args.split, cache_dir=args.cache_dir, offline=args.offline)
        print(json.dumps({'data_root': str(root), 'config': args.config, 'split': args.split, 'verified': True}))
    else:
        from .prompts import make_messages
        root = _data_root(args)
        rows = load_split(root, args.config, args.split, frozen=True)
        if args.row < 0 or args.row >= len(rows):
            raise InputError(f'row must be between 0 and {len(rows)-1}')
        record, = discover(args.config, args.split)
        row = rows[args.row]
        payload = {'id': prompt_id(args.config, args.split, args.row), 'rows': len(rows),
                   'level': record['level'], 'domain': record['domain'],
                   'dataset_sha256': record['parquet_sha256'],
                   'messages': make_messages(record['level'], record['domain'], row)}
        if args.show_reference:
            payload['answer'] = loads(row['answer']) if isinstance(row['answer'], str) else row['answer']
        print(json.dumps(payload, indent=2))


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
    _dataset_options(grade)
    grade.add_argument('--allow-partial', action='store_true', help='explicitly allow incomplete frozen split coverage')
    prepare = sub.add_parser('prepare', help='export frozen prompt IDs, hashes, and model messages')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--split', choices=('train', 'dev', 'eval'), default='eval')
    _dataset_options(prepare)
    prepare.add_argument('--output', type=Path, required=True)
    datasets = sub.add_parser('datasets', help='discover, download, and inspect frozen datasets')
    actions = datasets.add_subparsers(dest='dataset_command', required=True)
    listing = actions.add_parser('list', help='list packaged split identities (no network or data extra)')
    listing.add_argument('--config')
    listing.add_argument('--split', choices=('train', 'dev', 'eval'))
    listing.add_argument('--json', action='store_true')
    download = actions.add_parser('fetch', help='download a SHA-256 verified immutable split')
    download.add_argument('--config', required=True)
    download.add_argument('--split', choices=('train', 'dev', 'eval'), default='eval')
    download.add_argument('--cache-dir', type=Path)
    download.add_argument('--offline', action='store_true')
    inspect = actions.add_parser('inspect', help='show a registered task and messages')
    inspect.add_argument('--config', required=True)
    inspect.add_argument('--split', choices=('train', 'dev', 'eval'), default='eval')
    inspect.add_argument('--row', type=int, default=0)
    inspect.add_argument('--show-reference', action='store_true', help='include the answer for inspection only')
    _dataset_options(inspect)
    walkthrough = sub.add_parser('walkthrough', help='export a complete offline five-domain example')
    walkthrough.add_argument('--directory', type=Path, required=True)
    report = sub.add_parser('report', help='explain a saved evaluation receipt')
    report.add_argument('input', type=Path)
    args = parser.parse_args()
    exit_code = 0
    try:
        if args.command in ('datasets', 'walkthrough', 'report'):
            _utility(args)
            return
        if args.command == 'evaluate' and args.config is None and (args.cache_dir is not None or args.offline):
            raise InputError('--cache-dir and --offline require --config')
        if args.output.exists():
            raise InputError(f'output already exists: {args.output}')
        if args.command == 'prepare':
            records = list(prepare_requests(_data_root(args), args.config, args.split))
            text = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in records)
            summary = {'output': str(args.output), 'prompts': len(records)}
        else:
            raw = args.input.read_bytes()
            records, locations = read_records(raw)
            run = loads(args.run.read_bytes()) if args.run else None
            payload = evaluate(records, min_defined_prompts=args.min_defined_prompts,
                               run=run, data_root=_data_root(args) if args.config else args.data_root, config=args.config,
                               split=args.split, allow_partial=args.allow_partial, locations=locations)
            exit_code = 3 if payload['evaluation']['status'] == 'failed' else 0
            payload['input_sha256'] = hashlib.sha256(raw).hexdigest()
            text = json.dumps(payload, indent=2, allow_nan=False) + '\n'
            summary = {'output': str(args.output), 'cells': len(payload['cells']),
                       'dataset_kind': payload['dataset']['kind']}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            handle.write(text)
    except VerifierExecutionError as error:
        payload = {'schema': 'modebench-saved-responses-v3', 'software': software_identity(),
                   'evaluation': {'status': 'failed'}, 'failure': error.diagnostic, 'cells': []}
        if args.command == 'evaluate':
            payload['input_sha256'] = hashlib.sha256(raw).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps({'output': str(args.output), 'status': 'failed'}))
        raise SystemExit(3)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
