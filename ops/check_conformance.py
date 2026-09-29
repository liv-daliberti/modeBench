"""Compare current verifier behavior to frozen fixtures, optionally from a PR base."""
import argparse
import json
from pathlib import Path
import subprocess
from modebench.historical_prompts import grade_response

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = 'tests/fixtures/conformance-v1.json'


def check(corpus):
    count = 0
    for fixture in corpus['fixtures']:
        for case in fixture['cases']:
            actual = grade_response(fixture['level'],fixture['domain'],fixture['row'],case['response'])
            expected = case['expected']
            if actual['status'] not in ('correct','incorrect','malformed') or {k:actual[k] for k in expected} != expected:
                raise AssertionError(f"verifier drift: {fixture['config']}/{case['name']}: {actual}")
            count += 1
    return {'matched_cases':count,'covered_cells':len(corpus['fixtures']),'baseline_commit':corpus['baseline_commit']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,default=ROOT/FIXTURE)
    parser.add_argument('--base-ref',help='also test against immutable fixture bytes in the PR base commit')
    args=parser.parse_args()
    corpus=json.loads(args.fixtures.read_text())
    print(json.dumps(check(corpus)))
    if args.base_ref:
        subprocess.run(['git','cat-file','-e',f'{args.base_ref}^{{commit}}'],cwd=ROOT,check=True)
        probe=subprocess.run(['git','cat-file','-e',f'{args.base_ref}:{FIXTURE}'],cwd=ROOT,capture_output=True)
        if probe.returncode:
            print('PR base predates the first conformance corpus; current pinned baseline checked.')
        else:
            raw=subprocess.check_output(['git','show',f'{args.base_ref}:{FIXTURE}'],cwd=ROOT)
            print(json.dumps({'pr_base':args.base_ref,**check(json.loads(raw))}))
