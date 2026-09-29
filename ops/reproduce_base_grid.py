"""Recompute and check all frozen base-grid numerical cells from saved verified keys."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import statistics
from modebench.metrics import cell_summary, POOLED, PER_GROUP, rarefied_distinct, verified_mode_counts

ROOT = Path(__file__).resolve().parents[1]

def reproduce():
    expected = json.loads((ROOT / 'evidence/mode_diversity_base_grid.json').read_text())
    cells = []
    with gzip.open(ROOT / 'evidence/base_grid_keys.jsonl.gz', 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            results = record['prompt_results']
            summary = cell_summary(results, POOLED)
            per_group = cell_summary(results, PER_GROUP)
            depth3 = []
            for result in results:
                pooled = Counter()
                for draw in result['draws']:
                    pooled.update(verified_mode_counts(draw['attempts']))
                value = rarefied_distinct(pooled, 3)
                if value is not None:
                    depth3.append(value)
            cells.append({
                **{k: record[k] for k in ('model_label','level','domain','receipt','receipt_sha256')},
                'prompts': summary['prompts'],
                'pass8': statistics.fmean(r['pass8'] for r in results),
                'distinct8': statistics.fmean(r['distinct8'] for r in results),
                'pmd': summary['d_mode'], 'pmd_standard_error': summary['standard_error'],
                'rarefied_distinct_at_2': summary['rarefied_distinct_at_2'],
                'effective_modes': summary['effective_modes'],
                'defined_prompts': summary['defined_prompts'], 'support': summary['support'],
                'reportable': summary['reportable'],
                'rarefied_distinct_at_3': statistics.fmean(depth3) if depth3 else None,
                'defined_prompts_at_3': len(depth3), 'pmd_per_group': per_group['d_mode'],
                'defined_prompts_per_group': per_group['defined_prompts'],
            })
    cells.sort(key=lambda c: (c['model_label'], c['level'], c['domain']))
    if len(cells) != len(expected['cells']):
        raise AssertionError('base-grid cell count differs')
    for actual, reference in zip(cells, expected['cells']):
        if actual.keys() != reference.keys():
            raise AssertionError('base-grid schema differs')
        for key, value in actual.items():
            target = reference[key]
            # Python 3.10/3.11 statistics.stdev differ in the last floating bit.
            equal = math.isclose(value, target, rel_tol=1e-14, abs_tol=1e-15) if isinstance(value, float) and isinstance(target, float) else value == target
            if not equal:
                raise AssertionError(f"{actual['model_label']}/{actual['level']}/{actual['domain']}: {key}: {value} != {target}")
    return {'matched_cells': len(cells)}

if __name__ == '__main__':
    print(json.dumps(reproduce()))
