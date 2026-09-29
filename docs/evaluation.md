# Evaluation and metrics

## Input contract

Run `modebench evaluate INPUT.jsonl --output OUTPUT.json`. One JSON object per line represents one prompt:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | String recommended | Stable prompt identity, unique within its level/domain |
| `level` | Integer 1–5, or `"level1"` … `"level5"` | Task level |
| `domain` | String | `graph_coloring`, `countdown`, `python_factors`, `mathir`, or `pantry_plan`; `pantry` is an alias |
| `answer` | Object or JSON-encoded string | Exact executable reference specification from the frozen row |
| `responses` | Nonempty list of strings | Raw generations, including incorrect answers, malformed output, and refusals |

```json
{"id":"example","level":1,"domain":"countdown","answer":{"verifier":"countdown","numbers":[1,2,3],"target":6},"responses":["1+2+3","1*2*3","1+2","bad"]}
```

Use a separate input file for each model and generation condition: the CLI groups only by **level, domain, and response count k**, not by model metadata. It rejects duplicate prompt IDs within a level/domain. Extra fields are not used in scoring; keep collection metadata in a sidecar record.

There is no automatic retry, truncation, text repair, model sampling, or missing-response imputation. The versioned domain validators handle their registered formatting aliases. The loader does not authenticate a model identity or prove that your responses came from the recorded prompts.

## Output contract

The result contains `schema`, `input_sha256` (the raw input file's SHA-256), and `cells`. Each cell includes:

- `level`, `domain`, `k`, `accuracy`, `pass_at_k`, and `distinct_at_k`;
- `pcmd`, including `d_mode`, `defined_prompts`, `prompts`, `support`, `min_defined_prompts`, `reportable`, `standard_error`, `rarefied_distinct_at_2`, and `effective_modes`;
- `prompt_results`, with each prompt's metrics and each response's `verified`, `canonical_key`, and `graded_text`.

The CLI computes one group per prompt. It preserves all attempts. `canonical_key` is null when verification fails. For Level 1 Pantry, `graded_text` is the deterministic quantity allocation derived from the submitted support mask, rather than the original mask.

Outputs are created exclusively: an existing output path is an error. A computed PCMD value can remain present with `reportable: false`; consumers must respect that flag when producing tables or figures.

## Metric definitions

For a prompt with k saved responses, let K verify successfully and let `n_m` count those successes in mode m, so `sum(n_m) = K`.

| Metric | Per-prompt definition | Cell aggregation |
| --- | --- | --- |
| Accuracy | `K / k` | Equal mean over prompts |
| Empirical pass@k | 1 if `K > 0`, otherwise 0 | Equal mean over prompts |
| Distinct@k | Number of modes with positive count | Equal mean over prompts |
| PCMD | `1 - sum(n_m * (n_m - 1)) / (K * (K - 1))`, for `K >= 2` | Equal mean over defined prompts |

Here pass@k is the empirical success rate of the saved groups. The CLI does not implement the combinatorial estimator for subsampling k responses from a larger generation pool.

For verified-mode counts `{A: 2, B: 1}`, PCMD is `2/3`: two of the three unordered pairs have different modes. `{A: 3}` gives zero, while three singleton modes give one. With fewer than two successes it is undefined (`null`), rather than zero.

PCMD estimates success-conditional mode diversity. Unlike distinct@k, its population target does not directly contain the success probability. Finite-sample eligibility still depends on correctness: prompts with too few successes drop out of its defined population. Accuracy, support, and the eligible population must therefore accompany comparisons.

### Support and uncertainty

`defined_prompts` counts prompts with at least two verified responses. `support` divides that count by all prompts in the cell. The default `min_defined_prompts` is 30; fewer defined prompts produces `reportable: false`.

`standard_error` is the sample standard deviation of defined prompt-level PCMD estimates divided by the square root of their count. It is not a paired confidence interval for a method comparison. A single defined prompt has no standard error.

The CLI exposes `--min-defined-prompts` for explicitly registered alternatives. Record any change before analyzing outcomes; changing the threshold does not create evidence for an underpowered cell.

### Rarefaction and effective modes

The expected number of distinct modes in two verified draws is `1 + PCMD`. `effective_modes` computes `1 / (1 - mean_PCMD)` after aggregation. It returns null when PCMD is undefined or the empirical mean is one; do not interpret that null as zero effective modes. This nonlinear transform is not an unbiased estimate of an underlying mode count.

## Repeated groups

The Python metric API accepts multiple draws per prompt:

```python
from modebench.metrics import POOLED, PER_GROUP, prompt_mode_diversity

prompt = [{"attempts": [
    {"verified": True, "canonical_key": "mode-A"},
    {"verified": True, "canonical_key": "mode-B"},
    {"verified": False, "canonical_key": None},
]}]
assert prompt_mode_diversity(prompt, POOLED) == 1.0
assert prompt_mode_diversity(prompt, PER_GROUP) == 1.0
```

`POOLED` pools verified counts across a prompt's groups before computing PCMD. `PER_GROUP` computes PCMD for each eligible group, then averages defined groups within the prompt. The frozen base-grid reproducer reports both conventions.

Do not flatten four groups of eight into a single CLI group and label the resulting pass@32 or distinct@32 as pass@8 or distinct@8. Keep grouping in your analysis through the Python metric API when reproducing that protocol.

## Reporting results

Record and report:

1. Repository commit, dataset config/split and SHA-256, selected prompt IDs, and any exclusions.
2. Prompt condition, prompt hashes, domain adapter, and any syntax-constrained decoding.
3. Model identifier and immutable revision when available, decoding settings, token limit, seeds, and collection dates for hosted models.
4. Number of prompts and draws per prompt, repeated-group structure, failures, and collection completeness.
5. Accuracy, empirical pass@k, distinct@k, PCMD, eligible-prompt count, support, and reporting threshold.
6. Pairing and uncertainty method for comparisons; do not imply paired uncertainty from the per-cell standard error.

Mode identity is prompt-local. Do not merge canonical-key counts across different prompts as though they were samples from one solution distribution. Record Level 4 MathIR's unmatched status and any departure from the bundled historical Level 3 prompt condition.
