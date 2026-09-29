# ModeBench

**Can a model find more than one correct solution to the same problem?** ModeBench evaluates correctness and solution diversity using executable tasks with canonical outcome identities.

[![The five ModeBench domains, each showing two verified solution modes.](assets/modebench-domains.png)](assets/modebench-domains.png)

Five domains, five task levels, and **72 frozen dataset splits containing 15,552 rows**. Evaluate saved model responses on CPU without PyTorch, CUDA, or a training framework. [Re:Max / Re:Dr](https://github.com/liv-daliberti/remax) provides the separate optimization and training implementation.

| Domain | What is checked | What defines a mode |
| --- | --- | --- |
| Graph coloring | Every graph edge and fixed color | Complete color vector |
| Countdown | Exact arithmetic and use of the given operands | Normalized executed expression tree |
| Python factors | A restricted function returns proper divisors on the given inputs | Returned divisor vector |
| MathIR | Restricted equation actions execute correctly | Exact normalized state trajectory |
| PantryPlan | Ingredient and nutritional feasibility constraints | Ingredient support |

Mode identity is prompt-local. Different wording alone does not create a different mode; different Python programs can produce the same divisor vector.

[Installation](#installation) · [Evaluation](#evaluation) · [Metrics](#metrics) · [Datasets](#datasets) · [Reproducibility](#reproducibility) · [Contributing](#contributing)

## Installation

Use Python 3.10 or later on Linux or macOS; Windows users should use WSL. The external Python verifier uses POSIX process primitives. Run these commands from the checkout root:

```sh
git clone https://github.com/liv-daliberti/modeBench.git
cd modeBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
modebench evaluate examples/responses.jsonl --output outputs/example.json
```

The example makes no model calls. Its cells have accuracy `0.5`, pass@4 `1.0`, distinct@4 `2.0`, and PCMD `1.0`; their support is below the default reporting threshold. Choose a fresh output filename for each run—existing outputs are never overwritten.

| Installation | Includes |
| --- | --- |
| `python -m pip install -e .` | Validators, metrics, and CLI |
| `python -m pip install -e '.[data]'` | Also dataset loading and construction dependencies |
| `python -m pip install -e '.[dev]'` | Also the regression suite |

The Git checkout supplies data, evidence, and scripts; the wheel contains the Python package and CLI. To use the tested Linux/Python 3.10 dependencies, add `-c provenance/constraints-py310.txt` to the install command. The Python-factor worker executes a bounded, restricted language; it is not a general-purpose sandbox for arbitrary programs.

## Evaluation

Load a frozen split, render the registered prompt, and grade a response:

```python
from modebench.data import load_split
from modebench.prompts import make_messages, profile_metadata, prompt_sha256
from modebench.historical_prompts import grade_response

rows = load_split("data", "level1_countdown", "eval")
row = rows[0]
messages = make_messages(1, "countdown", row)
profile = profile_metadata(1, "countdown")
prompt_hash = prompt_sha256(1, "countdown", row)
# Send only `messages` to your model, then grade its returned text.
result = grade_response(1, "countdown", row, "YOUR_MODEL_RESPONSE")
print(result)  # verified, canonical_key, graded_text
```

Keep `answer`, support counts, and certified solutions out of model requests. Match saved responses by stable prompt ID, not asynchronous completion order. Preserve raw generations, refusals, and malformed answers; collection failures need a declared protocol and must not silently reduce the draw budget.

For batch evaluation, write one JSON object per prompt to a JSONL file:

```json
{"id":"example","level":1,"domain":"countdown","answer":{"verifier":"countdown","numbers":[1,2,3],"target":6},"responses":["1+2+3","1*2*3","1+2","bad"]}
```

```sh
modebench evaluate INPUT.jsonl --output OUTPUT.json
```

| Field | Contract |
| --- | --- |
| `id` | Stable prompt identity; string recommended, unique within level/domain |
| `level` | Integer 1–5 or `"level1"` through `"level5"` |
| `domain` | `graph_coloring`, `countdown`, `python_factors`, `mathir`, or `pantry_plan` (`pantry` is an alias) |
| `answer` | Exact executable reference from the frozen row, as an object or JSON-encoded string |
| `responses` | Nonempty list of raw response strings, including failed answers |

Use one input file per model and generation condition: grouping uses only level, domain, and response count `k`. Extra metadata fields are not used in scoring; retain collection metadata separately. The evaluator does not authenticate references against a dataset, model identity, or generation provenance.

Outputs contain a schema version, the input SHA-256, and cells with accuracy, pass@k, distinct@k, PCMD/support, and per-response results. Failed verification has a null canonical key. For Level 1 Pantry, `graded_text` is the deterministic allocation projected from the submitted six-bit support mask. Use `grade_response` or the CLI for this interface; the low-level key validator does not apply that projection.

<details>
<summary>Troubleshooting</summary>

| Symptom | Resolution |
| --- | --- |
| CLI not found | Activate the environment or run `python -m modebench.cli`. |
| Missing `datasets` | Install the `data` or `dev` extra. |
| Unknown split | Use `eval`, not `test`; check split availability below. |
| Dataset hash mismatch | Restore the file from the same revision; keep variants under separate identities. |
| Output already exists | Choose a fresh destination. |
| PCMD is null or unreportable | Check verified draws and eligible-prompt support; missing PCMD is not zero. |
| Python-factor execution fails on Windows | Use a POSIX environment such as WSL. |

</details>

## Metrics

For a prompt with `k` draws, let `K` verify and let `n_m` count verified responses in mode `m`.

| Metric | Per-prompt definition | Aggregation |
| --- | --- | --- |
| Accuracy | `K / k` | Equal mean over prompts |
| Empirical pass@k | `1` if any response verifies, otherwise `0` | Equal mean over prompts |
| Distinct@k | Number of distinct verified modes | Equal mean over prompts |
| PCMD | `1 - sum(n_m * (n_m - 1)) / (K * (K - 1))` | Equal mean over prompts with `K >= 2` |

PCMD measures success-conditional diversity. Counts `{A: 2, B: 1}` give `2/3`; one repeated mode gives zero. Fewer than two successes gives `null`, not zero. Eligibility depends on correctness, so always report accuracy, `defined_prompts`, `prompts`, `support`, and `reportable` alongside PCMD.

The default reporting threshold is 30 eligible prompts. A value may exist with `reportable: false`; respect that flag in tables and figures. `--min-defined-prompts` changes the threshold and should be registered before analysis. Empirical pass@k describes the saved groups, not the combinatorial estimator for subsampling a larger pool.

<details>
<summary>Repeated groups and uncertainty</summary>

The CLI treats each prompt's response list as one group. The Python metric API supports repeated groups:

```python
from modebench.metrics import POOLED, PER_GROUP, prompt_mode_diversity

draws = [{"attempts": [
    {"verified": True, "canonical_key": "A"},
    {"verified": True, "canonical_key": "B"},
    {"verified": False, "canonical_key": None},
]}]
assert prompt_mode_diversity(draws, POOLED) == 1.0
assert prompt_mode_diversity(draws, PER_GROUP) == 1.0
```

`POOLED` pools a prompt's groups; `PER_GROUP` averages its defined group scores. Do not flatten four groups of eight and label the resulting pass@32 as pass@8. Never pool mode counts across prompts.

`standard_error` is the sample standard deviation of defined prompt scores divided by the square root of their count. It is null for one defined prompt and is not a paired comparison interval. Finite-sample eligibility changes the population being compared.

`rarefied_distinct_at_2` equals `1 + PCMD`. `effective_modes` is `1 / (1 - mean_PCMD)`, computed after aggregation; it is null when PCMD is undefined or equals one, and is not an unbiased mode-count estimate.

</details>

## Datasets

Files are organized as [`data/<level>/<domain>/<split>.parquet`](data/). Configuration names remain `level1_countdown`, `level2_mathir`, and so on. The [manifest](data/manifest.json) records every configuration, split, row count, original subset, and hash. `load_split` verifies the Parquet SHA-256 and row count; pass the actual data-directory path when working outside the checkout.

| Configurations | Train | Dev | Eval |
| --- | ---: | ---: | ---: |
| Level 1 Countdown, graph coloring, MathIR, Python factors | 384 | — | 128 |
| Level 1 PantryPlan | 384 | 64 | 128 |
| Levels 2–5, all five domains | 384 | 128 | 128 |
| Level 1 graph-coloring single-answer diagnostic | — | — | 128 |

A dash means absent. The diagnostic lives under `data/level1/graph_coloring/unique_answer/` and uses configuration `level1_graph_coloring_unique_answer`; it is separate from the main benchmark.

Rows contain `problem`, `answer`, `modebench_task`, `answer_mode_count`, and `answer_mode_split`, plus domain-specific metadata. Python factors expects one restricted `lambda n: EXPR`; MathIR uses equation actions/menu selections; Level 1 Pantry uses a six-bit support mask with deterministic quantity projection. Other Pantry levels use allocation responses.

Prompts default to `registered_hints_v1`, matching the bundled data. The optional `python_level3_neutral_v1` changes Level 3 Python wording; its separately calibrated dataset is not included. Switching conditions changes the experiment. Structured chat messages do not reproduce training-time syntax masks; record free versus constrained decoding.

**Level 4 MathIR is not difficulty-matched, and Level 4 as a whole is not admitted.** The other four Level 4 domains passed their matching checks. All five Level 5 domains are admitted according to the retained manifests. Admission reflects a recorded calibration procedure, not a theorem of equal difficulty. See the [Levels 1–3](provenance/levels123_manifest.json) and [Levels 4–5](provenance/levels45_manifest.json) source records.

To materialize a frozen configuration for the separate training repository:

```sh
python ops/materialize_training_data.py --config level1_countdown --output outputs/countdown
```

This creates `train/` with subset `train` and `eval/` with subset `multi_answer`, preserving frozen rows and refusing an existing destination. Basic-domain generators remain in [ops/](ops/); consult their `--help`. Pantry construction requires an explicit ingredient-table JSON via `--ingredients`. Full higher-level construction, fitting, and admission workflows are not consolidated; generated variants require separate identities and admission evidence.

## Reproducibility

```sh
make check                          # Regression suite, frozen analysis, dataset hashes
python ops/verify_release.py         # Repository file inventory and hashes
```

Validation covers **112 tests**, **375 frozen base-grid cells**, and **72 splits / 15,552 rows**. CI runs on Python 3.10, 3.11, and 3.12. Installation, the saved-response example, and Countdown materialization were checked in a fresh Linux/Python 3.10 environment; [tested dependencies](provenance/constraints-py310.txt) are recorded.

The [verified-key archive](evidence/base_grid_keys.jsonl.gz) retains attempt flags/keys, repeated groups, recorded pass@8/distinct@8, and source receipt hashes. `ops/reproduce_base_grid.py` compares its recomputed summaries with the [frozen results](evidence/mode_diversity_base_grid.json), allowing only final-bit numeric tolerance (`rel_tol=1e-14`, `abs_tol=1e-15`); identifiers and counts match exactly. The archive lacks complete raw responses, so this check does not independently regrade generations or reproduce model sampling.

[Source provenance](provenance/source.json) records extracted-file identities; the [release manifest](provenance/release.json) binds final files. Extraction included uncommitted research changes, so the source commit alone is insufficient. Historical machine paths identify source artifacts and are not runtime dependencies. Equivalence to every historical verifier runtime remains unestablished; fresh model generation and training-checkpoint reproduction are outside these checks.

For reported results, retain raw responses and record the repository commit, dataset configuration/split/hash, selected prompt IDs and exclusions, prompt condition/hash, model revision, decoding settings, seeds/dates, draw/group structure, failures, metric support/threshold, and uncertainty method. Use the repository URL and exact commit in software citations; final scientific citation metadata remain pending.

## Contributing

Run `make check` before submitting changes. Report bugs through [GitHub issues](https://github.com/liv-daliberti/modeBench/issues) with the commit, Python version, domain/level, and a minimal response/reference example.

Explain changes to accepted responses, canonical identities, prompts, splits, or aggregation, and add focused behavioral tests. Keep frozen datasets and evidence immutable; changed conditions need new identities. Preserve the external Python execution boundary and keep the core independent of training frameworks.

After reviewing deliberate file changes, refresh the release inventory with `python ops/verify_release.py --refresh`, inspect its diff, then verify it again. Updating hashes records changed bytes; it does not establish scientific equivalence.

## License and attribution

Code is licensed under [Apache 2.0](LICENSE), with original notices retained. [Dataset source terms](provenance/licenses/data-sources.json) preserve attribution, including Pantry's USDA FoodData Central source. No separate blanket dataset license was assigned by this export; the code license does not assign new rights to upstream data.
