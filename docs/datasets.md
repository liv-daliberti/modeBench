# Datasets, prompts, and task interfaces

The Git checkout bundles 26 configurations: five domains at each of five levels, plus a Level 1 graph-coloring single-answer diagnostic. Together they contain 72 Parquet splits and 15,552 rows. The diagnostic is separate from the main multi-answer benchmark.

## Load frozen data

```python
from modebench.data import load_split

rows = load_split("data", "level1_countdown", "eval")
assert len(rows) == 128
```

`load_split` checks the Parquet SHA-256 against [data/manifest.json](../data/manifest.json) before loading and checks the row count afterward. Pass the actual `data/` path when working outside the checkout root. The wheel does not contain dataset files.

Split names are `train`, `dev`, and `eval`. Frozen split membership and source row content are retained. These data are not regenerated during installation or evaluation.

## Row contract

The common columns are:

| Column | Purpose |
| --- | --- |
| `problem` | Task text used to render the model prompt |
| `answer` | JSON-encoded executable verifier specification |
| `modebench_task` | Task/verifier identifier |
| `answer_mode_count` | Retained support-count metadata |
| `answer_mode_split` | Original answer-mode split label |

Individual configs retain additional construction and provenance columns. The verifier specification and support metadata are evaluation-side information. Do not include them in model requests or use certified solutions to initialize an online training bank.

## Configuration inventory

The table below is generated from the bundled manifest. A dash means that split is not included, rather than an empty dataset.

| Config | Train | Dev | Eval |
| --- | ---: | ---: | ---: |
| `level1_countdown` | 384 | — | 128 |
| `level1_graph_coloring` | 384 | — | 128 |
| `level1_graph_coloring_unique_answer` | — | — | 128 |
| `level1_mathir` | 384 | — | 128 |
| `level1_pantry_plan` | 384 | 64 | 128 |
| `level1_python_factors` | 384 | — | 128 |
| `level2_countdown` | 384 | 128 | 128 |
| `level2_graph_coloring` | 384 | 128 | 128 |
| `level2_mathir` | 384 | 128 | 128 |
| `level2_pantry_plan` | 384 | 128 | 128 |
| `level2_python_factors` | 384 | 128 | 128 |
| `level3_countdown` | 384 | 128 | 128 |
| `level3_graph_coloring` | 384 | 128 | 128 |
| `level3_mathir` | 384 | 128 | 128 |
| `level3_pantry_plan` | 384 | 128 | 128 |
| `level3_python_factors` | 384 | 128 | 128 |
| `level4_countdown` | 384 | 128 | 128 |
| `level4_graph_coloring` | 384 | 128 | 128 |
| `level4_mathir` | 384 | 128 | 128 |
| `level4_pantry_plan` | 384 | 128 | 128 |
| `level4_python_factors` | 384 | 128 | 128 |
| `level5_countdown` | 384 | 128 | 128 |
| `level5_graph_coloring` | 384 | 128 | 128 |
| `level5_mathir` | 384 | 128 | 128 |
| `level5_pantry_plan` | 384 | 128 | 128 |
| `level5_python_factors` | 384 | 128 | 128 |

Most main training splits contain 384 rows and evaluation splits 128. Level 1 Pantry's development set contains 64; the other Level 1 main domains have no bundled development split. Levels 2–5 have 128-row development splits for all domains.

## Domain interfaces

Use `make_messages` to render prompts and `grade_response` to apply the complete response interface:

```python
from modebench.prompts import make_messages, profile_metadata
from modebench.historical_prompts import grade_response

row = rows[0]  # loaded from level1_countdown above
messages = make_messages(1, "countdown", row)
profile = profile_metadata(1, "countdown")
result = grade_response(1, "countdown", row, "YOUR_MODEL_RESPONSE")
```

| Domain argument | Registered response and validation |
| --- | --- |
| `graph_coloring` | A color assignment, with supported compact/fill forms. Every edge and fixed color is checked; identity is the completed color vector. |
| `countdown` | An arithmetic expression using the exact given operand multiset. Execution uses exact rational arithmetic; identity is its normalized expression tree. |
| `python_factors` | One restricted `lambda n: EXPR`. External calls must return a nontrivial proper divisor for each input. Identity is the vector of returned integers, not source text or algorithm identity. |
| `mathir` | Restricted equation actions/menu selections. Validation and identity derive from the exact executed state trajectory. |
| `pantry_plan` | Level 1 uses a six-bit ingredient-support mask. A deterministic quantity-grid search projects that exact support to a feasible allocation. Later levels use the registered allocation surface. Identity is ingredient support. |

The low-level `validated_modebench_outcome_key(response, reference)` returns a canonical key only when the response verifies. It does not perform the Level 1 Pantry support-mask projection. Prefer the level/domain adapter for full benchmark evaluation.

Structured chat prompts do not automatically reproduce a trainer's constrained decoding masks. Save `profile_metadata` and distinguish free generation from syntax-constrained generation when reporting results.

## Prompt versions

`modebench.prompts.make_messages` defaults to `registered_hints_v1`, which matches the bundled historical data. Preserve exact system/user strings and the prompt hash with a run.

`python_level3_neutral_v1` remains available as an explicit compatibility condition. It changes Level 3 Python wording and was calibrated against a separate dataset that is not bundled here. Applying it to the historical Level 3 data is a changed experimental condition, not a reproduction of the frozen cohort. The default intentionally remains historical.

## Level admission

“Difficulty-matched” refers to the recorded calibration and confirmation procedure, not a theorem that two levels are equally difficult.

- Levels 1–3 retain the original frozen release and prompt identities.
- Level 4 passed matching checks in graph coloring, Countdown, Python factors, and PantryPlan. **MathIR is not difficulty-matched, and Level 4 as a whole is not admitted.** Keep this qualification with any Level 4 MathIR result.
- Level 5 is admitted in all five domains according to the retained release manifest.

The [Levels 1–3 source manifest](../provenance/levels123_manifest.json) and [Levels 4–5 source manifest](../provenance/levels45_manifest.json) retain the detailed source records. Historical machine paths in those records identify original artifacts; loading the bundled files does not require those paths.

## Training and new construction

To convert one frozen config to the local DatasetDict layout used by the separate training repository:

```sh
python ops/materialize_training_data.py --config level1_countdown --output outputs/countdown
```

The result contains `train/` with subset `train` and `eval/` with subset `multi_answer`. Choose a fresh destination; the materializer refuses to overwrite one. This operation preserves the selected frozen rows.

Basic-domain construction scripts are retained under `ops/`; see [reproducibility](reproducibility.md#construction-tools). Generating new data does not reproduce the full higher-level calibration and admission process. Keep variants separate from the bundled configs.

## Source terms

See [license provenance](../provenance/levels123_LICENSE_PROVENANCE.md) for retained source terms, including Pantry's USDA ingredient source. The export did not assign a separate dataset license. The repository's [Apache 2.0 license](../LICENSE) applies to code; upstream data terms remain separately documented.
