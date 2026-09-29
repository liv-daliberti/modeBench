# Reproducibility

This release supports several distinct checks. A successful saved-key analysis does not establish that a fresh model run will reproduce every historical checkpoint or response.

| Check | Command | What it establishes |
| --- | --- | --- |
| Regression suite | `python -m pytest -q` | Retained validation, mode identity, process boundary, metric, and API behavior |
| Dataset identity | `python ops/verify_data.py` | All 72 bundled Parquet files match recorded SHA-256 values |
| Frozen base-grid analysis | `python ops/reproduce_base_grid.py` | 375 cell summaries match the retained numerical record |
| Saved-response grading | `modebench evaluate INPUT.jsonl --output OUTPUT.json` | Current validators grade your supplied generations |
| All maintained checks | `make check` | Tests, base-grid reproduction, and dataset identity together |

All commands run from the checkout root after installation. The checks make no model-provider calls and launch no training jobs.

## Frozen analysis

[evidence/base_grid_keys.jsonl.gz](../evidence/base_grid_keys.jsonl.gz) is a compact projection of the original evaluation receipts. It retains per-attempt verified flags and canonical keys, repeated-group structure, recorded prompt-level pass@8/distinct@8, and each original receipt's path and SHA-256.

`ops/reproduce_base_grid.py` recomputes PCMD, support, both group conventions, rarefaction, and effective-mode summaries from these records, and aggregates recorded prompt-level pass@8 and distinct@8. It checks the resulting cells against [the frozen base-grid summary](../evidence/mode_diversity_base_grid.json).

The expected output is:

```json
{"matched_cells": 375}
```

Numeric comparisons allow `rel_tol=1e-14` and `abs_tol=1e-15` for final-bit variation in Python's statistical functions. Identifiers, counts, and schema must match exactly.

The archive does not contain all original response text. It cannot support independent response regrading by itself. Its source hashes preserve the identity of original receipts; the repository's release manifest binds the included compact archive.

## File and source identities

- [RELEASE_MANIFEST.json](../RELEASE_MANIFEST.json) lists final repository file hashes and sizes, excluding itself and generated caches/build outputs.
- [PROVENANCE.json](../PROVENANCE.json) maps the initial extracted source files to their research-checkout paths and source hashes. It distinguishes mechanical extraction hashes from reviewed final hashes.
- [data/manifest.json](../data/manifest.json) binds portable dataset configs, split labels, row counts, and Parquet identities.
- [provenance/](../provenance/) retains original export manifests and attribution. Absolute paths in these historical records are provenance, not runtime dependencies.

Verify the release manifest using only the standard library:

```sh
python ops/verify_release.py
```

The initial extraction came from a working tree that included uncommitted research changes. Its source commit alone is therefore insufficient to reconstruct it; preserve the file hashes and this repository's commit. See [validation](../VALIDATION.md) for the checks actually performed and [release scope](../RELEASE_STATUS.md) for remaining historical-runtime equivalence work.

## Construction tools

These retained scripts support basic-domain construction and inspection. Inspect each script's `--help` before using it. Some construction paths need upstream datasets or input files; they are not part of the offline quick start.

| Script | Purpose |
| --- | --- |
| `ops/make_modebench_data.py` | Graph-coloring/Countdown source materialization |
| `ops/make_exact_countdown_mode_data.py` | Exact multi-answer Countdown construction |
| `ops/make_python_factor_mode_data.py` | Factor-function rows with certified multiple modes |
| `ops/verify_python_factor_mode.py` | External factor-function validation CLI |
| `ops/make_mathir_action_menu_data.py` | MathIR action-menu construction |
| `ops/make_pantry_plan_mode_data.py` | Pantry constraint-task construction |
| `ops/make_pantry_plan_mode_data_v2.py` | Retained Pantry v2 construction |

The Pantry generators expect an ingredient-table JSON input (`--ingredients`); the historical default path is not included in this checkout. Use `--help` and supply that source input explicitly when constructing new data. Frozen Pantry evaluation does not need it.

The full Level 2–5 fitting, confirmation, and admission workflows are not consolidated in this release. Use bundled frozen splits for exact data identity; independently generated rows need a new dataset identity and their own admission evidence.

## New model runs

To make a new run reproducible, retain raw model responses as well as graded outputs. Record the inputs listed in the [reporting checklist](evaluation.md#reporting-results). Match immutable model revisions and exact prompts where available; hosted-model names alone may not identify stable deployments.

This repository's evaluation package is independent of the optimization method. Reproducing Re:Max/Re:Dr training requires the separate methods implementation, training environment, and the correct experiment's frozen protocol.

## Referencing this software

Use the repository URL and exact commit in software citations and experiment manifests:

```text
ModeBench. https://github.com/liv-daliberti/modeBench
Software version 0.1.0; Git commit: <the commit used for the experiment>.
```

This does not substitute for citing the associated scientific work when its final bibliographic metadata are available. This release does not invent a DOI or paper author list.
