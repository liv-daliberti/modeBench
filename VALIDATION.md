# Validation — 2026-09-29

## Standalone public repository

The documented installation was tested in a fresh Linux/Python 3.10 virtual environment with `python -m pip install -e '.[dev]'`. Exact dependency versions are recorded in [constraints-py310-tested.txt](constraints-py310-tested.txt).

- `make check`: **112 tests passed**, **375 base-grid cells matched**, and **72 frozen dataset splits / 15,552 rows** matched their recorded Parquet SHA-256 values.
- The saved-response CLI example completed successfully without model calls or GPU dependencies.
- Frozen Countdown materialization produced the expected 384 training rows and 128 evaluation rows.
- Python examples in the getting-started, datasets, and evaluation guides executed successfully.
- Documentation links and heading anchors were checked. The README illustration is an unchanged copy of the paper's five-domain figure, with its source SHA-256 recorded in `PROVENANCE.json`.
- The release-file inventory and checksums pass `python ops/verify_release.py`.

The standalone run exposed three Pantry tests that still referenced the original research checkout. Their fixtures now load the bundled, hash-verified Level 1 Pantry development split; all 112 tests pass from this repository's own root.

## Earlier extraction checks

The initial combined ModeBench/Re:Max suite passed 240 tests in the research Python 3.10 environment. Both packages built as wheels without dependency resolution. An isolated installed-wheel check exercised Countdown and the external Python-factor worker without importing PyTorch or the parent research package.

## Interpretation

These checks establish packaging behavior, retained regression behavior, frozen dataset bytes, and saved-key numerical reproduction. They do not rerun model generation, certify every historical verifier snapshot, or reproduce training checkpoints. The GitHub workflow also defines Python 3.11 and 3.12 checks; this local validation record covers Python 3.10.
