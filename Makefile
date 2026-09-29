PYTHON ?= python
.PHONY: check test reproduce conformance quality levels
check: test reproduce conformance quality levels
	$(PYTHON) ops/verify_data.py
test:
	$(PYTHON) -m pytest -q
reproduce:
	$(PYTHON) ops/reproduce_base_grid.py

conformance:
	$(PYTHON) ops/check_conformance.py

quality:
	$(PYTHON) ops/check_quality.py

levels:
	$(PYTHON) ops/summarize_levels.py --check
