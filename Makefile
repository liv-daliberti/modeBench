PYTHON ?= python
.PHONY: check test reproduce conformance quality
check: test reproduce conformance quality
	$(PYTHON) ops/verify_data.py
test:
	$(PYTHON) -m pytest -q
reproduce:
	$(PYTHON) ops/reproduce_base_grid.py

conformance:
	$(PYTHON) ops/check_conformance.py

quality:
	$(PYTHON) ops/check_quality.py
