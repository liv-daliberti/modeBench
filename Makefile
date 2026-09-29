PYTHON ?= python
.PHONY: check test reproduce conformance
check: test reproduce conformance
	$(PYTHON) ops/verify_data.py
test:
	$(PYTHON) -m pytest -q
reproduce:
	$(PYTHON) ops/reproduce_base_grid.py

conformance:
	$(PYTHON) ops/check_conformance.py
