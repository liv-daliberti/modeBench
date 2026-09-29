PYTHON ?= python
.PHONY: check test reproduce
check: test reproduce
	$(PYTHON) ops/verify_data.py
test:
	$(PYTHON) -m pytest -q
reproduce:
	$(PYTHON) ops/reproduce_base_grid.py
