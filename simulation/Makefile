PY := ./.venv/bin/python
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test:
	$(PY) -m pytest tests -q

smoke:
	$(PY) -m grhd.cli smoke
