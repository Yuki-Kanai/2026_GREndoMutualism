# GRHD — simulation and analysis

Code for Y. Kanai and C. Furusawa, *Genome reduction increases endosymbiont
sensitivity and promotes mutualism evolution*. The model and the numerical
and statistical methods are described in the Supporting Information (SI); this
repository implements them.

```
simulation/   Python: the host–endosymbiont model and its solvers (SI S2–S6)
  src/grhd/     model, LP/QP (Levels 1–2), MILP (Level 3), quantities
  config/       base.yaml (all constants and solver settings) + run designs
  scripts/      launchers, summary tables, GA and solver checks
  tests/        validation tests
  tables/       summary tables written by the pipeline
analysis/     R: every simulation figure of the main text and SI (SI S9)
  R/            shared library and cache builder
  notebooks/    one notebook per figure
  input/        evolution of r (SI S5.3, Fig. S7), as a spreadsheet
  tables/       statistics behind each figure
```

`simulation/README.md` maps the SI sections onto the code and lists the run
designs; `analysis/README.md` maps each figure onto its notebook and data.

Not included: the comparative genomics of Fig. 1 (SI S1) and the simulation
code for the evolution of `r` (SI S5.3); the output of the latter is in
`analysis/input/evolution_r_series_v2.xlsx`.

## Simulation output

The output of the simulation (about 2.2 GB of parquet, 616 files, one
`<run>_cells.parquet` and one `<run>_vectors.parquet` per run) is not part of
this repository. The analysis reads it from `simulation/data/consolidated/`;
place the files (or a symlink to them) there, or set `GRHD_RAW_DIR`.

## Quick start

```bash
# simulation (Python 3.11)
cd simulation
python3.11 -m venv .venv && ./.venv/bin/pip install -e '.[milp,dev]'
./.venv/bin/python -m pytest tests -q

# analysis (R 4.3), with the simulation output in place
cd ../analysis
micromamba create -f env/environment.yml && micromamba activate GRHDanalysis
Rscript R/10_build_cache.R
bash run_all.sh
```

## License

MIT, see `LICENSE`.
