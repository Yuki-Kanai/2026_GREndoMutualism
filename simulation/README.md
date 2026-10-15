# Simulation (Python)

The host–endosymbiont model of *Genome reduction increases endosymbiont
sensitivity and promotes mutualism evolution*, as described in the SI:

| SI section | here |
|---|---|
| S2 generation of hosts, endosymbionts and internal environments | `src/grhd/generate.py`, `src/grhd/rng.py` |
| S3, S5.1 Levels 1 and 2 (max `μ^S`, min ‖`w^S`‖², max `μ^H`) | `src/grhd/lp.py` |
| S4, S5.2 Level 3 (host evolution as an SOS1 MILP from the KKT conditions) | `src/grhd/milp.py` |
| S6 quantities computed from the solutions | `src/grhd/analyze.py` |
| one run (blocks of `(r, R^S)`, parquet output) | `src/grhd/pipeline.py`, `src/grhd/cli.py` |

Level 4 (evolution of `r`, SI S5.3) and the comparative genomics of SI S1 are
not part of this repository.

## Setup

Python 3.11. The exact package versions used are in `requirements-lock.txt`.

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -e '.[milp,dev]'
./.venv/bin/python -m pytest tests -q
```

Solvers (SI S5): HiGHS 1.15.1 (`highspy`) for the LPs, `quadprog` 0.1.13 for
the minimum-norm QP, and CPLEX 22.2 (`cplex`) for the Level 3 MILP. SCIP
(`pyscipopt`) is used only by the scripts that re-solve instances CPLEX
reported infeasible, and `clarabel`, `osqp` and `piqp` only by the QP backend
benchmarks.

## Parameters

`config/base.yaml` holds every model constant, tolerance and solver setting:
`m = 10`, `k = 4`, `r^H = 24`, `c^E = (1, 0, …, 0)`, `‖c^H‖₁ = 0.01` before
evolution, the host acceptance window `μ^H_0 ∈ [0.05, 0.1]`, the tolerances of
SI S6, the HiGHS options (`ipm`, crossover off, presolve off), and the CPLEX
options (presolve off, deterministic time limit 1.5 × 10⁵ ticks,
`ε = 10⁻⁹`). Every random draw is seeded from `master_seed` and the identity of
the host, endosymbiont or environment (`src/grhd/rng.py`), so results do not
depend on execution order or the number of workers.

A design file, `config/design_*.yaml`, sets what one run computes:

| design | stage | used for |
|---|---|---|
| `design_pre.yaml` | before evolution (Levels 1–2), `r = 1..100`, 300 `R^S` × 100 `c^H` | Fig. 3, 5, 6A/B |
| `design_post_full.yaml` | after evolution (Level 3), `r = 1..100`, 300 `R^S` | Fig. 4, 5; hosts 0–9 of Fig. S2A, S3 |
| `design_a4.yaml` | after evolution, 15 values of `r`, 100 `R^S`, one host per `--host-id` | hosts 10–19 of Fig. S2A; Fig. S9 |
| `design_r5.yaml` | after evolution, `--m`, `--r-h`, `--host-id` | Fig. 6C, S6, S8D |
| `design_post_ksweep.yaml`, `design_a1_ksweep.yaml` | after / before evolution, `--k 2, 8, 16` | Fig. S2C, S8C |
| `design_a1_post.yaml`, `design_a1_fixed.yaml` | after / before evolution, fixed set of `--n-rxn` reactions | Fig. S2D |
| `design_sw_pre.yaml`, `design_sw_post*.yaml` | reaction directions shared by host and endosymbionts, one world per `--host-id` | Fig. S9 |
| `design_post.yaml` | after evolution, 15 values of `r`, 300 `R^S` | first post-evolution run, later extended by `design_post_full.yaml` |

## Running

```bash
export PYTHONPATH=src
./.venv/bin/python -m grhd.cli run --design config/design_pre.yaml --workers 20
./.venv/bin/python -m grhd.cli run --design config/design_post_full.yaml --chunk 10 --workers 20
./.venv/bin/python -m grhd.cli run --design config/design_r5.yaml --m 15 --r-h 24 --host-id 3 --chunk 10
```

A run is split into blocks of `--chunk` endosymbionts at one `r`. Each block
is written to `data/raw/<stage>_<tag>/cells_r<RRR>_s<SSSSS>.parquet` (one row
per host–endosymbiont–environment) and `vectors_r<RRR>_s<SSSSS>.parquet` (the
solution vectors), and is skipped if it already exists, so runs are resumable
and can be split over machines with `--shard i --n-shards n`. The tag is
`<scheme>[_k<k>|_n<n_rxn>][_m<m>_rH<rH>][_h<host>|_w<world>]`, e.g.
`post_paper`, `post_paper_h7`, `post_paper_m15_rH24_h3`, `post_k_sweep_k8`,
`post_shared_world_w12`. Keep `--chunk` the same when extending an existing
run (`post_paper` and `post_paper_h*` used `--chunk 10`). A manifest and log of each
run go to `runs/<run_id>/`.

`scripts/consolidate_parquet.py` merges the blocks of each run into
`data/consolidated/<stage>_<tag>_cells.parquet` and `..._vectors.parquet`, the
layout the R analysis reads.

## Scripts

| | scripts |
|---|---|
| launching runs | `launch.sh`, `run_r5.sh`, `run_r5_par.sh`, `run_a4_milp.sh`, `posthost_queue.sh`, `run_a1_lp.sh`, `run_a1_milp.sh`, `run_sw.sh`, `sw_run_worlds.sh`, `sw_par.sh` |
| genetic algorithms for Level 3 (Fig. S4) | `ga_top_layer.py`, `ga_curve.py`, `ga_curve_summary.py`, `ga_rescue.py`, `ga_ray_queue.sh`, `bench_ga_modes.py` |
| re-solving infeasible instances with SCIP | `a1_milp_rescue.py`, `sw_rescue.py`, `phaseB_scip_crosscheck.py` |
| summary tables (`tables/`) | `make_tables_pre.py`, `make_tables_post.py`, `a4_analyze.py`, `a4_hosts.py`, `a4_pre_level2.py`, `r5_analyze.py`, `sw_worlds.py`, `phaseA_*.py`, `phaseB_*.py`, `a1_lp_peaks.py`, `a1_milp_verdict.py`, `zero_cH_rule_check.py`, `joint_growth_vs_mu0.py`, `mut_vs_mu0.py` |
| solver checks and benchmarks | `bench_*.py`, `r5_bench_milp.py`, `v10_sensitivity.py`, `verify_qp_backend.py`, `qp_backends*.py`, `diag_*.py` |
| data management | `consolidate_parquet.py`, `quarantine_aborted_hosts.py`, `r5_integrity.py` |
| figures and reports of the pipeline itself | `make_figure*.py`, `reports_to_pdf.py` |

`tables/README.md` describes the columns of the main tables. The paper
figures are made by the R analysis in `../analysis`, not by these scripts.

Comments in the code refer to internal design notes (`01_model_spec.md`,
`05_conventions.md` and others) that are not distributed; the SI is the
reference description of the model.
