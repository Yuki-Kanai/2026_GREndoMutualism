# Analysis (R)

Figures and statistics for *Genome reduction increases endosymbiont
sensitivity and promotes mutualism evolution*, computed in R from the output of
the simulation in `../simulation`. The statistical methods are described in
SI Section S9; the quantities analysed (`b`, `u`, `r₊`, `α`, `s`, excreted
metabolites) are defined in SI Section S6.

The analysis only reads the simulation directory and never writes into it.

## Setup

```bash
micromamba create -f env/environment.yml          # R 4.3.3, arrow, tidyverse, ...
micromamba activate GRHDanalysis
Rscript -e 'IRkernel::installspec(name = "ir_grhd", displayname = "R (GRHDanalysis)")'
```

`run_all.sh` expects the environment at `~/miniconda3/envs/GRHDanalysis`; set
`GRHD_ENV_PREFIX` if it is elsewhere.

Paths are set in `R/00_config.R`, relative to this directory:

| | default | override |
|---|---|---|
| simulation code, configs, tables | `../simulation` | `GRHD_SIM_DIR` |
| simulation output (parquet) | `../simulation/data/consolidated` | `GRHD_RAW_DIR` |
| this analysis | `.` | `GRHD_ANA_DIR` |

## Running

```bash
Rscript R/10_build_cache.R     # parquet -> cache/ (about 10 min)
bash run_all.sh                # notebooks 00-07 -> figures/, tables/
bash run_all.sh 08             # notebook 08 (Fig. S9) -> figures_sw/, tables_sw/
bash run_all.sh 02 05          # only the notebooks starting with 02 / 05
```

`R/10_build_cache.R --parts pre,post,...` rebuilds selected parts of the cache
(`pre`, `vectors`, `post`, `r5`, `hosts`, `controls`, `ga`); `--force`
overwrites existing parts. A full `run_all.sh` clears `figures/` and `tables/`
first, so that no figure outlives the code that made it.

Each notebook is kept as `notebooks/NAME.R` (the source, jupytext "percent"
format, runs under `Rscript`). `run_all.sh` converts it to
`notebooks/NAME.ipynb` with `tools/percent2ipynb.py` and executes it.

## Layout

```
R/
  00_config.R        paths and constants
  01_theme.R         figure style; save_fig() writes pdf + png + meta.json
  02_io.R            reading the simulation output
  03_stats.R         bootstrap, correlations, OLS, HDR contours
  10_build_cache.R   reads the parquet once and writes cache/
notebooks/           one notebook per figure (see below)
input/               evolution_r_series_v2.xlsx (input of notebook 07)
tables/              statistics behind each figure, one CSV per table
tables_sw/           statistics of notebook 08
tools/               percent2ipynb.py
env/environment.yml  R environment
cache/, figures/, figures_sw/, logs/   generated, not tracked
```

## Figures

Simulation runs are named `<stage>_<scheme>[_m<m>_rH<rH>][_h<host>|_w<world>]`
(see `../simulation/README.md`). Unless stated otherwise, `m = 10`, `k = 4`,
`r^H = 24`, host 0.

### Main text

| figure | notebook | output | data |
|---|---|---|---|
| Fig. 3A–C | `01_fig3_before_evolution` | `Fig3_combined` | `pre_paper` |
| Fig. 4A–C | `02_fig4_after_evolution` | `Fig4_combined` | `post_paper` |
| Fig. 5A, 5D, 5E | `03_fig5_mechanism` | `Fig5_combined` | `pre_paper`, `post_paper` |
| Fig. 6A, 6B | `04_fig6_structure_and_m` | `Fig6_combined` | `pre_paper` |
| Fig. 6C | `04_fig6_structure_and_m` | `Fig6_combined` | `post_paper[_h1..9]` (m = 10), `post_paper_m<m>_rH24[_h1..9]` |
| evolution of `r` | `07_evolution_dynamics` | `EvoFig1_r_vs_generation_paper_g5000` | `input/evolution_r_series_v2.xlsx` |

Fig. 1 (comparative genomics), Fig. 2 (model overview), and Fig. 5B and 5C
(schematics) are not produced from simulation output.

### Supplementary

| figure | notebook | output | data |
|---|---|---|---|
| S1 | `02_fig4_after_evolution` | `Fig4_parasitic_subset_combined` | `post_paper`, `pre_paper` |
| S2 | `05_supplementary` | `SFig1_combined` | A: `post_paper`, `post_paper_h1..19`; B: computed; C: `post_k_sweep_k{2,8,16}`; D: `post_fixed_repertoire_n40` |
| S3 | `05_supplementary` | `SFig2_raw_muH_nine_hosts` | nine of `post_paper`, `post_paper_h1..9` (the hosts run at `r = 1..100`, 300 `R^S` per `r`) |
| S4 | `06_ga_vs_milp` | `SFig5_combined` | `post_paper`; `../simulation/tables/ga_curve/`, `GA_validate.csv` |
| S5 | `05_supplementary` | `SFig4C_s_gate_and_extent` | `pre_paper`, `post_paper` |
| S6 | `04_fig6_structure_and_m` | `SFig_rstar_vs_m_both_designs` | `post_paper_m<m>_rH24[_h*]`, `post_paper_m<m>_rH<round(2.4m)>[_h1..4]` |
| S7 | `07_evolution_dynamics` | `EvoSFig1_combined2_paper_g5000` | `input/evolution_r_series_v2.xlsx` |
| S8 | `05_supplementary` | `SFig3_combined` | A, B: `post_paper`; C: `post_k_sweep_k*`; D: the `m` sweeps of Fig. 6C and S6 |
| S9 | `08_sfig_shared_tournament` | `SFig_SW_shared_tournament` | `pre/post_shared_world_w0..19`, `post_paper[_h1..19]` |

Output names are the figure files, and the prefix of the table files, in
`figures/` and `tables/`. Notebooks also write additional panels not used in
the paper (`*_extra_*`, `*_check`, single panels of the combined figures).

Notebook `00_data_map` lists the simulation runs on disk, the file schema, and
which notebook uses which run.

## Conventions

As in SI Section S9:

* Curves are the mean over host–endosymbiont pairs at each `r`. The peak `r*`
  is the `r` with the largest mean, ties broken towards the smaller `r`.
* Confidence intervals are percentile bootstrap intervals (`B = 2000`),
  resampling endosymbionts `R^S` with replacement independently within each
  `r`.
* Highest density regions (Fig. 3C, 4C) use a Gaussian KDE with a fixed
  isotropic bandwidth `h = 2×10⁻³` and `0.2` (standard deviation `h/4`,
  `KDE_BW_CONVENTION = "mass_h"`) on a 256 × 256 grid, with contours at 25,
  50 and 75 % of the mass.
* Pairs whose Level 2 problem is infeasible are excluded. Level 3 problems that
  reached the solver time limit are represented by the best solution found;
  Fig. S8 brackets the effect.
* A pair is mutualistic when `μ^H > μ^H_0 + 10⁻²` and `μ^S > 10⁻²` after host
  evolution.

All random operations (subsampling, bootstrap, jitter) are seeded from
`MASTER_SEED` in `R/00_config.R`. Each figure is written with a `.meta.json`
recording its source data, constants and sample sizes, and
`cache/MANIFEST.json` records what the cache was built from.

## Figure style

`R/01_theme.R` sets the style of every figure: Okabe–Ito colours, PNAS
figure sizes (`save_fig(..., size = "small" | "medium" | "main" | ...)`), and
italic variables with upright descriptive superscripts (`μ^H`). The font is the
first installed family in `FONT_PREFS` that has an italic face covering Greek
(Arial, then Liberation Sans, ...); the family used is recorded in each
`.meta.json`.
