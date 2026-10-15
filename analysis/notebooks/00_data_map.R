# %% [markdown]
# # 00 — Where the data is
#
# **Purpose.** Before any figure, establish what was simulated, where it is
# stored, and what one row of each file means. Everything below is produced by
# reading the files, not by copying numbers from anywhere else.
#
# **Two directories.**
#
# | | path | role |
# |---|---|---|
# | simulation | `<repo>/simulation` (`SIM_DIR`) | numerical experiments (Python). **Read-only from here.** |
# | analysis | `<repo>/analysis` (`ANA_DIR`) | this notebook, the figures, the statistics tables |
#
# Nothing in this analysis ever writes into the simulation directory.
#
# **Model, in one paragraph.** A host and a symbiont share a pool of `m = 10`
# metabolites (the last one is biomass). The host expresses `r^H = 24`
# regulatory modes, the symbiont `r` of them — `r` is the *symbiont genome
# size*, the x axis of nearly every figure. Given a host environment `c^H`,
# Level 1 gives the host growth `μ^H`, Level 2 the symbiont's optimal exchange
# fluxes `f^S*` and growth `μ^S`, Level 3 (a MILP) lets the **host evolve**
# `c^H` to maximise its own growth. `μ^H_0` is the host growth with no symbiont
# at all: above it the symbiont helps, below it the symbiont is a parasite.

# %%
R_DIR <- if (dir.exists(file.path(getwd(), "..", "R"))) {
  normalizePath(file.path(getwd(), "..", "R"))
} else normalizePath("R")
for (f in c("00_config.R", "01_theme.R", "02_io.R", "03_stats.R")) {
  source(file.path(R_DIR, f))
}
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(purrr)})
options(repr.matrix.max.rows = 40)
fig_size(7, 4)

cat("simulation :", SIM_DIR, "\n")
cat("analysis   :", ANA_DIR, "\n")
cat("R          :", R.version.string, "\n")
cat("arrow      :", as.character(packageVersion("arrow")), "\n")

# %% [markdown]
# ## 1. The runs
#
# `data/raw/<run>/` is one simulated experiment. There are four families:
#
# | run pattern | stage | what it is | used by |
# |---|---|---|---|
# | `pre_paper` | before host evolution | one host, `r = 1..100`, `N_S = 300` symbionts per `r`, `N_c = 100` sampled host environments each | **Fig.3, Fig.5D/5E, Fig.6A/6B** |
# | `post_paper`, `post_paper_h1..h9` | after host evolution | the evolved `c^H*` for each symbiont; `h1..h9` are nine further hosts | **Fig.4, Fig.5A**, and Fig.6C at `m = 10` |
# | `post_paper_m<M>_rH<H>[_h<i>]` | after host evolution | the `m` sweep: one directory per (metabolite count `m`, host genome `r^H`, host) | **Fig.6C** |
# | `pre_*` / `post_fixed_repertoire*`, `pre_k_sweep_*` | controls | referee controls, not part of the main figures | — |

# %%
runs <- available_runs()
fam <- function(x) {
  dplyr::case_when(
    x == "pre_paper" ~ "pre_paper (main, before evolution)",
    grepl("^post_paper(_h[0-9]+)?$", x) ~ "post_paper (main, after evolution)",
    grepl("^post_paper_m", x) ~ "post_paper_m* (m sweep, Fig.6C)",
    TRUE ~ "controls (not in the main figures)")
}
tibble(run = runs, family = fam(runs)) |>
  count(family, name = "n_directories") |>
  arrange(desc(n_directories))

# %% [markdown]
# ## 2. File naming — `r` is in the file name
#
# ```
# data/raw/<run>/cells_r<RRR>_s<SSSSS>.parquet
# data/raw/<run>/vectors_r<RRR>_s<SSSSS>.parquet
#                       ^^^^  ^^^^^^
#                       |     first symbiont id in this block
#                       symbiont genome size r, zero padded to 3 digits
# ```
#
# Because `r` is in the name, asking for one `r` never touches the other 99.
# That is what `read_cells(run, r)` in `R/02_io.R` exploits.

# %%
head(basename(block_files("pre_paper", "cells", r = 10)), 8)

# %% [markdown]
# ## 3. What one row means
#
# * **`cells_*.parquet` — one row per *cell*.** A cell is one
#   (symbiont `R^S`, host environment `c^H`) pair. Before evolution each
#   symbiont contributes `N_c = 100` rows (`c_id = 0..99`); after evolution the
#   host environment is the single optimum `c^H*`, so each symbiont contributes
#   exactly one row and `c_id` is null.
# * **`vectors_*.parquet` — the solution vectors of those same cells**
#   (`c^H`, `f^S*`, `w^S*`, `bounded_mask`) as list columns of length `m-1`.
#   Only Fig.5D/5E needs them.
#
# The columns the figures use:
#
# | column | meaning |
# |---|---|
# | `r` | symbiont genome size (number of regulatory modes it keeps) |
# | `symbiont_id`, `c_id` | which symbiont, which sampled host environment |
# | `mu_H_0` | host growth **without** any symbiont (the reference line) |
# | `mu_H`, `mu_S` | host and symbiont growth for this cell |
# | `delta_mu_H` | `mu_H - mu_H_0` |
# | `n_bounded` (`b`), `n_unbounded` (`u`) | metabolites whose exchange is / is not pinned by the host |
# | `ub` | `u * b` |
# | `n_excreted` | metabolites the symbiont excretes |
# | `alpha`, `s` | `sqrt(b)`, and the **sensitivity** `s` (Frobenius norm of the response matrix) |
# | `trivial_ub`, `flag_infeasible`, `flag_detlimit` | quality flags (see §5) |

# %%
schema_table <- function(path) {
  sch <- arrow::open_dataset(path)$schema
  tibble(column = names(sch),
         type = vapply(sch$fields, function(f) f$type$ToString(), ""))
}
print(schema_table(block_files("pre_paper", "cells", 10)[1]), n = 40)

# %%
schema_table(block_files("pre_paper", "vectors", 10)[1])

# %% [markdown]
# ## 4. How much data, per run

# %%
inv_pre <- run_inventory("pre_paper")
inv_post <- run_inventory("post_paper")
cat(sprintf("pre_paper : %d values of r, %s cells in total, %.0f MB of cells_*\n",
            nrow(inv_pre), format(sum(inv_pre$n_rows), big.mark = ","), sum(inv_pre$mb)))
cat(sprintf("post_paper: %d values of r, %s cells in total, %.1f MB of cells_*\n",
            nrow(inv_post), format(sum(inv_post$n_rows), big.mark = ","), sum(inv_post$mb)))
inv_post |> select(r, n_blocks, n_rows, mb)

# %% [markdown]
# **Read the `post_paper` table above carefully — and expect it to change.**
# The original R4 design grid was
# `r in {1,2,3,5,8,10,12,15,20,25,30,40,60,80,100}` with `N_S = 300` symbionts
# each. Further values of `r` are being added as the simulation runs, with
# their own (smaller) `N_S`, so the grid is uneven and still growing.
#
# The analysis therefore **uses every `r` present on disk by default**
# (`POST_R_SELECTION <- "all"` in `R/00_config.R`) and prints and plots the
# sample size per `r` instead of hiding it. Set
# `POST_R_SELECTION <- "design_grid"` to freeze the analysis back onto the
# original 15 values, which is what the Python pipeline's tables were built on.
#
# After the simulation writes new blocks, refresh the cache with
#
# ```bash
# Rscript R/10_build_cache.R --parts post,r5 --force
# ```

# %% [markdown]
# ## 5. The three quality flags, and what we do with them
#
# | flag | meaning | default treatment |
# |---|---|---|
# | `flag_infeasible` | Level 2 had no feasible solution, so `mu_H` is `NaN` for that cell | **excluded** from figures and statistics; the exclusion rate is reported per `r`, and Fig.3B is also drawn with the paper's convention (`μ^H = μ^S = 0`) for comparison |
# | `flag_detlimit` | the Level 3 MILP stopped at its deterministic time limit and returned a feasible-but-possibly-suboptimal incumbent | **kept**, and Fig.4A is drawn again with these instances removed. They bias `μ^H` *downwards* and happen more often at large `r`, so both versions must be seen |
# | `trivial_ub` | the symbiont has `u*b = 0`, i.e. no non-trivial exchange structure | kept; reported as a fraction |

# %%
fl <- read_cells("pre_paper", c(1, 10, 40, 100),
                 c("r", "flag_infeasible", "trivial_ub")) |>
  group_by(r) |>
  summarise(n = n(), frac_infeasible = mean(flag_infeasible),
            frac_trivial_ub = mean(trivial_ub), .groups = "drop")
fl

# %%
pc <- read_cells("post_paper", NULL, c("r", "flag_detlimit", "solver_status")) |>
  filter(r %in% POST_R_GRID)
cat(sprintf("post_paper on the design grid: %d cells, detlimit %.2f %%\n",
            nrow(pc), 100 * mean(pc$flag_detlimit)))
pc |> count(solver_status)

# %% [markdown]
# ## 6. The ancestral host growth `μ^H_0`
#
# One number sets the meaning of every panel: a symbiont is a **mutualist** for
# this host when `μ^H > μ^H_0` (and `μ^S > 0`), a **parasite** otherwise.
# It is stored on every row, so we read it rather than hard-code it.

# %%
mu_H_0 <- read_mu_H_0("pre_paper")
cat(sprintf("mu_H_0 = %.17g\n", mu_H_0))

# %% [markdown]
# ## 7. The cache this analysis builds
#
# The figure notebooks do not read the 1.6 GB of raw blocks. A single script
#
# ```bash
# Rscript R/10_build_cache.R
# ```
#
# walks the raw output once and writes the small tidy tables below into
# `cache/`. Delete `cache/` and re-run it to rebuild everything from scratch.
#
# | cache file | one row is | used by |
# |---|---|---|
# | `pre_summary` | one (`r`, variable): mean / median / trimmed mean / SD / quartiles | Fig.3A/3B, Fig.6A/6B |
# | `pre_flags` | one `r`: cell counts, infeasible rate, saturation rate | §5, Table R1 |
# | `pre_convention` | one `r`: mean `μ^H` raw vs. paper convention | Fig.3B inset |
# | `pre_points` | one subsampled cell (300 per `r`, fixed seed) | scatter layers |
# | `pre_hdr_points` | one cell of a separate 300-cell draw at `r ∈ {5,10,20,40}` | Fig.3C |
# | `pre_by_symbiont` | one symbiont `R^S`: its summary over the 100 sampled `c^H` | every bootstrap (the resampling unit is the symbiont) |
# | `pre_sd_bu` | one symbiont: `SD_b`, `SD_u` of `f^S*` | Fig.5D/5E |
# | `post_cells` | one evolved symbiont | Fig.4 |
# | `post_join` | one evolved symbiont + its pre-evolution summary | Fig.5A |
# | `r5_cells` | one evolved symbiont of the `m` sweep | Fig.6C |

# %%
if (file.exists(file.path(CACHE_DIR, "MANIFEST.json"))) {
  man <- jsonlite::read_json(file.path(CACHE_DIR, "MANIFEST.json"))
  cat("cache built:", man$built, "\n\n")
  purrr::map_dfr(man$cache_files, function(x) tibble::as_tibble(x))
} else {
  cat("No cache yet. Run:  Rscript R/10_build_cache.R\n")
}

# %% [markdown]
# ## 8. Which notebook makes which figure
#
# | notebook | figure | main data |
# |---|---|---|
# | `01_fig3_before_evolution` | Fig.3A/3B/3C | `pre_summary`, `pre_points`, `pre_hdr_points` |
# | `02_fig4_after_evolution` | Fig.4A/4B/4C | `post_cells` |
# | `03_fig5_mechanism` | Fig.5A/5D/5E | `post_join`, `pre_sd_bu`, `pre_by_symbiont` |
# | `04_fig6_structure_and_m` | Fig.6A/6B/6C | `pre_summary`, `pre_by_symbiont`, `r5_cells` |
#
# Each of them writes `figures/<name>.pdf`, `.png` and a `.meta.json` recording
# the constants it used, plus one or more `tables/<name>.csv`.

# %%
sessionInfo()
