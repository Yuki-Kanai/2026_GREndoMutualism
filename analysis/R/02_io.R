## ---------------------------------------------------------------------------
## 02_io.R : reading the simulation output.
##
## LAYOUT OF THE SIMULATION OUTPUT (read-only; see notebooks/00_data_map.ipynb)
##
## Original (cluster) layout, one file per r-block per symbiont-block:
##   <SIM_DIR>/data/raw/<run>/cells_r<RRR>_s<SSSSS>.parquet
##   <SIM_DIR>/data/raw/<run>/vectors_r<RRR>_s<SSSSS>.parquet
##
## Consolidated layout (what this code reads): the raw blocks were merged into
## one parquet per run by simple row concatenation (same columns, "r" still a column) --
##   <RAW_DIR>/<run>_cells.parquet
##   <RAW_DIR>/<run>_vectors.parquet
## so selecting one r is now a row filter on that column instead of a file
## selection. See RAW_DIR in 00_config.R for exactly where <RAW_DIR> points.
##
##   <run>    one simulated experiment, e.g. "pre_paper", "post_paper",
##            "post_paper_h7", "post_paper_m15_rH36"
##
##   cells_*    ONE ROW PER CELL.  A "cell" is one (symbiont R^S, host
##              environment c^H) pair: the unit that carries mu_H, mu_S, s, ...
##              Pre-evolution runs have N_c rows per symbiont (one per sampled
##              c^H); post-evolution runs have exactly one row per symbiont
##              (the evolved c^H*), so c_id is null there.
##   vectors_*  the solution vectors of the same cells (c^H, f^S*, w^S*,
##              bounded_mask) as list columns.  Only needed for Fig.5D/5E.
## ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(tibble)
  library(stringr)
})

## The one consolidated file for a (run, kind). `run` is usually a bare run
## name (resolved under RAW_DIR), but some callers (e.g. r5_block_dir() in
## 00_config.R) already build a fully qualified "<RAW_DIR>/<tag>" path -- keep
## honouring that the way the old per-run-subdirectory run_dir() did.
consolidated_file <- function(run, kind) {
  base <- if (grepl("/", run, fixed = TRUE)) run else file.path(RAW_DIR, run)
  sprintf("%s_%s.parquet", base, kind)
}

## Kept for compatibility with call sites that open the file directly via
## arrow::open_dataset(block_files(run, kind), ...): with consolidated
## storage this is just the one file. `r` no longer selects files -- it is
## applied as a row filter in read_cells()/read_vectors() instead.
block_files <- function(run, kind = c("cells", "vectors"), r = NULL) {
  kind <- match.arg(kind)
  f <- consolidated_file(run, kind)
  if (!file.exists(f)) character(0) else f
}

## The r values present in a run. No longer free (it means reading the
## column), but it is one column of an otherwise untouched dataset.
available_r <- function(run, kind = "cells") {
  f <- block_files(run, kind)
  if (!length(f)) return(integer(0))
  rs <- arrow::open_dataset(f, format = "parquet") |>
    dplyr::select(r) |> dplyr::distinct() |> dplyr::collect() |> dplyr::pull(r)
  sort(unique(as.integer(rs)))
}

## Which runs exist under RAW_DIR (consolidated storage has no per-run
## subdirectories any more -- a run is identified by its "<run>_cells.parquet").
available_runs <- function() {
  f <- list.files(RAW_DIR, pattern = "_cells\\.parquet$", full.names = FALSE)
  sort(sub("_cells\\.parquet$", "", f))
}

## Read cells.  `cols` keeps the transfer small -- always pass it.
read_cells <- function(run, r = NULL, cols = NULL) {
  f <- block_files(run, "cells")
  if (!length(f)) {
    stop(sprintf("no consolidated cells parquet for run '%s'%s", run,
                 if (is.null(r)) "" else sprintf(" at r = %s", paste(r, collapse = ","))))
  }
  r_sel <- r
  ds <- arrow::open_dataset(f, format = "parquet")
  if (!is.null(r_sel)) ds <- dplyr::filter(ds, r %in% r_sel)
  out <- if (is.null(cols)) ds else dplyr::select(ds, dplyr::all_of(cols))
  dplyr::collect(out)
}

## Read solution vectors (list columns).  Always pass a single r: these are big.
read_vectors <- function(run, r, cols = c("r", "symbiont_id", "c_id",
                                          "f_S_star", "bounded_mask")) {
  f <- block_files(run, "vectors")
  if (!length(f)) stop(sprintf("no consolidated vectors parquet for run '%s' at r = %s", run, r))
  r_sel <- r
  arrow::open_dataset(f, format = "parquet") |>
    dplyr::filter(r %in% r_sel) |>
    dplyr::select(dplyr::all_of(cols)) |>
    dplyr::collect()
}

## Columns we actually use.  Everything else stays on disk.
CELL_COLS_PRE <- c("r", "symbiont_id", "c_id", "mu_H_0", "mu_S", "mu_H",
                   "delta_mu_H", "n_bounded", "n_unbounded", "ub", "n_excreted",
                   "r_plus", "n_degenerate", "alpha", "s", "cond_b",
                   "trivial_ub", "flag_infeasible", "c_H_l1")
CELL_COLS_POST <- c("r", "symbiont_id", "m", "r_H", "mu_H_0", "mu_S", "mu_H",
                    "delta_mu_H", "n_bounded", "n_unbounded", "ub", "n_excreted",
                    "r_plus", "alpha", "s", "trivial_ub", "flag_infeasible",
                    "flag_detlimit", "solver_status", "mip_gap")

## An inventory of a run: one row per r, with row counts.
## `n_blocks` is always 1 and `mb` is the whole consolidated file's size
## (repeated per r) now that a run is one file, not one file per r-block.
run_inventory <- function(run, kind = "cells") {
  f <- block_files(run, kind)
  if (!length(f)) return(tibble())
  counts <- arrow::open_dataset(f, format = "parquet") |>
    dplyr::count(r, name = "n_rows") |>
    dplyr::collect() |>
    dplyr::arrange(r)
  tibble(run = run, kind = kind, r = counts$r, n_blocks = 1L,
         n_rows = counts$n_rows, mb = round(file.size(f) / 1024^2, 2))
}

## The ancestral (symbiont-free) host growth rate mu^H_0.  It is stored on every
## cell of the run, so we read it from the data rather than hard-coding it.
read_mu_H_0 <- function(run = "pre_paper", r = 1L) {
  v <- unique(read_cells(run, r, c("mu_H_0"))$mu_H_0)
  stopifnot(length(v) == 1)
  v
}

## Read a table produced by the simulation repo's Python pipeline.
## Used only for cross-checking: no figure in this analysis depends on it.
read_sim_table <- function(name) {
  p <- file.path(SIM_TAB_DIR, name)
  if (!file.exists(p)) return(NULL)
  readr::read_csv(p, show_col_types = FALSE)
}

## Cache helpers -------------------------------------------------------------
cache_path <- function(name) file.path(CACHE_DIR, paste0(name, ".parquet"))
cache_exists <- function(name) file.exists(cache_path(name))
cache_write <- function(df, name) {
  arrow::write_parquet(df, cache_path(name))
  message(sprintf("cache: %s  (%d rows x %d cols)", cache_path(name), nrow(df), ncol(df)))
  invisible(cache_path(name))
}
cache_read <- function(name) {
  p <- cache_path(name)
  if (!file.exists(p)) {
    stop(sprintf("cache '%s' is missing. Run:  Rscript R/10_build_cache.R", name))
  }
  arrow::read_parquet(p)
}
