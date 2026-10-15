#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## 10_build_cache.R : turn the raw per-cell parquet blocks into a handful of
## small, tidy tables that the figure notebooks can open in a second.
##
##   Rscript R/10_build_cache.R                 # everything that is missing
##   Rscript R/10_build_cache.R --parts pre     # one part only
##   Rscript R/10_build_cache.R --force         # rebuild even if present
##
## Parts
##   pre      data/raw/pre_paper      -> pre_*       (Fig.3, Fig.6A/6B)
##   vectors  data/raw/pre_paper      -> pre_sd_bu   (Fig.5D/5E)
##   post     data/raw/post_paper     -> post_*      (Fig.4, Fig.5A)
##   r5       data/raw/post_paper_m*  -> r5_cells    (Fig.6C)
##   hosts    data/raw/post_paper_h*  -> post_hosts_cells        (supplement)
##   controls data/raw/pre_k_sweep_*, *_fixed_repertoire_*
##                                    -> ksweep_*, fixedrep_*    (supplement)
##   ga       <SIM_DIR>/tables/ga_curve -> ga_cells              (supplement)
##
## Nothing here is written outside <ANA_DIR>/cache.
## ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(tibble); library(purrr)
  library(arrow); library(readr); library(stringr); library(jsonlite)
})

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "R"
source(file.path(here, "00_config.R"))
source(file.path(here, "02_io.R"))
source(file.path(here, "03_stats.R"))

argv  <- commandArgs(TRUE)
FORCE <- "--force" %in% argv
PARTS <- if ("--parts" %in% argv) {
  strsplit(argv[which(argv == "--parts") + 1], ",")[[1]]
} else c("pre", "vectors", "post", "r5", "hosts", "controls", "ga")

need <- function(name) FORCE || !cache_exists(name)
say  <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))

MANIFEST <- list()
note <- function(key, value) MANIFEST[[key]] <<- value

## ===========================================================================
## PART "pre"  --  data/raw/pre_paper, one row per (symbiont, c^H) cell
## ===========================================================================
if ("pre" %in% PARTS && (need("pre_summary") || need("pre_points") ||
                         need("pre_by_symbiont") || need("pre_flags"))) {
  say("pre: scanning data/raw/pre_paper ...")
  rs <- available_r("pre_paper")
  stopifnot(length(rs) > 0)

  sum_l <- pts_l <- hdr_l <- sym_l <- flag_l <- conv_l <- corr_l <- vector("list", length(rs))
  pooled_mu <- vector("list", length(rs))   # slim (r, mu_S, mu_H) for pooled rho

  num_cols <- c("mu_S", "mu_H", "delta_mu_H", "s", "alpha", "n_bounded",
                "n_unbounded", "ub", "n_excreted", "r_plus", "n_degenerate",
                "cond_b")

  for (i in seq_along(rs)) {
    rr <- rs[i]
    d <- read_cells("pre_paper", rr, CELL_COLS_PRE)
    infeas <- d$flag_infeasible
    ok <- !infeas
    v <- d[ok, , drop = FALSE]

    ## ---- descriptive statistics, infeasible cells excluded ---------------
    sum_l[[i]] <- map_dfr(num_cols, ~ summarise_vec(v[[.x]]) |>
                            mutate(variable = .x, .before = 1)) |>
      mutate(r = rr, .before = 1)

    ## ---- flags and rates --------------------------------------------------
    sat_tol <- TOL_ABS + TOL_REL * d$c_H_l1
    flag_l[[i]] <- tibble(
      r = rr,
      n_cells = nrow(d),
      n_symbionts = dplyr::n_distinct(d$symbiont_id),
      n_c = dplyr::n_distinct(d$c_id),
      mu_H_0 = d$mu_H_0[1],
      n_infeasible = sum(infeas),
      frac_infeasible = mean(infeas),
      frac_saturated = mean(abs(d$mu_S - d$c_H_l1) <= sat_tol, na.rm = TRUE),
      frac_trivial_ub = mean(v$trivial_ub),
      frac_mu_H_above_0 = mean(v$mu_H > d$mu_H_0[1])
    )

    ## ---- the paper's convention: infeasible -> mu_H = mu_S = 0 -----------
    conv_l[[i]] <- tibble(
      r = rr,
      mu_H_mean_raw = mean(v$mu_H), mu_S_mean_raw = mean(v$mu_S),
      mu_H_mean_convention = mean(ifelse(infeas, 0, d$mu_H)),
      mu_S_mean_convention = mean(ifelse(infeas, 0, d$mu_S))
    )

    ## ---- within-r correlations (on all valid cells of this r) ------------
    corr_l[[i]] <- bind_rows(
      corr_one(v$mu_H, v$mu_S, "spearman") |> mutate(method = "spearman"),
      corr_one(v$mu_H, v$mu_S, "pearson")  |> mutate(method = "pearson")
    ) |> mutate(r = rr, .before = 1)

    ## ---- scatter subsample, and a separate draw for the HDR panels -------
    set.seed(seed_from("pre_scatter", rr))
    pts_l[[i]] <- v[sort(sample.int(nrow(v), min(SCATTER_N, nrow(v)))), ] |>
      select(r, symbiont_id, c_id, mu_S, mu_H, delta_mu_H, s, ub,
             n_bounded, n_excreted, alpha, trivial_ub)
    if (rr %in% HDR_R_VALUES) {
      set.seed(seed_from("pre_hdr", rr))
      hdr_l[[i]] <- v[sort(sample.int(nrow(v), min(SCATTER_N, nrow(v)))), ] |>
        select(r, symbiont_id, c_id, mu_S, mu_H, delta_mu_H)
    }

    ## ---- per-symbiont aggregates (the resampling unit of every bootstrap)
    sym_l[[i]] <- v |>
      group_by(r, symbiont_id) |>
      summarise(
        n_c_valid = dplyr::n(),
        mu_H_max = max(mu_H), mu_H_median = median(mu_H), mu_H_mean = mean(mu_H),
        mu_S_mean = mean(mu_S), mu_S_max = max(mu_S),
        delta_mu_H_mean = mean(delta_mu_H),
        s_mean = mean(s), s_median = median(s),
        alpha_mean = mean(alpha),
        b_mean = mean(n_bounded), u_mean = mean(n_unbounded),
        ub_mean = mean(ub), n_excreted_mean = mean(n_excreted),
        r_plus_mean = mean(r_plus), cond_b_median = median(cond_b),
        frac_trivial_ub = mean(trivial_ub),
        .groups = "drop") |>
      mutate(
        mu_H_0 = d$mu_H_0[1],
        ## "parasitic before host evolution": the best of the N_c sampled host
        ## environments still leaves the host worse off than with no symbiont
        parasitic_pre = mu_H_max < d$mu_H_0[1],
        trivial_pre = frac_trivial_ub > 0.5)

    pooled_mu[[i]] <- tibble(r = rr, mu_H = v$mu_H, mu_S = v$mu_S)
    if (rr %% 20 == 0) say("pre:   r = %d done (%d of %d)", rr, i, length(rs))
  }

  cache_write(bind_rows(sum_l), "pre_summary")
  cache_write(bind_rows(flag_l), "pre_flags")
  cache_write(bind_rows(conv_l), "pre_convention")
  cache_write(bind_rows(pts_l), "pre_points")
  cache_write(bind_rows(hdr_l), "pre_hdr_points")
  cache_write(bind_rows(sym_l), "pre_by_symbiont")

  ## pooled correlation over every valid cell of every r -------------------
  say("pre: pooled correlations over all cells ...")
  pm <- bind_rows(pooled_mu)
  pooled <- bind_rows(
    corr_one(pm$mu_H, pm$mu_S, "spearman") |> mutate(method = "spearman"),
    corr_one(pm$mu_H, pm$mu_S, "pearson")  |> mutate(method = "pearson")
  ) |> mutate(r = NA_integer_, .before = 1)
  cache_write(bind_rows(mutate(pooled, scope = "pooled"),
                        mutate(bind_rows(corr_l), scope = "within_r")),
              "pre_correlations")
  note("pre_n_cells", nrow(pm))
  rm(pm, pooled_mu); invisible(gc())
}

## ===========================================================================
## PART "vectors"  --  SD_b / SD_u per symbiont, for Fig.5D / 5E
## ===========================================================================
## For one symbiont R^S, the N_c sampled host environments give N_c optimal
## exchange flux vectors f^S*.  The paper compares the SPREAD of those vectors
## with alpha and with s.  Both alpha and s are norms, so the spread must be a
## norm too: we use the root-sum-of-variances over the bounded (resp. the
## unbounded) components.  Two classifications are stored:
##   SD_b / SD_u        a component counts as bounded for this symbiont if it
##                      was bounded in more than half of its N_c cells
##                      (majority vote); variance over all N_c cells, 1/(N_c-1);
##   SD_b_pc / SD_u_pc  per composition: each component's variance (1/#K) is
##                      taken over the cells in which it is bounded (resp.
##                      unbounded) only -- the definition used in the paper.
if ("vectors" %in% PARTS && need("pre_sd_bu")) {
  say("vectors: computing SD_b / SD_u from f^S* ...")
  rs <- available_r("pre_paper", "vectors")
  out <- vector("list", length(rs))
  for (i in seq_along(rs)) {
    rr <- rs[i]
    d <- read_vectors("pre_paper", rr)
    d <- d[order(d$symbiont_id, d$c_id), ]
    nc <- length(d$f_S_star[[1]])
    F  <- matrix(unlist(d$f_S_star, use.names = FALSE), ncol = nc, byrow = TRUE)
    Bm <- matrix(as.numeric(unlist(d$bounded_mask, use.names = FALSE)),
                 ncol = nc, byrow = TRUE)
    sid <- d$symbiont_id
    n   <- as.numeric(table(sid))
    s1  <- rowsum(F, sid); s2 <- rowsum(F^2, sid)
    vr  <- (s2 - s1^2 / n) / (n - 1)                  # per-component variance
    bmaj <- (rowsum(Bm, sid) / n) > 0.5               # majority-vote bounded set
    ## Per-composition classification: for component j, the variance (1/#K) is
    ## taken only over the c^H in which j is bounded (resp. unbounded); a
    ## component never in that class contributes 0.
    pc_var <- function(M) {
      nk <- rowsum(M, sid)
      m1 <- rowsum(F * M, sid) / pmax(nk, 1)
      m2 <- rowsum(F^2 * M, sid) / pmax(nk, 1)
      ifelse(nk > 0, pmax(m2 - m1^2, 0), 0)
    }
    out[[i]] <- tibble(
      r = rr,
      symbiont_id = as.integer(rownames(s1)),
      n_c = n,
      SD_b = sqrt(rowSums(vr * bmaj)),
      SD_u = sqrt(rowSums(vr * !bmaj)),
      b_majority = rowSums(bmaj),
      SD_b_pc = sqrt(rowSums(pc_var(Bm))),
      SD_u_pc = sqrt(rowSums(pc_var(1 - Bm))))
    if (rr %% 20 == 0) say("vectors: r = %d done (%d of %d)", rr, i, length(rs))
  }
  cache_write(bind_rows(out), "pre_sd_bu")
}

## ===========================================================================
## PART "post"  --  data/raw/post_paper, one row per evolved symbiont
## ===========================================================================
if ("post" %in% PARTS && (need("post_cells") || need("post_join"))) {
  say("post: reading data/raw/post_paper ...")
  d <- read_cells("post_paper", NULL, CELL_COLS_POST) |>
    mutate(design_grid = r %in% POST_R_GRID)
  cache_write(d, "post_cells")

  sym <- cache_read("pre_by_symbiont")
  j <- d |>
    select(r, symbiont_id, mu_H_post = mu_H, mu_S_post = mu_S, mu_H_0,
           n_bounded, n_excreted, ub, s, alpha, trivial_ub,
           flag_detlimit, design_grid) |>
    left_join(sym |> select(r, symbiont_id, mu_H_pre_max = mu_H_max,
                            mu_H_pre_median = mu_H_median,
                            mu_H_pre_mean = mu_H_mean,
                            ub_pre_mean = ub_mean, s_pre_mean = s_mean,
                            alpha_pre_mean = alpha_mean, b_pre_mean = b_mean,
                            n_excreted_pre_mean = n_excreted_mean,
                            parasitic_pre, trivial_pre),
              by = c("r", "symbiont_id"))
  cache_write(j, "post_join")
  note("post_n_cells", nrow(d))
  note("post_n_unmatched_pre", sum(is.na(j$mu_H_pre_max)))
}

## ===========================================================================
## PART "r5"  --  Fig.6C: r* against the number of metabolites m
## ===========================================================================
## Two host designs (08_r5_two_designs.md of the simulation repo):
##   A  r^H = round(2.4 m)  -- host ability held fixed relative to m
##   B  r^H = 24            -- the setting of the original paper
## They coincide at m = 10, where the blocks are the main post_paper run.
if ("r5" %in% PARTS && need("r5_cells")) {
  say("r5: walking data/raw/post_paper_m* ...")
  dirs <- available_runs()
  mm <- str_match(dirs, "^post_paper_m([0-9]+)_rH([0-9]+)(_h([0-9]+))?$")
  present <- tibble(dir = dirs, m = as.integer(mm[, 2]), r_H = as.integer(mm[, 3]))
  present <- present[!is.na(present$m), ]
  ms_all <- sort(unique(present$m))

  rows <- list(); k <- 0L
  for (design in names(R5_N_HOSTS)) {
    ms <- sort(unique(c(ms_all, 10L)))
    for (m in ms) {
      rH <- r_H_of(design, m)
      if (m != 10L && !any(present$m == m & present$r_H == rH)) next
      for (host in seq_len(R5_N_HOSTS[[design]]) - 1L) {
        p <- r5_block_dir(design, m, host)
        f <- block_files(p, "cells")
        if (!length(f)) next
        dd <- arrow::open_dataset(f, format = "parquet") |>
          select(r, symbiont_id, m, r_H, mu_H_0, mu_H, mu_S, n_bounded,
                 n_excreted, s, flag_detlimit, flag_infeasible) |>
          collect() |>
          mutate(design = design, host_id = host, .before = 1)
        k <- k + 1L; rows[[k]] <- dd
      }
      say("r5: design %s, m = %d done", design, m)
    }
  }
  r5 <- bind_rows(rows)
  cache_write(r5, "r5_cells")
  note("r5_n_cells", nrow(r5))
}

## ===========================================================================
## PART "hosts"  --  the same experiment as Fig.4, for every sampled host
## ===========================================================================
## data/raw/post_paper (host 0) and post_paper_h1..hN, all at m = 10, k = 4,
## r^H = 24.  Supplementary figure: is the peak a property of the model or of
## the one host that Fig.4 happens to use?
if ("hosts" %in% PARTS && need("post_hosts_cells")) {
  say("hosts: reading post_paper and post_paper_h* ...")
  dirs <- c("post_paper", grep("^post_paper_h[0-9]+$", available_runs(), value = TRUE))
  out <- list()
  for (d in dirs) {
    h <- if (d == "post_paper") 0L else as.integer(sub(".*_h", "", d))
    f <- block_files(d, "cells")
    if (!length(f)) next
    out[[d]] <- arrow::open_dataset(f, format = "parquet") |>
      select(r, symbiont_id, m, r_H, mu_H_0, mu_H, mu_S, delta_mu_H,
             n_bounded, n_excreted, ub, s, alpha, r_plus, flag_detlimit,
             flag_infeasible) |>
      collect() |>
      mutate(host_id = h, .before = 1)
  }
  hosts <- bind_rows(out)
  cache_write(hosts, "post_hosts_cells")
  note("n_hosts_m10", dplyr::n_distinct(hosts$host_id))
}

## ===========================================================================
## PART "controls"  --  the two generation schemes that separate "control
## granularity" from "metabolic repertoire"  (referee point M2)
## ===========================================================================
##   k_sweep           k reactions per mode, k in {2, 4, 8, 16}; k = 4 is the
##                     main run, so pre_paper is folded in as k = 4.
##   fixed_repertoire  a reaction set Omega of fixed size n_rxn is split into
##                     r blocks, one per mode.  The repertoire is then CONSTANT
##                     in r and only the granularity changes -- the "no fixed
##                     k, redistribute the reactions" design.
PRE_CTRL_COLS <- c("r", "symbiont_id", "mu_H_0", "mu_S", "mu_H", "delta_mu_H",
                   "n_bounded", "n_unbounded", "ub", "n_excreted", "r_plus",
                   "alpha", "s", "trivial_ub", "flag_infeasible")

## Per-symbiont summary of one pre-evolution run, r by r.
pre_symbionts_of <- function(run) {
  rs <- available_r(run)
  map_dfr(rs, function(rr) {
    d <- read_cells(run, rr, PRE_CTRL_COLS)
    v <- d[!d$flag_infeasible, , drop = FALSE]
    if (!nrow(v)) return(NULL)
    v |>
      group_by(r, symbiont_id) |>
      summarise(n_c_valid = dplyr::n(),
                mu_H_max = max(mu_H), mu_H_mean = mean(mu_H),
                mu_S_mean = mean(mu_S), delta_mu_H_mean = mean(delta_mu_H),
                s_mean = mean(s), s_median = median(s), alpha_mean = mean(alpha),
                b_mean = mean(n_bounded), u_mean = mean(n_unbounded),
                ub_mean = mean(ub), n_excreted_mean = mean(n_excreted),
                r_plus_mean = mean(r_plus), frac_trivial_ub = mean(trivial_ub),
                .groups = "drop") |>
      mutate(mu_H_0 = d$mu_H_0[1], frac_infeasible = mean(d$flag_infeasible))
  })
}

if ("controls" %in% PARTS && need("ksweep_by_symbiont")) {
  say("controls: k sweep ...")
  ks <- c(pre_paper = 4L, pre_k_sweep_k2 = 2L, pre_k_sweep_k8 = 8L,
          pre_k_sweep_k16 = 16L)
  ks <- ks[names(ks) %in% available_runs()]
  out <- imap(ks, function(kk, run) {
    say("controls:   %s (k = %d)", run, kk)
    pre_symbionts_of(run) |> mutate(k = kk, run = run, .before = 1)
  })
  cache_write(bind_rows(out), "ksweep_by_symbiont")
}

if ("controls" %in% PARTS && need("fixedrep_by_symbiont")) {
  say("controls: fixed repertoire (pre) ...")
  runs <- grep("^pre_fixed_repertoire_n[0-9]+$", available_runs(), value = TRUE)
  out <- map(runs, function(run) {
    n_rxn <- as.integer(sub(".*_n", "", run))
    say("controls:   %s (n_rxn = %d)", run, n_rxn)
    pre_symbionts_of(run) |> mutate(n_rxn = n_rxn, run = run, .before = 1)
  })
  cache_write(bind_rows(out), "fixedrep_by_symbiont")
}

POST_CTRL_COLS <- c("r", "symbiont_id", "m", "r_H", "mu_H_0", "mu_H", "mu_S",
                    "delta_mu_H", "n_bounded", "n_excreted", "ub", "s", "alpha",
                    "r_plus", "flag_detlimit", "mip_gap")

if ("controls" %in% PARTS && need("fixedrep_post_cells")) {
  runs <- grep("^post_fixed_repertoire_n[0-9]+$", available_runs(), value = TRUE)
  if (length(runs)) {
    say("controls: fixed repertoire (post) ...")
    out <- map(runs, function(run) {
      arrow::open_dataset(block_files(run, "cells"), format = "parquet") |>
        select(dplyr::all_of(POST_CTRL_COLS)) |>
        collect() |>
        mutate(n_rxn = as.integer(sub(".*_n", "", run)), run = run, .before = 1)
    })
    cache_write(bind_rows(out), "fixedrep_post_cells")
  }
}

## The k sweep AFTER host evolution.  This did not exist when the supplement
## was first written; k = 4 is the main post_paper run, folded in here so the
## four values of k sit in one table.
if ("controls" %in% PARTS && need("ksweep_post_cells")) {
  runs <- grep("^post_k_sweep_k[0-9]+$", available_runs(), value = TRUE)
  if (length(runs)) {
    say("controls: k sweep (post) ...")
    out <- map(runs, function(run) {
      arrow::open_dataset(block_files(run, "cells"), format = "parquet") |>
        select(dplyr::all_of(POST_CTRL_COLS)) |>
        collect() |>
        mutate(k = as.integer(sub(".*_k", "", run)), run = run, .before = 1)
    })
    main <- arrow::open_dataset(block_files("post_paper", "cells"), format = "parquet") |>
      select(dplyr::all_of(POST_CTRL_COLS)) |> collect() |>
      mutate(k = K_REACTIONS, run = "post_paper", .before = 1)
    cache_write(bind_rows(c(out, list(main))), "ksweep_post_cells")
  }
}

## ===========================================================================
## PART "ga"  --  the genetic-algorithm solution of Level 3
## ===========================================================================
## The simulation repository solves the host's problem a second way, with a GA
## instead of the MILP, on the SAME symbionts (host 0, k = 4, symbiont_id
## 0..99 of post_paper).  Two modes:
##   ga_full  evolves c^H itself -- no structure assumed
##   ga_ray   evolves only the direction of c^H and line-searches the
##            magnitude, using the homogeneity of Level 1 in c^H
## One CSV per (mode, r) under <SIM_DIR>/tables/ga_curve.  Joining them to the
## MILP cells gives a PAIRED comparison: same symbiont, two solvers.
if ("ga" %in% PARTS && need("ga_cells")) {
  gdir <- file.path(SIM_TAB_DIR, "ga_curve")
  if (!dir.exists(gdir)) {
    say("ga: %s not found - skipped", gdir)
  } else {
    say("ga: reading %s ...", gdir)
    f <- list.files(gdir, pattern = "\\.csv$", full.names = TRUE)
    ga <- map_dfr(f, ~ readr::read_csv(.x, show_col_types = FALSE))
    milp <- read_cells("post_paper", NULL,
                       c("r", "symbiont_id", "mu_H", "flag_detlimit", "mip_gap",
                         "mu_H_0", "solver_status")) |>
      rename(mu_H_milp = mu_H)
    ga <- ga |>
      select(method, r, symbiont_id, mu_H_ga = mu_H, c_H_l1, seconds, n_eval,
             pop, generations) |>
      inner_join(milp, by = c("r", "symbiont_id"))
    cache_write(ga, "ga_cells")
    note("ga_n_cells", nrow(ga))
    note("ga_core_hours", round(sum(ga$seconds) / 3600, 2))
  }
}

## ===========================================================================
## manifest
## ===========================================================================
man <- c(
  list(built = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
       sim_dir = SIM_DIR, ana_dir = ANA_DIR, parts = PARTS,
       R_version = paste(R.version$major, R.version$minor, sep = "."),
       arrow_version = as.character(utils::packageVersion("arrow"))),
  MANIFEST,
  list(cache_files = map(sort(list.files(CACHE_DIR, "\\.parquet$", full.names = TRUE)),
                         ~ list(file = basename(.x),
                                mb = round(file.size(.x) / 1024^2, 3),
                                rows = nrow(arrow::open_dataset(.x))))))
write_json(man, file.path(CACHE_DIR, "MANIFEST.json"), auto_unbox = TRUE, pretty = TRUE)
say("done. manifest -> %s", file.path(CACHE_DIR, "MANIFEST.json"))
