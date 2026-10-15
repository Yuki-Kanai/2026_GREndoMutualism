# %% [markdown]
# # 08 — SFig (draft): does the peak survive a shared reaction direction?
#
# **The objection.** In the model every `R^S` and every `R^H` draws its own
# tournament: for each pair of metabolites the direction `M_i → M_j` or
# `M_j → M_i` is chosen afresh for every individual. A symbiont can therefore
# carry the *reverse* of a host reaction. In real chemistry the direction is
# set by thermodynamics and is shared by everyone. If the host's benefit comes
# from the symbiont supplying directions the host cannot run, a shared
# direction could weaken the complementation, and with it the peak.
#
# **The control.** One tournament per "world" is drawn, and the host and every
# symbiont of that world take their modes from it (`scheme = shared_world`).
# Everything else is held at the main setting: `m = 10`, `k = 4`, `r^H = 24`,
# host window `μ^H_0 ∈ [0.05, 0.10]`, the same MILP and solver settings as A4.
#
# | | run | design |
# |---|---|---|
# | **shared** | `pre_shared_world_w0..19` | Level 1 + 2, `r = 1..100`, 100 symbionts × 10 `c^H` per `r` |
# | | `post_shared_world_w0..19` | MILP, the A4 grid of 15 `r`, 100 symbionts per `r` |
# | **per-individual** (baseline) | `post_paper`, `post_paper_h1..19` | A4: the same 15 `r` × first 100 symbionts, 20 hosts |
#
# One host per world, so the 20 worlds compare one-to-one in design with the
# 20 A4 hosts. The two are **not paired**: a world's host is a different draw
# from the A4 host with the same id.
#
# **What is checked, in order of importance.**
#
# 1. Does the post-evolution `μ^H` still peak at an intermediate `r`, world by
#    world? Same criterion as G7: the peak beats **both** ends with a
#    bootstrap CI that excludes zero, and is not at an end.
# 2. How far does the mutualism fraction fall?
# 3. Does the peak of `s` (before evolution, Fig.6A) still sit near the peak
#    of `μ^H`?
#
# **Note on a tempting argument that does not hold.** With `w ≥ 0` every mode
# lies in the cone of its tournament's reactions, so it is tempting to say a
# shared world caps what the symbiont can add. It does not: a strongly
# connected tournament reaches `j → i` through a path, so its cone is the
# whole plane `1ᵀx = 0`. The effect, if any, is quantitative, not a hard
# ceiling, which is why it has to be measured.
#
# Outputs go to `figures_sw/` and `tables_sw/`, not to `figures/` and
# `tables/`, so this notebook can be merged without touching anything else.
# It is deliberately not in the default list of `run_all.sh`; run it with
# `bash run_all.sh 08`.

# %%
R_DIR <- if (dir.exists(file.path(getwd(), "..", "R"))) {
  normalizePath(file.path(getwd(), "..", "R"))
} else normalizePath("R")
for (f in c("00_config.R", "01_theme.R", "02_io.R", "03_stats.R")) source(file.path(R_DIR, f))
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(purrr); library(patchwork)})
fig_size(7, 3.2)

## Isolated output directories (see the header).  save_fig() / save_tab() read
## these globals at call time.
FIG_DIR <- file.path(ANA_DIR, "figures_sw")
TAB_DIR <- file.path(ANA_DIR, "tables_sw")
for (d in c(FIG_DIR, TAB_DIR)) dir.create(d, showWarnings = FALSE, recursive = TRUE)

SW_WORLDS <- eval(parse(text = Sys.getenv("SW_WORLDS", "0:19")))   # env override = smoke test only
N_S_POST  <- 100L                    # A4 used the first 100 symbionts per r
X_LOG     <- TRUE                    # the post grid is log-spaced; FALSE = linear as in Fig.4A
MUT_DELTA <- 1e-2                    # SFig4C definition: mu_H > mu_H_0 + 0.01 and mu_S > 0.01
MUT_THETA <- 1e-6                    # old Fig.4 definition (mu_S > 1e-6), kept as a check column
SCHEME_LAB <- c(per_individual = "Per-individual direction (main)",
                shared = "Shared direction")
SCHEME_COL <- c(per_individual = COL$mean2, shared = COL$mean)

## The shared-world runs are read from the consolidated one-file-per-run copies
## (<run>_cells.parquet, written and verified by scripts/consolidate_parquet.py
## of the simulation repo).  Locally they sit in RAW_DIR next to every other
## consolidated run; read_cells() is the fallback.
CONS_DIR <- RAW_DIR
cons_path <- function(run) file.path(CONS_DIR, paste0(run, "_cells.parquet"))
run_exists <- function(run) file.exists(cons_path(run)) || length(block_files(run, "cells")) > 0
read_run <- function(run, cols) {
  if (file.exists(cons_path(run))) {
    arrow::read_parquet(cons_path(run), col_select = dplyr::all_of(cols))
  } else read_cells(run, cols = cols)
}

# %% [markdown]
# ## Data
#
# Post-evolution cells for both schemes, cut to the same design. Each row is
# one evolved (host, symbiont) pair; `unit` is the A4 host id or the world id.

# %%
POST_COLS <- c("r", "symbiont_id", "mu_H_0", "mu_H", "mu_S", "s", "n_excreted",
               "flag_detlimit")
sw_post <- map_dfr(SW_WORLDS, function(w) {
  read_run(sprintf("post_shared_world_w%d", w),
           cols = c(POST_COLS, "solver_status", "milp_path")) |>
    mutate(unit = w, .before = 1)
})
pi_post <- cache_read("post_hosts_cells") |>
  filter(r %in% POST_R_GRID, symbiont_id < N_S_POST) |>
  rename(unit = host_id) |>
  select(unit, all_of(POST_COLS))

post <- bind_rows(
  mutate(pi_post, scheme = "per_individual"),
  mutate(select(sw_post, unit, all_of(POST_COLS)), scheme = "shared")
) |>
  mutate(scheme = factor(scheme, levels = names(SCHEME_LAB)),
         mu_H_rel = mu_H / mu_H_0,
         mutualistic = as.numeric(mu_H > mu_H_0 + MUT_DELTA & mu_S > MUT_DELTA))

## Every (scheme, unit, r) must carry exactly N_S_POST symbionts.
cover <- post |> count(scheme, unit, r)
cover |> group_by(scheme) |>
  summarise(units = n_distinct(unit), r_values = n_distinct(r),
            min_n = min(n), max_n = max(n), .groups = "drop")

# %% [markdown]
# ### Solver bookkeeping (always reported)
#
# The time-limit rate per `r` matters because a truncated cell is a **lower
# bound** on `μ^H`. If the shared worlds time out more often in the peak band,
# the comparison is biased against them.

# %%
sw_post |> count(solver_status) |> mutate(frac = n / sum(n))
sw_post |> mutate(milp_path = coalesce(milp_path, "none")) |> count(milp_path)
cat(sprintf("mu_H < mu_H_0 violations (shared): %d of %d\n",
            sum(sw_post$mu_H < sw_post$mu_H_0 - 1e-9), nrow(sw_post)))
detl <- post |> group_by(scheme, r) |>
  summarise(frac_detlimit = mean(flag_detlimit), .groups = "drop")
save_tab(detl, "SW_detlimit_by_r")
detl |> pivot_wider(names_from = scheme, values_from = frac_detlimit) |>
  filter(r >= 8, r <= 40)

# %% [markdown]
# ## 1. The peak of `μ^H`, unit by unit
#
# For each host (per-individual) and each world (shared): the bootstrap peak
# of the mean over symbionts, and whether it survives the G7 criterion. The
# same is done for `μ^H/μ^H_0` (which removes the spread in `μ^H_0` between
# units) and for the mutualism fraction.

# %%
peak_verdict <- function(d, value, label) {
  pk <- boot_curve_peak(d, value, label = label)$peak
  ends <- range(d$r)
  pk |> mutate(beats_both_ends = diff_peak_minus_first_lo > 0 &
                 diff_peak_minus_last_lo > 0,
               interior = !(peak_at %in% ends),
               survives = beats_both_ends & interior)
}
verd <- post |>
  group_by(scheme, unit) |>
  group_modify(function(d, k) {
    bind_rows(lapply(c("mu_H", "mu_H_rel", "mutualistic"), function(v)
      peak_verdict(d, v, paste("SW", k$scheme, k$unit, v, sep = "_")) |>
        mutate(variable = v)))
  }) |>
  ungroup() |>
  left_join(post |> distinct(scheme, unit, mu_H_0), by = c("scheme", "unit"))
save_tab(verd, "SW_peak_by_unit")

verd_sum <- verd |>
  group_by(scheme, variable) |>
  summarise(units = n(), survives = sum(survives),
            r_star_median = median(peak_at), r_star_min = min(peak_at),
            r_star_max = max(peak_at),
            peak_median = median(peak_value),
            peak_minus_worse_end_median = median(pmin(diff_peak_minus_first,
                                                      diff_peak_minus_last)),
            .groups = "drop")
save_tab(verd_sum, "SW_peak_summary")
verd_sum

# %% [markdown]
# ### How much smaller is the peak?
#
# The peak *amplitude* of one unit = peak minus the higher of its two ends
# (the smaller of the two differences). Compared between schemes with medians
# and Cliff's δ (shared vs per-individual; negative = the shared peaks are
# lower). Unpaired, because the units are different draws.

# %%
amp <- verd |>
  mutate(amplitude = pmin(diff_peak_minus_first, diff_peak_minus_last)) |>
  select(scheme, unit, variable, amplitude, peak_value, peak_at)
amp_cmp <- amp |>
  group_by(variable) |>
  summarise(
    median_amp_per_individual = median(amplitude[scheme == "per_individual"]),
    median_amp_shared = median(amplitude[scheme == "shared"]),
    ratio_of_medians = median_amp_shared / median_amp_per_individual,
    cliffs_delta_amp = cliffs_delta(amplitude[scheme == "shared"],
                                    amplitude[scheme == "per_individual"]),
    wilcox_p_amp = suppressWarnings(stats::wilcox.test(
      amplitude[scheme == "shared"], amplitude[scheme == "per_individual"])$p.value),
    median_r_star_per_individual = median(peak_at[scheme == "per_individual"]),
    median_r_star_shared = median(peak_at[scheme == "shared"]),
    .groups = "drop")
save_tab(amp_cmp, "SW_amplitude_comparison")
amp_cmp

# %% [markdown]
# ### Curves: per unit, and the mean over units
#
# The mean over units carries a bootstrap 95 % band in which the **unit** is
# resampled (20 hosts or 20 worlds), so it reflects host-to-host spread, not
# symbiont-to-symbiont spread.

# %%
curves <- post |>
  group_by(scheme, unit, r) |>
  summarise(mu_H = mean(mu_H), mu_H_rel = mean(mu_H_rel),
            mutualistic = mean(mutualistic), s = mean(s),
            n_excreted = mean(n_excreted), mu_H_0 = first(mu_H_0),
            n = dplyr::n(), .groups = "drop")
save_tab(curves, "SW_curves_by_unit")

mean_curves <- curves |>
  pivot_longer(c(mu_H, mu_H_rel, mutualistic, s, n_excreted),
               names_to = "variable", values_to = "value") |>
  group_by(scheme, variable) |>
  group_modify(function(d, k) {
    boot_curve_peak(d, "value", unit = NULL,
                    label = paste("SWmean", k$scheme, k$variable, sep = "_"))$curve
  }) |>
  ungroup()
save_tab(mean_curves, "SW_mean_curves")
mean_curves |> filter(variable %in% c("mu_H_rel", "mutualistic")) |>
  group_by(scheme, variable) |> slice_max(value, n = 1) |> ungroup() |>
  select(scheme, variable, r_at_max = r, max = value, ci_lo, ci_hi)

# %% [markdown]
# ## 2. The mutualism fraction
#
# Fraction of evolved pairs in which both partners grow, `μ^H > μ^H_0 + 0.01`
# and `μ^S > 0.01` (the definition of SFig4C), pooled over units, per `r`, and its maximum over `r`.
# The old Fig.4 definition (`μ^H > μ^H_0` and `μ^S > 10⁻⁶`) is kept as a
# check column.

# %%
mut_pooled <- post |>
  group_by(scheme, r) |>
  summarise(frac_mutualistic = mean(mutualistic),
            frac_mutualistic_fig4 = mean(mu_H > mu_H_0 & mu_S > MUT_THETA),
            .groups = "drop")
save_tab(mut_pooled, "SW_mutualism_by_r")
mut_pooled |> group_by(scheme) |>
  summarise(max_frac = max(frac_mutualistic),
            r_at_max = r[which.max(frac_mutualistic)],
            frac_at_r1 = frac_mutualistic[r == 1],
            frac_at_r100 = frac_mutualistic[r == 100], .groups = "drop")

# %% [markdown]
# ## 3. Sensitivity `s` before evolution (Fig.6A), and whether its peak
# lines up with the peak of `μ^H`
#
# Shared worlds: one `s` curve per world (its symbionts, before evolution).
# Per-individual: `s` before evolution does not depend on the host, so there
# is one curve (the Fig.6A curve, 300 symbionts × 100 `c^H`); it is also shown
# cut to the first 100 symbionts so the sample matches.
#
# `s` is heavy-tailed (see notebook 04), so the median over symbionts is
# reported next to the mean.

# %%
PRE_COLS <- c("r", "symbiont_id", "mu_H_0", "mu_S", "mu_H", "s", "n_excreted",
              "flag_infeasible")
sw_pre_sym <- map_dfr(SW_WORLDS, function(w) {
  read_run(sprintf("pre_shared_world_w%d", w), cols = PRE_COLS) |>
    group_by(r, symbiont_id) |>
    summarise(s_mean = mean(s), n_excreted_mean = mean(n_excreted),
              mu_H_mean = mean(mu_H[!flag_infeasible]),
              n_infeasible = sum(flag_infeasible),
              mu_H_0 = first(mu_H_0), .groups = "drop") |>
    mutate(unit = w, .before = 1)
})
cat(sprintf("pre, shared: %d worlds, %d symbiont rows, Level-2 infeasible cells: %d\n",
            n_distinct(sw_pre_sym$unit), nrow(sw_pre_sym), sum(sw_pre_sym$n_infeasible)))

sw_s <- sw_pre_sym |>
  group_by(unit, r) |>
  summarise(s_median = median(s_mean), s_mean = mean(s_mean),   # median first: summarise() shadows
            mu_H_rel_pre = mean(mu_H_mean / mu_H_0, na.rm = TRUE), .groups = "drop")
pi_sym <- cache_read("pre_by_symbiont")
pi_s <- bind_rows(
  pi_sym |> group_by(r) |>
    summarise(s_median = median(s_mean), s_mean = mean(s_mean), .groups = "drop") |>
    mutate(sample = "all 300 symbionts"),
  pi_sym |> filter(symbiont_id < N_S_POST) |> group_by(r) |>
    summarise(s_median = median(s_mean), s_mean = mean(s_mean), .groups = "drop") |>
    mutate(sample = "first 100 symbionts"))
save_tab(sw_s, "SW_pre_s_by_world")
save_tab(pi_s, "SW_pre_s_per_individual")

## r* of s per world, next to that world's r* of mu_H
s_star <- sw_s |> group_by(unit) |>
  summarise(r_star_s_mean = r[which.max(s_mean)],
            r_star_s_median = r[which.max(s_median)], .groups = "drop") |>
  left_join(verd |> filter(scheme == "shared", variable == "mu_H") |>
              select(unit, r_star_mu_H = peak_at, survives), by = "unit")
save_tab(s_star, "SW_rstar_s_vs_muH")
pi_s_star <- pi_s |> group_by(sample) |>
  summarise(r_star_s_mean = r[which.max(s_mean)],
            r_star_s_median = r[which.max(s_median)], .groups = "drop")
pi_s_star
s_star |> summarise(worlds = n(),
                    median_r_star_s_mean = median(r_star_s_mean),
                    median_r_star_s_median = median(r_star_s_median),
                    median_r_star_mu_H = median(r_star_mu_H),
                    spearman_s_mean_vs_mu_H = cor(r_star_s_mean, r_star_mu_H,
                                                  method = "spearman"))

# %% [markdown]
# ### Before evolution: `μ^H/μ^H_0` per world (for completeness)
#
# Against the A4 Level-2 table of the Python pipeline (per-individual, 20
# hosts). Only a sanity check that the shared worlds are not degenerate
# before evolution; no conclusion rests on it.

# %%
a4_pre <- read_sim_table("A4_pre_level2_by_r.csv")
pre_cmp <- bind_rows(
  sw_s |> group_by(r) |> summarise(mu_H_rel_pre = mean(mu_H_rel_pre)) |>
    mutate(scheme = "shared"),
  if (!is.null(a4_pre)) a4_pre |> group_by(r) |>
    summarise(mu_H_rel_pre = mean(mu_H_mean / mu_H_0, na.rm = TRUE)) |>
    mutate(scheme = "per_individual"))
save_tab(pre_cmp, "SW_pre_muH_rel")
pre_cmp |> filter(r %in% c(1, 5, 10, 15, 25, 50, 100)) |>
  pivot_wider(names_from = scheme, values_from = mu_H_rel_pre)

# %% [markdown]
# ## The figure (draft SFig)
#
# * **A / B** post-evolution `μ^H` against `r`, one thin line per unit and the
#   mean over units in bold; A per-individual (A4 hosts), B shared worlds.
#   Same y axis.
# * **C** `r*` of `μ^H` per unit. Filled = the peak survives the G7
#   criterion, open = it does not; the black bar is the median over units.
#   No legend (state it in the caption); colours are the schemes, as in A / B.
#   The colour key of the whole figure sits inside D.
#
# The x axis is labelled just `r` (italic) everywhere except C.
# * **D** mean over units of `μ^H/μ^H_0`, both schemes, unit-bootstrap band.
# * **E** mutualism fraction, mean over units, both schemes.
# * **F** sensitivity `s` before evolution: shared worlds (thin) and their
#   mean, against the per-individual Fig.6A curve.

# %%
## Formatting follows notebook 05 (SFig1): panel type size BASE_PT_PANEL,
## "mean ..." axis wording, one colour key inside a panel (D), and the y
## title only once per row where the panels share a y axis.
PT <- BASE_PT_PANEL
SCHEME_SHORT <- c(per_individual = "per-individual",
                  shared = "shared")
xr <- scale_x_r(log = X_LOG, name = expression(italic(r)))
ylim_mu <- range(curves$mu_H)
mean_mu <- mean_curves |> filter(variable == "mu_H")
LAB_MEAN_MU_H_POST <- mx("mean"~mu^H~"after evolution")
key_inside <- function(x, y, just = c(0, 1)) {
  legend_inside(x, y) +
    theme(legend.justification = just,
          legend.background = element_blank(),
          legend.key.height = unit(2.6, "mm"))
}

panel_units <- function(sch, ylab = LAB_MEAN_MU_H_POST) {
  ggplot(filter(curves, scheme == sch), aes(r, mu_H, group = unit)) +
    geom_line(colour = SCHEME_COL[[sch]], alpha = 0.35, linewidth = 0.35) +
    geom_line(data = filter(mean_mu, scheme == sch), aes(r, value),
              inherit.aes = FALSE, colour = "black", linewidth = 0.8) +
    xr + coord_cartesian(ylim = ylim_mu) +
    labs(y = ylab, title = paste(SCHEME_SHORT[[sch]], "direction")) +
    theme_grhd(PT) + theme(plot.title = element_text(size = PT, hjust = 0.5))
}
pA <- panel_units("per_individual")
pB <- panel_units("shared", ylab = NULL)        # same y axis as A

## C: r* per unit, one row per scheme.  Filled = survives the G7 criterion,
## open = does not; the black bar is the median over units.  The x range is cut
## to where the r* lie (log axis, as in A / B).
rstar <- verd |> filter(variable == "mu_H")
rstar_med <- rstar |> group_by(scheme) |> summarise(peak_at = median(peak_at), .groups = "drop")
pC <- ggplot(rstar, aes(peak_at, scheme, colour = scheme)) +
  geom_point(aes(shape = survives),
             position = position_jitter(width = 0, height = 0.28, seed = 1),
             size = 1.2, stroke = 0.45) +
  geom_point(data = rstar_med, shape = "|", size = 5, colour = "black") +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1), guide = "none") +
  scale_colour_manual(values = SCHEME_COL, guide = "none") +
  scale_y_discrete(labels = c(per_individual = "per-\nindividual", shared = "shared"),
                   expand = expansion(add = 0.45)) +
  scale_x_log10(name = mx(italic(r)*"* of "*mu^H), breaks = c(5, 10, 20, 40),
                limits = c(3, 45)) +
  annotation_logticks(sides = "b", size = 0.2, short = unit(0.4, "mm"),
                      mid = unit(0.7, "mm"), long = unit(1.1, "mm")) +
  labs(y = NULL) +
  theme_grhd(PT) + theme(axis.text.y = element_text(lineheight = 0.85))

band_panel <- function(var, ylab) {
  d <- filter(mean_curves, variable == var)
  ggplot(d, aes(r, value, colour = scheme, fill = scheme)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), colour = NA, alpha = 0.2) +
    geom_line(linewidth = 0.8) +
    scale_colour_manual(values = SCHEME_COL, labels = SCHEME_SHORT, name = NULL) +
    scale_fill_manual(values = SCHEME_COL, labels = SCHEME_SHORT, name = NULL) +
    xr + labs(y = ylab) + theme_grhd(PT)
}
pD <- band_panel("mu_H_rel", mx("mean"~mu^H/mu[0]^H~"after evolution")) +
  geom_hline(yintercept = 1, colour = COL$ref, linewidth = 0.3) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.22))) +   # room for the key
  key_inside(0.02, 1.04)                         # the one colour key of the figure
pE <- band_panel("mutualistic", "fraction mutualistic") + theme(legend.position = "none")

sw_s_mean <- sw_s |> group_by(r) |> summarise(s_mean = mean(s_mean), .groups = "drop")
ycut_s <- as.numeric(quantile(c(sw_s$s_mean, pi_s$s_mean), 0.99)) * 1.05
pF <- ggplot() +
  geom_line(data = sw_s, aes(r, s_mean, group = unit),
            colour = SCHEME_COL[["shared"]], alpha = 0.25, linewidth = 0.3) +
  geom_line(data = sw_s_mean, aes(r, s_mean, colour = "shared"), linewidth = 0.8) +
  geom_line(data = filter(pi_s, sample == "all 300 symbionts"),
            aes(r, s_mean, colour = "per_individual"), linewidth = 0.8) +
  scale_colour_manual(values = SCHEME_COL, labels = SCHEME_SHORT, name = NULL) +
  xr + coord_cartesian(ylim = c(0, ycut_s)) +
  labs(y = mx("mean"~italic(s)~"before evolution")) +
  theme_grhd(PT) + theme(legend.position = "none")

fig_size(10, 6.5)
pSW <- (pA | pB | pC) / (pD | pE | pF) +
  plot_annotation(tag_levels = "A") & tag_theme()
pSW

# %%
meta_sw <- list(
  scheme = "shared_world (one tournament per world, shared by host and symbionts)",
  worlds = length(SW_WORLDS), N_S_post = N_S_POST, r_grid_post = POST_R_GRID,
  pre_design = "r = 1..100, N_S = 100, N_c = 10",
  baseline = "A4 hosts 0..19, first 100 symbionts, same r grid",
  mutualism = "mu_H > mu_H_0 + 0.01 & mu_S > 0.01 (as SFig4C)",
  x_log = X_LOG,
  survive_criterion = "peak beats both ends (bootstrap CI excludes 0) and is interior")
save_fig(pSW, "SFig_SW_shared_tournament", width = 180, height = 115, meta = meta_sw)
save_fig(pD, "SFig_SW_panel_muH_rel", width = 90, height = 60, meta = meta_sw)

# %% [markdown]
# ## Across `m`: `r` × `μ^H` at `m = 5, 10, 15`
#
# **Question.** Does the shared direction hurt more at small `m`? At `m = 5`
# there are only 10 reaction pairs and the host's 24 modes already use all of
# them, so a symbiont in a shared world can only re-combine host reactions;
# at `m = 15` (105 pairs) the host covers roughly 60 %, and the symbiont has
# more room.
#
# | `m` | per-individual (baseline) | shared |
# |---|---|---|
# | 5 | R5-B `post_paper_m5_rH24[_h1..9]`, 10 hosts | `post_shared_world_m5_rH24_w*`, 10 worlds |
# | 10 | A4, 20 hosts (as above) | 20 worlds (as above) |
# | 15 | R5-B `post_paper_m15_rH24[_h1..9]`, 10 hosts | `post_shared_world_m15_rH24_w*`, 10 worlds |
#
# All at `k = 4`, `r^H = 24`, 100 symbionts per `r`; within one `m` the two
# schemes share the same `r` grid. Only units with all `r` complete are used
# (the count is printed), so a partial run does not bias the curves.
#
# Rows of the figure: per-individual units, shared units (thin line per unit,
# mean over units in black), and the two means overlaid with a
# unit-bootstrap 95 % band. Within one `m` the y axis is shared.

# %%
M_SET <- c(5L, 10L, 15L)
SW_WORLDS_M <- list(`5` = c(0:7, 9, 11), `15` = 0:9)   # tables/SW_worlds_m{5,15}.csv, use == TRUE
M_COLS <- c("r", "symbiont_id", "mu_H_0", "mu_H", "mu_S", "flag_detlimit")

read_units <- function(runs, units, scheme, m) {
  map2_dfr(runs, units, function(run, u) {
    if (!run_exists(run)) return(NULL)
    read_run(run, cols = M_COLS) |>
      filter(symbiont_id < N_S_POST) |>
      mutate(unit = u, scheme = scheme, m = m)
  })
}
## Per-individual baseline at m = 5, 15: R5-B (r^H = 24) hosts 0..9, from the
## analysis cache r5_cells (built by R/10_build_cache.R from post_paper_m*_rH24*).
r5_pi <- cache_read("r5_cells") |>
  filter(design == "B", m %in% c(5L, 15L), host_id <= 9, symbiont_id < N_S_POST) |>
  transmute(r, symbiont_id, mu_H_0, mu_H, mu_S, flag_detlimit, unit = host_id,
            scheme = "per_individual", m = as.integer(m))
pm <- bind_rows(
  filter(r5_pi, m == 5L),
  read_units(sprintf("post_shared_world_m5_rH24_w%d", SW_WORLDS_M[["5"]]),
             SW_WORLDS_M[["5"]], "shared", 5L),
  post |> transmute(r, symbiont_id, mu_H_0, mu_H, mu_S, flag_detlimit, unit,
                    scheme = as.character(scheme), m = 10L),
  filter(r5_pi, m == 15L),
  read_units(sprintf("post_shared_world_m15_rH24_w%d", SW_WORLDS_M[["15"]]),
             SW_WORLDS_M[["15"]], "shared", 15L)
)

## Cells CPLEX still reported infeasible after re-solving with numerical
## emphasis and presolve carry no mu_H. As in SI S5.2 they are discarded, after
## the completeness check below so that one such cell does not drop its unit.
pm |> filter(is.na(mu_H)) |> select(m, scheme, unit, r, symbiont_id, mu_H_0) |> print()

pm <- pm |>
  mutate(scheme = factor(scheme, levels = names(SCHEME_LAB)),
         mutualistic = as.numeric(mu_H > mu_H_0 + MUT_DELTA & mu_S > MUT_DELTA))

## keep a unit only if every r of its m-grid carries N_S_POST symbionts
grid_m <- pm |> distinct(m, scheme, r) |> count(m, scheme, name = "n_r")
complete_units <- pm |> count(m, scheme, unit, r) |>
  group_by(m, scheme, unit) |>
  summarise(n_r = n(), full = all(n == N_S_POST), .groups = "drop") |>
  left_join(grid_m, by = c("m", "scheme"), suffix = c("", "_grid")) |>
  filter(full, n_r == n_r_grid)
pm <- semi_join(pm, complete_units, by = c("m", "scheme", "unit"))
complete_units |> count(m, scheme, name = "units_complete")
cat(sprintf("infeasible cells discarded: %d\n", sum(is.na(pm$mu_H))))
pm <- filter(pm, !is.na(mu_H))

# %%
verd_m <- pm |>
  group_by(m, scheme, unit) |>
  group_modify(function(d, k) {
    bind_rows(lapply(c("mu_H", "mutualistic"), function(v)
      peak_verdict(d, v, paste("SWm", k$m, k$scheme, k$unit, v, sep = "_")) |>
        mutate(variable = v)))
  }) |>
  ungroup()
save_tab(verd_m, "SW_m_peak_by_unit")
verd_m_sum <- verd_m |>
  group_by(m, scheme, variable) |>
  summarise(units = n(), survives = sum(survives),
            r_star_median = median(peak_at), r_star_min = min(peak_at),
            r_star_max = max(peak_at), peak_median = median(peak_value),
            amplitude_median = median(pmin(diff_peak_minus_first, diff_peak_minus_last)),
            .groups = "drop")
save_tab(verd_m_sum, "SW_m_peak_summary")
verd_m_sum |> filter(variable == "mu_H") |> select(-variable)

## shared / per-individual, per m: how much of the peak amplitude is left
verd_m_sum |>
  select(m, scheme, variable, amplitude_median, peak_median) |>
  pivot_wider(names_from = scheme, values_from = c(amplitude_median, peak_median)) |>
  mutate(amplitude_ratio = amplitude_median_shared / amplitude_median_per_individual)

## detlimit rate per m and scheme (a truncated cell is a lower bound on mu_H)
pm |> group_by(m, scheme) |>
  summarise(frac_detlimit = mean(flag_detlimit),
            max_by_r = max(tapply(flag_detlimit, r, mean)), .groups = "drop")

# %%
curves_m <- pm |>
  group_by(m, scheme, unit, r) |>
  summarise(mu_H = mean(mu_H), mutualistic = mean(mutualistic), .groups = "drop")
mean_m <- curves_m |>
  group_by(m, scheme) |>
  group_modify(function(d, k) {
    boot_curve_peak(d, "mu_H", unit = NULL,
                    label = paste("SWm_mean", k$m, k$scheme, sep = "_"))$curve
  }) |>
  ungroup()
save_tab(curves_m, "SW_m_curves_by_unit")
save_tab(mean_m, "SW_m_mean_curves")

## Rows: per-individual units, shared units, both means.  The y title is
## drawn in the first column only (each m keeps its own y range), the x title
## in the bottom row only, and the colour key once, inside G.
panel_m_units <- function(mm, sch, ylim) {
  ggplot(filter(curves_m, m == mm, scheme == sch), aes(r, mu_H, group = unit)) +
    geom_line(colour = SCHEME_COL[[sch]], alpha = 0.4, linewidth = 0.35) +
    geom_line(data = filter(mean_m, m == mm, scheme == sch), aes(r, value),
              inherit.aes = FALSE, colour = "black", linewidth = 0.8) +
    xr + coord_cartesian(ylim = ylim) + theme_grhd(PT)
}
panel_m_means <- function(mm, ylim) {
  ggplot(filter(mean_m, m == mm), aes(r, value, colour = scheme, fill = scheme)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), colour = NA, alpha = 0.2) +
    geom_line(linewidth = 0.8) +
    scale_colour_manual(values = SCHEME_COL, labels = SCHEME_SHORT, name = NULL) +
    scale_fill_manual(values = SCHEME_COL, labels = SCHEME_SHORT, name = NULL) +
    xr + coord_cartesian(ylim = ylim) + theme_grhd(PT)
}
cols_m <- lapply(seq_along(M_SET), function(i) {
  mm <- M_SET[i]
  ylim <- range(filter(curves_m, m == mm)$mu_H)
  ylab <- if (i == 1) LAB_MEAN_MU_H_POST else NULL
  top <- panel_m_units(mm, "per_individual", ylim) +
    labs(y = ylab) + ggtitle(bquote(italic(m) == .(mm))) +
    theme(plot.title = element_text(size = PT, hjust = 0.5),
          axis.title.x = element_blank())       # the scale's name beats labs(x = NULL)
  mid <- panel_m_units(mm, "shared", ylim) + labs(y = ylab) +
    theme(axis.title.x = element_blank())
  bot <- panel_m_means(mm, ylim) + labs(y = ylab) +
    (if (i == 1) key_inside(0.98, 1.02, just = c(1, 1)) else theme(legend.position = "none"))
  list(top, mid, bot)
})
fig_size(10, 8)
pM <- wrap_plots(c(map(cols_m, 1), map(cols_m, 2), map(cols_m, 3)),
                 ncol = 3, byrow = TRUE) +
  plot_annotation(tag_levels = "A") & tag_theme()
pM

# %%
meta_m <- list(
  scheme = "per_individual vs shared_world, m = 5, 10, 15",
  units = complete_units |> count(m, scheme) |> mutate(key = paste(m, scheme)) |>
    with(setNames(as.list(n), key)),
  rows = "1: per-individual units, 2: shared units, 3: both means with unit-bootstrap band",
  baseline = "m = 5, 15: R5-B (r^H = 24) hosts 0..9; m = 10: A4 hosts 0..19",
  N_S_post = N_S_POST, x_log = X_LOG)
save_fig(pM, "SFig_SW_m_comparison", width = 180, height = 150, meta = meta_m)
