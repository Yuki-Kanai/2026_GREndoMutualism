# %% [markdown]
# # 06 — The genetic algorithm against the MILP
#
# **Question.** Fig.4A rests on the Level 3 MILP. On 6 % of cells the MILP
# stops at its deterministic time limit and returns an incumbent that may not
# be optimal — all of them at `r ≥ 22`, i.e. on the **downhill side of the
# peak** (SFig.3). So: does a solver that cannot truncate agree?
#
# **What was run.** The simulation repository solves the host's problem a
# second way, with a genetic algorithm, on the **same symbionts** the MILP
# used (host 0, `k = 4`, `m = 10`, `symbiont_id` 0–99 of `post_paper`). Two
# modes:
#
# | mode | what it evolves |
# |---|---|
# | `ga_full` | `c^H` itself — the pure GA, no structure assumed |
# | `ga_ray` | only the **direction** of `c^H`; the magnitude is line-searched, using the fact that Level 1 is homogeneous in `c^H` |
#
# Because the two solvers ran on the same symbionts, the comparison is
# **paired**: the same instance, two answers, not two distributions.
#
# **What a comparison can and cannot show.** The evidence is asymmetric. A GA
# that finds a *worse* solution says nothing about the MILP — a heuristic is
# allowed to lose. A GA that finds a *better* one is proof that the MILP's
# incumbent was not optimal. So the informative number is **how often, and by
# how much, the GA wins** — separately where the MILP proved optimality and
# where it ran out of time.

# %%
R_DIR <- if (dir.exists(file.path(getwd(), "..", "R"))) {
  normalizePath(file.path(getwd(), "..", "R"))
} else normalizePath("R")
for (f in c("00_config.R", "01_theme.R", "02_io.R", "03_stats.R")) source(file.path(R_DIR, f))
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(purrr); library(patchwork)})
fig_size(7, 3.2)

ga <- cache_read("ga_cells")
mu_H_0 <- ga$mu_H_0[1]
ga |> group_by(method) |>
  summarise(n_cells = dplyr::n(), n_r = dplyr::n_distinct(r),
            r_min = min(r), r_max = max(r),
            nS_per_r = dplyr::n_distinct(symbiont_id),
            core_hours = round(sum(seconds) / 3600, 1), .groups = "drop")

# %%
cat(sprintf("GA budget: pop = %s, generations = %s, %s evaluations per cell (median)\n",
            paste(sort(unique(ga$pop)), collapse = "/"),
            paste(sort(unique(ga$generations)), collapse = "/"),
            format(median(ga$n_eval), big.mark = ",")))
cat(sprintf("paired against the MILP on %d cells; %d of them (%.2f %%) are MILP time-limited\n",
            nrow(ga), sum(ga$flag_detlimit), 100 * mean(ga$flag_detlimit)))

# %% [markdown]
# ## A — the curve, solved both ways
#
# The mean `μ^H` after evolution against `r`, once per solver, plus the
# **cellwise best** of the three (the tightest lower bound the study can
# state). If the peak were an artefact of the MILP, these would not lie on
# top of each other.

# %%
paired <- ga |>
  select(method, r, symbiont_id, mu_H_ga, mu_H_milp, flag_detlimit) |>
  pivot_wider(names_from = method, values_from = mu_H_ga)
has_ray <- "ga_ray" %in% names(paired)

long <- ga |> transmute(r, symbiont_id, method, mu_H = mu_H_ga) |>
  bind_rows(ga |> filter(method == "ga_full") |>
              transmute(r, symbiont_id, method = "milp", mu_H = mu_H_milp)) |>
  bind_rows(paired |>
              transmute(r, symbiont_id, method = "best",
                        mu_H = pmax(mu_H_milp, ga_full,
                                    if (has_ray) ga_ray else -Inf, na.rm = TRUE)))

curves <- long |> group_by(method) |>
  group_modify(~ boot_curve_peak(.x, "mu_H", unit = NULL,
                                 label = paste0("gacurve", .y$method))$curve) |>
  ungroup()
peaks <- long |> group_by(method) |>
  group_modify(~ boot_curve_peak(.x, "mu_H", unit = NULL,
                                 label = paste0("gapeak", .y$method))$peak) |>
  ungroup()
save_tab(curves, "SFig5A_curves")
save_tab(peaks, "SFig5A_peaks")
peaks |> select(method, peak_at, peak_ci_lo, peak_ci_hi, peak_value)

# %% [markdown]
# **The peak is at the same `r` for every solver.** That is the result: the
# main finding does not depend on how the host's problem is solved.

# %%
METH_COL <- c(milp = COL$mean, ga_full = COL$mean2, ga_ray = COL$accent,
              best = "black")
METH_LTY <- c(milp = "solid", ga_full = "22", ga_ray = "42", best = "solid")
## display names only; the tables keep the method codes
METH_LAB <- c(milp = "MILP", ga_full = "GA (full)", ga_ray = "GA (direction)",
              best = "best of three")
pA <- ggplot(curves, aes(r, value, colour = method, linetype = method)) +
  hline_mu0(mu_H_0, label = NULL) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = METH_COL, labels = METH_LAB, breaks = names(METH_LAB), name = NULL) +
  scale_linetype_manual(values = METH_LTY, labels = METH_LAB, breaks = names(METH_LAB), name = NULL) +
  scale_x_r(rmax = max(curves$r)) +
  labs(y = mx("mean"~mu^H~"after evolution")) +
  theme_grhd() + legend_inside(0.99, 0.99) +
  theme(legend.justification = c(1, 1), legend.key.size = unit(2.6, "mm"))
pA

# %% [markdown]
# ## B — the paired difference, cell by cell
#
# `μ^H(GA) − μ^H(MILP)` for the same symbiont. Above zero the GA found
# something the MILP missed. The cells the MILP could not finish are drawn in
# a different colour, because that is where the two are expected to part.

# %%
diffs <- ga |> mutate(d = mu_H_ga - mu_H_milp)
pB <- ggplot(diffs, aes(r, d)) +
  geom_hline(yintercept = 0, colour = COL$ref, linewidth = 0.4) +
  points_cells(filter(diffs, !flag_detlimit), aes(r, d), size = 0.35,
               alpha = 0.3, colour = COL$point) +
  raster_points(geom_point(data = filter(diffs, flag_detlimit), aes(r, d),
                           colour = COL$mean, size = 0.7, alpha = 0.85, shape = 16)) +
  facet_wrap(~ method, ncol = 1, labeller = as_labeller(METH_LAB)) +
  scale_x_r(rmax = max(diffs$r)) +
  labs(y = mx(mu^H*"(GA)"~-~mu^H*"(MILP)")) +
  theme_grhd(BASE_PT_PANEL)
fig_size(6, 5); pB

# %% [markdown]
# ## C — the two regimes, counted
#
# Where the MILP **proved optimality**, a GA win is either numerical noise or
# a sign that the two formulations are not solving quite the same problem.
# Where the MILP **ran out of time**, a GA win is exactly what the bracket in
# SFig.3 predicted.

# %%
win <- diffs |>
  mutate(regime = ifelse(flag_detlimit, "MILP time-limited", "MILP proved optimal")) |>
  group_by(method, regime) |>
  summarise(n = dplyr::n(),
            ga_wins = mean(d > 1e-9), ga_loses = mean(d < -1e-9),
            mean_diff = mean(d), median_diff = median(d),
            max_gain = max(d), max_loss = min(d),
            .groups = "drop")
save_tab(win, "SFig5C_win_rates")
win

# %% [markdown]
# The size of the wins matters as much as the count, so here is the same
# split as distributions rather than means.

# %%
pC <- diffs |>
  mutate(regime = ifelse(flag_detlimit, "MILP time-limited", "MILP optimal")) |>
  ggplot(aes(x = d, fill = regime)) +
  geom_vline(xintercept = 0, colour = COL$ref, linewidth = 0.4) +
  geom_histogram(bins = 60, position = "identity", alpha = 0.7, colour = NA) +
  scale_fill_manual(values = c("MILP optimal" = COL$point,
                               "MILP time-limited" = COL$mean), name = NULL) +
  scale_y_continuous(trans = "log1p", breaks = c(0, 10, 100, 1000)) +
  facet_wrap(~ method, ncol = 1, labeller = as_labeller(METH_LAB)) +
  labs(x = mx(mu^H*"(GA)"~-~mu^H*"(MILP)"), y = "cells") +
  theme_grhd(BASE_PT_PANEL) +
  theme(legend.position = "right", legend.key.size = unit(2.4, "mm")) +
  guides(fill = guide_legend(nrow = 2))
fig_size(6, 5); pC

# %% [markdown]
# ## D — what the correction does to the curve
#
# If the GA is used to repair the cells the MILP could not finish, and the
# MILP is kept everywhere else, how much does the downhill side of Fig.4A
# move? This is the honest version of the curve.

# %%
repaired <- paired |>
  mutate(mu_H_repaired = ifelse(flag_detlimit,
                                pmax(mu_H_milp, ga_full,
                                     if (has_ray) ga_ray else -Inf, na.rm = TRUE),
                                mu_H_milp))
rep_curve <- bind_rows(
  repaired |> transmute(r, symbiont_id, mu_H = mu_H_milp, series = "as reported"),
  repaired |> transmute(r, symbiont_id, mu_H = mu_H_repaired,
                        series = "GA-repaired")) |>
  group_by(series) |>
  group_modify(~ boot_curve_peak(.x, "mu_H", unit = NULL,
                                 label = paste0("rep", .y$series))$curve) |> ungroup()
save_tab(rep_curve, "SFig5D_repaired_curve")
## Name the columns explicitly.  pivot_wider orders them by first appearance,
## and group_by() upstream sorts the groups alphabetically, so addressing them
## by position silently swaps the two series (and flips the sign of the gain).
gapmax <- rep_curve |> select(series, r, value) |>
  pivot_wider(names_from = series, values_from = value) |>
  rename(as_reported = `as reported`, repaired = `GA-repaired`) |>
  mutate(gain = repaired - as_reported)
stopifnot(all(gapmax$gain >= -1e-12))   # a repair can only ever raise the curve
cat(sprintf("largest change to the mean curve: %+.4f at r = %g; mean change over r = %+.5f\n",
            max(gapmax$gain), gapmax$r[which.max(gapmax$gain)], mean(gapmax$gain)))
save_tab(gapmax, "SFig5D_repair_gain")

pD <- ggplot(rep_curve, aes(r, value, colour = series, linetype = series)) +
  hline_mu0(mu_H_0, label = NULL) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = setNames(c(COL$mean, COL$mean2),
                                        sort(unique(rep_curve$series))), name = NULL) +
  scale_linetype_manual(values = setNames(c("solid", "22"),
                                          sort(unique(rep_curve$series))), name = NULL) +
  scale_x_r(rmax = max(rep_curve$r)) +
  labs(y = mx("mean"~mu^H~"after evolution")) +
  theme_grhd() + legend_inside(0.40, 0.99) +
  theme(legend.key.size = unit(2.6, "mm"),
        legend.text = element_text(size = BASE_PT - 2))
pD

# %% [markdown]
# ## Cost, and which solver to use
#
# The simulation repository also timed the two solvers head to head on cells
# where the MILP proves optimality (`tables/GA_validate.csv`).

# %%
val <- read_sim_table("GA_validate.csv")
if (!is.null(val)) {
  vs <- val |> group_by(mode) |>
    summarise(n = dplyr::n(), r = paste(sort(unique(r)), collapse = ","),
              ratio_median = median(ratio), ratio_min = min(ratio),
              frac_ga_beats = mean(ga_beats_milp),
              sec_ga = median(seconds_ga), sec_milp = median(seconds_milp),
              slowdown = median(seconds_ga) / median(seconds_milp), .groups = "drop")
  save_tab(vs, "SFig5_cost")
  print(vs)
  cat(sprintf("\nGA total in the curve run: %.0f core-hours for %d cells; the MILP did the same work in about %.1f core-hours.\n",
              sum(ga$seconds) / 3600, nrow(ga),
              nrow(ga) * median(val$seconds_milp) / 3600))
}

# %% [markdown]
# **Read together:** the GA reproduces the MILP optimum to about five decimal
# places where the MILP can prove optimality, and it costs two to three
# orders of magnitude more time to do it. Its value is not as a replacement —
# it is an independent check, and a repair for the cells the MILP cannot
# finish.

# %% [markdown]
# ## Save

# %%
metaG <- list(data = "<SIM_DIR>/tables/ga_curve (+ GA_validate.csv)",
              cache = "ga_cells", mu_H_0 = mu_H_0,
              n_paired_cells = nrow(ga),
              methods = sort(unique(ga$method)),
              ga_core_hours = round(sum(ga$seconds) / 3600, 1),
              note = "paired comparison: same host, same symbionts, two solvers")
save_fig(pA, "SFig5A_curve_by_solver",   size = "small",  meta = metaG)
save_fig(pB, "SFig5B_paired_difference", size = "medium", meta = metaG)
save_fig(pC, "SFig5C_difference_hist",   size = "medium", meta = metaG)
save_fig(pD, "SFig5D_repaired_curve",    size = "small",  meta = metaG)
## the GA-repaired curve (pD) is kept as its own file but left out of the merged figure
sfig5 <- (pA | pB | pC) + plot_annotation(tag_levels = "A") & tag_theme()
save_fig(sfig5, "SFig5_combined", size = "main_tall", meta = metaG)
fig_size(8, 7); sfig5
