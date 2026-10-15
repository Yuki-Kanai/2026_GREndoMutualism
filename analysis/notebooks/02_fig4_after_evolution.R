# %% [markdown]
# # 02 — Fig.4: the main result. Host growth *after* the host evolves
#
# **Question.** If the host is allowed to optimise its own environment `c^H`
# against a given symbiont, how good can it get — and does that depend on the
# symbiont's genome size `r`?
#
# **The claim this figure has to support** is that the mean `μ^H` after host
# evolution is **largest at an intermediate `r`**: neither a nearly empty
# symbiont nor a nearly complete one is the best partner. Everything in
# notebooks 03 and 04 exists to explain *why* that peak is there; this
# notebook is where the peak itself is measured.
#
# **Panels.**
#
# * **4A** `μ^H` after evolution against `r`: cells, the mean (vermillion)
#   with a bootstrap 95 % band, and the grey symbiont-free line `μ^H_0`.
# * **4B** `μ^S` after evolution against `r`, drawn **raw** (no rewriting of
#   `μ^S` to 0).
# * **4C** `μ^H` against `μ^S` at `r ∈ {5, 10, 20, 40}` in a 2 × 2 grid, with
#   HDR bands.
#
# **Data.** `data/raw/post_paper`: one host, one evolved `c^H*` per symbiont.
# **The `r` grid is still being extended** — the simulation is adding values
# of `r` as this is written. This notebook reads whatever is on disk and
# prints the sample size per `r`; it never assumes a fixed grid. After the
# simulation writes new blocks, refresh with
# `Rscript R/10_build_cache.R --parts post,r5 --force` and re-run.

# %%
R_DIR <- if (dir.exists(file.path(getwd(), "..", "R"))) {
  normalizePath(file.path(getwd(), "..", "R"))
} else normalizePath("R")
for (f in c("00_config.R", "01_theme.R", "02_io.R", "03_stats.R")) source(file.path(R_DIR, f))
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(purrr); library(patchwork)})
fig_size(7, 3.2)

post <- post_select(cache_read("post_cells"))
mu_H_0 <- post$mu_H_0[1]
rmax_post <- max(post$r)

support <- post |> count(r, name = "n_symbionts")
cat(sprintf("%d evolved symbionts over %d values of r (selection = '%s'); mu_H_0 = %.6g\n",
            nrow(post), nrow(support), POST_R_SELECTION, mu_H_0))
support |> pivot_wider(names_from = r, values_from = n_symbionts)

# %% [markdown]
# ### Sample size per `r`, drawn rather than described
#
# The grid is uneven while the simulation fills it in, and an uneven `n` is
# exactly what a mean curve hides. So it is plotted next to the curve.

# %%
p4_n <- ggplot(support, aes(r, n_symbionts)) +
  geom_col(width = 1.2, fill = COL$ref) +
  scale_x_r(rmax = rmax_post) +
  labs(y = mx(italic(N)[S]~"per"~italic(r))) +
  theme_grhd()
p4_n

# %% [markdown]
# ### Gate check before anything is plotted
#
# By construction `c^H = 0` is always feasible for the host, so host evolution
# can never make the host worse than `μ^H_0`. If any cell violated that, it
# would be a solver failure and not a result. We check rather than assume.

# %%
viol <- filter(post, mu_H < mu_H_0 - 1e-12)
cat(sprintf("cells with mu_H < mu_H_0: %d of %d\n", nrow(viol), nrow(post)))
if (nrow(viol)) print(count(viol, r, solver_status))

# %% [markdown]
# ## Fig.4A — the peak

# %%
bp_all  <- boot_curve_peak(post, "mu_H", unit = NULL, label = "mu_H_all")
bp_nodl <- boot_curve_peak(filter(post, !flag_detlimit), "mu_H", unit = NULL,
                           label = "mu_H_no_detlimit")

p4A <- ggplot() +
  points_cells(post, aes(r, mu_H), size = 0.35, alpha = 0.30) +
  hline_mu0(mu_H_0, label_x = 100, hjust = 1, vjust = -0.7) +
  geom_ribbon(data = bp_all$curve, aes(r, ymin = ci_lo, ymax = ci_hi),
              fill = COL$mean, alpha = 0.30) +
  geom_line(data = bp_all$curve, aes(r, value, colour = "Mean"), linewidth = 0.8) +
  scale_colour_manual(values = c(Mean = COL$mean), name = NULL) +
  scale_x_r(rmax = rmax_post) +
  labs(y = LAB_MU_H_POST) +
  theme_grhd() + legend_inside(0.60, 0.92)
p4A

# %%
cat(sprintf("peak of the mean mu^H at r* = %g   (bootstrap 95%% CI [%g, %g], B = %d)\n",
            bp_all$peak$peak_at, bp_all$peak$peak_ci_lo, bp_all$peak$peak_ci_hi,
            bp_all$peak$B_used))
cat(sprintf("peak minus r = %g : %.4g   95%% CI [%.4g, %.4g]\n",
            bp_all$peak$first_r, bp_all$peak$diff_peak_minus_first,
            bp_all$peak$diff_peak_minus_first_lo, bp_all$peak$diff_peak_minus_first_hi))
cat(sprintf("peak minus r = %g : %.4g   95%% CI [%.4g, %.4g]\n",
            bp_all$peak$last_r, bp_all$peak$diff_peak_minus_last,
            bp_all$peak$diff_peak_minus_last_lo, bp_all$peak$diff_peak_minus_last_hi))

# %% [markdown]
# **Why both contrasts matter.** "Intermediate `r` is best" is a stronger
# statement than "large `r` is bad". It requires the peak to beat the
# *smallest* `r` as well as the largest, which is what the first contrast
# above tests.
#
# ### Check: does the time-limited MILP bias the curve?
#
# Some Level 3 instances stop at the deterministic time limit and return a
# feasible incumbent that may not be optimal. Those can only push `μ^H`
# **down**, and they happen more often at large `r` — exactly the direction
# that would manufacture a peak. So the curve is drawn again without them.

# %%
dl <- post |> group_by(r) |>
  summarise(n = n(), frac_detlimit = mean(flag_detlimit),
            mip_gap_median = median(mip_gap, na.rm = TRUE),
            mip_gap_max = max(mip_gap, na.rm = TRUE), .groups = "drop")

p4A_dl <- ggplot() +
  hline_mu0(mu_H_0, label = NULL) +
  geom_ribbon(data = bp_all$curve, aes(r, ymin = ci_lo, ymax = ci_hi),
              fill = COL$mean, alpha = 0.22) +
  geom_line(data = bp_all$curve, aes(r, value, colour = "all instances"), linewidth = 0.8) +
  geom_ribbon(data = bp_nodl$curve, aes(r, ymin = ci_lo, ymax = ci_hi),
              fill = COL$mean2, alpha = 0.22) +
  geom_line(data = bp_nodl$curve, aes(r, value, colour = "detlimit excluded"), linewidth = 0.8) +
  scale_colour_manual(values = c("all instances" = COL$mean,
                                 "detlimit excluded" = COL$mean2), name = NULL) +
  scale_x_r(rmax = rmax_post) +
  labs(y = mx("mean"~mu^H~"after evolution")) +
  theme_grhd() + legend_inside(0.38, 0.92)
p4A_dl

# %%
cat(sprintf("r* with all instances      : %g  CI [%g, %g]\n",
            bp_all$peak$peak_at, bp_all$peak$peak_ci_lo, bp_all$peak$peak_ci_hi))
cat(sprintf("r* with detlimit excluded  : %g  CI [%g, %g]\n",
            bp_nodl$peak$peak_at, bp_nodl$peak$peak_ci_lo, bp_nodl$peak$peak_ci_hi))
dl

# %% [markdown]
# ## Fig.4B — what happens to the symbiont
#
# Drawn raw. The host optimises its own growth only; a symbiont driven to
# `μ^S = 0` is a possible outcome, not something to be hidden.

# %%
bpS <- boot_curve_peak(post, "mu_S", unit = NULL, label = "mu_S_all")
p4B <- ggplot() +
  points_cells(post, aes(r, mu_S), size = 0.35, alpha = 0.30) +
  geom_ribbon(data = bpS$curve, aes(r, ymin = ci_lo, ymax = ci_hi),
              fill = COL$mean, alpha = 0.30) +
  geom_line(data = bpS$curve, aes(r, value, colour = "Mean"), linewidth = 0.8) +
  scale_colour_manual(values = c(Mean = COL$mean), name = NULL) +
  scale_x_r(rmax = rmax_post, name = mx(italic(r))) +
  labs(y = LAB_MU_S_POST) +
  theme_grhd() + legend_inside(0.60, 0.92)
p4B

# %% [markdown]
# ## Fig.4C — the joint distribution at four genome sizes

# %%
hdr_rs4 <- HDR_R_VALUES[HDR_R_VALUES %in% unique(post$r)]
cont4C <- map_dfr(hdr_rs4, function(rr) {
  d <- filter(post, r == rr)
  h <- hdr_contours(d$mu_S, d$mu_H, bw = KDE_BW_FIG4C)
  if (is.null(h)) return(NULL)
  mutate(h, r = rr)
})
lab4C <- tibble(r = hdr_rs4, label = lab_r(hdr_rs4))
p4C <- ggplot() +
  geom_hline(yintercept = mu_H_0, colour = COL$ref, linewidth = 0.4) +
  geom_hdr_bands(cont4C) +
  ## 300 cells per panel -- well under RASTER_MIN_N, so vector, not rasterised.
  geom_point(data = filter(post, r %in% hdr_rs4), aes(mu_S, mu_H),
             colour = COL$hdr_point, size = 0.35, alpha = 0.8, shape = 16) +
  facet_wrap(~ r, nrow = 2) +
  ## as in Fig.3C: contours are computed on a wider grid, then clipped back
  coord_cartesian(xlim = range(post$mu_S), ylim = range(post$mu_H)) +
  ## three breaks, not five: at 9 pt the tick labels of a half-width facet
  ## run into each other
  scale_x_continuous(breaks = c(0, 0.5, 1)) +
  labs(x = LAB_MU_S_POST, y = LAB_MU_H_POST) +
  theme_grhd(BASE_PT_PANEL) +
  ## square panels, and an axis line on every one of them; ticks and tick
  ## labels stay on the outside edge of the grid (see R/01_theme.R)
  square_panels() +
  panel_axis_lines() +
  ## the HDR bands sit top-left here, so the panel label goes top-right
  facet_label_inside(lab4C, corner = "topright", spacing = 2.6, parse = TRUE)
fig_size(4.5, 4.5); p4C

# %% [markdown]
# ## Statistics for Fig.4
#
# ### R4-1 Descriptives, with and without the time-limited instances

# %%
desc4 <- bind_rows(
  post |> mutate(subset = "all"),
  post |> filter(!flag_detlimit) |> mutate(subset = "no_detlimit")) |>
  group_by(subset, r) |>
  group_modify(~ bind_rows(
    summarise_vec(.x$mu_H) |> mutate(variable = "mu_H"),
    summarise_vec(.x$mu_S) |> mutate(variable = "mu_S"),
    summarise_vec(.x$delta_mu_H) |> mutate(variable = "delta_mu_H"))) |>
  ungroup() |> relocate(variable, .after = r)
save_tab(desc4, "Fig4_descriptives_by_r")
desc4 |> filter(subset == "all", variable == "mu_H") |>
  select(r, n, mean, median, sd, q25, q75)

# %% [markdown]
# ### R4-2 The peak and its interval
#
# Bootstrap: percentile, `B = 2000`, resampled **independently within each
# `r`** (a different `r` is a different draw of symbionts, so sharing indices
# across `r` would be wrong). Ties in the argmax go to the smallest `r`.

# %%
peaks4 <- bind_rows(mutate(bp_all$peak, subset = "all"),
                    mutate(bp_nodl$peak, subset = "no_detlimit"),
                    mutate(bpS$peak, subset = "all")) |>
  mutate(loess_peak = NA_real_)
peaks4$loess_peak[1] <- loess_peak(bp_all$curve$r, bp_all$curve$value)
peaks4$loess_peak[2] <- loess_peak(bp_nodl$curve$r, bp_nodl$curve$value)
save_tab(peaks4, "Fig4_peaks")
save_tab(bind_rows(mutate(bp_all$curve, subset = "all", variable = "mu_H"),
                   mutate(bp_nodl$curve, subset = "no_detlimit", variable = "mu_H"),
                   mutate(bpS$curve, subset = "all", variable = "mu_S")),
         "Fig4_curves_with_CI")
save_tab(support, "Fig4_support_by_r")
peaks4 |> select(variable, subset, peak_at, peak_ci_lo, peak_ci_hi, peak_value, loess_peak)

# %% [markdown]
# ### R4-3 How often is the evolved pair actually mutualistic?
#
# Mutualistic means both partners grow: `μ^H > μ^H_0 + 0.01` and
# `μ^S > 0.01`, the same definition as SFig4C (`MUT_DELTA` in notebook 05) and
# the shared-direction SFig (notebook 08). The old definition (`μ^H > μ^H_0` and `μ^S > 0`, at two
# thresholds on `μ^S`) is kept in the table as a check, not plotted.

# %%
MUT_DELTA <- 1e-2
mut <- post |> group_by(r) |>
  summarise(n = n(),
            frac_mutualistic = mean(mu_H > mu_H_0 + MUT_DELTA & mu_S > MUT_DELTA),
            frac_muH_gain = mean(mu_H > mu_H_0 + MUT_DELTA),
            frac_muH_gt_muH0 = mean(mu_H > mu_H_0),
            frac_mutualistic_1e6 = mean(mu_H > mu_H_0 & mu_S > 1e-6),
            frac_mutualistic_1e3 = mean(mu_H > mu_H_0 & mu_S > 1e-3),
            frac_muS_zero = mean(mu_S <= 1e-9), .groups = "drop")
save_tab(mut, "Fig4_mutualism_fraction_by_r")

p4_mut <- ggplot(mut, aes(r, frac_mutualistic)) +
  geom_line(colour = COL$mean, linewidth = 0.8) +
  scale_x_r(rmax = rmax_post) + ylim(0, 1) +
  labs(y = "fraction mutualistic") +
  theme_grhd()
fig_size(6, 3.2); p4_mut

# %% [markdown]
# ### Extra: the pairs that were parasitic before host evolution
#
# Not in the instruction set, but it is the panel that answers "does host
# evolution rescue the *bad* symbionts, or only improve the good ones?".
# Restricted to symbionts whose best of 100 pre-evolution environments still
# left the host below `μ^H_0`.

# %%
join <- post_select(cache_read("post_join"))
para <- filter(join, parasitic_pre)
cat(sprintf("%d of %d evolved symbionts were parasitic before evolution (%.1f %%)\n",
            nrow(para), nrow(join), 100 * nrow(para) / nrow(join)))
p4_para <- NULL
if (nrow(para) > 50) {
  bp_para <- boot_curve_peak(para, "mu_H_post", unit = NULL, label = "mu_H_parasitic")
  p4_para <- ggplot() +
    points_cells(para, aes(r, mu_H_post), size = 0.35, alpha = 0.35) +
    hline_mu0(mu_H_0, label = NULL) +
    geom_ribbon(data = bp_para$curve, aes(r, ymin = ci_lo, ymax = ci_hi),
                fill = COL$mean, alpha = 0.30) +
    geom_line(data = bp_para$curve, aes(r, value, colour = "Mean"), linewidth = 0.8) +
    scale_colour_manual(values = c(Mean = COL$mean), name = NULL) +
    scale_x_r(rmax = rmax_post) +
    labs(y = LAB_MU_H_POST,
         subtitle = "restricted to pairs parasitic before evolution") +
    theme_grhd() + theme(plot.subtitle = element_text(size = BASE_PT - 2.5)) +
    legend_inside(0.60, 0.92)
  save_tab(bp_para$curve, "Fig4_parasitic_subset_curve")
  cat(sprintf("peak of this subset at r* = %g  CI [%g, %g]\n",
              bp_para$peak$peak_at, bp_para$peak$peak_ci_lo, bp_para$peak$peak_ci_hi))
}
p4_para

# %% [markdown]
# The same subset as a two-panel figure in the layout of Fig.4A/4B: `μ^H` and
# `μ^S` after evolution, without the subtitle (the caption says which pairs
# are shown).

# %%
p4_paraAB <- NULL
if (nrow(para) > 50) {
  bp_paraS <- boot_curve_peak(para, "mu_S_post", unit = NULL, label = "mu_S_parasitic")
  save_tab(bp_paraS$curve, "Fig4_parasitic_subset_curve_muS")
  cat(sprintf("mu_S of this subset peaks at r = %g  CI [%g, %g]\n",
              bp_paraS$peak$peak_at, bp_paraS$peak$peak_ci_lo, bp_paraS$peak$peak_ci_hi))
  p4_paraA <- p4_para + labs(subtitle = NULL)
  p4_paraB <- ggplot() +
    points_cells(para, aes(r, mu_S_post), size = 0.35, alpha = 0.30) +
    geom_ribbon(data = bp_paraS$curve, aes(r, ymin = ci_lo, ymax = ci_hi),
                fill = COL$mean, alpha = 0.30) +
    geom_line(data = bp_paraS$curve, aes(r, value, colour = "Mean"), linewidth = 0.8) +
    scale_colour_manual(values = c(Mean = COL$mean), name = NULL, guide = "none") +
    scale_x_r(rmax = rmax_post, name = mx(italic(r))) +
    labs(y = LAB_MU_S_POST) +
    theme_grhd()
  p4_paraAB <- (p4_paraA | p4_paraB) +
    plot_annotation(tag_levels = "A") & tag_theme()
}
fig_size(7, 2.8); p4_paraAB

# %% [markdown]
# ### Extra: how many symbionts actually mattered?
#
# Not in the instruction set. Two thresholds on the same post-evolution data
# (host evolves `c^H` at a FIXED `r` -- no evolution of `r` here): any effect
# distinguishable from zero (`|Delta*mu^H|` outside the same tolerance the
# saturation test in `00_config.R` uses, `TOL_ABS + TOL_REL * |mu_H_0|`) and a
# much stronger bar, doubling the symbiont-free growth rate
# (`mu^H >= 2 mu^H_0`). Doubling needs `Delta*mu^H >= mu_H_0 > 0`, far above
# the "any effect" tolerance, so it is a strict subset -- "either" is exactly
# the "any effect" curve, not a distinct union. Both are plotted anyway, to
# show how much smaller the doubling bar is, not because they differ.

# %%
impact <- post |>
  mutate(delta_nonzero = abs(delta_mu_H) > (TOL_ABS + TOL_REL * abs(mu_H_0)),
         doubled = mu_H >= 2 * mu_H_0) |>
  group_by(r) |>
  summarise(n = n(),
            frac_delta_nonzero = mean(delta_nonzero),
            frac_doubled = mean(doubled),
            frac_either = mean(delta_nonzero | doubled), .groups = "drop")
save_tab(impact, "Fig4_extra_symbiont_impact_by_r")
cat(sprintf("overall (n = %d): %.1f%% any effect, %.1f%% doubled mu_H_0, %.1f%% either\n",
            nrow(post), 100 * weighted.mean(impact$frac_delta_nonzero, impact$n),
            100 * weighted.mean(impact$frac_doubled, impact$n),
            100 * weighted.mean(impact$frac_either, impact$n)))

p4_impact <- impact |>
  select(r, frac_delta_nonzero, frac_doubled) |>
  pivot_longer(-r) |>
  mutate(name = recode(name, frac_delta_nonzero = "Delta*mu^H != 0",
                       frac_doubled = "mu^H >= 2*mu[0]^H")) |>
  ggplot(aes(r, value, colour = name)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = c("Delta*mu^H != 0" = COL$mean,
                                 "mu^H >= 2*mu[0]^H" = COL$mean2),
                      name = NULL, labels = scales::parse_format()) +
  scale_x_r(rmax = rmax_post) + ylim(0, 1) +
  labs(y = "fraction of symbionts") +
  theme_grhd() + legend_inside(0.55, 0.92)
fig_size(6, 3.2); p4_impact

# %% [markdown]
# ## Save

# %%
meta4 <- list(data = "data/raw/post_paper", cache = "post_cells",
              mu_H_0 = mu_H_0, n_cells = nrow(post),
              r_grid = sort(unique(post$r)),
              r_selection = POST_R_SELECTION,
              n_per_r = "unequal while the grid is being filled; see Fig4_support_by_r.csv",
              bootstrap_B = BOOT_B, r_axis = "linear",
              detlimit = "kept in 4A/4B; Fig4A_detlimit_check shows both")
save_fig(p4A, "Fig4A_muH_vs_r", size = "small", meta = meta4)
save_fig(p4B, "Fig4B_muS_vs_r", size = "small", meta = meta4)
save_fig(p4A_dl, "Fig4A_detlimit_check", size = "small", meta = meta4)
save_fig(p4_mut, "Fig4_mutualism_fraction", size = "small", meta = meta4)
save_fig(p4_n, "Fig4_extra_support_per_r", size = "small", meta = meta4)
if (!is.null(p4_para)) save_fig(p4_para, "Fig4_extra_parasitic_subset", size = "small", meta = meta4)
save_fig(p4_impact, "Fig4_extra_symbiont_impact", size = "small",
         meta = c(meta4, list(tol_abs = TOL_ABS, tol_rel = TOL_REL,
                              nonzero_rule = "abs(delta_mu_H) > TOL_ABS + TOL_REL * abs(mu_H_0)",
                              doubled_rule = "mu_H >= 2 * mu_H_0")))
save_fig(p4C, "Fig4C_muH_vs_muS_HDR", size = "medium",
         meta = c(meta4, list(kde_bw = KDE_BW_FIG4C, kde_bw_convention = KDE_BW_CONVENTION,
                              kde_sigma = KDE_BW_FIG4C / 4, hdr_levels = HDR_LEVELS,
                              kde_grid = KDE_GRID, panels_r = hdr_rs4,
                              layout = "2 x 2")))
## C is a 2 x 2 of SQUARE panels, so its height fixes its width and any extra
## width allocated to it just becomes whitespace.  0.9 is what it actually
## occupies at a 5 cm figure height.
fig4 <- (p4A | p4B | p4C) + plot_layout(widths = c(1, 1, 0.9)) +
  plot_annotation(tag_levels = "A") & tag_theme()
save_fig(fig4, "Fig4_combined", size = "main", meta = meta4)
fig_size(10, 2.8); fig4

## Two panels as wide as Fig.4A/4B inside Fig4_combined (widths 1 : 1 : 0.9 of 18 cm)
if (!is.null(p4_paraAB))
  save_fig(p4_paraAB, "Fig4_parasitic_subset_combined",
           width = round(2 * PNAS_WIDTH[["large"]] / 2.9), height = FIG_H_MAIN, meta = meta4)

# %% [markdown]
# ## Cross-check against the Python pipeline
#
# The Python tables were built on the original 15-value design grid. The
# comparison below is therefore restricted to the values of `r` the two have
# in common; any `r` added since will simply not appear.

# %%
ref <- read_sim_table("R4_peak.csv")
if (!is.null(ref)) {
  rr <- ref |> filter(subset == "all", quantity == "mu_H", statistic == "mean")
  bp_grid <- boot_curve_peak(filter(post, r %in% POST_R_GRID), "mu_H",
                             unit = NULL, label = "mu_H_design_grid")
  cat(sprintf("Python (design grid) : r* = %g  CI [%g, %g]   peak value %.6g\n",
              rr$peak_at[1], rr$peak_ci_lo[1], rr$peak_ci_hi[1], rr$peak_value[1]))
  cat(sprintf("R      (design grid) : r* = %g  CI [%g, %g]   peak value %.6g\n",
              bp_grid$peak$peak_at, bp_grid$peak$peak_ci_lo,
              bp_grid$peak$peak_ci_hi, bp_grid$peak$peak_value))
  cat(sprintf("R      (all r on disk): r* = %g  CI [%g, %g]   peak value %.6g\n",
              bp_all$peak$peak_at, bp_all$peak$peak_ci_lo,
              bp_all$peak$peak_ci_hi, bp_all$peak$peak_value))
} else cat("tables/R4_peak.csv not found - skipped.\n")

# %%
ref2 <- read_sim_table("R4_curve.csv")
if (!is.null(ref2) && "mu_H_mean" %in% names(ref2)) {
  cmp <- bp_all$curve |> select(r, mean_R = value) |>
    inner_join(ref2 |> select(r, mean_py = mu_H_mean), by = "r") |>
    mutate(abs_diff = abs(mean_R - mean_py))
  cat(sprintf("mean mu_H per r, R vs Python: max |diff| = %.3g over %d shared values of r\n",
              max(cmp$abs_diff), nrow(cmp)))
}
