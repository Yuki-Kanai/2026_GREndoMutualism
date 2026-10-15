# %% [markdown]
# # 04 — Fig.6: the structural quantities, and how the optimum moves with `m`
#
# **Question.** Fig.4A put the best partner at an intermediate genome size.
# Fig.6 asks what is special about that size.
#
# **Panels.**
#
# * **6A** sensitivity `s` against `r` (before evolution), one point per
#   symbiont, with the mean over symbionts in vermillion. If `s` peaks at an
#   intermediate `r`, the symbiont there is the one whose exchange fluxes the
#   host can move the most.
# * **6B** the metabolites the host pins (`b`, blue) and the metabolites the
#   symbiont excretes (vermillion), against `r`; the two mean curves are
#   black, told apart by line type.
# * **6C** the genome size `r*` at which the mean post-evolution `μ^H` is
#   largest, against the number of metabolites `m`. One point per host, with
#   the reference line `r = m − 1`. The main panel shows design B only
#   (`r^H = 24`, the manuscript setting); design A goes to the supplement.
#
# **A warning about `s`.** `s` is heavy-tailed: when the symbiont's response
# matrix is nearly singular, `s` jumps by orders of magnitude and a single
# cell can own the mean. 6A is therefore drawn three ways — per-symbiont
# **mean** (the manuscript panel), per-symbiont **median**, and the **raw**
# cells — all with rasterised point layers so the tail is visible without
# turning the PDF into 30,000 vector objects.
#
# **Two host designs in 6C.** Sweeping `m` with the host genome fixed at
# `r^H = 24` changes the host population itself, not just its size. So the
# sweep is run twice: design **B** keeps `r^H = 24` (the manuscript setting)
# and design **A** keeps `r^H = round(2.4 m)`, holding the host's ability
# fixed relative to `m`. The two coincide at `m = 10` by construction.

# %%
R_DIR <- if (dir.exists(file.path(getwd(), "..", "R"))) {
  normalizePath(file.path(getwd(), "..", "R"))
} else normalizePath("R")
for (f in c("00_config.R", "01_theme.R", "02_io.R", "03_stats.R")) source(file.path(R_DIR, f))
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(purrr); library(patchwork)})
fig_size(7, 3.2)

S    <- cache_read("pre_summary")
pts  <- cache_read("pre_points")
sym  <- cache_read("pre_by_symbiont")

# %% [markdown]
# ## Fig.6A — sensitivity against genome size
#
# The manuscript panel: each point is one symbiont's **mean** `s` over its 100
# sampled host environments; the line is the mean over symbionts.

# %%
sS <- filter(S, variable == "s")
## The tail of s is long enough to squash the panel, so the y axis is clipped
## at the 99.5th percentile of the plotted points.  The number of symbionts
## above the clip is stated, and the full range is in the statistics table.
ycut_sym <- as.numeric(quantile(sym$s_mean, 0.995, na.rm = TRUE)) * 1.05
cat(sprintf("6A y axis clipped at s = %.2f; %d of %d symbionts (%.2f %%) lie above it\n",
            ycut_sym, sum(sym$s_mean > ycut_sym), nrow(sym),
            100 * mean(sym$s_mean > ycut_sym)))
p6A <- ggplot() +
  points_cells(sym, aes(r, s_mean), size = 0.25, alpha = 0.25) +
  geom_line(data = sS, aes(r, mean, colour = "Mean"), linewidth = 0.8) +
  scale_colour_manual(values = c(Mean = COL$mean), name = NULL) +
  scale_x_r() +
  coord_cartesian(ylim = c(0, ycut_sym)) +
  labs(y = mx("Mean sensitivity ("*italic(s)*")")) +
  theme_grhd() + legend_inside(0.58, 0.92)
p6A

# %% [markdown]
# The same panel with the **median** instead, for the same points — the
# summary that a heavy tail does not move.

# %%
p6A_med <- ggplot() +
  points_cells(sym, aes(r, s_median), size = 0.25, alpha = 0.25) +
  geom_line(data = sS, aes(r, median, colour = "Median"), linewidth = 0.8) +
  scale_colour_manual(values = c(Median = COL$mean), name = NULL) +
  scale_x_r() +
  coord_cartesian(ylim = c(0, as.numeric(quantile(sym$s_median, 0.995, na.rm = TRUE)) * 1.05)) +
  labs(y = mx("Median sensitivity ("*italic(s)*")")) +
  theme_grhd() + legend_inside(0.58, 0.92)
p6A_med

# %% [markdown]
# And the **raw** cells (300 per `r`, the same draw the Fig.3 scatter uses),
# with all three summaries over-plotted. The y axis is clipped at the 99.5th
# percentile so the tail does not flatten the panel; the full range is in the
# statistics table.

# %%
ycut <- as.numeric(quantile(pts$s, 0.995, na.rm = TRUE)) * 1.05
p6A_raw <- ggplot() +
  points_cells(pts, aes(r, s), size = 0.25, alpha = 0.20) +
  geom_line(data = sS, aes(r, mean, colour = "mean"), linewidth = 0.8) +
  geom_line(data = sS, aes(r, trimmed_mean, colour = "5 % trimmed mean"), linewidth = 0.7) +
  geom_line(data = sS, aes(r, median, colour = "median"), linewidth = 0.7) +
  scale_colour_manual(values = c("mean" = COL$mean,
                                 "5 % trimmed mean" = COL$accent,
                                 "median" = COL$mean2), name = NULL) +
  scale_x_r() +
  coord_cartesian(ylim = c(0, ycut)) +
  labs(y = mx("sensitivity"~italic(s)~"(raw cells)")) +
  theme_grhd() + legend_inside(0.52, 0.92)
p6A_raw

# %% [markdown]
# ### Where is the peak of `s`?
#
# The resampling unit is the symbiont `R^S`, drawn independently within each
# `r`.

# %%
pk_s      <- boot_curve_peak(sym, "s_median", unit = NULL, label = "s_median",
                             agg_fun = median)
pk_s_mean <- boot_curve_peak(sym, "s_mean", unit = NULL, label = "s_mean_of_symbiont",
                             agg_fun = median)
pk_s_mm   <- boot_curve_peak(sym, "s_mean", unit = NULL, label = "s_mean_of_mean")
peak_pooled_s <- sS$r[which.max(sS$median)]
cat(sprintf("(i)   median over pooled cells                     r* = %g   (curve in 6A-median)\n",
            peak_pooled_s))
cat(sprintf("(ii)  median over symbionts of per-symbiont MEDIAN  r* = %g   95%% CI [%g, %g]\n",
            pk_s$peak$peak_at, pk_s$peak$peak_ci_lo, pk_s$peak$peak_ci_hi))
cat(sprintf("(iii) median over symbionts of per-symbiont MEAN    r* = %g   95%% CI [%g, %g]\n",
            pk_s_mean$peak$peak_at, pk_s_mean$peak$peak_ci_lo, pk_s_mean$peak$peak_ci_hi))
cat(sprintf("(iv)  mean over symbionts of per-symbiont MEAN      r* = %g   95%% CI [%g, %g]   (the curve in 6A)\n",
            pk_s_mm$peak$peak_at, pk_s_mm$peak$peak_ci_lo, pk_s_mm$peak$peak_ci_hi))

# %% [markdown]
# **Four ways to say "the typical `s`", and they do not all agree.** (i) is
# the median over all cells at that `r`. (ii)–(iv) first reduce each symbiont
# to one number and then combine over symbionts; only these can carry a
# bootstrap interval, because the symbiont is the unit that was randomly
# drawn. Reducing by the *mean* first lets the heavy tail back in through the
# per-symbiont average and moves the peak to the right.
#
# The intervals overlap, which is the honest reading: **`s` peaks at an
# intermediate `r` somewhere around 8–14, and the data do not pin it down more
# finely than that.** Definition (iii) is what the Python pipeline reports in
# `tables/R2_peaks.csv`, and (iv) is the curve drawn in panel 6A. All four are
# in the table below; none is presented on its own.

# %% [markdown]
# ## Fig.6B — what the host pins, and what the symbiont gives back

# %%
mb <- filter(S, variable == "n_bounded")
me <- filter(S, variable == "n_excreted")
long_b <- bind_rows(mutate(mb, series = "Bounded"),
                    mutate(me, series = "Excreted"))
pts_b <- bind_rows(sym |> transmute(r, value = b_mean, series = "Bounded"),
                   sym |> transmute(r, value = n_excreted_mean, series = "Excreted"))

p6B <- ggplot() +
  raster_points(geom_point(data = pts_b, aes(r, value, colour = series),
                           size = 0.25, alpha = 0.22, shape = 16)) +
  geom_hline(yintercept = (M_METABOLITES - 1) / 2, colour = COL$ref,
             linetype = "dashed", linewidth = 0.4) +
  ## The mean curves are black, as in the manuscript; the series are told
  ## apart by line type, and by the colour of the points behind them.
  geom_line(data = long_b, aes(r, mean, linetype = series),
            colour = "black", linewidth = 0.8) +
  scale_colour_manual(values = c(Bounded = COL$bounded, Excreted = COL$excreted),
                      name = NULL) +
  scale_linetype_manual(values = c(Bounded = "22", Excreted = "solid"), name = NULL) +
  scale_x_r(name = mx(italic(r))) +
  labs(y = "Mean metabolite count") +
  theme_grhd() +
  theme(legend.position = "top", legend.justification = "center") +
  guides(colour = guide_legend(order = 1, override.aes = list(alpha = 1, size = 1.4)),
         linetype = guide_legend(order = 1))
p6B

# %%
pk_b  <- boot_curve_peak(sym, "b_mean", unit = NULL, label = "b_mean")
pk_e  <- boot_curve_peak(sym, "n_excreted_mean", unit = NULL, label = "n_excreted_mean")
pk_ub <- boot_curve_peak(sym, "ub_mean", unit = NULL, label = "ub_mean")
peaks6 <- bind_rows(pk_s$peak, pk_s_mean$peak, pk_s_mm$peak,
                    pk_b$peak, pk_e$peak, pk_ub$peak)
save_tab(peaks6, "Fig6_peaks_pre_evolution")
peaks6 |> select(variable, peak_at, peak_ci_lo, peak_ci_hi, peak_value)

# %% [markdown]
# The dashed line is `(m−1)/2`, the value of `b` that maximises `u·b` purely
# algebraically. A peak of `u·b` near it is arithmetic, not biology — which is
# why `u·b` is reported here but is not used on its own as evidence.

# %%
tab6 <- S |> filter(variable %in% c("s", "alpha", "n_bounded", "n_unbounded",
                                    "ub", "n_excreted", "r_plus", "cond_b")) |>
  arrange(variable, r)
save_tab(tab6, "Fig6_structural_descriptives_by_r")
tab6 |> filter(r %in% c(1, 5, 10, 13, 20, 40, 100),
               variable %in% c("s", "n_bounded", "n_excreted")) |>
  select(r, variable, n, mean, median, trimmed_mean, q25, q75, max)

# %% [markdown]
# ## Fig.6C — does the optimum track `m`?
#
# For each host we take the mean post-evolution `μ^H` at every `r` on the grid
# and read off `r*`, the `r` where it is largest (ties go to the smaller `r`).
# The bootstrap interval resamples the 100 symbionts of that host, at that `r`,
# independently for each `r`.

# %%
r5 <- cache_read("r5_cells")
r5 |> count(design, m, host_id) |> count(design, m, name = "n_hosts") |>
  pivot_wider(names_from = design, values_from = n_hosts, names_prefix = "design_")

# %% [markdown]
# **Completeness first.** The `m` sweep is still being filled in. A host whose
# blocks hold fewer than the nominal 100 symbionts would give a meaningless
# `r*`, so those are listed and dropped rather than silently averaged in.

# %%
support <- r5 |> count(design, m, host_id, r, name = "n_symbionts")
incomplete <- support |> filter(n_symbionts < 100)
ok_hosts <- support |> group_by(design, m, host_id) |>
  summarise(n_r = n(), min_support = min(n_symbionts), .groups = "drop") |>
  filter(min_support >= 100)
cat(sprintf("hosts with full support: %d of %d   (%d (design, m, host, r) blocks short of 100)\n",
            nrow(ok_hosts), nrow(distinct(support, design, m, host_id)), nrow(incomplete)))
if (nrow(incomplete)) print(incomplete |> count(design, m, name = "n_short_blocks"))
save_tab(support, "Fig6C_support_by_host_and_r")

# %%
r5ok <- r5 |> semi_join(ok_hosts, by = c("design", "m", "host_id"))
host_peaks <- r5ok |>
  group_by(design, m, host_id) |>
  group_modify(function(d, key) {
    bp <- boot_curve_peak(d, "mu_H", unit = NULL, B = BOOT_B,
                          label = paste(key$design, key$m, key$host_id, sep = "_"))
    cur <- bp$curve
    tibble(r_star = bp$peak$peak_at, r_star_ci_lo = bp$peak$peak_ci_lo,
           r_star_ci_hi = bp$peak$peak_ci_hi, peak_mu_H = bp$peak$peak_value,
           r_star_loess = loess_peak(cur$r, cur$value),
           mu_H_0 = d$mu_H_0[1], r_H = d$r_H[1],
           b_at_peak = mean(d$n_bounded[d$r == bp$peak$peak_at]),
           peak_beats_both_ends = bp$peak$diff_peak_minus_first_lo > 0 &
                                  bp$peak$diff_peak_minus_last_lo > 0)
  }) |> ungroup()
save_tab(host_peaks, "Fig6C_host_peaks")
cat(sprintf("%d hosts; peak beats BOTH ends for %d of them\n",
            nrow(host_peaks), sum(host_peaks$peak_beats_both_ends)))

# %% [markdown]
# The manuscript panel shows one design (`r^H = 24`) as open circles jittered
# by at most 0.2 in both axes, with the line `r = m − 1`. Both designs are
# drawn here, since the second one is the control that says whether the
# relation survives holding the host's relative ability fixed.

# %%
set.seed(seed_from("fig6C_jitter"))
mmax <- max(host_peaks$m)
hp_B <- filter(host_peaks, design == "B")

## Main-figure version: the manuscript's design (r^H = 24) only, in black.
p6C <- ggplot(hp_B, aes(m, r_star)) +
  geom_abline(slope = 1, intercept = -1, colour = "black",
              linetype = "dashed", linewidth = 0.4) +
  annotate("segment", x = mmax - 7.2, xend = mmax - 9, y = 8.6, yend = mmax - 11,
           colour = "black", linewidth = 0.3,
           arrow = grid::arrow(length = unit(1.2, "mm"), type = "closed")) +
  annotate("text", x = mmax - 7, y = 8.2, label = "italic(r) == italic(m) - 1", parse = TRUE,
           size = TXT(7), hjust = 0, vjust = 0.4, colour = "black",
           family = FONT_FAMILY) +
  geom_point(position = position_jitter(width = 0.2, height = 0.2),
             size = 1.2, shape = 1, stroke = 0.4, colour = "black") +
  scale_x_continuous(breaks = seq(5, 25, 5), limits = c(4, mmax + 1)) +
  scale_y_continuous(breaks = seq(5, 30, 5)) +
  labs(x = mx(italic(m)~"(No. of metabolites)"),
       y = mx(atop(italic(r)~"with maximum", "mean"~mu^H~"after evolution"))) +
  theme_grhd()
p6C

# %% [markdown]
# The control design (`r^H = 2.4 m`, holding the host's relative ability fixed)
# belongs in the supplement rather than the main figure, so the two are also
# drawn together here and saved separately as `SFig_rstar_vs_m_both_designs`.

# %%
set.seed(seed_from("fig6C_jitter"))
p6C_both <- ggplot(host_peaks, aes(m, r_star, colour = design, shape = design)) +
  geom_abline(slope = 1, intercept = -1, colour = "black",
              linetype = "dashed", linewidth = 0.4) +
  annotate("segment", x = mmax - 7.2, xend = mmax - 9, y = 8.6, yend = mmax - 11,
           colour = "black", linewidth = 0.3,
           arrow = grid::arrow(length = unit(1.2, "mm"), type = "closed")) +
  annotate("text", x = mmax - 7, y = 8.2, label = "italic(r) == italic(m) - 1", parse = TRUE,
           size = TXT(7), hjust = 0, vjust = 0.4, colour = "black",
           family = FONT_FAMILY) +
  geom_point(position = position_jitter(width = 0.2, height = 0.2),
             size = 1.2, alpha = 0.85, stroke = 0.4, fill = NA) +
  scale_colour_manual(values = c(A = COL$mean2, B = COL$mean),
                      labels = c(A = mx(A*":"~r^H == 2.4*m),
                                 B = mx(B*":"~r^H == 24)), name = NULL) +
  scale_shape_manual(values = c(A = 21, B = 1),
                     labels = c(A = mx(A*":"~r^H == 2.4*m),
                                B = mx(B*":"~r^H == 24)), name = NULL) +
  scale_x_continuous(breaks = seq(5, 25, 5), limits = c(4, mmax + 1)) +
  scale_y_continuous(breaks = seq(5, 30, 5)) +
  labs(x = mx(italic(m)~"(No. of metabolites)"),
       y = mx(atop(italic(r)~"with maximum", "mean"~mu^H~"after evolution"))) +
  theme_grhd() + legend_inside(0.03, 0.95)
p6C_both

# %%
reg <- host_peaks |> group_by(design) |>
  group_modify(~ slope_vs_reference(.x$m, .x$r_star, reference = 1)) |> ungroup()
save_tab(reg, "Fig6C_regression")
reg

# %%
per_m <- host_peaks |> group_by(design, m) |>
  summarise(n_hosts = n(), r_star_mean = mean(r_star), r_star_median = median(r_star),
            r_star_sd = sd(r_star), r_star_min = min(r_star), r_star_max = max(r_star),
            mu_H_0_median = median(mu_H_0), r_H = first(r_H),
            resid_vs_m_minus_1 = mean(r_star - (m - 1)), .groups = "drop")
save_tab(per_m, "Fig6C_by_m")
per_m

# %% [markdown]
# `resid_vs_m_minus_1` is the average distance of `r*` from the reference
# line. If it is flat in `m` but not zero, `r*` is proportional to `m` with a
# different offset than `m − 1`; if it grows with `m`, the proportionality
# itself fails. The panel below is what separates those two readings.

# %%
p6C_resid <- ggplot(host_peaks, aes(m, r_star - (m - 1), colour = design)) +
  geom_hline(yintercept = 0, colour = "black", linetype = "dashed", linewidth = 0.4) +
  geom_point(position = position_jitter(width = 0.2, height = 0), size = 1,
             alpha = 0.8, shape = 16) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.5, alpha = 0.15) +
  scale_colour_manual(values = c(A = COL$mean2, B = COL$mean), name = NULL) +
  scale_x_continuous(breaks = seq(5, 25, 5)) +
  labs(x = mx(italic(m)), y = mx(italic(r)*"* -"~(italic(m)-1))) +
  theme_grhd() + legend_inside(0.03, 0.18)
p6C_resid

# %% [markdown]
# ## Save

# %%
meta6 <- list(data = c("data/raw/pre_paper", "data/raw/post_paper_m*"),
              cache = c("pre_summary", "pre_by_symbiont", "r5_cells"),
              bootstrap_B = BOOT_B, r_axis = "linear",
              s_panels = "6A drawn three ways: per-symbiont mean, per-symbiont median, raw cells",
              raster_layer_dpi = RASTER_DPI,
              r5_designs = "main panel: design B (r^H = 24) only; design A in the supplement",
              r5_hosts_used = nrow(host_peaks),
              r5_min_support_per_block = 100)
save_fig(p6A, "Fig6A_s_vs_r", size = "small", meta = meta6)
save_fig(p6A_med, "Fig6A_s_vs_r_median", size = "small", meta = meta6)
save_fig(p6A_raw, "Fig6A_s_vs_r_raw_cells", size = "small", meta = meta6)
save_fig(p6B, "Fig6B_metabolites_vs_r", size = "small", meta = meta6)
save_fig(p6C, "Fig6C_rstar_vs_m", size = "small", meta = meta6)
save_fig(p6C_both, "SFig_rstar_vs_m_both_designs", size = "small", meta = meta6)
save_fig(p6C_resid, "Fig6C_residual_vs_m", size = "small", meta = meta6)
fig6 <- (p6A | p6B | p6C) + plot_annotation(tag_levels = "A") & tag_theme()
save_fig(fig6, "Fig6_combined", size = "main_tall", meta = meta6)
fig_size(10, 3.3); fig6

# %% [markdown]
# ## Cross-check against the Python pipeline

# %%
ref <- read_sim_table("R2_peaks.csv")
if (!is.null(ref)) {
  cat("The Python pipeline reduces each symbiont by its MEAN before taking the\n",
      "median over symbionts -- definition (iii) above. The comparable row here\n",
      "is s_mean_of_symbiont.\n\n", sep = "")
  print(ref |> select(quantity, statistic, peak_at, peak_ci_lo, peak_ci_hi))
  cat("\nR (this notebook):\n")
  print(peaks6 |> select(variable, peak_at, peak_ci_lo, peak_ci_hi))
}

# %%
ref5 <- read_sim_table("R5_host_peaks.csv")
if (!is.null(ref5)) {
  cmp <- host_peaks |> select(design, m, host_id, r_star_R = r_star) |>
    inner_join(ref5 |> select(design, m, host_id, r_star_py = r_star),
               by = c("design", "m", "host_id"))
  cat(sprintf("hosts present in both: %d; identical r*: %d\n",
              nrow(cmp), sum(cmp$r_star_R == cmp$r_star_py)))
  if (any(cmp$r_star_R != cmp$r_star_py)) print(filter(cmp, r_star_R != r_star_py))
} else cat("tables/R5_host_peaks.csv not found - skipped.\n")
