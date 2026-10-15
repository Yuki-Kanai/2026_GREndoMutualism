# %% [markdown]
# # 01 — Fig.3: host and symbiont growth *before* host evolution
#
# **Question.** With the host environment `c^H` drawn at random (not yet
# optimised), how do the symbiont's growth `μ^S` and the host's growth `μ^H`
# depend on the symbiont genome size `r`, and are the two in conflict?
#
# **Panels (matching the manuscript layout).**
#
# * **3A** `μ^S` against `r` — cells, plus the mean over cells (vermillion).
# * **3B** `μ^H` against `r` — same, with the symbiont-free host growth
#   `μ^H_0` drawn as a grey line. Below it the symbiont is a parasite.
# * **3C** `Δμ^H = μ^H − μ^H_0` against `μ^S` at `r ∈ {5, 10, 20, 40}`, in a
#   2 × 2 grid, with highest-density-region bands over the points.
#
# **Data.** `data/raw/pre_paper`: one host, `r = 1..100`, `N_S = 300`
# symbionts per `r`, `N_c = 100` sampled host environments per symbiont —
# 3,000,000 cells, of which 300 per `r` are drawn for the scatter.
#
# **Drawing conventions.** The `r` axis is linear, as in the manuscript. Every
# point layer is rasterised at `RASTER_DPI` (300) so that a 30,000-point panel
# stays a small PDF with vector axes and text.

# %%
R_DIR <- if (dir.exists(file.path(getwd(), "..", "R"))) {
  normalizePath(file.path(getwd(), "..", "R"))
} else normalizePath("R")
for (f in c("00_config.R", "01_theme.R", "02_io.R", "03_stats.R")) source(file.path(R_DIR, f))
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(purrr); library(patchwork)})
fig_size(7, 3.2)

S     <- cache_read("pre_summary")
pts   <- cache_read("pre_points")
hp    <- cache_read("pre_hdr_points")
flags <- cache_read("pre_flags")
conv  <- cache_read("pre_convention")
corrs <- cache_read("pre_correlations")
sym   <- cache_read("pre_by_symbiont")

mu_H_0 <- flags$mu_H_0[1]
cat(sprintf("mu_H_0 = %.6g   |   %s cells   |   %d values of r   |   %d points per r drawn\n",
            mu_H_0, format(sum(flags$n_cells), big.mark = ","), nrow(flags), SCATTER_N))

# %% [markdown]
# ## Fig.3A — symbiont growth

# %%
mS <- filter(S, variable == "mu_S")
p3A <- ggplot() +
  points_cells(pts, aes(r, mu_S)) +
  geom_line(data = mS, aes(r, mean, colour = "Mean"), linewidth = 0.8) +
  scale_colour_manual(values = c(Mean = COL$mean), name = NULL) +
  scale_x_r() +
  labs(y = LAB_MU_S) +
  theme_grhd() + legend_inside(0.60, 0.22)
p3A

# %% [markdown]
# ## Fig.3B — host growth, against the symbiont-free host

# %%
mH <- filter(S, variable == "mu_H")
p3B <- ggplot() +
  points_cells(pts, aes(r, mu_H)) +
  hline_mu0(mu_H_0, label_x = 28, vjust = -0.7) +
  geom_line(data = mH, aes(r, mean, colour = "Mean"), linewidth = 0.8) +
  scale_colour_manual(values = c(Mean = COL$mean), name = NULL) +
  scale_x_r(name = mx(italic(r))) +
  labs(y = LAB_MU_H) +
  theme_grhd() + legend_inside(0.58, 0.98)
p3B

# %% [markdown]
# ### Check: the two ways of treating infeasible cells
#
# When Level 2 has no feasible solution the cell carries no `μ^H`. We exclude
# it; the alternative convention sets `μ^H = μ^S = 0`. The second pulls the
# mean down wherever infeasibility is common, so the two curves are drawn
# together rather than one being shown on its own.

# %%
p3B_conv <- ggplot(conv, aes(r)) +
  hline_mu0(mu_H_0, label = NULL) +
  geom_line(aes(y = mu_H_mean_raw, colour = "infeasible excluded"), linewidth = 0.8) +
  geom_line(aes(y = mu_H_mean_convention, colour = "infeasible set to 0"), linewidth = 0.8) +
  scale_colour_manual(values = c("infeasible excluded" = COL$mean,
                                 "infeasible set to 0" = COL$mean2), name = NULL) +
  scale_x_r() +
  labs(y = mx("mean"~mu^H)) +
  theme_grhd() + legend_inside(0.28, 0.92)
p3B_conv

# %%
flags |>
  select(r, n_cells, frac_infeasible, frac_saturated, frac_trivial_ub) |>
  filter(r %in% c(1, 2, 5, 10, 20, 40, 60, 80, 100))

# %% [markdown]
# ## Fig.3C — is a fast symbiont necessarily a costly one?
#
# Each panel is an independent 300-cell draw at one `r`. The bands are
# highest-density regions carrying 25 / 50 / 75 % of the kernel density,
# computed with a **fixed isotropic bandwidth** (`KDE_BW_FIG3C = 2 × 10⁻³`,
# recorded in the figure metadata) on a 256 × 256 grid, so the four panels are
# directly comparable. The grey line is `Δμ^H = 0`: points below it are cells
# where the symbiont costs the host.

# %%
cont3C <- map_dfr(HDR_R_VALUES, function(rr) {
  d <- filter(hp, r == rr)
  h <- hdr_contours(d$mu_S, d$delta_mu_H, bw = KDE_BW_FIG3C)
  if (is.null(h)) return(NULL)
  mutate(h, r = rr)
})

e3 <- function(x) x * 1e3      # the axes are drawn in units of 1e-3
lab3C <- tibble(r = HDR_R_VALUES, label = lab_r(HDR_R_VALUES))
p3C <- ggplot() +
  geom_hline(yintercept = 0, colour = COL$ref, linewidth = 0.4) +
  geom_hdr_bands(cont3C) +
  ## 300 cells per panel -- well under RASTER_MIN_N, so vector, not rasterised.
  geom_point(data = hp, aes(mu_S, delta_mu_H),
             colour = COL$hdr_point, size = 0.35, alpha = 0.8, shape = 16) +
  facet_wrap(~ r, nrow = 2) +
  ## Three breaks per axis, not five: the panels are square and half-width, and
  ## at 9 pt a fourth tick label runs into the neighbouring panel's first one.
  scale_x_continuous(labels = e3, n.breaks = 3) +
  scale_y_continuous(labels = e3) +
  ## The KDE grid deliberately runs past the data so that the contours close;
  ## the panel is then clipped back to the data, so a band may leave the frame.
  coord_cartesian(xlim = range(hp$mu_S), ylim = range(hp$delta_mu_H)) +
  labs(x = mx(mu^S~"("%*%10^-3*")"),
       y = mx(Delta*mu^H~"("%*%10^-3*")")) +
  theme_grhd(BASE_PT_PANEL) +
  ## square panels, and an axis line on every one of them; ticks and tick
  ## labels stay on the outside edge of the grid (see R/01_theme.R)
  square_panels() +
  panel_axis_lines() +
  facet_label_inside(lab3C, spacing = 2.6, parse = TRUE)  # after the theme
fig_size(4.5, 4.5); p3C

# %% [markdown]
# ## Statistics for Fig.3
#
# ### R1-1 Descriptive statistics per `r`

# %%
tab_desc <- S |>
  filter(variable %in% c("mu_S", "mu_H", "delta_mu_H")) |>
  arrange(variable, r)
save_tab(tab_desc, "Fig3_descriptives_by_r")
tab_desc |> filter(r %in% c(1, 5, 10, 20, 40, 100)) |>
  select(r, variable, n, mean, median, sd, q25, q75)

# %% [markdown]
# ### R1-2 Is `μ^H` traded off against `μ^S`?
#
# Two numbers are needed, not one. **Pooled** over all `r`, the correlation is
# confounded by `r` itself (both quantities move with `r`). **Within** a fixed
# `r` it is not. We report both, and the distribution of the within-`r` values.

# %%
pooled <- filter(corrs, scope == "pooled")
within <- filter(corrs, scope == "within_r")
within_summary <- within |>
  group_by(method) |>
  summarise(n_r = n(), rho_median = median(rho, na.rm = TRUE),
            rho_q25 = quantile(rho, 0.25, na.rm = TRUE),
            rho_q75 = quantile(rho, 0.75, na.rm = TRUE),
            rho_min = min(rho, na.rm = TRUE), rho_max = max(rho, na.rm = TRUE),
            frac_negative = mean(rho < 0, na.rm = TRUE),
            fisher_pooled = fisher_pool(rho, n), .groups = "drop")
save_tab(corrs, "Fig3_correlations_full")
save_tab(within_summary, "Fig3_correlations_within_r_summary")
list(pooled = pooled |> select(method, rho, p, n), within_r = within_summary)

# %%
p3_rho <- ggplot(within, aes(r, rho, colour = method)) +
  geom_hline(yintercept = 0, colour = COL$ref, linewidth = 0.4) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = c(spearman = COL$mean, pearson = COL$mean2), name = NULL) +
  scale_x_r() +
  labs(y = mx("correlation"~(mu^H*","~mu^S)~"within one"~italic(r))) +
  theme_grhd() + legend_inside(0.58, 0.25)
fig_size(6, 3.2); p3_rho

# %% [markdown]
# ### R1-3 Exclusion and saturation rates
#
# `frac_infeasible` is the share of cells dropped; `frac_saturated` is the
# share where `μ^S` sits exactly on its upper bound `|c^H|_1` (tolerance
# `1e-8 + 1e-6·|c^H|_1`), i.e. the symbiont takes everything the host offers.
# The saturation curve is the mechanism behind the shape of 3A.

# %%
save_tab(flags, "Fig3_flags_by_r")
save_tab(conv, "Fig3_convention_comparison")
cat(sprintf("infeasible overall: %.4f %%   (max over r: %.3f %% at r = %d)\n",
            100 * sum(flags$n_infeasible) / sum(flags$n_cells),
            100 * max(flags$frac_infeasible), flags$r[which.max(flags$frac_infeasible)]))
cat(sprintf("largest gap between the two conventions: %.4g at r = %d\n",
            max(abs(conv$mu_H_mean_raw - conv$mu_H_mean_convention)),
            conv$r[which.max(abs(conv$mu_H_mean_raw - conv$mu_H_mean_convention))]))

# %%
p3_sat <- ggplot(flags, aes(r, frac_saturated)) +
  geom_line(colour = COL$mean, linewidth = 0.8) +
  scale_x_r() + ylim(0, 1) +
  labs(y = "fraction of cells at the\nsymbiont uptake bound") +
  theme_grhd()
p3_sat

# %% [markdown]
# ### R1-4 How often is the symbiont a parasite?
#
# Per `r`, the fraction of **symbionts** (not cells) whose best of the 100
# sampled host environments still leaves the host below `μ^H_0`. This is the
# subset the parasitic-pairs panel of the supplement restricts to.

# %%
par_frac <- sym |>
  group_by(r) |>
  summarise(n_symbionts = n(), frac_parasitic = mean(parasitic_pre), .groups = "drop")
save_tab(par_frac, "Fig3_parasitic_fraction_by_r")
p3_par <- ggplot(par_frac, aes(r, frac_parasitic)) +
  geom_line(colour = COL$mean, linewidth = 0.8) +
  scale_x_r() + ylim(0, 1) +
  labs(y = mx(atop("fraction of symbionts", "with max"~mu^H < mu[0]^H))) +
  theme_grhd()
p3_par

# %% [markdown]
# ## Save

# %%
meta3 <- list(data = "data/raw/pre_paper",
              cache = c("pre_summary", "pre_points", "pre_hdr_points"),
              mu_H_0 = mu_H_0,
              n_cells_total = sum(flags$n_cells),
              subsample_per_r = SCATTER_N,
              r_axis = "linear",
              infeasible = "excluded")
save_fig(p3A, "Fig3A_muS_vs_r", size = "small", meta = meta3)
save_fig(p3B, "Fig3B_muH_vs_r", size = "small", meta = meta3)
save_fig(p3B_conv, "Fig3B_convention_check", size = "small", meta = meta3)
save_fig(p3_rho, "Fig3_extra_within_r_correlation", size = "small", meta = meta3)
save_fig(p3_sat, "Fig3_extra_saturation_fraction", size = "small", meta = meta3)
save_fig(p3_par, "Fig3_extra_parasitic_fraction", size = "small", meta = meta3)
save_fig(p3C, "Fig3C_dmuH_vs_muS_HDR", size = "medium",
         meta = c(meta3, list(kde_bw = KDE_BW_FIG3C, kde_bw_convention = KDE_BW_CONVENTION,
                              kde_sigma = KDE_BW_FIG3C / 4, hdr_levels = HDR_LEVELS,
                              kde_grid = KDE_GRID, panels_r = HDR_R_VALUES,
                              layout = "2 x 2")))

## as in Fig.4: C is a 2 x 2 of square panels, so extra width is whitespace
fig3 <- (p3A | p3B | p3C) + plot_layout(widths = c(1, 1, 0.9)) +
  plot_annotation(tag_levels = "A") & tag_theme()
save_fig(fig3, "Fig3_combined", size = "main", meta = meta3)
fig_size(10, 2.8); fig3

# %% [markdown]
# ## Cross-check against the Python pipeline
#
# The same quantities were computed independently in Python
# (`<SIM_DIR>/tables/R1_stats.csv`). If the two agree to numerical precision,
# the R analysis layer is reading and reducing the data correctly. This is a
# check, not an input: no figure above depends on it.

# %%
ref <- read_sim_table("R1_stats.csv")
if (!is.null(ref)) {
  cmp <- S |> filter(variable == "mu_H") |> select(r, mean_R = mean) |>
    inner_join(ref |> select(r, mean_py = mu_H_mean), by = "r") |>
    mutate(abs_diff = abs(mean_R - mean_py))
  cat(sprintf("mu_H mean, R vs Python over %d values of r: max |diff| = %.3g\n",
              nrow(cmp), max(cmp$abs_diff)))
} else {
  cat("tables/R1_stats.csv not found in the simulation directory - skipped.\n")
}
