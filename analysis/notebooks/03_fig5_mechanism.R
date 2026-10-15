# %% [markdown]
# # 03 — Fig.5: what the host is actually exploiting
#
# **Question.** The peak in Fig.4A says an intermediate genome size makes the
# best partner. Two things have to be true for the mechanism to hold:
#
# 1. **5A** — how the pair does after host evolution should be related to, but
#    not determined by, how it did before. If post-evolution `μ^H` were simply
#    a relabelling of pre-evolution `μ^H`, host evolution would be adding
#    nothing.
# 2. **5D / 5E** — the quantities used to describe a symbiont, the
#    adaptability `α = √b` and the sensitivity `s`, really do control how much
#    the symbiont's exchange fluxes *move* when the host changes its
#    environment. A symbiont the host cannot move is a symbiont the host
#    cannot exploit.
#
# **Panels 5B and 5C are schematics** — the matrix decomposition of
# `∂f^S*/∂c^H`, and the cartoon contrasting adaptability with sensitivity.
# They carry no data, so they are not produced here; they stay as drawn
# artwork. Everything else on Fig.5 is below.
#
# **How the spread is defined (this matters).** `α` and `s` are *norms*. The
# matching measure of how much `f^S*` moves must therefore be a norm as well:
#
# ```
# SD_b = sqrt( Σ_j Var_{k ∈ K^b_j}( f^S*_j ) )
# SD_u = sqrt( Σ_j Var_{k ∈ K^u_j}( f^S*_j ) )
# ```
#
# where `K^b_j` (`K^u_j`) is the set of the 100 sampled `c^H` of that symbiont
# under which metabolite `j` is bounded (unbounded), the variance is the
# population variance (`1/#K`) over that set, and a metabolite with an empty set
# contributes 0 -- the classification is made **per composition**. This is the
# definition used in the paper, and Fig.5D/5E are drawn with it. The earlier
# **majority-vote** version (a metabolite is bounded for the symbiont if it is
# bounded in more than half of its `c^H`; variance over all 100, `1/(K-1)`) is
# kept as `SD_b_major` / `SD_u_major`, and its fits are in the same table.
# A plain per-component scalar SD would be nearly constant by
# construction — for bounded components `f^S*_i ≡ c^H_i`, and the components of
# `c^H` are exchangeable — and would make 5D flat for reasons that have nothing
# to do with the model.

# %%
R_DIR <- if (dir.exists(file.path(getwd(), "..", "R"))) {
  normalizePath(file.path(getwd(), "..", "R"))
} else normalizePath("R")
for (f in c("00_config.R", "01_theme.R", "02_io.R", "03_stats.R")) source(file.path(R_DIR, f))
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(purrr); library(patchwork)})
fig_size(6, 4)

join <- post_select(cache_read("post_join"))
sym  <- cache_read("pre_by_symbiont")
sdbu <- cache_read("pre_sd_bu")
mu_H_0 <- join$mu_H_0[1]
cat(sprintf("%d evolved symbionts matched to their pre-evolution summary (%d unmatched)\n",
            nrow(join), sum(is.na(join$mu_H_pre_max))))

# %% [markdown]
# ## Fig.5A — before against after
#
# The manuscript panel uses `5 ≤ r ≤ 25` and colours the points by `r`; both
# that range and the full grid are produced, because the restriction is a
# choice and the reader should see what it does. `max` over the 100 sampled
# `c^H` is the natural "best case before evolution"; the `median` version is
# given too, since the max of 100 draws is a statistic with its own upward
# bias.

# %%
plot_5A <- function(d, xcol, xlab) {
  rho <- corr_one(d[[xcol]], d$mu_H_post, "spearman")
  ggplot(d, aes(.data[[xcol]], mu_H_post, colour = r)) +
    geom_hline(yintercept = mu_H_0, colour = COL$ref, linewidth = 0.4) +
    raster_points(geom_point(size = 0.45, alpha = 0.5, shape = 16)) +
    annotate("text", x = min(d[[xcol]], na.rm = TRUE), y = mu_H_0,
             label = "Symbiont-free growth", size = TXT(7), hjust = 0, vjust = 1.5,
             colour = "black", family = FONT_FAMILY) +
    annotate("text", x = max(d[[xcol]], na.rm = TRUE),
             y = min(d$mu_H_post, na.rm = TRUE),
             label = sprintf("rho == %.2f", rho$rho), parse = TRUE,
             size = TXT(8), hjust = 1, vjust = -0.2, colour = "black",
             family = FONT_FAMILY) +
    scale_x_continuous(n.breaks = 2) +
    scale_colour_viridis_c(name = mx(italic(r))) +
    labs(x = xlab, y = LAB_MU_H_POST) +
    theme_grhd()
}
## random drawing order, so that no value of r is always painted on top (seeded)
set.seed(seed_from("fig5A_draw_order"))
j525 <- filter(join, r >= R3_R_RANGE[1], r <= R3_R_RANGE[2]) |> slice_sample(prop = 1)
p5A     <- plot_5A(j525, "mu_H_pre_max",
                   mx("Max."~mu^H~"before evolution"))
p5A_all <- plot_5A(join, "mu_H_pre_max",
                   mx("Max."~mu^H~"before evolution (all"~italic(r)*")"))
p5A_med <- plot_5A(j525, "mu_H_pre_median",
                   mx("Median"~mu^H~"before evolution"))
p5A

# %%
corr5A <- map_dfr(c("mu_H_pre_max", "mu_H_pre_median"), function(xc) {
  map_dfr(list(c("5<=r<=25", R3_R_RANGE), c("all", c(1, 1e9))), function(sb) {
    d <- filter(join, r >= as.numeric(sb[2]), r <= as.numeric(sb[3]))
    bind_rows(
      corr_one(d[[xc]], d$mu_H_post, "spearman") |> mutate(scope = "pooled"),
      partial_spearman(d[[xc]], d$mu_H_post, d$r) |>
        mutate(p = NA_real_, scope = "partial_on_r")) |>
      mutate(x = xc, subset = sb[1])
  })
})
save_tab(corr5A, "Fig5A_correlations")
corr5A |> select(x, subset, scope, rho, n)

# %% [markdown]
# The pooled `ρ` mixes the `r` gradient into the answer. `partial_on_r` takes
# it out; the within-`r` values below do the same one `r` at a time.

# %%
within5A <- j525 |> group_by(r) |>
  group_modify(~ corr_one(.x$mu_H_pre_max, .x$mu_H_post, "spearman")) |>
  ungroup()
save_tab(within5A, "Fig5A_correlations_within_r")
p5A_rho <- ggplot(within5A, aes(r, rho)) +
  geom_hline(yintercept = 0, colour = COL$ref, linewidth = 0.4) +
  geom_line(colour = COL$mean, linewidth = 0.8) +
  geom_point(colour = COL$mean, size = 1) +
  labs(x = mx(italic(r)), y = mx(rho~"within one"~italic(r))) +
  theme_grhd()
p5A_rho

# %% [markdown]
# ## Fig.5D / 5E — does `α` (resp. `s`) predict how much `f^S*` moves?

# %%
mech <- sdbu |>
  inner_join(sym |> select(r, symbiont_id, alpha_mean, s_mean, b_mean),
             by = c("r", "symbiont_id")) |>
  rename(SD_b_major = SD_b, SD_u_major = SD_u) |>
  mutate(SD_b = SD_b_pc, SD_u = SD_u_pc)      # the per-composition definition
mech525 <- filter(mech, r >= R3_R_RANGE[1], r <= R3_R_RANGE[2])
cat(sprintf("%d symbionts in total, %d with 5 <= r <= 25\n", nrow(mech), nrow(mech525)))

plot_mech <- function(d, xcol, ycol, xlab, ylab) {
  fit <- ols_fit(d[[xcol]], d[[ycol]])
  ggplot(d, aes(.data[[xcol]], .data[[ycol]])) +
    points_cells(d, aes(.data[[xcol]], .data[[ycol]]), size = 0.3, alpha = 0.25) +
    geom_abline(slope = fit$slope, intercept = fit$intercept,
                colour = COL$mean, linewidth = 0.7) +
    scale_y_continuous(labels = function(x) x * 1e3) +
    labs(x = xlab, y = ylab,
         subtitle = sprintf("slope = %.3g\nR2 = %.3f,  n = %d",
                            fit$slope, fit$r2, fit$n)) +
    theme_grhd() + theme(plot.subtitle = element_text(size = BASE_PT - 3.5))
}
p5D <- plot_mech(mech525, "alpha_mean", "SD_b",
                 mx("Mean adaptability ("*alpha*")"),
                 mx(SD(italic(f)[italic(b)]^"S*")~"("%*%10^-3*")"))
p5E <- plot_mech(mech525, "s_mean", "SD_u",
                 mx("Mean sensitivity ("*italic(s)*")"),
                 mx(SD(italic(f)[italic(u)]^"S*")~"("%*%10^-3*")"))
fig_size(8, 3.2); p5D | p5E

# %% [markdown]
# The same two panels coloured by `r`, without the regression line. Points
# are drawn in a random (seeded) order so that no value of `r` hides the others.

# %%
set.seed(seed_from("fig5DE_draw_order"))
mech525.shuffled <- mech525 |> slice_sample(prop = 1)
p5D.2 <- mech525.shuffled |>
  ggplot(aes(alpha_mean, SD_b, colour = r)) +
  ggrastr::rasterize(geom_point(size = 0.3, alpha = 0.5, shape = 16), dpi = 1000) +
  scale_y_continuous(labels = function(x) x * 1e3) +
  labs(x = mx("Mean adaptability ("*alpha*")"),
       y = mx(SD(italic(f)[italic(b)]^"S*")~"("%*%10^-3*")")) +
  theme_grhd() + scale_colour_viridis_c(name = mx(italic(r)))
p5E.2 <- mech525.shuffled |>
  ggplot(aes(s_mean, SD_u, colour = r)) +
  ggrastr::rasterize(geom_point(size = 0.3, alpha = 0.5, shape = 16), dpi = 1000) +
  scale_y_continuous(labels = function(x) x * 1e3) +
  labs(x = mx("Mean sensitivity ("*italic(s)*")"),
       y = mx(SD(italic(f)[italic(u)]^"S*")~"("%*%10^-3*")")) +
  theme_grhd() + scale_colour_viridis_c(name = mx(italic(r)))
fig_size(8, 3.2); p5D.2 | p5E.2

# %% [markdown]
# `s` is heavy-tailed (a near-singular response matrix sends it up by orders
# of magnitude), so the same pair is shown on log axes, where the straight-line
# fit above is not the right summary and the rank correlation is.

# %%
p5E_log <- mech525.shuffled |> filter(s_mean > 0, SD_u > 0) |>
  ggplot(aes(s_mean, SD_u, colour = r)) +
  ggrastr::rasterize(geom_point(size = 0.3, alpha = 0.5, shape = 16), dpi = 1000) +
  scale_x_log10() + scale_y_log10() +
  labs(x = mx("Mean sensitivity ("*italic(s)*")"),
       y = mx(SD(italic(f)[italic(u)]^"S*"))) +
  theme_grhd() + scale_colour_viridis_c(name = mx(italic(r)))
fig_size(5, 3.2); p5E_log

# %%
SD_DEFS <- c(per_composition = "", majority_vote = "_major")
tab5DE <- purrr::imap_dfr(SD_DEFS, function(suffix, def) {
  yb <- paste0("SD_b", suffix); yu <- paste0("SD_u", suffix)
  grid <- tibble::tribble(~x, ~y, ~subset,
    "alpha_mean", yb, "5<=r<=25", "alpha_mean", yb, "all_r",
    "s_mean",     yu, "5<=r<=25", "s_mean",     yu, "all_r")
  purrr::pmap_dfr(grid, function(x, y, subset) {
    d <- if (subset == "all_r") mech else mech525
    rho <- corr_one(d[[x]], d[[y]])$rho      # before mutate(), which masks x / y
    ols_fit(d[[x]], d[[y]]) |>
      mutate(x = x, y = substr(y, 1, 4), subset = subset,   # "SD_b" / "SD_u"
             spearman_rho = rho)
  }) |> mutate(SD_definition = def)
})
save_tab(tab5DE, "Fig5DE_fits")
tab5DE |> select(subset, x, y, n, slope, intercept, r2, spearman_rho)

# %% [markdown]
# ### Extra: the same two panels over the whole `r` range
#
# Not in the instruction set. The manuscript restricts to `5 ≤ r ≤ 25`; this
# shows what the restriction leaves out, with `r` on the colour scale.

# %%
p5DE_allr <- ggplot(mech, aes(alpha_mean, SD_b, colour = r)) +
  raster_points(geom_point(size = 0.3, alpha = 0.5, shape = 16)) +
  scale_colour_viridis_c(name = mx(italic(r))) +
  scale_y_continuous(labels = function(x) x * 1e3) +
  labs(x = mx("Mean adaptability ("*alpha*")"),
       y = mx(SD(italic(f)[italic(b)]^"S*")~"("%*%10^-3*")")) +
  theme_grhd()
p5DE_allr_u <- ggplot(mech, aes(s_mean, SD_u, colour = r)) +
  raster_points(geom_point(size = 0.3, alpha = 0.5, shape = 16)) +
  scale_colour_viridis_c(name = mx(italic(r))) +
  scale_y_continuous(labels = function(x) x * 1e3) +
  labs(x = mx("Mean sensitivity ("*italic(s)*")"),
       y = mx(SD(italic(f)[italic(u)]^"S*")~"("%*%10^-3*")")) +
  theme_grhd()
fig_size(8, 3.2); p5DE_allr | p5DE_allr_u

# %% [markdown]
# ## Save

# %%
meta5 <- list(data = c("data/raw/pre_paper", "data/raw/post_paper"),
              cache = c("post_join", "pre_sd_bu", "pre_by_symbiont"),
              mu_H_0 = mu_H_0,
              subset = "5 <= r <= 25 (manuscript); all-r versions also saved",
              panels_not_produced = "5B and 5C are schematics, not data panels",
              SD_definition = "root-sum-of-variances, majority-vote bounded set")
save_fig(p5A, "Fig5A_pre_vs_post_max", size = "square_45", meta = meta5)
save_fig(p5A_med, "Fig5A_pre_vs_post_median", size = "square_45", meta = meta5)
save_fig(p5A_all, "Fig5A_pre_vs_post_max_allr", size = "square_45", meta = meta5)
save_fig(p5A_rho, "Fig5A_extra_within_r_rho", size = "square_45", meta = meta5)
save_fig(p5D, "Fig5D_alpha_vs_SDb", size = "square_45", meta = meta5)
save_fig(p5E, "Fig5E_s_vs_SDu", size = "square_45", meta = meta5)
save_fig(p5D.2 + theme(legend.position = "none"), "Fig5D_alpha_vs_SDb_colorr",
         width = 45, height = 40, meta = meta5)
save_fig(p5E.2, "Fig5E_s_vs_SDu_colorr", width = 70, height = 40, meta = meta5)
save_fig(p5E_log, "Fig5E_s_vs_SDu_log", size = "square_45", meta = meta5)
save_fig(p5DE_allr, "Fig5D_extra_allr", size = "square_45", meta = meta5)
save_fig(p5DE_allr_u, "Fig5E_extra_allr", size = "square_45", meta = meta5)
fig5 <- (p5A | p5D | p5E) +
  plot_annotation(tag_levels = list(c("A", "D", "E"))) & tag_theme()
save_fig(fig5, "Fig5_combined", size = "main", meta = meta5)
fig_size(10, 2.8); fig5

# %% [markdown]
# ## Cross-check against the Python pipeline

# %%
ref <- read_sim_table("R3_stats.csv")
if (!is.null(ref)) {
  print(ref |> select(subset, x, y, slope, r2, spearman, n) |> head(6))
  cat("\nR (this notebook):\n")
  print(tab5DE |> select(subset, x, y, slope, r2, spearman_rho, n))
} else cat("tables/R3_stats.csv not found - skipped.\n")
