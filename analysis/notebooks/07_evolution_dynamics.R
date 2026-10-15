# %% [markdown]
# # 07 — Evolutionary dynamics of *r*
#
# **Question.** Starting five independent sets of populations at different
# `r_0`, does the regulatory-mode number `r` evolve toward the same place
# regardless of where it started — and does that place line up with the `r*`
# that the static (non-evolutionary) optimisation of Fig.4A already found?
#
# **Data.** `input/evolution_r_series_v2.xlsx` — the FINAL data set of the
# evolutionary simulation (dated 2026-10-03, all runs stopped; supersedes the
# 2026-09-25 snapshot `evolution_r_series.xlsx`, of which it is a strict
# superset -- mainly `r_0 = 15` extended from ~4300 to ~8300 generations and
# `r_0 = 60` by ~300): 160 independent
# Wright-Fisher populations (`N = 25` lineages each) per starting `r_0` in
# `{5, 15, 30, 45, 60}`, mutated by mode birth/death and resampled by fitness
# every generation. Raw `r` is recorded every generation, `mu^H` every 10.
# Populations were launched at different times, so the generation range
# differs by `r_0`, and roughly half of each set is further behind the other
# half at any given late generation (see the file's own `README`/`provenance`
# sheets). The file itself cautions that **trends over windows shorter than
# ~1500 generations have repeatedly reversed** -- read endpoints accordingly.
#
# **Figures**
# * **main** (`EvoFig1_r_vs_generation`) median `r` and its IQR against
#   generation, one line per `r_0`.
# * **supplement A** (`EvoSFig1A_raw_trajectories`) the raw per-population `r`
#   trajectories underneath that median/IQR, for a fixed random subset of 10
#   of the 160 populations per `r0` (all 160 are in the notebook as `pA_all`,
#   not saved -- overplotting all of them makes the figure unreadable).
# * **supplement B** (`EvoSFig1B_final_r_fraction`) the distribution of `r` at
#   each population's most recently observed generation — "after evolution",
#   such as it is so far.
# * **supplement C** (`EvoSFig1C_muH_vs_r_smoothed`) that same final-`r`
#   distribution, pooled over all `r_0`, directly above the static (no
#   evolution of `r`) `mu^H(r)` curve behind Fig.4A
#   (`tables/Fig4_curves_with_CI.csv`) — does the population mass sit where
#   `mu^H` is highest?
# * **supplement D** (`EvoSFig1D_dmuH_dr`) the same distribution above
#   `d(mu^H)/dr` of that static curve, to see whether it sits near the
#   zero-crossing (the static `r*`).
# * **supplement E** (`EvoSFig2_drdt_vs_dmuHdr`) a phase-velocity plot: the
#   mean per-generation `<dr>`, binned by the current `r`, pooled over every
#   step of every population's trajectory (not just endpoints) -- directly
#   comparable to `d(mu^H)/dr` in D, since both should cross zero at the same
#   `r` if selection on `r` is tracking that static landscape.
# * **supplement F** (`EvoSFig3_muH_ratio_vs_r`) `mu^H(t) / mu^H(0)` (baselined
#   on each population's OWN generation-0 `mu^H`, not the pipeline's shared
#   `mu_H_0`) against `r(t)`, every individual sample plotted -- no averaging,
#   no binning, no subsetting.
# * **supplement G** (`EvoSFig4_muH_impact_vs_r`) the stats of that same raw
#   data: binned on `r(t)`, the fraction of samples with any improvement over
#   their own start and the fraction that doubled it.
# * **supplement H** (`EvoSFig5_muH_impact_early_vs_all`) F and G again,
#   restricted to `generation <= 100`, next to F and G themselves -- the
#   `generation <= 100` panels are closer to `r`'s innate fitness
#   distribution, before thousands of generations of drift toward smaller
#   `r` bias the all-generations panels' high-`r` end. Panel D is G's
#   fraction by `r0` (bars, restricted to `abs(r - r0) < 5`, since that early
#   `r(t)` has barely moved from `r0`) stacked above `SD(mu^H)` by `r0` --
#   the same restriction, but variability rather than a threshold.

# %%
R_DIR <- if (dir.exists(file.path(getwd(), "..", "R"))) {
  normalizePath(file.path(getwd(), "..", "R"))
} else normalizePath("R")
for (f in c("00_config.R", "01_theme.R", "02_io.R")) source(file.path(R_DIR, f))
suppressPackageStartupMessages({
  library(readxl); library(readr); library(dplyr); library(tidyr); library(purrr)
  library(ggplot2); library(patchwork)
})
fig_size(7, 3.2)

EVO_XLSX  <- file.path(ANA_DIR, "input", "evolution_r_series_v2.xlsx")
R0_LEVELS <- c(5, 15, 30, 45, 60)
SRC_NOTE  <- "FINAL data set, all runs stopped as of 2026-10-03 (input/evolution_r_series_v2.xlsx); generation ranges differ by r0 (populations launched at different times); the file cautions that trends over windows shorter than ~1500 generations have repeatedly reversed."

## Rasterise an arbitrary layer (not just points -- see raster_points() in
## 01_theme.R, which is geom_point-specific).
rasterise_layer <- function(layer, dpi = RASTER_DPI) {
  if (HAVE_GGRASTR) ggrastr::rasterise(layer, dpi = dpi) else layer
}

## Okabe-Ito, discrete, for r0 -- not viridis: r0 is five unordered starting
## CONDITIONS here, not a continuous gradient (contrast Fig.5A's viridis r
## gradient, which really is one). Grey is reserved for reference lines and
## vermillion for the single "Mean" curve elsewhere in this notebook, so
## r0 gets the other five.
R0_COLOURS <- setNames(unname(cbp[c("orange", "skyblue", "green", "blue", "purple")]),
                       as.character(R0_LEVELS))
scale_colour_r0 <- function(..., guide = "legend") {
  scale_colour_manual(name = expression(italic(r)[0]), values = R0_COLOURS, guide = guide, ...)
}
scale_fill_r0 <- function(..., guide = "legend") {
  scale_fill_manual(name = expression(italic(r)[0]), values = R0_COLOURS, guide = guide, ...)
}

## "r0 == 5" as a facet strip, with the 0 a proper subscript -- mirrors
## lab_r() in 01_theme.R, for label_parsed rather than label_both's plain text.
lab_r0 <- function(r0) sprintf("italic(r)[0] == %s", format(r0))

# %% [markdown]
# ## Load
#
# `summary` already has median/Q1/Q3 by `(r_0, generation)` -- used as-is for
# the main figure. The `r=<r0>` sheets are one row per generation, one column
# per population; reshaped to long form for the raw trajectories and the
# final-generation distribution. (`muH_r=<r0>` is not used here: C and D
# compare where r ends up against the STATIC, no-evolution mu^H(r) curve, not
# against mu^H measured under evolution.)

# %%
summ <- read_excel(EVO_XLSX, sheet = "summary") |>
  rename(n_pop = `n populations`, mean_r = `mean r`, median_r = `median r`,
         q1 = Q1, q3 = Q3, pct_11_20 = `% in 11-20`, pct_below_20 = `% below 20`,
         mean_muH = `mean muH`)

read_long <- function(sheet, value_name) {
  read_excel(EVO_XLSX, sheet = sheet) |>
    pivot_longer(-generation, names_to = "population", values_to = value_name) |>
    filter(!is.na(.data[[value_name]]))
}
raw_r <- map_dfr(R0_LEVELS, ~ read_long(sprintf("r=%d", .x), "r") |> mutate(r0 = .x))

# %% [markdown]
# ## Main — median *r* and its IQR against generation, per starting *r_0*

# %%
p_main <- summ |>
  mutate(r0 = factor(r_0, levels = R0_LEVELS)) |>
  ggplot(aes(generation, median_r, colour = r0, fill = r0)) +
  geom_ribbon(aes(ymin = q1, ymax = q3), colour = NA, alpha = 0.15) +
  geom_line(linewidth = 0.5) +
  scale_colour_r0() +
  scale_fill_r0(guide = "none") +
  scale_x_continuous(name = "Generation") +
  scale_y_continuous(name = expression(italic(r))) +
  theme_grhd()
save_fig(p_main, "EvoFig1_r_vs_generation", width = 60, height = 50,
         meta = list(data = "input/evolution_r_series_v2.xlsx (sheet: summary)",
                     note = SRC_NOTE,
                     line = "median r per r0", ribbon = "Q1-Q3 per r0"))
p_main

# %% [markdown]
# ## Supplement A — the raw values under that median/IQR
#
# All 160 populations per `r0` overplotted is hard to read (`pA_all` below,
# notebook only, not saved), so the saved figure is a fixed random subset of
# 10 populations per `r0` -- `set.seed(MASTER_SEED)` makes the draw
# reproducible. At only 50 lines total, full generation resolution is fine
# (no thinning needed) and the layer is small enough to stay vector -- see
# the RASTER_MIN_N discussion in 01_theme.R for the point-layer version of
# the same reasoning.

# %%
raw_r_lab <- raw_r |>
  mutate(r0_lab = factor(lab_r0(r0), levels = lab_r0(R0_LEVELS)),
         r0 = factor(r0, levels = R0_LEVELS))

set.seed(MASTER_SEED)
pop_subset <- raw_r_lab |> distinct(r0, population) |>
  group_by(r0) |> slice_sample(n = 10) |> ungroup()
raw_r_sample <- raw_r_lab |> semi_join(pop_subset, by = c("r0", "population")) |>
  group_by(r0) |> mutate(lineage = factor(dense_rank(population))) |> ungroup()
## Colour tells the 10 independent lineages of one r0 apart; the same 10
## colours repeat in every facet, since lineages are not matched across r0.
LINEAGE_COL <- c("#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
                 "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC")

pA <- ggplot(raw_r_sample, aes(generation, r, group = interaction(r0, population))) +
  geom_line(aes(colour = lineage), alpha = 0.85, linewidth = 0.3) +
  scale_colour_manual(values = LINEAGE_COL, guide = "none") +
  facet_wrap(~r0_lab, nrow = 1, labeller = label_parsed) +
  ## Every 2000 generations, at this facet width, ran the tick labels
  ## together -- 3 breaks instead.
  scale_x_continuous(name = "Generation", breaks = scales::pretty_breaks(n = 3)) +
  scale_y_continuous(name = expression(italic(r))) +
  theme_grhd()
save_fig(pA, "EvoSFig1A_raw_trajectories", width = PNAS_WIDTH[["large"]], height = 65,
         meta = list(data = "input/evolution_r_series_v2.xlsx (sheets: r=<r0>)",
                     note = paste(SRC_NOTE, "A fixed random subset of 10 of 160 populations per r0 (seed = MASTER_SEED); see pA_all in the notebook for all 160."),
                     n_populations_per_r0_shown = 10,
                     n_populations_per_r0_total = 160))
pA

# %% [markdown]
# All 160 populations per `r0` -- notebook only (not saved, and rasterised:
# this one really does have enough lines to need it).

# %%
raw_r_thin <- raw_r_lab |> filter(generation %% 5 == 0)   # thinned for plotting only
pA_all <- ggplot(raw_r_thin, aes(generation, r, group = interaction(r0, population))) +
  rasterise_layer(geom_line(aes(colour = r0), alpha = 0.12, linewidth = 0.12)) +
  scale_colour_r0(guide = "none") +
  facet_wrap(~r0_lab, nrow = 1, labeller = label_parsed) +
  scale_x_continuous(name = "Generation", breaks = scales::pretty_breaks(n = 3)) +
  scale_y_continuous(name = expression(italic(r))) +
  theme_grhd()
pA_all

# %% [markdown]
# ## Supplement B — fraction of *r* "after evolution"
#
# "After evolution" here means each population's own most recently observed
# generation (the tail of its trajectory) — not a common generation, because
# the runs are different lengths (see the caution above).

# %%
final_r <- raw_r |>
  group_by(r0, population) |>
  slice_max(generation, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(r0_lab = factor(lab_r0(r0), levels = lab_r0(R0_LEVELS)),
         r0 = factor(r0, levels = R0_LEVELS))

pB <- ggplot(final_r, aes(r)) +
  ## density * width = count / (count in this facet); count / sum(count) would
  ## divide by all five facets together
  geom_histogram(aes(y = after_stat(density * width)),
                 binwidth = 2, boundary = 0, fill = "grey35", colour = "white", linewidth = 0.1) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
  facet_wrap(~r0_lab, nrow = 1, scales = "free_x", labeller = label_parsed) +
  labs(x = expression(italic(r)~"(most recent generation observed)"),
       y = "Fraction of populations") +
  theme_grhd() + theme(panel.spacing.x = unit(5, "mm"))
save_fig(pB, "EvoSFig1B_final_r_fraction", width = PNAS_WIDTH[["large"]], height = 60,
         meta = list(data = "input/evolution_r_series_v2.xlsx (sheets: r=<r0>)",
                     note = paste(SRC_NOTE,
                                  "\"Final\" = each population's own most recent generation, not a common one."),
                     n_populations_per_r0 = 160))
pB
save_tab(final_r |> select(r0, population, generation, r), "Evo_final_r_by_population")

# %% [markdown]
# ## Supplement C — final-*r* distribution against the static `mu^H(r)` curve
#
# Not the same populations' own `mu^H` under evolution -- the STATIC,
# no-evolution-of-`r` `mu^H(r)` behind Fig.4A (`tables/Fig4_curves_with_CI.csv`,
# `subset == "all"`), i.e. what `mu^H` would be if the host were optimised
# once for a fixed `r`, restricted to the `r` range the evolutionary data
# actually cover. The distribution on top pools all 160x5 populations' most
# recently observed `r` (Supplement B, un-faceted) so the two panels share
# one `r` axis: does the population mass sit where `mu^H` is highest?

# %%
r_range <- c(floor(min(final_r$r)), ceiling(max(final_r$r)))

p_density_r <- ggplot(final_r, aes(r)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 2, boundary = 0,
                 fill = "grey60", colour = "white", linewidth = 0.1) +
  scale_x_continuous(name = NULL, limits = r_range) +
  ## The default 0.01 steps crowded 6 labels into a panel 1/3.2 the height of
  ## the one below it -- 3 breaks instead.
  scale_y_continuous(name = "Density", breaks = scales::pretty_breaks(n = 3)) +
  theme_grhd() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

ref_curve <- read_csv(file.path(TAB_DIR, "Fig4_curves_with_CI.csv"), show_col_types = FALSE) |>
  filter(variable == "mu_H", subset == "all") |>
  transmute(r, muH = value) |>
  filter(r >= r_range[1], r <= r_range[2])

## Fig.4A's raw per-r bootstrap mean at integer r is smooth enough to read as
## a value, but a bare central difference of point-to-point bootstrap noise
## is not (it amplifies exactly the noise a value-vs-r plot hides) -- hence
## the derivative should be taken on a smoothed copy, not the raw curve. Both
## are shown in C and D together, so the smoothing is never hidden.
ref_smooth <- ref_curve |>
  mutate(muH = predict(loess(muH ~ r, data = ref_curve, span = 0.15, degree = 2)))

CURVE_COLS <- c(Raw = COL$mean, Smoothed = COL$mean2)
CURVE_LTY  <- c(Raw = "22", Smoothed = "solid")
curves_c <- bind_rows(mutate(ref_curve, curve = "Raw"), mutate(ref_smooth, curve = "Smoothed"))

p_muH_static <- ggplot(curves_c, aes(r, muH, colour = curve, linetype = curve)) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(name = NULL, values = CURVE_COLS) +
  scale_linetype_manual(name = NULL, values = CURVE_LTY) +
  scale_x_continuous(name = expression(italic(r)), limits = r_range) +
  scale_y_continuous(name = LAB_MU_H_POST) +
  theme_grhd() + theme(legend.position = "bottom")

pC <- (p_density_r / p_muH_static) + plot_layout(heights = c(1, 2.2))
save_fig(pC, "EvoSFig1C_muH_vs_r_smoothed", size = "medium",
         meta = list(data = "input/evolution_r_series_v2.xlsx (sheets: r=<r0>) + tables/Fig4_curves_with_CI.csv",
                     note = paste(SRC_NOTE, "Top: final-r distribution pooled over all r0. Bottom: the STATIC (no evolution of r) mu^H(r), not these populations' own mu^H -- raw and loess-smoothed (span 0.15)."),
                     r_range = paste(r_range, collapse = "-")))
pC
save_tab(final_r |> transmute(r0, population, r), "Evo_final_r_pooled")
save_tab(curves_c, "Evo_muH_static_reference")

# %% [markdown]
# ## Supplement D — final-*r* distribution against `d(mu^H)/dr` of that curve
#
# Central-difference derivative of both curves in C: raw (noisy, because
# differencing amplifies the bootstrap noise a value-vs-r plot hides) and
# smoothed (the one to actually read).

# %%
deriv_of <- function(d) {
  d <- d |> arrange(r)
  tibble(r = (d$r[-1] + d$r[-nrow(d)]) / 2, dmuH = diff(d$muH) / diff(d$r))
}
derivs <- bind_rows(
  deriv_of(ref_curve)  |> mutate(curve = "Raw"),
  deriv_of(ref_smooth) |> mutate(curve = "Smoothed")
)

p_dmuH_static <- ggplot(derivs, aes(r, dmuH, colour = curve, linetype = curve)) +
  geom_hline(yintercept = 0, colour = COL$ref, linewidth = 0.4) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(name = NULL, values = CURVE_COLS) +
  scale_linetype_manual(name = NULL, values = CURVE_LTY) +
  scale_x_continuous(name = expression(italic(r)), limits = r_range) +
  scale_y_continuous(name = mx((Delta*mu^H) / (Delta*italic(r)))) +
  theme_grhd() + theme(legend.position = "bottom")

pD <- (p_density_r / p_dmuH_static) + plot_layout(heights = c(1, 2.2))
save_fig(pD, "EvoSFig1D_dmuH_dr", size = "medium",
         meta = list(data = "derived from Evo_muH_static_reference",
                     note = paste(SRC_NOTE, "Central difference of both the raw and the loess-smoothed (span 0.15) copy of the static curve -- neither is an analytic derivative."),
                     method = "loess(span = 0.15, degree = 2), then central difference of both raw and smoothed"))
pD
save_tab(derivs, "Evo_dmuH_dr_static")

# %% [markdown]
# ## Supplement E — the empirical phase velocity `<dr/dt>` against `d(mu^H)/dr`
#
# Pooling every step of every population's raw trajectory (all `r0`, all 160
# replicates each) gives one-generation increments `(r, dr)` at every
# generation any population was ever at that `r` -- not just endpoints, so
# unlike B/C/D this covers the full `r` range densely, including where the
# final-r distribution is thin. Binning the midpoint `r` of each step
# (`(r_t + r_{t+1}) / 2`, same convention as the central differences in D)
# and averaging `dr` within each bin gives the mean per-generation `<dr>` as
# a function of `r` -- a discrete-time phase velocity, directly comparable to
# `d(mu^H)/dr`: if selection on `r` tracks the static fitness landscape,
# `<dr>` should be positive where that landscape's slope is positive, cross
# zero near the same `r`, and be negative beyond it.

# %%
BIN_W <- 2   # matches the histogram binwidth used in B/C/D
step_r <- raw_r |>
  arrange(r0, population, generation) |>
  group_by(r0, population) |>
  mutate(dg = generation - lag(generation), dr = r - lag(r), r_mid = (r + lag(r)) / 2) |>
  ungroup() |>
  filter(!is.na(dg))

phase <- step_r |>
  mutate(r_bin = BIN_W * floor(r_mid / BIN_W) + BIN_W / 2) |>
  group_by(r_bin) |>
  summarise(n = n(), mean_dr = mean(dr / dg), se = stats::sd(dr / dg) / sqrt(n), .groups = "drop") |>
  ## Same n >= 100 convention as SFig1C's k-sweep curves (README): the tail
  ## beyond the r_range window is not being hidden, just not extrapolated.
  filter(n >= 100, r_bin >= r_range[1], r_bin <= r_range[2])

p_dr_dt <- ggplot(phase, aes(r_bin, mean_dr)) +
  geom_hline(yintercept = 0, colour = COL$ref, linewidth = 0.4) +
  geom_ribbon(aes(ymin = mean_dr - se, ymax = mean_dr + se), fill = COL$mean, alpha = 0.25) +
  geom_line(colour = COL$mean, linewidth = 0.7) +
  scale_x_continuous(name = NULL, limits = r_range) +
  scale_y_continuous(name = mx("Mean"~Delta*italic(r)~"per generation")) +
  theme_grhd() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

p_dmuH_smooth_only <- ggplot(filter(derivs, curve == "Smoothed"), aes(r, dmuH)) +
  geom_hline(yintercept = 0, colour = COL$ref, linewidth = 0.4) +
  geom_line(colour = COL$mean2, linewidth = 0.7) +
  scale_x_continuous(name = expression(italic(r)), limits = r_range) +
  scale_y_continuous(name = mx((Delta*mu^H) / (Delta*italic(r)))) +
  theme_grhd()

pE <- (p_dr_dt / p_dmuH_smooth_only) + plot_layout(heights = c(1, 1))
save_fig(pE, "EvoSFig2_drdt_vs_dmuHdr", size = "medium",
         meta = list(data = "input/evolution_r_series_v2.xlsx (sheets: r=<r0>), all steps of all 800 populations pooled",
                     note = paste(SRC_NOTE, "Top: mean per-generation dr, binned on the midpoint r of each step, +/- SE, bins with n < 100 dropped. Bottom: the smoothed static d(mu^H)/dr from D, repeated for alignment."),
                     bin_width = BIN_W, min_n_per_bin = 100))
pE
save_tab(phase, "Evo_drdt_by_r_bin")

# %% [markdown]
# ## Supplement F — `mu^H(t) / mu^H(0)` against `r(t)`, every individual sample
#
# `x` is each population's own CURRENT `r` at the generation `mu^H` was
# sampled -- not generation itself. `y`'s baseline is each population's OWN
# `mu^H` at generation 0 (not the pipeline's shared `mu_H_0`). No averaging,
# no binning, no subsetting to a readable few: every one of the ~596,000
# `(r0, population, generation)` samples is one point in the cloud below, so
# the spread at a given `r` is the actual spread across individuals, not a
# summary of it. Rasterised -- this is exactly the many-points case
# `RASTER_MIN_N` (01_theme.R) is for.

# %%
raw_mu <- map_dfr(R0_LEVELS, ~ read_long(sprintf("muH_r=%d", .x), "muH") |> mutate(r0 = .x))

muH0_by_pop <- raw_mu |>
  group_by(r0, population) |>
  slice_min(generation, n = 1, with_ties = FALSE) |>
  transmute(r0, population, muH0 = muH) |>
  ungroup()

ratio_vs_r <- raw_r |>
  inner_join(raw_mu, by = c("r0", "population", "generation")) |>
  inner_join(muH0_by_pop, by = c("r0", "population")) |>
  mutate(ratio = muH / muH0)
save_tab(ratio_vs_r |> select(r0, population, generation, r, muH, muH0, ratio),
         "Evo_muH_ratio_vs_r_by_sample")

## Axis and legend labels as plotmath (keys stay plain strings in the tables)
LAB_RT    <- mx(italic(r)*"("*italic(t)*")")
LAB_RATIO <- mx(mu^H*"("*italic(t)*")"/mu^H*"(0)")
THR_KEYS  <- c("mu^H(t) > mu^H(0)", "mu^H(t) >= 2*mu^H(0)")
THR_LABS  <- mx(mu^H*"("*italic(t)*")" > mu^H*"(0)",
                mu^H*"("*italic(t)*")" >= 2*mu^H*"(0)")

pF <- ggplot() +
  geom_hline(yintercept = 1, colour = COL$ref, linewidth = 0.4) +
  geom_hline(yintercept = 2, colour = COL$ref, linewidth = 0.4, linetype = "22") +
  points_cells(ratio_vs_r, aes(r, ratio), size = 0.2, alpha = 0.05) +
  scale_x_continuous(name = LAB_RT) +
  labs(y = LAB_RATIO) +
  theme_grhd()
save_fig(pF, "EvoSFig3_muH_ratio_vs_r", size = "medium",
         meta = list(data = "input/evolution_r_series_v2.xlsx (sheets: r=<r0>, muH_r=<r0>)",
                     note = paste(SRC_NOTE, "Baseline is each population's OWN mu^H at generation 0, not the pipeline's shared mu_H_0. Every sample plotted, all r0 and all 160 populations each pooled, no binning or subsetting.",
                                  "Reference lines: solid at ratio = 1 (no change), dashed at ratio = 2 (doubled)."),
                     n_points = nrow(ratio_vs_r)))
pF

# %% [markdown]
# ## Supplement G — the stats of that same raw data: proportion vs `r(t)`
#
# Same data as F exactly (`ratio_vs_r`: every sample, own-generation-0
# baseline, `x = r(t)`) -- binned on `r(t)` (`BIN_W4 = 2`, matching B/C/D) and
# reduced to the fraction of samples clearing two bars: any improvement over
# that population's own start (`mu^H(t) > mu^H(0)`), and doubling it
# (`mu^H(t) >= 2*mu^H(0)`). Bins with fewer than 100 samples are dropped
# (same convention as Supplement E), not hidden -- the underlying counts are
# in the saved table.

# %%
BIN_W4 <- 2
impact_vs_r <- ratio_vs_r |>
  mutate(gt1 = muH > muH0, doubled = muH >= 2 * muH0,
         r_bin = BIN_W4 * floor(r / BIN_W4) + BIN_W4 / 2) |>
  group_by(r_bin) |>
  summarise(n = n(), frac_gt1 = mean(gt1), frac_doubled = mean(doubled), .groups = "drop")
save_tab(impact_vs_r, "Evo_muH_impact_by_r_bin")

pG <- impact_vs_r |>
  filter(n >= 100) |>
  select(r_bin, frac_gt1, frac_doubled) |>
  pivot_longer(-r_bin) |>
  mutate(name = recode(name, frac_gt1 = "mu^H(t) > mu^H(0)",
                       frac_doubled = "mu^H(t) >= 2*mu^H(0)")) |>
  ggplot(aes(r_bin, value, colour = name)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = setNames(c(COL$mean, COL$mean2), THR_KEYS),
                      breaks = THR_KEYS, labels = THR_LABS, name = NULL) +
  ylim(0, 1) +
  labs(x = LAB_RT, y = "fraction of samples") +
  theme_grhd() + legend_inside(0.65, 0.15)
save_fig(pG, "EvoSFig4_muH_impact_vs_r", size = "medium",
         meta = list(data = "derived from Evo_muH_impact_by_r_bin (same raw samples as EvoSFig3_muH_ratio_vs_r)",
                     note = paste(SRC_NOTE, "Baseline is each population's OWN mu^H at generation 0, not the pipeline's shared mu_H_0. Bins with n < 100 dropped, not hidden -- see the table for counts."),
                     bin_width = BIN_W4, min_n_per_bin = 100))
pG

# %% [markdown]
# ## Supplement H — early generations only, and the innate-vs-drifted contrast
#
# Comparing F and G (all generations) against the same two statistics
# restricted to `generation <= 100` shows why both windows matter, not just
# the long one: at large `r`, F/G are dominated by populations that have
# already spent thousands of generations drifting toward smaller `r` (the
# main-figure story), so the "fraction improved" curve at high `r` is built
# from whatever is left there late, which is a biased sample of what large
# `r` can actually do. Early generations, before that drift has had time to
# happen, are closer to `r`'s innate fitness distribution.
#
# But restricting to `generation <= 100` also means `r(t)` has barely moved
# from `r0` yet (`abs(r - r0) < 5` still keeps 88% of those rows), so a
# continuous bin-on-`r` curve like G would mostly be five narrow spikes at
# `r0 in {5, 15, 30, 45, 60}` with nothing in between -- not a curve, a
# five-point sample pretending to be one. Panel D is that five-point sample
# honestly: one bar per `r0`, restricted to `generation <= 100` AND
# `abs(r - r0) < 5`, ~1500 samples each.
#
# * **A** = F (all generations, every sample).
# * **B** = F restricted to `generation <= 100`.
# * **C** = G (all generations, binned on `r(t)`).
# * **D** = G's fraction, restricted the same way (top) AND SD of raw
#   `mu^H(t)` within each `r0` (bottom) -- both by `r0`, bars not a curve,
#   restricted to `generation <= 100` AND `abs(r - r0) < 5`, for the reason
#   above. Two different units (0-1 fraction vs `mu^H`), so two stacked
#   panels sharing one x axis, not one y scale forced to hold both.

# %%
pF_early <- ggplot() +
  geom_hline(yintercept = 1, colour = COL$ref, linewidth = 0.4) +
  geom_hline(yintercept = 2, colour = COL$ref, linewidth = 0.4, linetype = "22") +
  points_cells(filter(ratio_vs_r, generation <= 100), aes(r, ratio), size = 0.3, alpha = 0.15) +
  scale_x_continuous(name = LAB_RT, limits = c(0, 100)) +
  labs(y = LAB_RATIO) +
  theme_grhd()
save_fig(pF_early, "EvoSFig3B_muH_ratio_vs_r_early", size = "medium",
         meta = list(data = "input/evolution_r_series_v2.xlsx (sheets: r=<r0>, muH_r=<r0>)",
                     note = paste(SRC_NOTE, "Same as EvoSFig3_muH_ratio_vs_r, restricted to generation <= 100."),
                     generation_max = 100))
pF_early

# %%
NEAR_R0 <- 5   # abs(r - r0) <
impact_near_r0 <- ratio_vs_r |>
  filter(generation <= 100, abs(r - r0) < NEAR_R0) |>
  mutate(gt1 = muH > muH0, doubled = muH >= 2 * muH0) |>
  group_by(r0) |>
  summarise(n = n(), frac_gt1 = mean(gt1), frac_doubled = mean(doubled),
            sd_muH = stats::sd(muH), .groups = "drop") |>
  mutate(r0_lab = factor(lab_r0(r0), levels = lab_r0(R0_LEVELS)))
save_tab(impact_near_r0, "Evo_muH_impact_near_r0_early_gen")

## Both the proportion (two thresholds, as in G) and SD(mu^H) -- different
## units (0-1 fraction vs mu^H), so stacked as two panels sharing one x axis
## rather than forced onto one y scale.
pD_prop <- impact_near_r0 |>
  select(r0_lab, frac_gt1, frac_doubled) |>
  pivot_longer(-r0_lab) |>
  mutate(name = recode(name, frac_gt1 = "mu^H(t) > mu^H(0)",
                       frac_doubled = "mu^H(t) >= 2*mu^H(0)")) |>
  ggplot(aes(r0_lab, value, fill = name)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = setNames(c(COL$mean, COL$mean2), THR_KEYS),
                    breaks = THR_KEYS, labels = THR_LABS, name = NULL) +
  scale_x_discrete(name = NULL, labels = scales::parse_format()) +
  ylim(0, 1) +
  labs(y = "fraction of samples") +
  theme_grhd() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        legend.position = "top") +
  labs(tag = "D")

pD_sd <- ggplot(impact_near_r0, aes(r0_lab, sd_muH)) +
  geom_col(fill = COL$mean2, width = 0.6) +
  scale_x_discrete(name = NULL, labels = scales::parse_format()) +
  labs(y = mx("SD("*mu^H*")"~"within"~italic(r)[0])) +
  theme_grhd()

pD_bar <- (pD_prop / pD_sd) + plot_layout(heights = c(1, 1))
save_fig(pD_bar, "EvoSFig4B_muH_impact_and_sd_near_r0_early", size = "medium",
         meta = list(data = "derived from Evo_muH_impact_near_r0_early_gen",
                     note = paste(SRC_NOTE, sprintf("generation <= 100 AND abs(r - r0) < %d, grouped by r0 (~1500 samples each). Top: fraction clearing each of two thresholds (same as G). Bottom: SD of raw mu^H(t) within each group -- variability, not a threshold.", NEAR_R0)),
                     generation_max = 100, near_r0_window = NEAR_R0))
pD_bar

# %%
## pG's own legend_inside() was tuned for its own standalone width; at half
## that width (squeezed next to D in this combo) it overlaps D's y-axis
## label, so the combo gets its own copy with a plain bottom legend instead.
pG_combo <- pG + theme(legend.position = "bottom") + labs(tag = "C")
pHcombo <- ((pF + labs(tag = "A")) | (pF_early + labs(tag = "B"))) /
  (pG_combo | pD_bar) &
  tag_theme()
save_fig(pHcombo, "EvoSFig5_muH_impact_early_vs_all", width = PNAS_WIDTH[["large"]], height = 190,
         meta = list(note = SRC_NOTE))
pHcombo

# %% [markdown]
# ## Combined supplementary figure

# %%
## patchwork's automatic tag_levels would tag every LEAF panel, giving C-F for
## the two nested (density / curve) composites instead of one tag each; and
## wrap_elements() (which would make each nested composite opaque to that)
## rasterises it at a size too small for the density panel's own axis text.
## So: build one-off tagged copies with labs(tag = ...) by hand instead of
## auto-lettering, and only style them (size, position) with tag_theme().
comboS <- ((pA + labs(tag = "A")) / (pB + labs(tag = "B")) /
             ((p_density_r / (p_muH_static  + labs(tag = "C"))) + plot_layout(heights = c(1, 2.2)) |
              (p_density_r / (p_dmuH_static + labs(tag = "D"))) + plot_layout(heights = c(1, 2.2)))) &
  tag_theme()
save_fig(comboS, "EvoSFig1_combined", width = PNAS_WIDTH[["large"]], height = 210,
         meta = list(note = SRC_NOTE))
comboS

# %%
comboS2 <- ((pA + labs(tag = "A")) / (pB + labs(tag = "B"))) &
  tag_theme()
save_fig(comboS2, "EvoSFig1_combined2", width = PNAS_WIDTH[["large"]], height = 100,
         meta = list(note = SRC_NOTE))
comboS2

# %% [markdown]
# ## A number worth carrying forward
#
# Per-`r0` state at each condition's own last observed generation (not a
# common generation -- see caution above):

# %%
final_state_by_r0 <- summ |>
  group_by(r_0) |>
  slice_max(generation, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(r_0, generation, n_pop, median_r, q1, q3, mean_muH)
final_state_by_r0
save_tab(final_state_by_r0, "Evo_final_state_by_r0")

# %% [markdown]
# Per-`r0` state at a MATCHED generation, as the data file's own README asks:
# the latest `summary` generation at which every `r0` still has all 160
# populations (6500 in the 2026-10-03 data set; only 3950 in the 2026-09-25
# snapshot, where `r0 = 15` stopped short).

# %%
matched_gen <- summ |>
  filter(n_pop == 160) |>
  group_by(r_0) |>
  summarise(g = max(generation), .groups = "drop") |>
  pull(g) |> min()
matched_state_by_r0 <- summ |>
  filter(generation == matched_gen) |>
  select(r_0, generation, n_pop, mean_r, median_r, q1, q3, pct_11_20, pct_below_20, mean_muH)
matched_state_by_r0
save_tab(matched_state_by_r0, "Evo_matched_generation_state_by_r0")

# %% [markdown]
# ## Paper versions — one common generation window, `r0 = 60` excluded
#
# For the manuscript: the main figure and `EvoSFig1_combined2` again, with
# every `r0` cut at the same generation `g_max` (so all lines end together)
# and `r0 = 60` dropped (still drifting at the end of its run). Every
# remaining population reaches `g_max` (shortest run: 6505 generations), so
# panel B here is `r` AT `g_max` for all 160 populations per `r0` -- a common
# generation, unlike the per-population last generation of Supplement B.
# Same styling, same 10 sampled lineages per `r0` as above. Built for
# `g_max = 5000` and `6000`; the file names carry the cut-off.

# %%
R0_PAPER      <- setdiff(R0_LEVELS, 60)
PAPER_GEN_MAX <- c(5000, 6000)
SRC_NOTE_PAPER <- function(g_max) {
  paste(SRC_NOTE, sprintf("Paper version: r0 = 60 excluded; all r0 truncated at generation %d (all 160 populations of every remaining r0 reach it).", g_max))
}

for (g_max in PAPER_GEN_MAX) {
  summ_p <- summ |>
    filter(r_0 %in% R0_PAPER, generation <= g_max) |>
    mutate(r0 = factor(r_0, levels = R0_PAPER))
  stopifnot(all(summ_p$n_pop == 160))

  ## %+% swaps the data under the plots built above, keeping their styling
  ## the default breaks every 1000 ran together at this width for g_max = 5000
  p_main_p <- (p_main %+% summ_p) +
    scale_x_continuous(name = "Generation", breaks = seq(0, g_max, length.out = 3))
  save_fig(p_main_p, sprintf("EvoFig1_r_vs_generation_paper_g%d", g_max), width = 60, height = 50,
           meta = list(data = "input/evolution_r_series_v2.xlsx (sheet: summary)",
                       note = SRC_NOTE_PAPER(g_max),
                       line = "median r per r0", ribbon = "Q1-Q3 per r0",
                       r0 = paste(R0_PAPER, collapse = ","), generation_max = g_max))
  print(p_main_p)

  r_at_g <- raw_r_lab |> filter(r0 %in% R0_PAPER, generation == g_max)
  stopifnot(nrow(r_at_g) == 160 * length(R0_PAPER))

  ## 4 facets ending on a round number ran "6000" into the next facet's "0";
  ## the same 5 mm facet gap pB already uses
  pA_p <- (pA %+% filter(raw_r_sample, r0 %in% R0_PAPER, generation <= g_max)) +
    theme(panel.spacing.x = unit(5, "mm"))
  pB_p <- (pB %+% r_at_g) + labs(x = bquote(italic(r)~"at generation"~.(g_max)))
  combo2_p <- ((pA_p + labs(tag = "A")) / (pB_p + labs(tag = "B"))) & tag_theme()
  save_fig(combo2_p, sprintf("EvoSFig1_combined2_paper_g%d", g_max),
           width = PNAS_WIDTH[["large"]], height = 100,
           meta = list(data = "input/evolution_r_series_v2.xlsx (sheets: r=<r0>)",
                       note = paste(SRC_NOTE_PAPER(g_max),
                                    "A: fixed random subset of 10 of 160 populations per r0 (seed = MASTER_SEED), same as EvoSFig1A. B: r of all 160 populations per r0 at the common generation g_max."),
                       r0 = paste(R0_PAPER, collapse = ","), generation_max = g_max,
                       n_populations_per_r0_shown_A = 10, n_populations_per_r0_B = 160))
  print(combo2_p)

  save_tab(summ_p |> select(r_0, generation, n_pop, mean_r, median_r, q1, q3),
           sprintf("Evo_summary_paper_g%d", g_max))
  save_tab(r_at_g |> transmute(r0, population, generation, r),
           sprintf("Evo_r_at_gen%d_by_population_paper", g_max))
}
