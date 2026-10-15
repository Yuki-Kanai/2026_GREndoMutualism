# %% [markdown]
# # 05 — Supplementary figures
#
# The main figures rest on three choices that a referee will test, and this
# notebook is where each one is answered from data rather than from argument:
#
# | | the objection | what answers it |
# |---|---|---|
# | **S1A** | "the peak is a property of the one host you happened to draw" | the same experiment repeated over every sampled host |
# | **S1B** | "shrinking `r` at fixed `k` also shrinks the reaction repertoire, so the peak may just be a reaction-count effect" | how many distinct reactions `r` modes actually cover |
# | **S1C** | — same objection, settled experimentally | the `k` sweep, and the fixed-repertoire scheme in which the repertoire is held constant and only the granularity changes |
# | **S2** | "show me the raw spread behind those means" | raw `μ^H` against `r` for nine representative hosts |
# | **S3** | "is the MILP answer good enough — would a genetic algorithm differ?" | no GA exists, so instead: the proportion of runs that hit the deterministic time limit, and the resulting **bracket** on the `μ^H` curve |
# | **S4** | "what is `b` actually doing, and does `s` predict the evolved `μ^H`?" | `b` against `r₊`, and `s` against post-evolution `μ^H` |
#
# Everything is drawn with the same palette and the same rasterised point
# layers as the main figures, and saved at PNAS sizes
# (small 9 × 6 cm, medium 11 × 11 cm, large 18 × 22 cm).

# %%
R_DIR <- if (dir.exists(file.path(getwd(), "..", "R"))) {
  normalizePath(file.path(getwd(), "..", "R"))
} else normalizePath("R")
for (f in c("00_config.R", "01_theme.R", "02_io.R", "03_stats.R")) source(file.path(R_DIR, f))
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(purrr); library(patchwork)})
fig_size(7, 3.2)

## The main post-evolution run, loaded once: several sections compare against it.
post <- post_select(cache_read("post_cells"))

# %% [markdown]
# ## S1A — the peak, host by host
#
# `m = 10`, `k = 4`, `r^H = 24`, one curve per sampled host. Fig.4A is the
# curve for host 0. If the peak were an accident of that draw, the other
# curves would not share it.

# %%
## infeasible cells carry no mu_H and are excluded, as everywhere else
hosts_all <- cache_read("post_hosts_cells")
hosts <- filter(hosts_all, !flag_infeasible)
cat(sprintf("infeasible cells excluded: %d\n", sum(hosts_all$flag_infeasible)))
cat(sprintf("%d hosts, %d evolved symbionts, r = %d..%d\n",
            dplyr::n_distinct(hosts$host_id), nrow(hosts),
            min(hosts$r), max(hosts$r)))
hosts |> count(host_id, name = "n_cells") |>
  summarise(hosts = n(), min_cells = min(n_cells), max_cells = max(n_cells))

# %%
host_curves <- hosts |>
  group_by(host_id, r) |>
  summarise(mu_H_mean = mean(mu_H), n = dplyr::n(),
            mu_H_0 = first(mu_H_0), .groups = "drop")
host_peak <- host_curves |>
  group_by(host_id) |>
  slice_max(mu_H_mean, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(host_id, r_star = r, peak_mu_H = mu_H_mean, mu_H_0)
save_tab(host_curves, "SFig1A_host_curves")
save_tab(host_peak, "SFig1A_host_peaks")
cat(sprintf("r* over %d hosts: median %g, range %g-%g\n", nrow(host_peak),
            median(host_peak$r_star), min(host_peak$r_star), max(host_peak$r_star)))

# %%
rmax_h <- max(host_curves$r)
pS1A <- ggplot(host_curves, aes(r, mu_H_mean, group = host_id)) +
  geom_line(colour = COL$point, alpha = 0.45, linewidth = 0.4) +
  geom_line(data = filter(host_curves, host_id == 0),
            aes(colour = "host of Fig.4"), linewidth = 0.9) +
  geom_point(data = host_peak, aes(r_star, peak_mu_H), inherit.aes = FALSE,
             shape = 1, size = 1.1, stroke = 0.4, colour = COL$mean2) +
  scale_colour_manual(values = c("host of Fig.4" = COL$mean), name = NULL) +
  scale_x_r(rmax = rmax_h) +
  labs(y = mx("mean"~mu^H~"after evolution")) +
  theme_grhd() + legend_inside(0.45, 0.95)
pS1A

# %% [markdown]
# Each host has its own `μ^H_0` and its own ceiling, so the curves are also
# shown rescaled to each host's own maximum — that is the form in which "the
# peak sits at the same `r`" is actually readable.

# %%
pS1A_rel <- host_curves |>
  group_by(host_id) |> mutate(rel = mu_H_mean / max(mu_H_mean)) |> ungroup() |>
  ggplot(aes(r, rel, group = host_id)) +
  geom_line(colour = COL$point, alpha = 0.45, linewidth = 0.4) +
  geom_line(data = ~ filter(.x, host_id == 0), colour = COL$mean, linewidth = 0.9) +
  scale_x_r(rmax = rmax_h) +
  labs(y = mx("mean"~mu^H~"/ its maximum for that host")) +
  theme_grhd()
pS1A_rel

# %% [markdown]
# ## S1B — how many reactions does a genome of `r` modes actually cover?
#
# This one needs no simulation: it follows from the generation rule. The
# reaction pool of one individual has `N = C(m, 2)` reactions (one direction
# drawn per unordered pair). Each mode takes `k` of them **without**
# replacement, and modes are drawn **independently**, so a reaction may appear
# in several modes. The chance that a given reaction is missed by one mode is
# `1 − k/N`, so
#
# ```
# E[distinct reactions] = N · ( 1 − (1 − k/N)^r )
# ```
#
# The curve saturates: past a few tens of modes, adding another mode adds
# almost no new chemistry. **That is the whole of the referee's concern, made
# quantitative** — at `k = 4`, `m = 10` the repertoire is already ~90 % covered
# by `r ≈ 25`, so the decline of `μ^H` beyond the peak cannot be a
# repertoire effect.

# %%
N_pool <- choose(M_METABOLITES, 2)
k_grid <- c(2, 4, 8, 16)
r_grid <- 1:100
exact <- expand_grid(k = k_grid, r = r_grid) |>
  mutate(expected = N_pool * (1 - (1 - k / N_pool)^r),
         frac = expected / N_pool)

## a direct simulation of the same rule, as a check on the algebra
set.seed(seed_from("SFig1B_reaction_coverage"))
sim <- expand_grid(k = k_grid, r = c(1, 2, 5, 10, 20, 40, 60, 100)) |>
  rowwise() |>
  mutate(sim_mean = mean(replicate(400, length(unique(
    unlist(replicate(r, sample.int(N_pool, k), simplify = FALSE))))))) |>
  ungroup()
save_tab(exact, "SFig1B_reaction_coverage_exact")
save_tab(sim, "SFig1B_reaction_coverage_simulated")
cat(sprintf("pool size N = C(%d,2) = %d\n", M_METABOLITES, N_pool))
cat(sprintf("max |exact - simulated| = %.3f reactions\n",
            max(abs(sim$sim_mean -
                      N_pool * (1 - (1 - sim$k / N_pool)^sim$r)))))

# %%
pS1B <- ggplot(exact, aes(r, expected, colour = factor(k))) +
  geom_hline(yintercept = N_pool, colour = COL$ref, linetype = "dashed",
             linewidth = 0.4) +
  geom_line(linewidth = 0.8) +
  geom_point(data = sim, aes(r, sim_mean, colour = factor(k)),
             shape = 1, size = 1, stroke = 0.4) +
  scale_colour_manual(values = c(`2` = cbp[["blue"]], `4` = cbp[["vermillion"]],
                                 `8` = cbp[["green"]], `16` = cbp[["purple"]]),
                      name = mx(italic(k))) +
  scale_x_r() +
  labs(y = "distinct reactions in the genome") +
  theme_grhd() + legend_inside(0.72, 0.55)
pS1B

# %% [markdown]
# ## S1C / S1D — the two controls that separate granularity from repertoire
#
# * **`k` sweep** — same rule, `k ∈ {2, 4, 8, 16}` reactions per mode. A larger
#   `k` buys the same repertoire with fewer modes, so if the peak tracked the
#   repertoire it would move left as `k` grows.
# * **fixed repertoire** — a reaction set `Ω` of fixed size `n_rxn` is drawn
#   once and then split into `r` blocks, one per mode. Here the repertoire is
#   **constant in `r` by construction** and only the granularity varies.
#
# Which of these exist before and after host evolution is read off the disk
# rather than assumed, because the simulation keeps adding runs.

# %%
ks      <- cache_read("ksweep_by_symbiont")
fr      <- cache_read("fixedrep_by_symbiont")
fr_post <- if (cache_exists("fixedrep_post_cells")) cache_read("fixedrep_post_cells") else NULL
ks_post <- if (cache_exists("ksweep_post_cells"))   cache_read("ksweep_post_cells")   else NULL

avail <- bind_rows(
  ks |> count(k, name = "n_symbionts") |>
    mutate(scheme = "k sweep", stage = "before", level = as.character(k)),
  fr |> count(n_rxn, name = "n_symbionts") |>
    mutate(scheme = "fixed repertoire", stage = "before", level = as.character(n_rxn)),
  if (!is.null(ks_post)) ks_post |> count(k, name = "n_symbionts") |>
    mutate(scheme = "k sweep", stage = "after", level = as.character(k)),
  if (!is.null(fr_post)) fr_post |> count(n_rxn, name = "n_symbionts") |>
    mutate(scheme = "fixed repertoire", stage = "after", level = as.character(n_rxn))
) |> select(scheme, stage, level, n_symbionts)
save_tab(avail, "SFig1C_runs_available")
avail

# %% [markdown]
# **What exists, and what does not.** The reaction pool for `m = 10` holds
# `C(10,2) = 45` reactions, so `n_rxn = 40` is already 89 % of everything the
# model can offer and `n_rxn = 45` would be the whole pool. The runs on disk
# are `n_rxn ∈ {12, 24, 40}` before evolution and **only `n_rxn = 40` after**.
# There is no `n_rxn = 20` and no `n_rxn = 45` run; those would be new
# simulations, not new figures. Every `n_rxn` that does exist is plotted.

# %% [markdown]
# ### S1C — the `k` sweep after host evolution
#
# This is the panel the objection actually needs: not the mechanism quantity
# `s`, but `μ^H` itself. **Read the support and the time-limit rate underneath
# it** — the sweep is still filling in, and the MILP gets much harder as `k`
# grows (S3), so the four curves are not equally trustworthy.

# %%
pS1C_k <- NULL
if (!is.null(ks_post)) {
  ks_support <- ks_post |> group_by(k, r) |>
    summarise(n = dplyr::n(), frac_detlimit = mean(flag_detlimit), .groups = "drop")
  save_tab(ks_support, "SFig1C_ksweep_post_support")
  print(ks_support |> group_by(k) |>
          summarise(n_r = dplyr::n(), r_max = max(r), nS_min = min(n),
                    nS_median = median(n),
                    frac_detlimit = weighted.mean(frac_detlimit, n),
                    .groups = "drop"))

  ks_curve <- ks_post |> group_by(k, r) |>
    summarise(mu_H_mean = mean(mu_H), n = dplyr::n(), .groups = "drop") |>
    filter(n >= 100)                      # drop blocks that are still thin
  ks_post_peaks <- ks_post |> semi_join(ks_curve, by = c("k", "r")) |>
    group_by(k) |>
    group_modify(~ boot_curve_peak(.x, "mu_H", unit = NULL,
                                   label = paste0("kpost", .y$k))$peak) |> ungroup()
  save_tab(ks_curve, "SFig1C_ksweep_post_curves")
  save_tab(ks_post_peaks, "SFig1C_ksweep_post_peaks")
  print(ks_post_peaks |> select(k, peak_at, peak_ci_lo, peak_ci_hi, peak_value))

  pS1C_k <- ggplot(ks_curve, aes(r, mu_H_mean, colour = factor(k))) +
    hline_mu0(ks_post$mu_H_0[1], label = NULL) +
    geom_line(linewidth = 0.8) +
    scale_colour_manual(values = c(`2` = cbp[["blue"]], `4` = cbp[["vermillion"]],
                                   `8` = cbp[["green"]], `16` = cbp[["purple"]]),
                        name = mx(italic(k))) +
    scale_x_r(rmax = max(ks_curve$r)) +
    labs(y = mx("mean"~mu^H~"after evolution")) +
    theme_grhd() + legend_inside(0.80, 0.97) +
    theme(legend.key.size = unit(2.6, "mm"))
}
pS1C_k

# %% [markdown]
# The same sweep before evolution, on the mechanism quantity `s`.

# %%
ks_curve_pre <- ks |> group_by(k, r) |>
  summarise(s_mean = mean(s_mean), s_median = median(s_median),
            mu_H_mean = mean(mu_H_mean), b_mean = mean(b_mean),
            n = dplyr::n(), .groups = "drop")
save_tab(ks_curve_pre, "SFig1C_ksweep_pre_curves")
pS1C_k_pre <- ggplot(ks_curve_pre, aes(r, s_mean, colour = factor(k))) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = c(`2` = cbp[["blue"]], `4` = cbp[["vermillion"]],
                                 `8` = cbp[["green"]], `16` = cbp[["purple"]]),
                      name = mx(italic(k))) +
  scale_x_r() +
  labs(y = mx("mean sensitivity ("*italic(s)*")")) +
  theme_grhd() + legend_inside(0.72, 0.92)
pS1C_k_pre

# %%
ks_peaks <- ks |> group_by(k) |>
  group_modify(~ boot_curve_peak(.x, "s_mean", unit = NULL,
                                 label = paste0("k", .y$k))$peak) |>
  ungroup()
save_tab(ks_peaks, "SFig1C_ksweep_pre_peaks")
ks_peaks |> select(k, peak_at, peak_ci_lo, peak_ci_hi, peak_value)

# %% [markdown]
# ### S1D — the fixed repertoire, every `n_rxn` that exists
#
# Before evolution all three repertoire sizes are available, so all three are
# drawn. After evolution only `n_rxn = 40` has been run; it is compared
# against the paper scheme over the same range of `r`, which is the
# like-for-like test — same `r`, same host, repertoire held constant versus
# repertoire growing with `r`.

# %%
fr_curve <- fr |> group_by(n_rxn, r) |>
  summarise(s_mean = mean(s_mean), mu_H_mean = mean(mu_H_mean),
            b_mean = mean(b_mean), n = dplyr::n(), .groups = "drop")
save_tab(fr_curve, "SFig1D_fixedrep_pre_curves")
fr_lvls <- sort(unique(fr_curve$n_rxn))
n_rxn_cols <- setNames(c(cbp[["blue"]], cbp[["green"]],
                         cbp[["purple"]])[seq_along(fr_lvls)], fr_lvls)

pS1D_pre <- ggplot(fr_curve, aes(r, s_mean, colour = factor(n_rxn))) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = n_rxn_cols, name = mx(italic(n)[rxn])) +
  scale_x_continuous(name = mx(italic(r)), breaks = seq(0, 40, 10)) +
  labs(y = mx("mean sensitivity ("*italic(s)*")")) +
  theme_grhd() + legend_inside(0.70, 0.92)
pS1D_pre

# %%
pS1D_pre_muH <- ggplot(fr_curve, aes(r, mu_H_mean, colour = factor(n_rxn))) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = n_rxn_cols, name = mx(italic(n)[rxn])) +
  scale_x_continuous(name = mx(italic(r)), breaks = seq(0, 40, 10)) +
  labs(y = mx("mean"~mu^H~"before evolution")) +
  theme_grhd() + legend_inside(0.70, 0.92)
pS1D_pre_muH

# %%
pS1D <- NULL
if (!is.null(fr_post)) {
  rmax_fr <- max(fr_post$r)
  cmp <- bind_rows(
    fr_post |> transmute(r, mu_H, series = "fixed repertoire"),
    post |> filter(r <= rmax_fr) |> transmute(r, mu_H, series = "paper scheme"))
  cat(sprintf("fixed repertoire n_rxn = %s
",
              paste(sort(unique(fr_post$n_rxn)), collapse = ", ")))
  cmp_curve <- cmp |> group_by(series) |>
    group_modify(~ boot_curve_peak(.x, "mu_H", unit = NULL,
                                   label = paste0("frc", .y$series))$curve) |> ungroup()
  cmp_peak <- cmp |> group_by(series) |>
    group_modify(~ boot_curve_peak(.x, "mu_H", unit = NULL,
                                   label = paste0("frp", .y$series))$peak) |> ungroup()
  save_tab(cmp_curve, "SFig1D_fixedrep_post_curves")
  save_tab(cmp_peak, "SFig1D_fixedrep_post_peaks")
  print(cmp_peak |> select(series, peak_at, peak_ci_lo, peak_ci_hi, peak_value))

  sr <- sort(unique(cmp_curve$series))
  pS1D <- ggplot(cmp_curve, aes(r, value, colour = series, fill = series)) +
    hline_mu0(fr_post$mu_H_0[1], label = NULL) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), colour = NA, alpha = 0.25) +
    geom_line(linewidth = 0.8) +
    scale_colour_manual(values = setNames(c(COL$mean2, COL$mean), sr), name = NULL) +
    scale_fill_manual(values = setNames(c(COL$mean2, COL$mean), sr), name = NULL) +
    scale_x_continuous(name = mx(italic(r)), breaks = seq(0, 40, 10)) +
    labs(y = mx("mean"~mu^H~"after evolution")) +
    theme_grhd() + legend_inside(0.40, 0.99) +
    theme(legend.key.size = unit(2.6, "mm"))
}
pS1D

# %% [markdown]
# ### S1E — do the peaks of `s`, `ub` and the excreted count move with the scheme?
#
# The same two controls as S1C/S1D, but for the structural quantities of
# Fig.6A/6B **before** host evolution (per-symbiont means over the 100 random
# `c^H`, then averaged per `r`): the `k` sweep, and the fixed repertoire next
# to the paper scheme over the same range of `r`. The `μ^H` peaks after
# evolution (S1C/S1D) are put in the same table so the two can be read side
# by side. Peaks are the bootstrap argmax (`B = 2000`, resampled within `r`).

# %%
S1E_VARS <- c(s_mean = "mean s", s_median = "median s", ub_mean = "mean ub",
              n_excreted_mean = "mean excreted")
s1e_peaks <- function(d, tag) {
  imap_dfr(S1E_VARS, function(lab, v) {
    agg <- if (v == "s_median") median else mean
    p <- boot_curve_peak(d, v, unit = NULL, agg_fun = agg, label = paste("S1E", tag, v))$peak
    tibble(quantity = lab, peak_at = p$peak_at, ci_lo = p$peak_ci_lo, ci_hi = p$peak_ci_hi,
           peak_value = p$peak_value, r_max = max(d$r))
  })
}
fr_rmax <- fr |> group_by(n_rxn) |> summarise(r_max = max(r), .groups = "drop")
s1e_tab <- bind_rows(
  ks |> group_by(level = k) |> group_modify(~ s1e_peaks(.x, paste0("k", .y$level))) |>
    ungroup() |> mutate(scheme = "k sweep"),
  fr |> group_by(level = n_rxn) |> group_modify(~ s1e_peaks(.x, paste0("fr", .y$level))) |>
    ungroup() |> mutate(scheme = "fixed repertoire"),
  map_dfr(fr_rmax$n_rxn, function(n) {
    ks |> filter(k == K_REACTIONS, r <= fr_rmax$r_max[fr_rmax$n_rxn == n]) |>
      s1e_peaks(paste0("paper_le", n)) |> mutate(level = n, scheme = "paper scheme, same r range")
  }),
  if (exists("ks_post_peaks")) ks_post_peaks |>
    transmute(scheme = "k sweep", level = k, quantity = "mu_H after evolution", peak_at,
              ci_lo = peak_ci_lo, ci_hi = peak_ci_hi, peak_value, r_max = NA_real_),
  if (exists("cmp_peak")) cmp_peak |>
    transmute(scheme = ifelse(series == "fixed repertoire", "fixed repertoire",
                              "paper scheme, same r range"),
              level = unique(fr_post$n_rxn)[1], quantity = "mu_H after evolution", peak_at,
              ci_lo = peak_ci_lo, ci_hi = peak_ci_hi, peak_value, r_max = NA_real_)
) |>
  relocate(scheme, level, quantity) |>
  arrange(factor(scheme, c("k sweep", "fixed repertoire", "paper scheme, same r range")),
          factor(quantity, c("mu_H after evolution", unname(S1E_VARS))), level)
save_tab(s1e_tab, "SFig1E_structure_peaks")
s1e_tab |> select(scheme, level, quantity, peak_at, ci_lo, ci_hi)

# %% [markdown]
# The curves behind that table: one column per control, one row per quantity
# (mean `s`, mean `ub`, mean excreted count), open circles at each curve's
# peak. In the fixed-repertoire column the paper scheme is the dashed curve,
# over the same range of `r`.

# %%
S1E_PLOT <- c(s_mean = "mean s", ub_mean = "mean ub", n_excreted_mean = "mean excreted")
S1E_YLAB <- list(s_mean = mx("mean"~italic(s)), ub_mean = mx("mean"~italic(u)*italic(b)),
                 n_excreted_mean = "mean excreted")
long_curves <- function(d) {   # d carries a `series` column
  d |> group_by(series, r) |>
    summarise(across(all_of(names(S1E_PLOT)), mean), .groups = "drop") |>
    tidyr::pivot_longer(all_of(names(S1E_PLOT)), names_to = "var", values_to = "value")
}
cur_k  <- long_curves(mutate(ks, series = paste0("k = ", k)))
cur_fr <- bind_rows(
  long_curves(mutate(fr, series = paste0("n_rxn = ", n_rxn))),
  long_curves(mutate(filter(ks, k == K_REACTIONS, r <= max(fr$r)), series = "paper scheme")))
peak_pts <- function(cur) cur |> group_by(series, var) |> slice_max(value, n = 1, with_ties = FALSE) |> ungroup()

K_COL  <- c(`k = 2` = cbp[["blue"]], `k = 4` = cbp[["vermillion"]],
            `k = 8` = cbp[["green"]], `k = 16` = cbp[["purple"]])
FR_COL <- c(setNames(unname(n_rxn_cols), paste0("n_rxn = ", names(n_rxn_cols))),
            `paper scheme` = cbp[["vermillion"]])
FR_LTY <- c(setNames(rep("solid", length(n_rxn_cols)), paste0("n_rxn = ", names(n_rxn_cols))),
            `paper scheme` = "22")
K_LAB  <- mx(italic(k) == 2, italic(k) == 4, italic(k) == 8, italic(k) == 16)
FR_LAB <- as.expression(c(
  lapply(names(n_rxn_cols), function(n) bquote(italic(n)[rxn] == .(as.numeric(n)))),
  list(quote("paper scheme ("*italic(k) == 4*")"))))

s1e_panel <- function(cur, v, cols, ltys, labs_, xlab, show_legend) {
  d <- filter(cur, var == v); pk <- peak_pts(d)
  ggplot(d, aes(r, value, colour = series, linetype = series)) +
    geom_line(linewidth = 0.7) +
    geom_point(data = pk, shape = 1, size = 1.6, stroke = 0.6, show.legend = FALSE) +
    scale_colour_manual(values = cols, breaks = names(cols), labels = labs_, name = NULL) +
    scale_linetype_manual(values = ltys, breaks = names(cols), labels = labs_, name = NULL) +
    labs(x = xlab, y = S1E_YLAB[[v]]) +
    theme_grhd(BASE_PT_PANEL) +
    (if (show_legend) legend_inside(0.99, 0.99) else theme(legend.position = "none")) +
    theme(legend.justification = c(1, 1), legend.key.size = unit(2.4, "mm"))
}
K_LTY <- setNames(rep("solid", 4), names(K_COL))
s1e_rows <- lapply(names(S1E_PLOT), function(v) {
  last <- v == tail(names(S1E_PLOT), 1)
  (s1e_panel(cur_k, v, K_COL, K_LTY, K_LAB,
             if (last) mx(italic(r)) else NULL, v == names(S1E_PLOT)[1]) +
     scale_x_continuous(limits = c(0, 100))) |
  (s1e_panel(cur_fr, v, FR_COL, FR_LTY, FR_LAB,
             if (last) mx(italic(r)) else NULL, v == names(S1E_PLOT)[1]) +
     scale_x_continuous(limits = c(0, max(fr$r))))
})
pS1E <- wrap_plots(s1e_rows, ncol = 1) +
  plot_annotation(tag_levels = "A") & tag_theme()
fig_size(7, 7); pS1E

# %% [markdown]
# ## S2 — the raw spread behind the means, for nine hosts
#
# Nine hosts spanning the range of `μ^H_0`, each with its own raw cells and
# its own mean curve. Same axes everywhere, so the panels can be compared.
#
# Only hosts whose run is **complete** — the same `r = 1..100` grid and
# `N_S = 300` as host 0 — are candidates: a raw-spread figure with holes in `r`
# would show the sampling, not the spread.

# %%
## "complete" = the same r grid and the same N_S per r as host 0, the Fig.4A run.
## Counted before infeasible cells are dropped, so one infeasible cell does not
## disqualify a run.
ref_grid <- hosts_all |> filter(host_id == 0L) |> count(r, name = "n_ref")
complete_hosts <- hosts_all |> count(host_id, r) |>
  left_join(ref_grid, by = "r") |>
  group_by(host_id) |>
  summarise(complete = n_distinct(r) == nrow(ref_grid) && isTRUE(all(n == n_ref)),
            .groups = "drop") |>
  filter(complete) |> pull(host_id)
cat("hosts with a complete run:", paste(sort(complete_hosts), collapse = ", "), "\n")

s2_pool <- host_peak |> filter(host_id %in% complete_hosts) |> arrange(mu_H_0)
h9 <- s2_pool |>
  slice(round(seq(1, nrow(s2_pool), length.out = min(9, nrow(s2_pool))))) |>
  pull(host_id)
cat("hosts shown:", paste(h9, collapse = ", "), "\n")
d9 <- filter(hosts, host_id %in% h9) |>
  mutate(host = factor(host_id, levels = h9,
                       labels = sprintf("host %d", h9)))
c9 <- host_curves |> filter(host_id %in% h9) |>
  mutate(host = factor(host_id, levels = h9, labels = sprintf("host %d", h9)))
s2_hosts <- c9 |> group_by(host_id) |>
  summarise(mu_H_0 = first(mu_H_0), n_r = dplyr::n(), n_cells = sum(n),
            r_star = r[which.max(mu_H_mean)], peak_mu_H = max(mu_H_mean),
            .groups = "drop") |>
  arrange(match(host_id, h9))
save_tab(s2_hosts, "SFig2_hosts_shown")
s2_hosts
lab9 <- distinct(c9, host) |> mutate(label = as.character(host))

pS2 <- ggplot() +
  points_cells(d9, aes(r, mu_H), size = 0.3, alpha = 0.3) +
  geom_line(data = c9, aes(r, mu_H_mean), colour = COL$mean, linewidth = 0.7) +
  geom_hline(data = distinct(c9, host, mu_H_0), aes(yintercept = mu_H_0),
             colour = COL$ref, linewidth = 0.35) +
  facet_wrap(~ host, nrow = 3) +
  scale_x_r(rmax = rmax_h) +
  labs(y = LAB_MU_H_POST) +
  theme_grhd(BASE_PT_PANEL) +
  facet_label_inside(lab9, size = TXT(7), corner = "topright")   # after the theme: blanks the strip
fig_size(6, 5); pS2

# %% [markdown]
# ## S3 — how good is the optimiser?
#
# **A genetic-algorithm comparison is not available.** Host evolution is
# solved here as a mixed-integer program only; there is no GA implementation
# anywhere in the simulation repository, so there is nothing to compare
# against. That panel would mean writing a GA for Level 3 and running it —
# a new experiment, not a new figure.
#
# What *can* be shown, and answers the same worry, is the effect of the
# **deterministic time limit**. Some Level 3 instances stop before the solver
# has proved optimality and return a feasible incumbent. For a maximisation
# that incumbent is a **lower bound** on the true `μ^H`, and the solver's
# relative gap turns into an **upper bound**:
#
# ```
# lower = mu_H (as recorded)        upper = min( mu_H * (1 + mip_gap), 1 )
# ```
#
# The cap at 1 is the model's own bound: the host takes up at most `c^E` and
# `|c^E|_1 = 1`, exactly as `μ^S ≤ |c^H|_1` bounds the symbiont. At the hardest
# `r` the solver's gap is loose enough to exceed it, and the tighter of the two
# bounds is the honest one.
#
# So the true mean curve is bracketed. Panel **A** is the proportion of
# instances that hit the limit, per `r`; panel **B** puts the lower bound (what
# every other figure uses), the optimistic upper bound, and the
# limit-free subset on the same axes. If the three agree, the time limit is
# not shaping the result.

# %%
dl_by_r <- post |> group_by(r) |>
  summarise(n = dplyr::n(), n_detlimit = sum(flag_detlimit),
            frac_detlimit = mean(flag_detlimit),
            mip_gap_median = median(mip_gap[flag_detlimit], na.rm = TRUE),
            mip_gap_max = max(c(0, mip_gap[flag_detlimit]), na.rm = TRUE),
            .groups = "drop")
save_tab(dl_by_r, "SFig3_detlimit_by_r")
cat(sprintf("overall: %d of %d instances hit the time limit (%.2f %%)\n",
            sum(dl_by_r$n_detlimit), sum(dl_by_r$n),
            100 * sum(dl_by_r$n_detlimit) / sum(dl_by_r$n)))
cat(sprintf("worst r: %g with %.1f %% (median gap %.3g, max gap %.3g)\n",
            dl_by_r$r[which.max(dl_by_r$frac_detlimit)],
            100 * max(dl_by_r$frac_detlimit),
            dl_by_r$mip_gap_median[which.max(dl_by_r$frac_detlimit)],
            dl_by_r$mip_gap_max[which.max(dl_by_r$frac_detlimit)]))
dl_by_r |> filter(frac_detlimit > 0)

# %%
pS3A <- ggplot(dl_by_r, aes(r, 100 * frac_detlimit)) +
  geom_col(width = 1.1, fill = COL$mean2) +
  scale_x_r(rmax = max(post$r)) +
  labs(y = "instances at the time limit (%)") +
  theme_grhd()
pS3A

# %% [markdown]
# The bracket. `upper` replaces each time-limited incumbent by the solver's
# own bound on it and leaves every proven-optimal instance alone, so it is the
# most favourable curve the data can support; `lower` is the incumbent, which
# is what Fig.4A plots.

# %%
## The host's uptake is capped by the environment it lives in
## (`f^H <= c^E`, and `c^E = (1, 0, ..., 0)`), so `mu_H <= |c^E|_1 = 1` the same
## way `mu_S <= |c^H|_1` bounds the symbiont.  The solver's gap does not know
## that, and at the hardest r it is loose enough to put the "upper bound" above
## 1; the model bound is the tighter of the two, so the two are combined.
MU_H_MAX <- 1
post_b <- post |>
  mutate(mu_H_lower = mu_H,
         mu_H_upper_raw = ifelse(flag_detlimit & is.finite(mip_gap),
                                 mu_H * (1 + mip_gap), mu_H),
         mu_H_upper = pmin(mu_H_upper_raw, MU_H_MAX))
cat(sprintf("observed max mu_H = %.6f ; solver-gap upper bound capped at 1 for %d of %d time-limited cells
",
            max(post_b$mu_H), sum(post_b$mu_H_upper_raw > MU_H_MAX),
            sum(post_b$flag_detlimit)))
bp_lo   <- boot_curve_peak(post_b, "mu_H_lower", unit = NULL, label = "muH_lower")
bp_up   <- boot_curve_peak(post_b, "mu_H_upper", unit = NULL, label = "muH_upper")
bp_none <- boot_curve_peak(filter(post_b, !flag_detlimit), "mu_H_lower",
                           unit = NULL, label = "muH_no_detlimit")
brack <- bind_rows(mutate(bp_lo$curve,   series = "lower bound"),
                   mutate(bp_up$curve,   series = "upper bound"),
                   mutate(bp_none$curve, series = "proven optimal"))
save_tab(brack, "SFig3_muH_bracket")
save_tab(bind_rows(mutate(bp_lo$peak, series = "lower"),
                   mutate(bp_up$peak, series = "upper"),
                   mutate(bp_none$peak, series = "proven optimal")),
         "SFig3_muH_bracket_peaks")

cat(sprintf("largest gap between the lower and the upper mean curve: %.4g (at r = %g)\n",
            max(bp_up$curve$value - bp_lo$curve$value),
            bp_lo$curve$r[which.max(bp_up$curve$value - bp_lo$curve$value)]))
for (nm in c("lower", "upper", "optimal only")) {
  b <- switch(nm, lower = bp_lo, upper = bp_up, bp_none)
  cat(sprintf("r* (%-12s) = %g   95%% CI [%g, %g]\n", nm,
              b$peak$peak_at, b$peak$peak_ci_lo, b$peak$peak_ci_hi))
}

# %%
pS3B <- ggplot(brack, aes(r, value, colour = series, linetype = series)) +
  hline_mu0(post$mu_H_0[1], label = NULL) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = c("lower bound" = COL$mean,
                                 "upper bound" = COL$mean2,
                                 "proven optimal" = COL$accent), name = NULL) +
  scale_linetype_manual(values = c("lower bound" = "solid",
                                   "upper bound" = "22",
                                   "proven optimal" = "42"), name = NULL) +
  scale_x_r(rmax = max(post$r)) +
  labs(y = mx("mean"~mu^H~"after evolution")) +
  theme_grhd() + legend_inside(0.42, 0.99) +
  theme(legend.key.size = unit(2.6, "mm"))
pS3B

# %% [markdown]
# ### Where the time limit bites: by `k`, and by `m`
#
# The difficulty of the Level 3 MILP is not a constant of the study. It grows
# with the number of reactions per mode `k` and with the number of metabolites
# `m`, so any comparison **across** `k` or **across** `m` is also a comparison
# across solver difficulty. That has to be on the table before those sweeps
# are read.
#
# (There is nothing to show for the pre-evolution runs: Levels 1 and 2 are
# LPs and a QP, with no time limit to hit.)

# %%
dl_k <- NULL
if (!is.null(ks_post)) {
  dl_k <- ks_post |> group_by(k, r) |>
    summarise(n = dplyr::n(), frac_detlimit = mean(flag_detlimit),
              mip_gap_median = median(mip_gap[flag_detlimit], na.rm = TRUE),
              .groups = "drop")
  save_tab(dl_k, "SFig3_detlimit_by_k")
  ## NOTE: summarise() evaluates in order and each name shadows the column,
  ## so every use of the vectors r / n / frac_detlimit has to come before the
  ## line that replaces them with a scalar.
  print(dl_k |> group_by(k) |>
          summarise(r_max = max(r),
                    worst_r = r[which.max(frac_detlimit)],
                    worst_frac = max(frac_detlimit),
                    frac_detlimit = weighted.mean(frac_detlimit, n),
                    n_total = sum(n), .groups = "drop"))
}

# %%
pS3C <- NULL
if (!is.null(dl_k)) {
  pS3C <- ggplot(dl_k, aes(r, 100 * frac_detlimit, colour = factor(k))) +
    geom_line(linewidth = 0.7) +
    scale_colour_manual(values = c(`2` = cbp[["blue"]], `4` = cbp[["vermillion"]],
                                   `8` = cbp[["green"]], `16` = cbp[["purple"]]),
                        name = mx(italic(k))) +
    scale_x_r(rmax = max(dl_k$r)) +
    labs(y = "instances at the time limit (%)") +
    theme_grhd() + legend_inside(0.03, 0.99) +
    theme(legend.key.size = unit(2.6, "mm"))
}
pS3C

# %%
r5 <- cache_read("r5_cells")
## The time limit is only reached at r >= 22, and the r grid differs by m
## (m = 10: every r = 1..100; other m: 18-22 values, dense near m - 1), so a
## rate pooled over each m's own grid mostly measures that grid.  This panel
## alone uses the r values sampled for EVERY (design, m, host); r5 itself is
## left whole for Fig.6C and everything else.
r_common <- r5 |> distinct(design, m, host_id, r) |>
  group_by(r) |> summarise(n_blocks = dplyr::n(), .groups = "drop") |>
  filter(n_blocks == nrow(distinct(r5, design, m, host_id))) |> pull(r) |> sort()
cat("r values shared by every (design, m, host):", paste(r_common, collapse = ", "), "\n")
dl_m <- r5 |> filter(r %in% r_common) |> group_by(design, m) |>
  summarise(n = dplyr::n(), frac_detlimit = mean(flag_detlimit),
            frac_infeasible = mean(flag_infeasible), .groups = "drop") |>
  mutate(r_grid = paste(r_common, collapse = " "))
save_tab(dl_m, "SFig3_detlimit_by_m")
pS3D <- ggplot(dl_m, aes(m, 100 * frac_detlimit, colour = design, shape = design)) +
  geom_line(linewidth = 0.7) + geom_point(size = 1.1) +
  scale_colour_manual(values = c(A = COL$mean2, B = COL$mean), name = NULL,
                      labels = c(A = mx(r^H == 2.4*m),
                                 B = mx(r^H == 24))) +
  scale_shape_manual(values = c(A = 16, B = 17), name = NULL,
                     labels = c(A = mx(r^H == 2.4*m),
                                B = mx(r^H == 24))) +
  scale_x_continuous(breaks = seq(5, 25, 5)) +
  labs(x = mx(italic(m)~"(No. of metabolites)"),
       y = "instances at the time limit (%)") +
  theme_grhd() + legend_inside(0.03, 0.99) +
  theme(legend.key.size = unit(2.6, "mm"))
pS3D

# %% [markdown]
# **How to use this.** Where the rate is near zero the curve can be read at
# face value. Where it is not, the honest statement is the bracket of panel B,
# and a difference between two `k` (or two `m`) that is smaller than the
# spread between the lower and the upper bound is not a difference the data
# can support.

# %% [markdown]
# ## S4 — two diagnostics worth having
#
# ### S4A `b` against `r₊`
#
# `r₊` is the number of modes carrying non-zero weight in the optimal symbiont
# metabolism — the part of the genome that is actually used. `b` is the number
# of metabolites the host pins. Plotting them against each other shows whether
# "more genome in use" translates into "more host control", before and after
# host evolution.

# %%
sym <- cache_read("pre_by_symbiont")
pS4A_pre <- ggplot(sym, aes(r_plus_mean, b_mean, colour = r)) +
  raster_points(geom_point(size = 0.3, alpha = 0.45, shape = 16)) +
  scale_colour_viridis_c(name = mx(italic(r))) +
  labs(x = mx("mean"~italic(r)["+"]),
       y = mx("mean"~italic(b)),
       subtitle = "before host evolution") +
  theme_grhd() + theme(plot.subtitle = element_text(size = BASE_PT - 2.5))

pS4A_post <- ggplot(post, aes(r_plus, n_bounded, colour = r)) +
  raster_points(geom_point(size = 0.4, alpha = 0.5, shape = 16)) +
  scale_colour_viridis_c(name = mx(italic(r))) +
  labs(x = mx(italic(r)["+"]~"(modes in use)"),
       y = mx(italic(b)~"(bounded metabolites)"),
       subtitle = "after host evolution") +
  theme_grhd() + theme(plot.subtitle = element_text(size = BASE_PT - 2.5))
fig_size(8, 3.2); pS4A_pre | pS4A_post

# %%
tab_S4A <- bind_rows(
  corr_one(sym$r_plus_mean, sym$b_mean) |> mutate(stage = "pre", scope = "pooled"),
  partial_spearman(sym$r_plus_mean, sym$b_mean, sym$r) |>
    mutate(p = NA_real_, stage = "pre", scope = "partial_on_r"),
  corr_one(post$r_plus, post$n_bounded) |> mutate(stage = "post", scope = "pooled"),
  partial_spearman(post$r_plus, post$n_bounded, post$r) |>
    mutate(p = NA_real_, stage = "post", scope = "partial_on_r"))
save_tab(tab_S4A, "SFig4A_b_vs_rplus_correlations")
tab_S4A |> select(stage, scope, rho, n)

# %% [markdown]
# ### S4B sensitivity against the evolved `μ^H`
#
# The mechanism claims that a symbiont the host can *move* is a symbiont the
# host can *exploit*. The direct test is whether a symbiont's sensitivity,
# measured **before** evolution, predicts the host's growth **after** it.
# Pooling over `r` would let the `r` gradient answer for us, so the partial
# correlation on `r` and the within-`r` values are given alongside.

# %%
join <- post_select(cache_read("post_join"))
pS4B <- ggplot(join, aes(s_pre_mean, mu_H_post, colour = r)) +
  raster_points(geom_point(size = 0.4, alpha = 0.55, shape = 16)) +
  geom_hline(yintercept = join$mu_H_0[1], colour = COL$ref, linewidth = 0.4) +
  scale_colour_viridis_c(name = mx(italic(r))) +
  scale_x_log10(labels = scales::label_log()) +
  labs(x = mx("mean"~italic(s)~"before evolution"),
       y = LAB_MU_H_POST) +
  theme_grhd()

pS4B_post <- ggplot(post, aes(s, mu_H, colour = r)) +
  raster_points(geom_point(size = 0.4, alpha = 0.55, shape = 16)) +
  geom_hline(yintercept = post$mu_H_0[1], colour = COL$ref, linewidth = 0.4) +
  scale_colour_viridis_c(name = mx(italic(r))) +
  scale_x_log10(labels = scales::label_log()) +
  labs(x = mx(italic(s)~"at the evolved"~c^H),
       y = LAB_MU_H_POST) +
  theme_grhd()
fig_size(8, 3.2); pS4B | pS4B_post

# %%
tab_S4B <- bind_rows(
  corr_one(join$s_pre_mean, join$mu_H_post) |>
    mutate(x = "s before evolution", scope = "pooled"),
  partial_spearman(join$s_pre_mean, join$mu_H_post, join$r) |>
    mutate(p = NA_real_, x = "s before evolution", scope = "partial_on_r"),
  corr_one(post$s, post$mu_H) |> mutate(x = "s at the evolved c^H", scope = "pooled"),
  partial_spearman(post$s, post$mu_H, post$r) |>
    mutate(p = NA_real_, x = "s at the evolved c^H", scope = "partial_on_r"))
within_S4B <- join |> group_by(r) |>
  group_modify(~ corr_one(.x$s_pre_mean, .x$mu_H_post)) |> ungroup()
save_tab(tab_S4B, "SFig4B_s_vs_muH_post_correlations")
save_tab(within_S4B, "SFig4B_s_vs_muH_post_within_r")
tab_S4B |> select(x, scope, rho, n)

# %% [markdown]
# ### How big is the effect, not just how reliable
#
# A rank correlation says the two move together. It does not say how much host
# growth is at stake, and with `n` in the tens of thousands a small `\u03c1` would
# still be "significant".
#
# **Look at the distribution of `s` first, because it decides which effect
# size is even meaningful.** After host evolution most symbionts sit at
# exactly `s = 0`, and a tertile split of a variable that is mostly ties is
# degenerate \u2014 both cut points land on zero and the "top third" becomes the
# whole sample. So the effect is reported in two pieces: the step from
# `s = 0` to `s > 0`, and the gradient *within* `s > 0`.

# %%
zero_s <- post |>
  summarise(n = dplyr::n(),
            frac_s_zero = mean(s <= 0),
            frac_at_mu0_given_s0 = mean(abs(mu_H - mu_H_0) < 1e-9 & s <= 0) / mean(s <= 0),
            frac_s0_given_at_mu0 = mean(abs(mu_H - mu_H_0) < 1e-9 & s <= 0) /
              mean(abs(mu_H - mu_H_0) < 1e-9))
save_tab(zero_s, "SFig4B_zero_s")
cat(sprintf("s = 0 for %.1f %% of evolved symbionts.\n", 100 * zero_s$frac_s_zero))
cat(sprintf("of those, %.2f %% sit at exactly mu^H_0; of the symbionts at exactly mu^H_0, %.2f %% have s = 0.\n",
            100 * zero_s$frac_at_mu0_given_s0, 100 * zero_s$frac_s0_given_at_mu0))

# %% [markdown]
# That is the headline effect and it is not a correlation at all: **a symbiont
# the host cannot move (`s = 0`) leaves the host exactly where it started.**
# The gradient among the symbionts it *can* move is the second, smaller story.

# %%
post_pos <- filter(post, s > 0)
eff_pooled <- bind_rows(
  effect_slope(join$s_pre_mean, join$mu_H_post, log_x = TRUE) |>
    mutate(x = "s before evolution (s > 0)", scope = "pooled"),
  effect_slope(post_pos$s, post_pos$mu_H, log_x = TRUE) |>
    mutate(x = "s at the evolved c^H (s > 0)", scope = "pooled"))
save_tab(eff_pooled, "SFig4B_effect_slopes")
eff_pooled |> select(x, slope, slope_ci_lo, slope_ci_hi, std_slope, r2, n)

# %%
contr <- bind_rows(
  contrast_groups(post$mu_H[post$s <= 0], post$mu_H[post$s > 0],
                  label = "s_zero_vs_positive") |>
    mutate(comparison = "s = 0  vs  s > 0", set = "all evolved symbionts"),
  group_contrast(post_pos$s, post_pos$mu_H, label = "s_tertiles_positive") |>
    mutate(comparison = "bottom vs top third of s", set = "s > 0 only"),
  group_contrast(post$s, post$mu_H, label = "s_tertiles_all") |>
    mutate(comparison = "bottom vs top third of s", set = "all (degenerate)"))
save_tab(contr, "SFig4B_effect_contrasts")
cat(sprintf("for scale: mu^H_0 = %.4f, mean mu^H after evolution = %.3f\n",
            post$mu_H_0[1], mean(post$mu_H)))
contr |> select(comparison, set, n_low, n_high, mean_low, mean_high, diff,
                diff_ci_lo, diff_ci_hi, cohens_d, cliffs_delta, degenerate_split)

# %% [markdown]
# The last row is kept deliberately, with its `degenerate_split` flag set: it
# is what a tertile split of `s` reports if the ties are not noticed, and it
# understates the effect by a factor of three. The middle row is the one to
# quote for the gradient.
#
# The same, one `r` at a time, within `s > 0`. If the effect survives here it
# is not the `r` gradient in disguise.

# %%
first_ok <- function(v) { v <- v[!is.na(v)]; if (length(v)) v[1] else NA_real_ }
eff_within <- post_pos |> group_by(r) |>
  filter(dplyr::n() >= 30) |>
  group_modify(~ bind_rows(
    effect_slope(.x$s, .x$mu_H, log_x = TRUE),
    group_contrast(.x$s, .x$mu_H, B = 400, label = paste0("within", .y$r)) |>
      select(diff, diff_ci_lo, diff_ci_hi, cliffs_delta))) |>
  ungroup() |>
  group_by(r) |>
  summarise(across(c(slope, r2, diff, diff_ci_lo, diff_ci_hi, cliffs_delta),
                   first_ok),
            .groups = "drop")
save_tab(eff_within, "SFig4B_effect_within_r")
eff_within |> filter(r %in% c(2, 5, 10, 13, 20, 30, 50))

# %%
pS4B_eff <- ggplot(eff_within, aes(r, diff)) +
  geom_hline(yintercept = 0, colour = COL$ref, linewidth = 0.4) +
  geom_ribbon(aes(ymin = diff_ci_lo, ymax = diff_ci_hi), fill = COL$mean, alpha = 0.25) +
  geom_line(colour = COL$mean, linewidth = 0.8) +
  scale_x_r(rmax = max(eff_within$r)) +
  labs(y = mx(atop("top third minus bottom third of"~italic(s),
                           "in"~mu^H~"after evolution ("*italic(s)>0*")"))) +
  theme_grhd()
pS4B_eff

# %% [markdown]
# And the shape of the relation itself, as binned means rather than a fitted
# line. Deciles are taken **within `s > 0`**; a decile plot over all cells puts
# seven of the ten bins on the same point, which is true but tells the reader
# nothing about the gradient. The `s = 0` group is drawn separately, at the
# left, as the flat baseline it is.

# %%
dec <- post_pos |>
  mutate(bin = ntile(s, 10)) |>
  group_by(bin) |>
  summarise(s_median = median(s), mu_H_mean = mean(mu_H),
            se = sd(mu_H) / sqrt(dplyr::n()), n = dplyr::n(), .groups = "drop")
zero_grp <- post |> filter(s <= 0) |>
  summarise(mu_H_mean = mean(mu_H), se = sd(mu_H) / sqrt(dplyr::n()),
            n = dplyr::n())
save_tab(dec, "SFig4B_s_deciles")
save_tab(zero_grp, "SFig4B_s_zero_group")
dec

# %%
pS4B_dec <- ggplot(dec, aes(s_median, mu_H_mean)) +
  hline_mu0(post$mu_H_0[1], label = NULL) +
  geom_hline(yintercept = zero_grp$mu_H_mean, colour = COL$mean2,
             linetype = "22", linewidth = 0.5) +
  annotate("text", x = min(dec$s_median), y = zero_grp$mu_H_mean,
           label = sprintf("s = 0  (n = %s)", format(zero_grp$n, big.mark = ",")),
           size = TXT(7), hjust = 0, vjust = -0.6, colour = COL$mean2,
           family = FONT_FAMILY) +
  geom_errorbar(aes(ymin = mu_H_mean - 1.96 * se, ymax = mu_H_mean + 1.96 * se),
                width = 0, colour = COL$mean, linewidth = 0.4) +
  geom_line(colour = COL$mean, linewidth = 0.7) +
  geom_point(colour = COL$mean, size = 1.2) +
  scale_x_log10() +
  labs(x = mx("sensitivity"~italic(s)~"at the evolved"~c^H~"(decile median,"~italic(s)>0*")"),
       y = mx("mean"~mu^H~"after evolution")) +
  theme_grhd()
pS4B_dec

# %% [markdown]
# The within-`r` rank correlation between the pre-evolution `s` and the
# post-evolution `μ^H`, for comparison with the effect sizes above.

# %%
pS4B_rho <- ggplot(within_S4B, aes(r, rho)) +
  geom_hline(yintercept = 0, colour = COL$ref, linewidth = 0.4) +
  geom_line(colour = COL$mean, linewidth = 0.8) +
  geom_point(colour = COL$mean, size = 1) +
  scale_x_r(rmax = max(join$r)) +
  labs(y = mx(rho~"("*italic(s)*" before,"~mu^H~"after) within one"~italic(r))) +
  theme_grhd()
fig_size(6, 3.2); pS4B_rho

# %% [markdown]
# ### S4C — `s` as a gate, and as a predictor of extent
#
# The pooled correlation mixes two things: whether a host can exploit a
# symbiont **at all**, and **how far** it gets when it can. They are separated
# here, using only the **pre-evolution** `s` (the `s` at the evolved `c^H`
# depends on what the host chose, so it cannot show "potential").
#
# * **gate** — the share of pairs that become mutualistic, i.e. `μ^H` after
#   evolution above `μ^H_0` by more than `MUT_DELTA` and `μ^S` after evolution
#   above `MUT_DELTA` (both partners grow; the same definition as Fig.4 and
#   the shared-direction SFig), for `s = 0` against `s > 0`, and within
#   `s > 0` against `s`;
# * **extent** — `μ^H` after evolution among the mutualistic pairs only.
#
# Among `s > 0`, `s` still falls with `r`, so every `s` effect is also given
# **adjusted for `r`**: bins of `s` are quintiles of the rank *within each `r`*,
# and the regressions carry a natural spline in `r` (4 df).
#
# Upper-quantile regression is not done. The 90th percentile runs into the
# model bound `μ^H ≤ 1` in the top bins (it is reported per bin instead), so a
# "ceiling set by `s`" could not be told apart from that bound.

# %%
suppressPackageStartupMessages(library(splines))
MUT_DELTA <- 1e-2   # mutualistic: mu_H > mu_H_0 + MUT_DELTA and mu_S > MUT_DELTA
g <- join |>
  mutate(mut = mu_H_post > mu_H_0 + MUT_DELTA & mu_S_post > MUT_DELTA,
         s_pos = s_pre_mean > 0,
         log_s = ifelse(s_pos, log10(s_pre_mean), NA_real_))

## ---- the gate: s = 0 against s > 0 ----------------------------------------
gate_all <- g |> group_by(s_pos) |>
  summarise(n = dplyr::n(), n_mut = sum(mut), p_mut = mean(mut),
            mu_H_mean = mean(mu_H_post), .groups = "drop")
gate_by_r <- g |> group_by(r) |>
  summarise(n_s0 = sum(!s_pos), n_spos = sum(s_pos),
            p_mut_s0   = if (any(!s_pos)) mean(mut[!s_pos]) else NA_real_,
            p_mut_spos = if (any(s_pos))  mean(mut[s_pos])  else NA_real_,
            frac_s_pos = mean(s_pos), .groups = "drop")
save_tab(gate_all, "SFig4C_gate_overall")
save_tab(gate_by_r, "SFig4C_gate_by_r")
gate_all
cat(sprintf("mutualistic pairs that had s = 0 before evolution: %d of %d (%.1f %%)\n",
            sum(g$mut & !g$s_pos), sum(g$mut), 100 * sum(g$mut & !g$s_pos) / sum(g$mut)))
cat(sprintf("r with p_mut(s > 0) > p_mut(s = 0), where both groups exist: %d of %d\n",
            sum(gate_by_r$p_mut_spos > gate_by_r$p_mut_s0, na.rm = TRUE),
            sum(!is.na(gate_by_r$p_mut_spos) & !is.na(gate_by_r$p_mut_s0))))

# %% [markdown]
# Bins of `s` among `s > 0`: pooled deciles (confounded with `r`) and
# quintiles of the within-`r` rank (adjusted for `r`). `s = 0` is the
# reference row. `p_mut` carries an exact binomial 95 % CI; the extent is the
# mean `μ^H` of the mutualistic pairs in the bin, with a normal 95 % CI.

# %%
gp <- g |> filter(s_pos) |>
  mutate(bin_pooled = ntile(s_pre_mean, 10)) |>
  group_by(r) |>
  mutate(bin_within_r = if (dplyr::n() >= 5) ntile(s_pre_mean, 5) else NA_integer_) |>
  ungroup()

bin_summary <- function(d) {
  d |> summarise(n = dplyr::n(), n_mut = sum(mut),
                 s_median = median(s_pre_mean), r_median = median(r),
                 p_mut = n_mut / n,
                 p_mut_lo = stats::binom.test(n_mut, n)$conf.int[1],
                 p_mut_hi = stats::binom.test(n_mut, n)$conf.int[2],
                 extent_mean = if (n_mut > 0) mean(mu_H_post[mut]) else NA_real_,
                 extent_se = if (n_mut > 1) sd(mu_H_post[mut]) / sqrt(n_mut) else NA_real_,
                 mu_H_q50 = quantile(mu_H_post, 0.5), mu_H_q90 = quantile(mu_H_post, 0.9),
                 .groups = "drop")
}
s_bins <- bind_rows(
  g |> filter(!s_pos) |> bin_summary() |> mutate(scheme = "s = 0", bin = 0L),
  gp |> group_by(bin = bin_pooled) |> bin_summary() |> mutate(scheme = "pooled decile"),
  gp |> filter(!is.na(bin_within_r)) |> group_by(bin = bin_within_r) |> bin_summary() |>
    mutate(scheme = "within-r quintile")) |>
  mutate(extent_lo = extent_mean - 1.96 * extent_se,
         extent_hi = extent_mean + 1.96 * extent_se) |>
  relocate(scheme, bin)
save_tab(s_bins, "SFig4C_s_bins")
s_bins |> select(scheme, bin, n, s_median, r_median, p_mut, extent_mean, mu_H_q90)

# %% [markdown]
# The same two parts as regressions, among `s > 0` only, with and without
# `log10 s`, both carrying `ns(r, 4)`:
#
# * **gate** — logistic regression of "mutualistic"; odds ratio per 10-fold
#   `s`, the likelihood-ratio test for adding `log10 s`, and the AUC with and
#   without it;
# * **extent** — linear regression of `μ^H` among the mutualistic pairs; slope
#   per 10-fold `s`, and the share of the variance left after `r` that
#   `log10 s` explains (partial `R²`).

# %%
auc <- function(y, p) {
  rk <- rank(p); n1 <- sum(y); n0 <- sum(!y)
  (sum(rk[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
lg0 <- glm(mut ~ ns(r, df = 4), family = binomial, data = gp)
lg1 <- glm(mut ~ log_s + ns(r, df = 4), family = binomial, data = gp)
lg_s <- glm(mut ~ log_s, family = binomial, data = gp)
lg_ci <- exp(confint.default(lg1)["log_s", ])
lrt <- anova(lg0, lg1, test = "Chisq")

gm <- filter(gp, mut)
ex0 <- lm(mu_H_post ~ ns(r, df = 4), data = gm)
ex1 <- lm(mu_H_post ~ log_s + ns(r, df = 4), data = gm)
ex_ci <- confint(ex1)["log_s", ]
rss0 <- sum(residuals(ex0)^2); rss1 <- sum(residuals(ex1)^2)
ex_within <- gm |> group_by(r) |> filter(dplyr::n() >= 20) |>
  summarise(rho = suppressWarnings(cor(s_pre_mean, mu_H_post, method = "spearman")),
            n = dplyr::n(), .groups = "drop")

s_models <- tibble::tribble(
  ~part,    ~quantity,                                  ~value,                     ~ci_lo,        ~ci_hi,
  "gate",   "odds ratio per 10-fold s (adjusted for r)", exp(coef(lg1)[["log_s"]]), lg_ci[[1]],    lg_ci[[2]],
  "gate",   "LRT p for adding log10 s to r",            lrt$`Pr(>Chi)`[2],          NA,            NA,
  "gate",   "AUC, r only",                              auc(gp$mut, fitted(lg0)),   NA,            NA,
  "gate",   "AUC, log10 s only",                        auc(gp$mut, fitted(lg_s)),  NA,            NA,
  "gate",   "AUC, r + log10 s",                         auc(gp$mut, fitted(lg1)),   NA,            NA,
  "extent", "slope per 10-fold s (adjusted for r)",     coef(ex1)[["log_s"]],       ex_ci[[1]],    ex_ci[[2]],
  "extent", "R2, r only",                               summary(ex0)$r.squared,     NA,            NA,
  "extent", "R2, r + log10 s",                          summary(ex1)$r.squared,     NA,            NA,
  "extent", "partial R2 of log10 s given r",            (rss0 - rss1) / rss0,       NA,            NA,
  "extent", "median within-r Spearman rho (r with >= 20 mutualistic pairs)",
                                                        median(ex_within$rho),      min(ex_within$rho), max(ex_within$rho)
) |> mutate(n_gate = nrow(gp), n_extent = nrow(gm))
save_tab(s_models, "SFig4C_s_models")
s_models |> select(part, quantity, value, ci_lo, ci_hi)

# %% [markdown]
# ### Does the gate or the extent make the peak of Fig.4A?
#
# At every `r`, the mean `μ^H` after evolution splits exactly into
#
# ```
# mean μ^H(r) = P(s > 0 | r) · E[μ^H | s > 0, r] + P(s = 0 | r) · E[μ^H | s = 0, r]
# ```
#
# and `E[μ^H | s = 0, r]` stays at `μ^H_0`. Holding one factor at its average
# over `r` and letting the other vary shows which of them puts the peak where
# it is.

# %%
dec_r <- g |> group_by(r) |>
  summarise(n = dplyr::n(), mu_H_mean = mean(mu_H_post), p_spos = mean(s_pos),
            n_spos = sum(s_pos),
            e_spos = if (any(s_pos)) mean(mu_H_post[s_pos]) else NA_real_,
            e_s0 = if (any(!s_pos)) mean(mu_H_post[!s_pos]) else NA_real_,
            mu_H_0 = first(mu_H_0), .groups = "drop")
p_bar <- weighted.mean(dec_r$p_spos, dec_r$n)
e1_bar <- weighted.mean(dec_r$e_spos, dec_r$n_spos, na.rm = TRUE)
e0 <- dec_r$mu_H_0[1]
dec_r <- dec_r |>
  mutate(only_gate_varies   = p_spos * e1_bar + (1 - p_spos) * e0,
         only_extent_varies = p_bar * ifelse(is.na(e_spos), e0, e_spos) + (1 - p_bar) * e0,
         share_gain_from_spos = p_spos * (ifelse(is.na(e_spos), e0, e_spos) - e0) /
           (mu_H_mean - e0))
dec_peaks <- tibble(
  curve = c("mean mu_H (Fig.4A)", "P(s > 0 | r)", "E[mu_H | s > 0, r]",
            "only the gate varies", "only the extent varies"),
  column = c("mu_H_mean", "p_spos", "e_spos", "only_gate_varies", "only_extent_varies")) |>
  mutate(peak_at = vapply(column, function(v) dec_r$r[which.max(dec_r[[v]])], numeric(1)),
         peak_value = vapply(column, function(v) max(dec_r[[v]], na.rm = TRUE), numeric(1)))
save_tab(dec_r, "SFig4C_decomposition_by_r")
save_tab(dec_peaks, "SFig4C_decomposition_peaks")
dec_peaks
cat(sprintf("E[mu_H | s = 0, r] - mu_H_0: max %.2g over r\n", max(abs(dec_r$e_s0 - e0), na.rm = TRUE)))

# %%
lab_bins <- c("0" = "s = 0", "1" = "Q1", "2" = "Q2", "3" = "Q3", "4" = "Q4", "5" = "Q5")
bins_plot <- s_bins |> filter(scheme %in% c("s = 0", "within-r quintile")) |>
  mutate(bin = factor(bin, levels = 0:5, labels = lab_bins))
pS4C_gate <- ggplot(bins_plot, aes(bin, p_mut)) +
  geom_errorbar(aes(ymin = p_mut_lo, ymax = p_mut_hi), width = 0.2, colour = COL$mean) +
  geom_point(colour = COL$mean, size = 1.4) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = mx(italic(s)~"before evolution (quintile within"~italic(r)*")"),
       y = "mutualistic pairs (fraction)") +
  theme_grhd()
pS4C_extent <- ggplot(filter(bins_plot, bin != "s = 0"), aes(bin, extent_mean)) +
  geom_errorbar(aes(ymin = extent_lo, ymax = extent_hi), width = 0.2, colour = COL$mean) +
  geom_point(aes(shape = "mean"), colour = COL$mean, size = 1.4) +
  geom_point(aes(y = mu_H_q90, shape = "90th percentile (all s > 0)"),
             colour = COL$point, size = 1.2) +
  scale_shape_manual(values = c("mean" = 16, "90th percentile (all s > 0)" = 1), name = NULL) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = mx(italic(s)~"before evolution (quintile within"~italic(r)*")"),
       y = mx(mu^H~"of mutualistic pairs")) +
  theme_grhd() + legend_inside(0.02, 0.02) +
  theme(legend.justification = c(0, 0), legend.key.size = unit(2.4, "mm"))
dec_long <- dec_r |>
  select(r, `mean mu^H` = mu_H_mean, `P(s > 0)` = p_spos,
         `mean mu^H given s > 0` = e_spos) |>
  pivot_longer(-r, names_to = "curve", values_to = "value")
DEC_BREAKS <- c("mean mu^H", "P(s > 0)", "mean mu^H given s > 0")
DEC_LABELS <- mx("mean"~mu^H, italic(P)*"("*italic(s)>0*")", "mean"~mu^H~"given"~italic(s)>0)
pS4C_dec <- ggplot(dec_long, aes(r, value, colour = curve, linetype = curve)) +
  hline_mu0(e0, label = NULL) +
  geom_line(linewidth = 0.7, na.rm = TRUE) +
  scale_colour_manual(values = setNames(c(COL$mean, cbp[["skyblue"]], "black"), DEC_BREAKS),
                      breaks = DEC_BREAKS, labels = DEC_LABELS, name = NULL) +
  scale_linetype_manual(values = setNames(c("solid", "solid", "22"), DEC_BREAKS),
                        breaks = DEC_BREAKS, labels = DEC_LABELS, name = NULL) +
  scale_x_r(rmax = max(dec_r$r)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(y = "value") +
  theme_grhd() + legend_inside(0.99, 0.99) +
  theme(legend.justification = c(1, 1), legend.key.size = unit(2.4, "mm"))
fig_size(9, 3.2); pS4C_gate | pS4C_extent | pS4C_dec

# %% [markdown]
# ## Save

# %%
metaS <- list(data = c("data/raw/post_paper_h*", "data/raw/pre_k_sweep_*",
                       "data/raw/*_fixed_repertoire_*"),
              cache = c("post_hosts_cells", "ksweep_by_symbiont",
                        "fixedrep_by_symbiont", "fixedrep_post_cells"),
              raster_layer_dpi = RASTER_DPI, r_axis = "linear",
              note = "no GA implementation exists; S3 shows the MILP time-limit bracket instead")

## The individual panels, in case any of them is wanted on its own ----------
save_fig(pS1A,     "SFig1A_muH_peak_by_host",     size = "small", meta = metaS)
save_fig(pS1A_rel, "SFig1A_muH_peak_by_host_rel", size = "small", meta = metaS)
save_fig(pS1B,     "SFig1B_reaction_coverage",    size = "small", meta = metaS)
if (!is.null(pS1C_k)) save_fig(pS1C_k, "SFig1C_ksweep_post_muH", size = "small", meta = metaS)
save_fig(pS1C_k_pre,   "SFig1C_ksweep_pre_sensitivity",  size = "small", meta = metaS)
save_fig(pS1D_pre,     "SFig1D_fixedrep_pre_sensitivity", size = "small", meta = metaS)
save_fig(pS1D_pre_muH, "SFig1D_fixedrep_pre_muH",         size = "small", meta = metaS)
if (!is.null(pS1D))  save_fig(pS1D,  "SFig1D_fixedrep_post_muH",  size = "small", meta = metaS)
save_fig(pS1E, "SFig1E_structure_peaks_by_scheme", width = PNAS_WIDTH[["large"]], height = 150,
         meta = metaS)
if (!is.null(pS3C))  save_fig(pS3C,  "SFig3C_detlimit_by_k",      size = "small", meta = metaS)
save_fig(pS3D,       "SFig3D_detlimit_by_m",      size = "small", meta = metaS)
save_fig(pS4B_eff,   "SFig4B_effect_within_r",    size = "small", meta = metaS)
save_fig(pS4B_dec,   "SFig4B_s_deciles",          size = "small", meta = metaS)
save_fig(pS2,      "SFig2_raw_muH_nine_hosts",    size = "medium", meta = metaS)
save_fig(pS3A,     "SFig3A_detlimit_fraction",    size = "small", meta = metaS)
save_fig(pS3B,     "SFig3B_muH_bracket",          size = "small", meta = metaS)
save_fig(pS4B_rho, "SFig4B_within_r_rho",         size = "small", meta = metaS)
sfig4c <- (pS4C_gate | pS4C_extent | pS4C_dec) +
  plot_annotation(tag_levels = "A") & tag_theme()
save_fig(sfig4c, "SFig4C_s_gate_and_extent", size = "main_tall", meta = metaS)

# %% [markdown]
# ### SFig.1, merged
#
# One figure, four panels, at the PNAS medium size (11 × 11 cm):
#
# * **A** the peak, host by host
# * **B** how many distinct reactions `r` modes cover (exact + simulated)
# * **C** the `k` sweep
# * **D** the fixed-repertoire scheme, where the repertoire is constant in `r`

# %%
pC <- if (!is.null(pS1C_k)) pS1C_k else pS1C_k_pre
pD <- if (!is.null(pS1D)) pS1D else pS1D_pre
sfig1 <- (pS1A | pS1B) / (pC | pD) +
  plot_annotation(tag_levels = "A") & tag_theme()
save_fig(sfig1, "SFig1_combined", size = "medium", meta = metaS)
fig_size(8, 7); sfig1

# %% [markdown]
# ### SFig.3, merged — optimiser quality

# %%
sfig3 <- (pS3A | pS3B) / ((if (!is.null(pS3C)) pS3C else plot_spacer()) | pS3D) +
  plot_annotation(tag_levels = "A") & tag_theme()
save_fig(sfig3, "SFig3_combined", size = "medium", meta = metaS)
fig_size(8, 7); sfig3

# %% [markdown]
# ### SFig.4, merged

# %%
## b against r+ after evolution (pS4A_post) is left out of the merged figure;
## its correlations are still in SFig4A_b_vs_rplus_correlations.
sfig4 <- (pS4A_pre | pS4B | pS4B_post) +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "A") & tag_theme()
save_fig(sfig4, "SFig4_combined", size = "main_tall", meta = metaS)
fig_size(8, 7); sfig4
