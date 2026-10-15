## ---------------------------------------------------------------------------
## 03_stats.R : the statistics used by the main figures.
##
## Conventions fixed by instructions/02_reproduction_tasks.md of the
## simulation repo, and reproduced here so that the R and the Python analysis
## answer the same question in the same way:
##
##   * bootstrap: percentile, B = 2000, resampling unit = the symbiont R^S,
##     drawn INDEPENDENTLY WITHIN EACH r  (different r are different draws of
##     R^S, so sharing indices across r would be wrong);
##   * argmax ties are broken towards the SMALLEST r;
##   * heavy-tailed quantities (the sensitivity s) are summarised by the median
##     and by a two-sided 5 % trimmed mean, with the raw mean only alongside;
##   * HDR contours: Gaussian KDE with a FIXED isotropic bandwidth on a
##     256 x 256 grid; the contour level is the smallest density whose
##     super-level set carries the requested probability mass.
## ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(purrr)
})

## Deterministic seed from a set of labels, so every table is reproducible.
seed_from <- function(...) {
  h <- digest::digest(paste(c(MASTER_SEED, ...), collapse = "|"), algo = "xxhash32")
  as.integer(strtoi(substr(h, 1, 7), 16L) %% 2147483647L) + 1L
}

trimmed_mean <- function(x, prop = TRIM_PROP) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  mean(x, trim = prop)
}

## One row of descriptive statistics for one vector.
summarise_vec <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) {
    return(tibble(n = 0L, mean = NA_real_, median = NA_real_, trimmed_mean = NA_real_,
                  sd = NA_real_, q25 = NA_real_, q75 = NA_real_,
                  min = NA_real_, max = NA_real_))
  }
  q <- stats::quantile(x, c(0.25, 0.75), names = FALSE, type = 7)
  tibble(n = length(x), mean = mean(x), median = stats::median(x),
         trimmed_mean = trimmed_mean(x), sd = stats::sd(x),
         q25 = q[1], q75 = q[2], min = min(x), max = max(x))
}

## Descriptive statistics of several columns, by one grouping column.
group_summary <- function(df, by, cols) {
  map_dfr(cols, function(cc) {
    df |>
      group_by(.data[[by]]) |>
      group_modify(~ summarise_vec(.x[[cc]])) |>
      ungroup() |>
      mutate(variable = cc, .after = dplyr::all_of(by))
  })
}

## ---------------------------------------------------------------------------
## Bootstrap of a curve and of its argmax
## ---------------------------------------------------------------------------
## df     long data: one row per cell
## value  the column to average
## group  the x axis of the curve (default r)
## unit   the resampling unit; cells of the same unit are averaged first
## unit_fun  how cells of one symbiont are reduced to one number
## agg_fun   how the symbionts of one r are combined into the curve value
##           (use median/median for heavy-tailed quantities such as s)
boot_curve_peak <- function(df, value, group = "r", unit = "symbiont_id",
                            B = BOOT_B, alpha = 0.05, label = value,
                            unit_fun = mean, agg_fun = mean) {
  groups <- sort(unique(df[[group]]))
  per_unit <- lapply(groups, function(g) {
    d <- df[df[[group]] == g, , drop = FALSE]
    v <- if (!is.null(unit) && unit %in% names(d) && any(!is.na(d[[unit]]))) {
      tapply(d[[value]], d[[unit]], function(z) unit_fun(z[is.finite(z)]))
    } else {
      d[[value]]
    }
    as.numeric(v[is.finite(v)])
  })
  names(per_unit) <- as.character(groups)

  set.seed(seed_from("boot_curve_peak", label, group, unit))
  fast_mean <- identical(agg_fun, mean)
  boot <- matrix(NA_real_, nrow = B, ncol = length(groups))
  for (j in seq_along(groups)) {
    x <- per_unit[[j]]
    if (!length(x)) next
    idx <- matrix(sample.int(length(x), B * length(x), replace = TRUE), nrow = B)
    rs <- matrix(x[idx], nrow = B)
    boot[, j] <- if (fast_mean) rowMeans(rs) else apply(rs, 1, agg_fun)
  }

  point <- vapply(per_unit, function(x) if (length(x)) agg_fun(x) else NA_real_, 0)
  curve <- tibble(
    !!group := groups,
    value = as.numeric(point),
    ci_lo = apply(boot, 2, stats::quantile, probs = alpha / 2, na.rm = TRUE),
    ci_hi = apply(boot, 2, stats::quantile, probs = 1 - alpha / 2, na.rm = TRUE),
    n = vapply(per_unit, length, 0L)
  )

  ok <- stats::complete.cases(boot)
  bo <- boot[ok, , drop = FALSE]
  jpk <- which.max(point)                       # ties -> smallest group key
  arg <- groups[max.col(bo, ties.method = "first")]
  peak <- tibble(
    variable = label,
    peak_at = groups[jpk],
    peak_value = as.numeric(point[jpk]),
    peak_ci_lo = unname(stats::quantile(arg, alpha / 2)),
    peak_ci_hi = unname(stats::quantile(arg, 1 - alpha / 2)),
    B_used = nrow(bo)
  )
  for (nm in c("first", "last")) {
    jend <- if (nm == "first") 1L else length(groups)
    d <- bo[, jpk] - bo[, jend]
    peak[[paste0("diff_peak_minus_", nm)]] <- as.numeric(point[jpk] - point[jend])
    peak[[paste0("diff_peak_minus_", nm, "_lo")]] <- unname(stats::quantile(d, alpha / 2))
    peak[[paste0("diff_peak_minus_", nm, "_hi")]] <- unname(stats::quantile(d, 1 - alpha / 2))
    peak[[paste0(nm, "_r")]] <- groups[jend]
  }
  list(curve = curve, peak = peak)
}

## Peak of a LOESS-smoothed curve (reported next to the discrete argmax).
loess_peak <- function(x, y, span = 0.4) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 5) return(NA_real_)
  fit <- stats::loess(y[ok] ~ x[ok], span = span, degree = 2)
  xs <- seq(min(x[ok]), max(x[ok]), length.out = 400)
  yh <- stats::predict(fit, newdata = data.frame(`x[ok]` = xs, check.names = FALSE))
  xs[which.max(yh)]
}

## ---------------------------------------------------------------------------
## Correlations
## ---------------------------------------------------------------------------
corr_one <- function(x, y, method = c("spearman", "pearson")) {
  method <- match.arg(method)
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 5) return(tibble(rho = NA_real_, p = NA_real_, n = sum(ok)))
  ct <- suppressWarnings(stats::cor.test(x[ok], y[ok], method = method,
                                         exact = FALSE))
  tibble(rho = unname(ct$estimate), p = ct$p.value, n = sum(ok))
}

## Pooled and within-group correlation, the pair that 02 R1 asks for.
corr_pooled_and_within <- function(df, x, y, group = "r",
                                   method = c("spearman", "pearson")) {
  method <- match.arg(method)
  pooled <- corr_one(df[[x]], df[[y]], method) |>
    mutate(scope = "pooled", group_value = NA_real_, .before = 1)
  within <- df |>
    group_by(.data[[group]]) |>
    group_modify(~ corr_one(.x[[x]], .x[[y]], method)) |>
    ungroup() |>
    rename(group_value = dplyr::all_of(group)) |>
    mutate(scope = "within", .before = 1)
  bind_rows(pooled, within) |>
    mutate(x = x, y = y, method = method)
}

## Rank-transform all three, regress out rank(z), correlate the residuals.
partial_spearman <- function(x, y, z) {
  ok <- is.finite(x) & is.finite(y) & is.finite(z)
  if (sum(ok) < 5) return(tibble(rho = NA_real_, n = sum(ok)))
  rx <- rank(x[ok]); ry <- rank(y[ok]); rz <- rank(z[ok])
  ex <- stats::residuals(stats::lm(rx ~ rz))
  ey <- stats::residuals(stats::lm(ry ~ rz))
  if (stats::sd(ex) == 0 || stats::sd(ey) == 0) return(tibble(rho = NA_real_, n = sum(ok)))
  tibble(rho = stats::cor(ex, ey), n = sum(ok))
}

## Fisher-z pooling of within-group rho, weights n - 3.
fisher_pool <- function(rho, n) {
  ok <- is.finite(rho) & is.finite(n) & n > 3 & abs(rho) < 1
  if (!any(ok)) return(NA_real_)
  z <- atanh(rho[ok]); w <- n[ok] - 3
  tanh(sum(w * z) / sum(w))
}

## ---------------------------------------------------------------------------
## Effect sizes
## ---------------------------------------------------------------------------
## A correlation says "the two move together"; it does not say how much of the
## outcome is at stake.  These give the size of the thing, in the units of the
## outcome, so a reader can decide whether it matters.

## Cliff's delta = P(a > b) - P(a < b).  Non-parametric, scale-free, and it
## does not assume the two groups have a sensible mean -- which matters here
## because mu^H after evolution is spread over most of [0, 1].
cliffs_delta <- function(a, b) {
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  if (!length(a) || !length(b)) return(NA_real_)
  rk <- rank(c(a, b))
  U  <- sum(rk[seq_along(a)]) - length(a) * (length(a) + 1) / 2
  2 * U / (length(a) * length(b)) - 1
}

## Slope of y on x (optionally on log10 x, for a heavy-tailed predictor such
## as s), with its interval, the standardised slope and R^2.
effect_slope <- function(x, y, log_x = FALSE) {
  ok <- is.finite(x) & is.finite(y) & (if (log_x) x > 0 else TRUE)
  if (sum(ok) < 5) {
    return(tibble(slope = NA_real_, slope_ci_lo = NA_real_, slope_ci_hi = NA_real_,
                  std_slope = NA_real_, r2 = NA_real_, n = sum(ok), x_scale = NA_character_))
  }
  xx <- if (log_x) log10(x[ok]) else x[ok]
  yy <- y[ok]
  fit <- stats::lm(yy ~ xx); sm <- summary(fit); ci <- stats::confint(fit)
  tibble(slope = unname(stats::coef(fit)[2]),
         slope_ci_lo = ci[2, 1], slope_ci_hi = ci[2, 2],
         std_slope = unname(stats::coef(fit)[2] * stats::sd(xx) / stats::sd(yy)),
         r2 = sm$r.squared, n = sum(ok),
         x_scale = if (log_x) "log10(x): slope is per 10-fold increase" else "x")
}

## The contrast between two groups that are already defined, with a bootstrap
## interval, Cohen's d and Cliff's delta.  Use this whenever the split is
## categorical ("s = 0" against "s > 0"); a quantile split cannot express it.
contrast_groups <- function(lo, hi, B = BOOT_B, alpha = 0.05,
                            label = "contrast", degenerate = FALSE) {
  lo <- lo[is.finite(lo)]; hi <- hi[is.finite(hi)]
  if (length(lo) < 2 || length(hi) < 2) {
    return(tibble(label = label, n_low = length(lo), n_high = length(hi),
                  mean_low = NA_real_, mean_high = NA_real_, diff = NA_real_,
                  diff_ci_lo = NA_real_, diff_ci_hi = NA_real_,
                  cohens_d = NA_real_, cliffs_delta = NA_real_,
                  degenerate_split = degenerate))
  }
  set.seed(seed_from("contrast_groups", label))
  bs <- replicate(B, mean(sample(hi, length(hi), TRUE)) -
                     mean(sample(lo, length(lo), TRUE)))
  tibble(label = label, n_low = length(lo), n_high = length(hi),
         mean_low = mean(lo), mean_high = mean(hi),
         diff = mean(hi) - mean(lo),
         diff_ci_lo = unname(stats::quantile(bs, alpha / 2)),
         diff_ci_hi = unname(stats::quantile(bs, 1 - alpha / 2)),
         cohens_d = (mean(hi) - mean(lo)) /
           sqrt((stats::var(hi) + stats::var(lo)) / 2),
         cliffs_delta = cliffs_delta(hi, lo),
         degenerate_split = degenerate)
}

## Top-tertile minus bottom-tertile contrast in y, with a bootstrap interval,
## Cohen's d and Cliff's delta.  This is the effect size that reads directly:
## "symbionts in the top third of s leave the host this much better off".
group_contrast <- function(x, y, probs = c(1/3, 2/3), B = BOOT_B, alpha = 0.05,
                           label = "contrast") {
  empty <- tibble(label = label, n_low = NA_integer_, n_high = NA_integer_,
                  mean_low = NA_real_, mean_high = NA_real_, diff = NA_real_,
                  diff_ci_lo = NA_real_, diff_ci_hi = NA_real_,
                  cohens_d = NA_real_, cliffs_delta = NA_real_,
                  degenerate_split = NA)
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]; y <- y[ok]
  if (length(x) < 20) return(empty)
  q  <- stats::quantile(x, probs, names = FALSE)
  ## A quantile split is meaningless when the predictor is mostly ties: both
  ## cut points land on the same value and the "high" group ends up being the
  ## whole sample.  Flag it rather than returning a number that looks fine.
  degenerate <- isTRUE(all.equal(q[1], q[2]))
  lo <- y[x <= q[1]]; hi <- y[x >= q[2]]
  if (!length(lo) || !length(hi)) return(empty)
  contrast_groups(lo, hi, B = B, alpha = alpha, label = label,
                  degenerate = degenerate)
}

## ---------------------------------------------------------------------------
## OLS, for Fig.5D / 5E
## ---------------------------------------------------------------------------
ols_fit <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) {
    return(tibble(slope = NA_real_, intercept = NA_real_, r2 = NA_real_,
                  resid_sd = NA_real_, p_slope = NA_real_, n = sum(ok)))
  }
  fit <- stats::lm(y[ok] ~ x[ok])
  s <- summary(fit)
  tibble(slope = unname(stats::coef(fit)[2]), intercept = unname(stats::coef(fit)[1]),
         r2 = s$r.squared, resid_sd = s$sigma,
         p_slope = unname(s$coefficients[2, 4]), n = sum(ok))
}

## Regression of r* on m, with a test of "slope = reference" (Fig.6C).
slope_vs_reference <- function(x, y, reference = 1) {
  ok <- is.finite(x) & is.finite(y)
  fit <- stats::lm(y[ok] ~ x[ok])
  s <- summary(fit)
  est <- unname(stats::coef(fit)[2]); se <- unname(s$coefficients[2, 2])
  tstat <- (est - reference) / se
  dfree <- fit$df.residual
  tibble(slope = est, se = se, intercept = unname(stats::coef(fit)[1]),
         r2 = s$r.squared, reference_slope = reference,
         ci_lo = est - stats::qt(0.975, dfree) * se,
         ci_hi = est + stats::qt(0.975, dfree) * se,
         t_vs_reference = tstat,
         p_vs_reference = 2 * stats::pt(-abs(tstat), dfree), n = sum(ok))
}

## ---------------------------------------------------------------------------
## Highest-density regions (Fig.3C, Fig.4C)
## ---------------------------------------------------------------------------
## Gaussian KDE with a fixed isotropic bandwidth `bw` (NOT a rule-of-thumb:
## the value is chosen once, recorded in the figure metadata, and used for
## every panel so that panels are comparable).  Returns contour paths ready
## for geom_path().
## `bw` is interpreted according to `convention`:
##   "mass_h"  the bandwidth argument of MASS::kde2d / ggdensity, which is
##             FOUR standard deviations wide (sigma = bw / 4).  This is what a
##             bandwidth quoted in an R-drawn figure normally means, and it is
##             the default here so the bands match the manuscript.
##   "sigma"   bw is the Gaussian standard deviation directly.  This is what
##             the Python pipeline used, and it makes the bands 4x wider.
## Whichever is chosen, the sigma actually used is returned as an attribute
## and written into the figure metadata, so the panel is never ambiguous.
hdr_contours <- function(x, y, bw, levels = HDR_LEVELS, grid = KDE_GRID,
                         pad = 0.08, convention = KDE_BW_CONVENTION) {
  convention <- match.arg(convention, c("mass_h", "sigma"))
  bw <- if (convention == "mass_h") bw / 4 else bw
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]; y <- y[ok]
  if (length(x) < 5) return(NULL)
  xr <- diff(range(x)); if (xr == 0) xr <- 1
  yr <- diff(range(y)); if (yr == 0) yr <- 1
  ## The grid must extend far enough past the data for the kernel to decay,
  ## otherwise a contour runs into the edge, comes back as an open path, and
  ## is closed by a straight line across the panel.  Three bandwidths is
  ## enough for a Gaussian kernel.
  padx <- max(pad * xr, 3 * bw)
  pady <- max(pad * yr, 3 * bw)
  xs <- seq(min(x) - padx, max(x) + padx, length.out = grid)
  ys <- seq(min(y) - pady, max(y) + pady, length.out = grid)
  inv2 <- 1 / (2 * bw * bw)
  dens <- matrix(0, nrow = grid, ncol = grid)
  for (i in seq_along(x)) {
    dens <- dens + outer(exp(-(xs - x[i])^2 * inv2), exp(-(ys - y[i])^2 * inv2))
  }
  dens <- dens / sum(dens)
  flat <- sort(as.vector(dens), decreasing = TRUE)
  cum <- cumsum(flat)
  cuts <- vapply(levels, function(lv) {
    k <- findInterval(lv / 100, cum) + 1L
    flat[min(k, length(flat))]
  }, 0)
  out <- map2_dfr(cuts, levels, function(cut, lv) {
    cl <- grDevices::contourLines(xs, ys, dens, levels = cut)
    if (!length(cl)) return(NULL)
    map_dfr(seq_along(cl), function(i) {
      tibble(level = lv, piece = paste(lv, i, sep = "-"),
             x = cl[[i]]$x, y = cl[[i]]$y)
    })
  })
  attr(out, "cuts") <- setNames(cuts, levels)
  attr(out, "sigma") <- bw
  attr(out, "convention") <- convention
  out
}

## Reproducible subsample of `n` rows per group (scatter panels).
subsample_per_group <- function(df, group = "r", n = SCATTER_N, tag = "scatter") {
  df |>
    group_by(.data[[group]]) |>
    group_modify(function(d, key) {
      set.seed(seed_from(tag, group, key[[1]]))
      if (nrow(d) <= n) d else d[sort(sample.int(nrow(d), n)), , drop = FALSE]
    }) |>
    ungroup()
}
