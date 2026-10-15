"""Aggregation, bootstrap CIs, peak location and effect sizes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .rng import derive


# --------------------------------------------------------------------------- #
# basic per-group summaries
# --------------------------------------------------------------------------- #
def trimmed_mean(x, prop=0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(sps.trim_mean(x, prop))


def summarise(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    fin = x[np.isfinite(x)]
    if fin.size == 0:
        return {k: np.nan for k in
                ("n", "mean", "median", "trimmed_mean_5", "sd", "q25", "q75",
                 "q10", "q90", "min", "max")} | {"n": 0, "n_nan": int(x.size)}
    return {
        "n": int(fin.size), "n_nan": int(x.size - fin.size),
        "mean": float(np.mean(fin)), "median": float(np.median(fin)),
        "trimmed_mean_5": trimmed_mean(fin, 0.05),
        "sd": float(np.std(fin, ddof=1)) if fin.size > 1 else np.nan,
        "q10": float(np.quantile(fin, 0.10)), "q25": float(np.quantile(fin, 0.25)),
        "q75": float(np.quantile(fin, 0.75)), "q90": float(np.quantile(fin, 0.90)),
        "min": float(fin.min()), "max": float(fin.max()),
    }


def group_summary(df: pd.DataFrame, by: str, cols, extra=None) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=True):
        rec = {by: key}
        for c in cols:
            if c not in g.columns:
                continue
            for k, v in summarise(g[c]).items():
                rec[f"{c}_{k}"] = v
        if extra:
            rec.update(extra(g))
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def _rng(master_seed, table_id, statistic):
    return derive(master_seed, "boot", table_id=table_id, statistic=statistic)


def bootstrap_ci(x, stat=np.mean, B=2000, alpha=0.05, master_seed=0,
                 table_id="t", statistic="stat"):
    """Percentile bootstrap CI over the resampling unit given in x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = _rng(master_seed, table_id, statistic)
    idx = rng.integers(0, x.size, size=(B, x.size))
    boot = np.array([stat(x[i]) for i in idx])
    point = float(stat(x))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def bootstrap_curve_and_peak(df, value, group="r", unit="symbiont_id",
                             stat=np.mean, B=2000, alpha=0.05, master_seed=0,
                             table_id="peak"):
    """Per-group CI of `stat`, and the bootstrap distribution of argmax.

    Resampling is done *independently within each group* (02 R2): different r
    values are different draws of R^S, so sharing indices across r is wrong.
    Ties in the argmax are broken towards the smallest group key.
    """
    groups = sorted(df[group].unique())
    # unit-level values inside each group
    per_unit = {}
    for gkey in groups:
        g = df[df[group] == gkey]
        if unit in g.columns and g[unit].notna().any():
            v = g.groupby(unit)[value].mean()
            per_unit[gkey] = v.to_numpy(dtype=float)
        else:
            per_unit[gkey] = pd.to_numeric(g[value], errors="coerce").to_numpy(float)
        per_unit[gkey] = per_unit[gkey][np.isfinite(per_unit[gkey])]

    rng = _rng(master_seed, table_id, value)
    boot = np.empty((B, len(groups)))
    for j, gkey in enumerate(groups):
        x = per_unit[gkey]
        if x.size == 0:
            boot[:, j] = np.nan
            continue
        idx = rng.integers(0, x.size, size=(B, x.size))
        boot[:, j] = stat(x[idx], axis=1)

    point = np.array([stat(per_unit[g0]) if per_unit[g0].size else np.nan
                      for g0 in groups])
    lo = np.nanquantile(boot, alpha / 2, axis=0)
    hi = np.nanquantile(boot, 1 - alpha / 2, axis=0)
    curve = pd.DataFrame({group: groups, "value": point, "ci_lo": lo, "ci_hi": hi,
                          "n": [per_unit[g0].size for g0 in groups]})

    valid = ~np.isnan(boot).any(axis=1)
    arg = np.array(groups)[np.argmax(boot[valid], axis=1)]
    peak = {
        "peak_at": groups[int(np.nanargmax(point))],
        "peak_value": float(np.nanmax(point)),
        "peak_ci_lo": float(np.quantile(arg, alpha / 2)) if arg.size else np.nan,
        "peak_ci_hi": float(np.quantile(arg, 1 - alpha / 2)) if arg.size else np.nan,
        "B_used": int(valid.sum()),
    }
    # contrast between the peak and each endpoint, on the same bootstrap draws
    jpk = int(np.nanargmax(point))
    for label, jend in (("first", 0), ("last", len(groups) - 1)):
        d = boot[valid, jpk] - boot[valid, jend]
        peak[f"diff_peak_minus_{label}"] = float(point[jpk] - point[jend])
        peak[f"diff_peak_minus_{label}_lo"] = float(np.quantile(d, alpha / 2))
        peak[f"diff_peak_minus_{label}_hi"] = float(np.quantile(d, 1 - alpha / 2))
        peak[f"{label}_r"] = groups[jend]
    return curve, peak


def loess_peak(x, y, frac=0.4, log_x=True):
    """argmax of a LOWESS smooth of the group curve (reported alongside the raw one).

    Smoothing is done against log(x) by default.  The r grid spans two decades and
    every figure uses a log x-axis, so a window of constant width in log r is the
    one that means the same thing at r = 5 and at r = 80.

    A guard is needed because LOWESS is a *local linear* fit: at the left edge of a
    steeply rising curve it extrapolates upward, and the smooth can exceed the data.
    On the r = 1..100 step-1 grid with frac = 0.4 in linear r the smoothed value at
    r = 1 came out as 0.675 against a data maximum of 0.512, putting the argmax at
    the boundary.  (This stayed hidden on the 15-21 point grids, where the window
    held only 6-8 points.)  So: if the smooth overshoots the data, halve frac and
    retry; give up rather than report a boundary artefact.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except Exception:                                   # noqa: BLE001
        return np.nan
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if log_x:
        ok &= x > 0
    if ok.sum() < 4:
        return np.nan
    xx = np.log(x[ok]) if log_x else x[ok]
    yy = y[ok]
    tol = 1e-9 + 1e-3 * (np.nanmax(yy) - np.nanmin(yy))
    f = float(frac)
    while f >= 0.05:
        sm = lowess(yy, xx, frac=f, return_sorted=True)
        if np.nanmax(sm[:, 1]) <= np.nanmax(yy) + tol:
            xp = float(sm[np.argmax(sm[:, 1]), 0])
            return float(np.exp(xp)) if log_x else xp
        f /= 2.0
    return np.nan


# --------------------------------------------------------------------------- #
# correlations and effect sizes
# --------------------------------------------------------------------------- #
def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    rho, p = sps.spearmanr(x[ok], y[ok])
    return float(rho), float(p), int(ok.sum())


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    rr, p = sps.pearsonr(x[ok], y[ok])
    return float(rr), float(p), int(ok.sum())


def partial_spearman(x, y, z):
    """Rank-transform all three, regress out rank(z), correlate the residuals.

    This is the primary definition fixed in 03_additional_analyses.md A3-b.
    """
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 5:
        return np.nan, int(ok.sum())
    rx, ry, rz = (sps.rankdata(v[ok]) for v in (x, y, z))
    Z = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex, ey = rx - Z @ bx, ry - Z @ by
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(ex, ey)[0, 1]), int(ok.sum())


def fisher_pool(rhos, ns):
    """Fisher-z pooling of within-group rho with n-3 weights."""
    rhos = np.asarray(rhos, float)
    ns = np.asarray(ns, float)
    ok = np.isfinite(rhos) & (ns > 3) & (np.abs(rhos) < 1)
    if not ok.any():
        return np.nan
    z = np.arctanh(rhos[ok])
    w = ns[ok] - 3
    return float(np.tanh(np.sum(w * z) / np.sum(w)))


def cliffs_delta(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    # rank-based O(n log n) form
    allv = np.concatenate([a, b])
    rk = sps.rankdata(allv)
    ra = rk[: a.size].sum()
    u = ra - a.size * (a.size + 1) / 2.0
    return float(2 * u / (a.size * b.size) - 1)


def hodges_lehmann(a, b, max_pairs=4_000_000, rng=None):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    if a.size * b.size <= max_pairs:
        return float(np.median(a[:, None] - b[None, :]))
    rng = rng or np.random.default_rng(0)
    ia = rng.integers(0, a.size, max_pairs)
    ib = rng.integers(0, b.size, max_pairs)
    return float(np.median(a[ia] - b[ib]))


def quartile_contrast(x, y):
    """Median of y in the top vs bottom quartile of x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 8:
        return {"q1_median": np.nan, "q4_median": np.nan, "contrast": np.nan,
                "cliffs_delta": np.nan, "hl_shift": np.nan, "n": int(x.size)}
    lo, hi = np.quantile(x, [0.25, 0.75])
    y1, y4 = y[x <= lo], y[x >= hi]
    return {"q1_median": float(np.median(y1)), "q4_median": float(np.median(y4)),
            "contrast": float(np.median(y4) - np.median(y1)),
            "cliffs_delta": cliffs_delta(y4, y1),
            "hl_shift": hodges_lehmann(y4, y1), "n": int(x.size)}


# --------------------------------------------------------------------------- #
# 2-D HDR contours for the KDE panels
# --------------------------------------------------------------------------- #
def hdr_contours(x, y, bw, levels=(25, 50, 75), grid=256, pad=0.08):
    """Gaussian KDE with a fixed isotropic bandwidth; return the density grid and
    the contour levels enclosing the requested probability mass."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        return None
    xr = x.max() - x.min() or 1.0
    yr = y.max() - y.min() or 1.0
    xs = np.linspace(x.min() - pad * xr, x.max() + pad * xr, grid)
    ys = np.linspace(y.min() - pad * yr, y.max() + pad * yr, grid)
    XX, YY = np.meshgrid(xs, ys)
    dens = np.zeros_like(XX)
    inv2 = 1.0 / (2 * bw * bw)
    for xi, yi in zip(x, y):
        dens += np.exp(-((XX - xi) ** 2 + (YY - yi) ** 2) * inv2)
    dens /= dens.sum()
    flat = np.sort(dens.ravel())[::-1]
    cum = np.cumsum(flat)
    cuts = []
    for lv in levels:
        k = np.searchsorted(cum, lv / 100.0)
        cuts.append(float(flat[min(k, flat.size - 1)]))
    return xs, ys, dens, sorted(cuts)


# --------------------------------------------------------------------------- #
# paired bootstrap over a shared resampling unit (07_run2_workorder.md Phase A)
# --------------------------------------------------------------------------- #
def paired_group_bootstrap(df, group, unit, value_cols, stat=np.mean, B=2000,
                           alpha=0.05, master_seed=0, table_id="paired",
                           baseline=None):
    """Bootstrap several columns of the SAME cells on one set of resamples.

    Resampling is independent within each group (different r are different draws
    of R^S) but shared across `value_cols`, so differences between columns are
    paired and their CIs are not inflated by between-column independence.

    Returns a long frame: one row per (group, column) with the point estimate,
    its CI, and — when `baseline` is given — the paired difference from that
    column with its own CI.
    """
    rng = derive(master_seed, "boot", table_id=table_id,
                 statistic="|".join(value_cols))
    out = []
    for gkey, g in df.groupby(group, sort=True):
        if unit is not None and unit in g.columns:
            g = g.groupby(unit, as_index=False).mean(numeric_only=True)
        arrs = {c: pd.to_numeric(g[c], errors="coerce").to_numpy(float)
                for c in value_cols}
        n = len(g)
        if n == 0:
            continue
        idx = rng.integers(0, n, size=(B, n))
        boot, point = {}, {}
        for c, x in arrs.items():
            point[c] = float(stat(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan
            boot[c] = stat(np.where(np.isfinite(x[idx]), x[idx], np.nan), axis=1)
        for c in value_cols:
            rec = {group: gkey, "column": c, "n": int(n), "value": point[c],
                   "ci_lo": float(np.nanquantile(boot[c], alpha / 2)),
                   "ci_hi": float(np.nanquantile(boot[c], 1 - alpha / 2))}
            if baseline is not None and c != baseline:
                d = boot[c] - boot[baseline]
                rec["diff_vs_baseline"] = point[c] - point[baseline]
                rec["diff_lo"] = float(np.nanquantile(d, alpha / 2))
                rec["diff_hi"] = float(np.nanquantile(d, 1 - alpha / 2))
            out.append(rec)
    return pd.DataFrame(out)


def bootstrap_effect_ci(x, y, fn, B=2000, alpha=0.05, master_seed=0,
                        table_id="effect", statistic="fn"):
    """Percentile CI of a two-sample statistic fn(x_hi, x_lo) by resampling pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 8:
        return np.nan, np.nan, np.nan
    rng = derive(master_seed, "boot", table_id=table_id, statistic=statistic)
    point = fn(x, y)
    idx = rng.integers(0, x.size, size=(B, x.size))
    vals = np.array([fn(x[i], y[i]) for i in idx], dtype=float)
    return (float(point), float(np.nanquantile(vals, alpha / 2)),
            float(np.nanquantile(vals, 1 - alpha / 2)))


def quartile_contrast_ci(x, y, B=2000, alpha=0.05, master_seed=0,
                         table_id="qc", mu_ref=None):
    """Top- vs bottom-quartile contrast of y across x, with bootstrap CIs."""
    def _contrast(xx, yy):
        lo, hi = np.quantile(xx, [0.25, 0.75])
        a, b = yy[xx >= hi], yy[xx <= lo]
        if a.size == 0 or b.size == 0:
            return np.nan
        return float(np.median(a) - np.median(b))

    def _cliff(xx, yy):
        lo, hi = np.quantile(xx, [0.25, 0.75])
        return cliffs_delta(yy[xx >= hi], yy[xx <= lo])

    def _hl(xx, yy):
        lo, hi = np.quantile(xx, [0.25, 0.75])
        return hodges_lehmann(yy[xx >= hi], yy[xx <= lo])

    base = quartile_contrast(x, y)
    out = dict(base)
    for name, fn in (("contrast", _contrast), ("cliffs_delta", _cliff),
                     ("hl_shift", _hl)):
        p, lo, hi = bootstrap_effect_ci(x, y, fn, B=B, alpha=alpha,
                                        master_seed=master_seed,
                                        table_id=table_id, statistic=name)
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = lo, hi
    if mu_ref:
        out["contrast_over_muH0"] = out["contrast"] / mu_ref
    return out
