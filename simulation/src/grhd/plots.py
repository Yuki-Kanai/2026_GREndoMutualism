"""Figures for R1-R5.

Per 05_conventions.md section 8 these functions never read data/raw/ directly:
mean curves, bands and peaks come from the CSV tables, and the scatter layers
come from a pre-thinned plotdata parquet written by the analysis step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import plotstyle as ps
from .stats import hdr_contours


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def scatter_layer(ax, x, y, color=None, size=1.4, alpha=0.30, rasterized=True):
    ax.scatter(x, y, s=size, c=color or ps.C["scatter"], alpha=alpha,
               linewidths=0, rasterized=rasterized, zorder=1)


def mean_curve(ax, x, y, lo=None, hi=None, color=None, label=None, lw=1.3,
               band_alpha=0.25, marker=None):
    color = color or ps.C["mean"]
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=band_alpha, linewidth=0,
                        zorder=2)
    ax.plot(x, y, color=color, lw=lw, label=label, zorder=3, marker=marker,
            markersize=2.4, markeredgewidth=0, solid_capstyle="round")


def hline(ax, y, label=None, color=None, ls=(0, (3.5, 2.0))):
    ax.axhline(y, color=color or ps.C["ref"], lw=0.8, ls=ls, zorder=2,
               label=label)


def log_r_axis(ax, rmax=100):
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlim(0.85, rmax * 1.12)
    ax.minorticks_off()


def hdr_panel(ax, x, y, bw, levels=(25, 50, 75), grid=256, cmap_colors=None,
              point_color=None, show_points=True):
    if show_points:
        scatter_layer(ax, x, y, color=point_color or "#AAB2BA", size=2.0, alpha=0.45)
    out = hdr_contours(x, y, bw=bw, levels=levels, grid=grid)
    if out is None:
        return
    xs, ys, dens, cuts = out
    cols = cmap_colors or ["#9CC3E5", "#3C7DBF", "#123F66"]
    ax.contour(xs, ys, dens, levels=cuts, colors=cols[: len(cuts)],
               linewidths=[0.7, 0.9, 1.1][: len(cuts)], zorder=4)


# --------------------------------------------------------------------------- #
# R1 — Fig.3
# --------------------------------------------------------------------------- #
def fig3(curves: pd.DataFrame, pts: pd.DataFrame, mu_H_0: float, out_dir: Path,
         hdr_rs=(5, 10, 20, 40), bw=2e-3, n_cells=None, fig_id="R1_fig3",
         meta=None):
    ps.use()
    fig = plt.figure(figsize=(ps.COL2, 3.55))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.62)

    axA = fig.add_subplot(gs[0, :2])
    sub = pts
    scatter_layer(axA, sub["r"], sub["mu_S"])
    mean_curve(axA, curves["r"], curves["mu_S_mean"],
               curves.get("mu_S_ci_lo"), curves.get("mu_S_ci_hi"))
    ps.tidy(axA, "Symbiont genome size, $r$", r"Symbiont growth  $\mu^{S}$")
    log_r_axis(axA)
    ps.panel_label(axA, "A", dx=-0.115)

    axB = fig.add_subplot(gs[0, 2:])
    scatter_layer(axB, sub["r"], sub["mu_H"])
    mean_curve(axB, curves["r"], curves["mu_H_mean"],
               curves.get("mu_H_ci_lo"), curves.get("mu_H_ci_hi"),
               color=ps.C["mean2"])
    hline(axB, mu_H_0)
    axB.annotate(r"$\mu^{H}_{0}$", xy=(1.0, mu_H_0), xytext=(1.5, mu_H_0),
                 va="bottom", ha="left", fontsize=6.5, color=ps.C["ref"])
    ps.tidy(axB, "Symbiont genome size, $r$", r"Host growth  $\mu^{H}$")
    log_r_axis(axB)
    ps.panel_label(axB, "B", dx=-0.115)

    axC = []
    for i, rr in enumerate(hdr_rs):
        ax = fig.add_subplot(gs[1, i])
        d = pts[pts["r"] == rr]
        # plot in units of 1e-3 so no offset text is needed
        hdr_panel(ax, d["mu_S"].to_numpy() * 1e3,
                  d["delta_mu_H"].to_numpy() * 1e3, bw=bw * 1e3)
        ax.axhline(0.0, color=ps.C["ref"], lw=0.6, ls=(0, (3.5, 2.0)), zorder=2)
        ax.set_title(f"$r$ = {rr}", loc="center", fontsize=7)
        ax.set_xlabel(r"$\mu^{S}$  ($\times 10^{-3}$)")
        axC.append(ax)
        if i == 0:
            ax.set_ylabel(r"$\Delta\mu^{H}$  ($\times 10^{-3}$)")
            ps.panel_label(ax, "C", dx=-0.42)
        else:
            ax.tick_params(labelleft=False)
    ylo = min(a.get_ylim()[0] for a in axC)
    yhi = max(a.get_ylim()[1] for a in axC)
    for ax in axC:
        ax.set_ylim(ylo, yhi)
    return ps.save(fig, fig_id, out_dir, meta)


# --------------------------------------------------------------------------- #
# R2 — Fig.6A / 6B / S2B
# --------------------------------------------------------------------------- #
def fig6AB_S2B(curves: pd.DataFrame, pts: pd.DataFrame, out_dir: Path,
               fig_id="R2_fig6AB_S2B", meta=None):
    ps.use()
    fig, axes = plt.subplots(1, 3, figsize=(ps.COL2, 1.95))
    fig.subplots_adjust(wspace=0.40)

    ax = axes[0]
    scatter_layer(ax, pts["r"], pts["s"], size=1.2, alpha=0.22)
    # s is heavy tailed (near-singular R^S_{b+}); clip the axis so the location
    # of the peak is readable and state how many points fall outside
    if "s_q90" in curves:
        ytop = float(np.nanmax(curves["s_q90"])) * 1.45
        n_off = int((pts["s"] > ytop).sum())
        ax.set_ylim(-0.02 * ytop, ytop)
        if n_off:
            ax.text(0.985, 0.10,
                    f"{n_off}/{len(pts)} plotted\ncells above axis",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.4, color="#666666", linespacing=1.3)
    for q, a in ((("s_q10", "s_q90"), 0.16), (("s_q25", "s_q75"), 0.26)):
        if q[0] in curves:
            ax.fill_between(curves["r"], curves[q[0]], curves[q[1]],
                            color=ps.C["mean"], alpha=a, linewidth=0, zorder=2)
    mean_curve(ax, curves["r"], curves["s_median"], color=ps.C["mean"],
               label="median")
    ax.plot(curves["r"], curves["s_mean"], color=ps.C["mean2"], lw=1.0,
            ls=(0, (4, 1.6)), label="mean", zorder=3)
    ax.plot(curves["r"], curves["s_trimmed_mean_5"], color=ps.C["accent"],
            lw=1.0, ls=(0, (1.4, 1.4)), label="5% trimmed mean", zorder=3)
    ps.tidy(ax, "Symbiont genome size, $r$", r"Sensitivity  $s=\|S\|_{F}$")
    log_r_axis(ax)
    ax.legend(loc="upper left", fontsize=5.8)
    ps.panel_label(ax, "A", dx=-0.30)

    ax = axes[1]
    mean_curve(ax, curves["r"], curves["n_bounded_mean"],
               curves.get("n_bounded_q25"), curves.get("n_bounded_q75"),
               color=ps.C["mean"], label="bounded metabolites, $b$")
    mean_curve(ax, curves["r"], curves["n_excreted_mean"],
               curves.get("n_excreted_q25"), curves.get("n_excreted_q75"),
               color=ps.C["accent"], label="excreted metabolites")
    ps.tidy(ax, "Symbiont genome size, $r$", "Number of metabolites")
    log_r_axis(ax)
    ax.legend(loc="center right")
    ps.panel_label(ax, "B", dx=-0.30)

    ax = axes[2]
    jr = np.random.default_rng(0)
    jit = jr.uniform(-0.35, 0.35, len(pts))      # ub is an integer; jitter +-0.35
    scatter_layer(ax, pts["r"], pts["ub"] + jit, size=0.9, alpha=0.13)
    mean_curve(ax, curves["r"], curves["ub_mean"],
               curves.get("ub_q25"), curves.get("ub_q75"), color=ps.C["green"])
    m = int(meta.get("m", 10)) if meta else 10
    ax.axhline(((m - 1) // 2) * ((m - 1) - (m - 1) // 2), color=ps.C["ref"],
               lw=0.7, ls=(0, (3.5, 2)), zorder=2)
    ps.tidy(ax, "Symbiont genome size, $r$", r"$u\,b$")
    log_r_axis(ax)
    ps.panel_label(ax, "C", dx=-0.30)
    return ps.save(fig, fig_id, out_dir, meta)


# --------------------------------------------------------------------------- #
# R3 — Fig.5D / 5E
# --------------------------------------------------------------------------- #
def fig5DE(per_symbiont: pd.DataFrame, fits: dict, out_dir: Path,
           fig_id="R3_fig5DE", meta=None):
    ps.use()
    fig, axes = plt.subplots(1, 2, figsize=(ps.COL15, 2.0))
    fig.subplots_adjust(wspace=0.36)
    for ax, (xcol, ycol, xlab, ylab, letter, col) in zip(axes, [
        ("alpha_mean", "SD_b", r"Adaptability  $\langle\alpha\rangle=\langle\sqrt{b}\rangle$",
         r"$\mathrm{SD}_{b}$", "D", ps.C["mean"]),
        ("s_mean", "SD_u", r"Sensitivity  $\langle s\rangle$",
         r"$\mathrm{SD}_{u}$", "E", ps.C["mean2"]),
    ]):
        d = per_symbiont.dropna(subset=[xcol, ycol])
        ax.scatter(d[xcol], d[ycol], s=2.2, c=col, alpha=0.30, linewidths=0,
                   rasterized=True)
        f = fits.get(ycol)
        if f and np.isfinite(f["slope"]):
            xs = np.linspace(d[xcol].min(), d[xcol].max(), 100)
            ax.plot(xs, f["intercept"] + f["slope"] * xs, color="#111111", lw=1.0)
            ax.text(0.03, 0.95,
                    f"slope = {f['slope']:.3g}\n$R^2$ = {f['r2']:.3f}\n"
                    f"$\\rho_S$ = {f['spearman']:.3f}   $n$ = {f['n']}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6.0)
        ps.tidy(ax, xlab, ylab)
        ps.panel_label(ax, letter, dx=-0.20)
    return ps.save(fig, fig_id, out_dir, meta)


# --------------------------------------------------------------------------- #
# R4 — Fig.4
# --------------------------------------------------------------------------- #
def fig4(curves: pd.DataFrame, pts: pd.DataFrame, mu_H_0: float, out_dir: Path,
         hdr_rs=(5, 10, 20, 40), bw=0.2, fig_id="R4_fig4", meta=None,
         title_suffix=""):
    ps.use()
    fig = plt.figure(figsize=(ps.COL2, 3.55))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.62)

    axA = fig.add_subplot(gs[0, :2])
    scatter_layer(axA, pts["r"], pts["mu_H"], size=1.8, alpha=0.30)
    mean_curve(axA, curves["r"], curves["mu_H_mean"],
               curves.get("mu_H_ci_lo"), curves.get("mu_H_ci_hi"),
               color=ps.C["mean2"], marker="o")
    hline(axA, mu_H_0)
    axA.annotate(r"$\mu^{H}_{0}$", xy=(1.0, mu_H_0), xytext=(1.2, mu_H_0),
                 va="bottom", ha="left", fontsize=6.5, color=ps.C["ref"])
    ps.tidy(axA, "Symbiont genome size, $r$", r"Host growth  $\mu^{H}$")
    axA.set_title("after host evolution", loc="left", fontsize=6.5,
                  color="#555555")
    log_r_axis(axA, rmax=float(curves["r"].max()))
    ps.panel_label(axA, "A", dx=-0.115)

    axB = fig.add_subplot(gs[0, 2:])
    scatter_layer(axB, pts["r"], pts["mu_S"], size=1.8, alpha=0.30)
    mean_curve(axB, curves["r"], curves["mu_S_mean"],
               curves.get("mu_S_ci_lo"), curves.get("mu_S_ci_hi"),
               color=ps.C["mean"], marker="o")
    ps.tidy(axB, "Symbiont genome size, $r$", r"Symbiont growth  $\mu^{S}$")
    axB.set_title("after host evolution", loc="left", fontsize=6.5,
                  color="#555555")
    log_r_axis(axB, rmax=float(curves["r"].max()))
    ps.panel_label(axB, "B", dx=-0.115)

    for i, rr in enumerate(hdr_rs):
        ax = fig.add_subplot(gs[1, i])
        d = pts[pts["r"] == rr]
        hdr_panel(ax, d["mu_S"].to_numpy(), d["mu_H"].to_numpy(), bw=bw)
        hline(ax, mu_H_0)
        ax.set_title(f"$r$ = {rr}", loc="center", fontsize=7)
        ax.set_xlabel(r"$\mu^{S}$")
        if i == 0:
            ax.set_ylabel(r"$\mu^{H}$")
            ps.panel_label(ax, "C", dx=-0.42)
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=7, x=0.01, ha="left")
    return ps.save(fig, fig_id, out_dir, meta)


def fig5A(pts: pd.DataFrame, xcol, out_dir: Path, fig_id="R4_fig5A", meta=None,
          xlabel=None, rho_text=None, mu_H_0=None):
    ps.use()
    fig, ax = plt.subplots(figsize=(ps.COL1, 1.95))
    d = pts.dropna(subset=[xcol, "mu_H_post"])
    sc = ax.scatter(d[xcol], d["mu_H_post"], c=d["r"], s=3.2, cmap="viridis",
                    linewidths=0, alpha=0.85, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("$r$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    # the two axes live on very different scales (pre-evolution mu^H sits just
    # below mu^H_0, post-evolution mu^H spans [mu^H_0, 1]), so an identity line
    # would compress the data into a sliver; mark mu^H_0 on both axes instead.
    if mu_H_0 is not None:
        ax.axvline(mu_H_0, color=ps.C["ref"], lw=0.7, ls=(0, (3.5, 2)))
        ax.axhline(mu_H_0, color=ps.C["ref"], lw=0.7, ls=(0, (3.5, 2)))
    xr = float(d[xcol].max() - d[xcol].min()) or 1e-4
    ax.set_xlim(d[xcol].min() - 0.08 * xr, d[xcol].max() + 0.08 * xr)
    if rho_text:
        ax.text(0.03, 0.96, rho_text, transform=ax.transAxes, va="top",
                fontsize=6.0)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
    ax.xaxis.get_offset_text().set_fontsize(5.5)
    ps.tidy(ax, xlabel or xcol, r"$\mu^{H}$ after host evolution")
    return ps.save(fig, fig_id, out_dir, meta)


def figS2A(pts: pd.DataFrame, out_dir: Path, fig_id="R4_figS2A", meta=None,
           rho_text=None):
    ps.use()
    fig, ax = plt.subplots(figsize=(ps.COL1, 1.95))
    for flag, col, lab in ((False, ps.C["mean"], "non-trivial"),
                           (True, ps.C["accent"], r"trivial ($ub<1$)")):
        d = pts[pts["trivial_pre"] == flag]
        ax.scatter(d["ub_pre_mean"], d["mu_H_post"], s=3.0, c=col, alpha=0.55,
                   linewidths=0, label=lab, rasterized=True)
    if rho_text:
        ax.text(0.03, 0.96, rho_text, transform=ax.transAxes, va="top", fontsize=6.0)
    ax.legend(loc="lower right")
    ps.tidy(ax, r"pre-evolution $\langle u\,b\rangle$",
            r"$\mu^{H}$ after host evolution")
    return ps.save(fig, fig_id, out_dir, meta)


def fig6C(per_host: pd.DataFrame, out_dir: Path, fig_id="R5_fig6C", meta=None,
          jitter=0.12, seed=0):
    ps.use()
    fig, ax = plt.subplots(figsize=(ps.COL1, 1.95))
    rng = np.random.default_rng(seed)
    jx = per_host["m"] + rng.uniform(-jitter, jitter, len(per_host))
    ax.scatter(jx, per_host["r_star"], s=6, c=ps.C["mean"], alpha=0.7,
               linewidths=0)
    ms = np.sort(per_host["m"].unique())
    med = [per_host.loc[per_host["m"] == mm, "r_star"].median() for mm in ms]
    ax.plot(ms, med, color=ps.C["mean2"], lw=1.1, marker="o", markersize=2.6)
    ax.plot(ms, ms - 1, color=ps.C["ref"], lw=0.8, ls=(0, (3.5, 2)),
            label="$r = m-1$")
    ax.legend(loc="upper left")
    ps.tidy(ax, "Number of metabolites, $m$",
            r"$r^{*}$ (argmax of mean $\mu^{H}$)")
    return ps.save(fig, fig_id, out_dir, meta)


def fig_overlay_schemes(curves_by_scheme: dict, ycol, ylabel, out_dir, fig_id,
                        xcol="r", xlabel="Symbiont genome size, $r$", meta=None,
                        logx=True):
    ps.use()
    fig, ax = plt.subplots(figsize=(ps.COL1, 1.95))
    for i, (name, c) in enumerate(curves_by_scheme.items()):
        col = ps.SERIES[i % len(ps.SERIES)]
        lo = c.get(f"{ycol}_ci_lo")
        hi = c.get(f"{ycol}_ci_hi")
        mean_curve(ax, c[xcol], c[ycol], lo, hi, color=col, label=name)
    if logx:
        log_r_axis(ax, rmax=max(float(c[xcol].max()) for c in curves_by_scheme.values()))
    ax.legend(loc="best")
    ps.tidy(ax, xlabel, ylabel)
    return ps.save(fig, fig_id, out_dir, meta)
