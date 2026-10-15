"""Figures for R1 (Fig.3), R2 (Fig.6A/6B/S2B) and R3 (Fig.5D/5E).

Reads only tables/*.csv and data/derived/plotdata_*.parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd import plots, plotstyle as ps

ap = argparse.ArgumentParser()
ap.add_argument("--tables", default="tables")
ap.add_argument("--derived", default="data/derived")
ap.add_argument("--figures", default="figures")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
TAB, DER, FIG = Path(a.tables), Path(a.derived), Path(a.figures)
FIG.mkdir(parents=True, exist_ok=True)

r1 = pd.read_csv(TAB / "R1_stats.csv")
r2 = pd.read_csv(TAB / "R2_stats.csv")
pts = pd.read_parquet(DER / "plotdata_pre.parquet")
hdr = pd.read_parquet(DER / "plotdata_pre_hdr.parquet")
meta_in = json.loads((TAB / "R1_meta.json").read_text())
mu_H_0 = meta_in["mu_H_0"]
m = meta_in["m"]

base_meta = {"config": "config/base.yaml", "font": ps.SANS,
             "n_cells_total": meta_in["n_cells"], "m": m, "mu_H_0": mu_H_0,
             "subsample_per_r": meta_in["subsample_per_r"],
             "bootstrap_B": meta_in["bootstrap_B"]}

# ---- Fig.3 -----------------------------------------------------------------
p = plots.fig3(
    r1, pts.assign(**{}), mu_H_0, FIG,
    hdr_rs=(5, 10, 20, 40), bw=float(cfg["stats"]["kde_bw_fig3c"]),
    fig_id="R1_fig3",
    meta=base_meta | {"tables": ["R1_stats.csv"],
                      "plotdata": ["plotdata_pre.parquet"],
                      "kde_bw": cfg["stats"]["kde_bw_fig3c"],
                      "hdr_levels": cfg["stats"]["hdr_levels"],
                      "kde_grid": cfg["stats"]["kde_grid"],
                      "panels": {"A": "mu_S vs r", "B": "mu_H vs r",
                                 "C": "delta_mu_H vs mu_S, HDR contours"}})
print("wrote", p)

# Fig.3C on its own, from the dedicated 300-cell draw
ps.use()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(ps.COL2, 1.75))
fig.subplots_adjust(wspace=0.42)
for ax, rr in zip(axes, (5, 10, 20, 40)):
    d = hdr[hdr["r"] == rr]
    plots.hdr_panel(ax, d["mu_S"].to_numpy(), d["delta_mu_H"].to_numpy(),
                    bw=float(cfg["stats"]["kde_bw_fig3c"]))
    ax.axhline(0.0, color=ps.C["ref"], lw=0.6, ls=(0, (3.5, 2)))
    ax.set_title(f"$r$ = {rr}   ($n$ = {len(d)})", fontsize=6.8)
    ax.set_xlabel(r"$\mu^{S}$")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
    ax.xaxis.get_offset_text().set_fontsize(5.5)
    ax.yaxis.get_offset_text().set_fontsize(5.5)
axes[0].set_ylabel(r"$\Delta\mu^{H}$")
ps.panel_label(axes[0], "C", dx=-0.30)
print("wrote", ps.save(fig, "R1_fig3C", FIG,
                       base_meta | {"plotdata": ["plotdata_pre_hdr.parquet"],
                                    "kde_bw": cfg["stats"]["kde_bw_fig3c"]}))

# ---- Fig.3B with the paper's convention (infeasible -> mu_H = mu_S = 0) -----
conv = TAB / "A3a_convention_curve.csv"
if conv.exists():
    c = pd.read_csv(conv)
    fig, ax = plt.subplots(figsize=(ps.COL1, 1.95))
    plots.mean_curve(ax, r1["r"], r1["mu_H_mean"], color=ps.C["mean2"],
                     label="raw (infeasible excluded)")
    plots.mean_curve(ax, c["r"], c["mu_H_mean_convention"], color=ps.C["mean"],
                     label="paper convention (infeasible = 0)")
    plots.hline(ax, mu_H_0)
    plots.log_r_axis(ax)
    ax.legend(loc="best")
    ps.tidy(ax, "Symbiont genome size, $r$", r"Host growth  $\mu^{H}$")
    print("wrote", ps.save(fig, "R1_fig3B_convention", FIG, base_meta))

# ---- Fig.6A / 6B / S2B ------------------------------------------------------
p = plots.fig6AB_S2B(
    r2, pts, FIG, fig_id="R2_fig6AB_S2B",
    meta=base_meta | {"tables": ["R2_stats.csv", "R2_peaks.csv"],
                      "plotdata": ["plotdata_pre.parquet"],
                      "panels": {"A": "s vs r (median, mean, trimmed mean, "
                                      "10-90 and 25-75 bands)",
                                 "B": "mean b and mean n_excreted vs r",
                                 "C": "ub vs r"}})
print("wrote", p)

# ---- b-axis re-plot (A2-D4 / R2 b-axis version) -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(ps.COL2, 1.9))
fig.subplots_adjust(wspace=0.40)
for ax, (col, lab, colr) in zip(axes, [
        ("s_median", r"median sensitivity  $s$", ps.C["mean"]),
        ("n_excreted_mean", "mean excreted metabolites", ps.C["accent"]),
        ("ub_mean", r"mean $u\,b$", ps.C["green"])]):
    sc = ax.scatter(r2["n_bounded_mean"], r2[col], c=r2["r"], s=7,
                    cmap="viridis", linewidths=0)
    ax.axvline((m - 1) / 2, color=ps.C["ref"], lw=0.7, ls=(0, (3.5, 2)))
    ps.tidy(ax, r"mean bounded metabolites  $\langle b\rangle$", lab)
cb = fig.colorbar(sc, ax=axes, pad=0.012, fraction=0.03)
cb.set_label("$r$", fontsize=7)
cb.ax.tick_params(labelsize=6)
print("wrote", ps.save(fig, "R2_b_axis", FIG,
                       base_meta | {"tables": ["R2_stats.csv"],
                                    "reference_line": "(m-1)/2"}))

# ---- Fig.5D / 5E ------------------------------------------------------------
sdp = TAB / "R3_per_symbiont.csv"
if sdp.exists():
    sd = pd.read_csv(sdp)
    st = pd.read_csv(TAB / "R3_stats.csv")
    for subset, suffix in (("5<=r<=25", ""), ("all_r", "_allr")):
        d = sd if subset == "all_r" else sd[(sd["r"] >= 5) & (sd["r"] <= 25)]
        fits = {}
        for ycol in ("SD_b", "SD_u"):
            row = st[(st["subset"] == subset) & (st["y"] == ycol)]
            if len(row):
                fits[ycol] = row.iloc[0].to_dict()
        p = plots.fig5DE(d, fits, FIG, fig_id=f"R3_fig5DE{suffix}",
                         meta=base_meta | {"tables": ["R3_per_symbiont.csv",
                                                      "R3_stats.csv"],
                                           "subset": subset,
                                           "SD_definition": "root-sum-of-variances "
                                           "over the majority-vote bounded set "
                                           "(primary); see R3_stats.csv for the "
                                           "sequential, pooled-scalar and "
                                           "mean-component variants"})
        print("wrote", p)
