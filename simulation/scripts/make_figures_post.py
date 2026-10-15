"""Figures for R4: Fig.4A/4B/4C, Fig.S1, Fig.5A, Fig.S2A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

curve = pd.read_csv(TAB / "R4_curve.csv")
stats = pd.read_csv(TAB / "R4_stats.csv")
peaks = pd.read_csv(TAB / "R4_peak.csv")
corr = pd.read_csv(TAB / "R4_correlations.csv")
pts = pd.read_parquet(DER / "plotdata_post.parquet")
join = pd.read_parquet(DER / "plotdata_post_join.parquet")
meta_in = json.loads((TAB / "R4_meta.json").read_text())
mu_H_0 = meta_in["mu_H_0"]

base = {"config": "config/base.yaml", "font": ps.SANS, "stage": "post",
        "mu_H_0": mu_H_0, "n_post_cells": meta_in["n_post_cells"],
        "N_S": meta_in["N_S"], "bootstrap_B": meta_in["bootstrap_B"],
        "frac_detlimit": meta_in["frac_detlimit"]}

hdr_rs = [r for r in (5, 10, 20, 40) if r in set(pts["r"])]

# ---- Fig.4 ------------------------------------------------------------------
print("wrote", plots.fig4(
    curve, pts, mu_H_0, FIG, hdr_rs=tuple(hdr_rs),
    bw=float(cfg["stats"]["kde_bw_fig4c"]), fig_id="R4_fig4",
    meta=base | {"tables": ["R4_curve.csv", "R4_peak.csv"],
                 "plotdata": ["plotdata_post.parquet"],
                 "kde_bw": cfg["stats"]["kde_bw_fig4c"],
                 "hdr_levels": cfg["stats"]["hdr_levels"],
                 "note": "all solved instances, including detlimit incumbents; "
                         "see R4_stats.csv subset=no_detlimit for the "
                         "detlimit-excluded curve"}))

# ---- Fig.4A with and without detlimit ---------------------------------------
ps.use()
fig, ax = plt.subplots(figsize=(ps.COL1, 1.95))
for label, col, lab in (("all", ps.C["mean2"], "all instances"),
                        ("no_detlimit", ps.C["mean"], "detlimit excluded")):
    d = stats[stats["subset"] == label]
    plots.mean_curve(ax, d["r"], d["mu_H_mean"], d["mu_H_ci_lo"], d["mu_H_ci_hi"],
                     color=col, label=lab, marker="o")
plots.hline(ax, mu_H_0)
plots.log_r_axis(ax, rmax=float(curve["r"].max()))
ax.legend(loc="best")
ps.tidy(ax, "Symbiont genome size, $r$", r"mean $\mu^{H}$ after evolution")
print("wrote", ps.save(fig, "R4_fig4A_detlimit", FIG,
                       base | {"tables": ["R4_stats.csv"]}))

# ---- Fig.S1: pairs that were parasitic before host evolution ----------------
sub = pts.merge(join[["r", "symbiont_id", "parasitic_pre"]],
                on=["r", "symbiont_id"], how="left")
sub = sub[sub["parasitic_pre"].fillna(False)]
if len(sub) > 20:
    cur = []
    for rr, g in sub.groupby("r"):
        cur.append({"r": rr, "mu_H_mean": g["mu_H"].mean(),
                    "mu_S_mean": g["mu_S"].mean(), "n": len(g)})
    cur = pd.DataFrame(cur)
    cur.to_csv(TAB / "R4_figS1_curve.csv", index=False)
    print("wrote", plots.fig4(
        cur, sub, mu_H_0, FIG, hdr_rs=tuple(hdr_rs),
        bw=float(cfg["stats"]["kde_bw_fig4c"]), fig_id="R4_figS1",
        meta=base | {"restriction": "symbionts whose pre-evolution max mu_H "
                                    "over the N_c sampled c^H was < mu_H_0",
                     "n": int(len(sub))},
        title_suffix="restricted to pairs that were parasitic before host evolution"))

# ---- Fig.5A (max and median versions) ---------------------------------------
for xcol, fid, xlab in (
        ("mu_H_pre_max", "R4_fig5A_max",
         r"max $\mu^{H}$ before evolution (over $N_c$ sampled $c^{H}$)"),
        ("mu_H_pre_median", "R4_fig5A_median",
         r"median $\mu^{H}$ before evolution")):
    for subset, suffix in (("5<=r<=25", ""), ("all", "_allr")):
        d = join if subset == "all" else join[(join["r"] >= 5) & (join["r"] <= 25)]
        row = corr[(corr["x"] == xcol) & (corr["subset"] == subset)]
        txt = None
        if len(row):
            rr = row.iloc[0]
            txt = (f"$\\rho_S$ = {rr['spearman_pooled']:.3f} (pooled)\n"
                   f"partial on $r$ = {rr['partial_spearman_on_r']:.3f}\n"
                   f"$n$ = {int(rr['n'])}")
        print("wrote", plots.fig5A(d, xcol, FIG, fig_id=fid + suffix,
                                   meta=base | {"subset": subset,
                                                "tables": ["R4_correlations.csv"]},
                                   xlabel=xlab, rho_text=txt, mu_H_0=mu_H_0))

# ---- Fig.S2A ----------------------------------------------------------------
row = corr[(corr["x"] == "ub_pre_mean") & (corr["subset"] == "all")]
txt = None
if len(row):
    rr = row.iloc[0]
    txt = (f"$\\rho_S$ = {rr['spearman_pooled']:.3f} (pooled)\n"
           f"partial on $r$ = {rr['partial_spearman_on_r']:.3f}\n"
           f"$n$ = {int(rr['n'])}")
print("wrote", plots.figS2A(join, FIG, fig_id="R4_figS2A",
                            meta=base | {"tables": ["R4_correlations.csv"]},
                            rho_text=txt))

# ---- post-evolution mechanism quantities ------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(ps.COL2, 1.95))
fig.subplots_adjust(wspace=0.40)
for ax, (col, lab, colr) in zip(axes, [
        ("n_bounded_mean", r"mean bounded metabolites $b$", ps.C["mean"]),
        ("n_excreted_mean", "mean excreted metabolites", ps.C["accent"]),
        ("s_median", r"median sensitivity $s$", ps.C["green"])]):
    plots.mean_curve(ax, curve["r"], curve[col], color=colr, marker="o")
    plots.log_r_axis(ax, rmax=float(curve["r"].max()))
    ps.tidy(ax, "Symbiont genome size, $r$", lab)
for ax, L in zip(axes, "ABC"):
    ps.panel_label(ax, L, dx=-0.30)
print("wrote", ps.save(fig, "R4_post_structure", FIG,
                       base | {"tables": ["R4_curve.csv"],
                               "note": "structural quantities measured at the "
                                       "evolved c^H*, after the ||c^H||_1 "
                                       "tie-break"}))

# ---- mutualism fraction -----------------------------------------------------
fig, ax = plt.subplots(figsize=(ps.COL1, 1.95))
d = stats[stats["subset"] == "all"]
plots.mean_curve(ax, d["r"], d["frac_mutualistic_1e6"], color=ps.C["mean"],
                 label=r"$\mu^{S} > 10^{-6}$", marker="o")
plots.mean_curve(ax, d["r"], d["frac_mutualistic_1e3"], color=ps.C["mean2"],
                 label=r"$\mu^{S} > 10^{-3}$", marker="o")
plots.log_r_axis(ax, rmax=float(d["r"].max()))
ax.set_ylim(-0.03, 1.03)
ax.legend(loc="best")
ps.tidy(ax, "Symbiont genome size, $r$",
        r"fraction mutualistic ($\mu^{H}>\mu^{H}_{0}$)")
print("wrote", ps.save(fig, "R4_mutualism_fraction", FIG,
                       base | {"tables": ["R4_stats.csv"]}))
