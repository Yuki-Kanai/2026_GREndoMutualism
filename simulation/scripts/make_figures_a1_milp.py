"""Figures for A1-MILP (gate G2b).  Run after a1_milp_verdict.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grhd import plotstyle as ps
from grhd.cfgio import load_config
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--tables", default="tables")
ap.add_argument("--figures", default="figures")
ap.add_argument("--n-rxn", type=int, default=40)
ap.add_argument("--mu-h0", type=float, default=0.051711)
a = ap.parse_args()

TAB, FIG = Path(a.tables), Path(a.figures)
cur = pd.read_csv(TAB / "A1_muH_curves.csv")
pk = pd.read_csv(TAB / "A1_muH_peaks.csv")
ps.use()

ARM = {"paper": dict(color=ps.C["mean"], ls="-", label=r"paper ($k=4$)"),
       f"fixed_n{a.n_rxn}": dict(color=ps.C["mean2"], ls=(0, (4, 1.5)),
                                 label=rf"fixed repertoire ($n_{{\rm rxn}}={a.n_rxn}$)")}


def draw(ax, quantity, stat, axis):
    for arm, st in ARM.items():
        g = cur[(cur.arm == arm) & (cur.quantity == quantity)
                & (cur.statistic == stat) & (cur.axis == axis)].sort_values("group_value")
        if g.empty:
            continue
        ax.plot(g["group_value"], g["value"], color=st["color"], ls=st["ls"],
                label=st["label"], lw=1.2)
        ax.fill_between(g["group_value"], g["ci_lo"], g["ci_hi"],
                        color=st["color"], alpha=0.18, lw=0)
        p = pk[(pk.arm == arm) & (pk.quantity == quantity)
               & (pk.statistic == stat) & (pk.axis == axis)]
        if not p.empty:
            ax.plot([p["peak_at"].iloc[0]], [p["peak_value"].iloc[0]], marker="v",
                    ms=3.4, color=st["color"], mec="none")
            ax.errorbar([p["peak_at"].iloc[0]], [p["peak_value"].iloc[0]],
                        xerr=[[p["peak_at"].iloc[0] - p["peak_ci_lo"].iloc[0]],
                              [p["peak_ci_hi"].iloc[0] - p["peak_at"].iloc[0]]],
                        color=st["color"], elinewidth=0.7, capsize=1.6, lw=0)


fig, axes = plt.subplots(1, 4, figsize=(ps.COL2, 2.05))
fig.subplots_adjust(wspace=0.42)

ax = axes[0]
draw(ax, "mu_H", "mean", "r")
ax.axhline(a.mu_h0, color=ps.C["ref"], lw=0.6, ls=":")
ax.text(1.1, a.mu_h0 * 1.25, r"$\mu^{H}_{0}$", fontsize=5.8, color=ps.C["ref"])
ax.set_xscale("log")
ax.legend(loc="upper right", fontsize=5.4)
ps.tidy(ax, r"Number of modes, $r$", r"mean post-evolution $\mu^{H}$")
ps.panel_label(ax, "A", dx=-0.30)

ax = axes[1]
draw(ax, "mutual", "fraction", "r")
ax.set_xscale("log"); ax.set_ylim(-0.03, 1.0)
ps.tidy(ax, r"Number of modes, $r$", "fraction mutualistic")
ps.panel_label(ax, "B", dx=-0.30)

ax = axes[2]
draw(ax, "no_gain", "fraction", "r")
ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
ps.tidy(ax, r"Number of modes, $r$", r"fraction with $\mu^{H}=\mu^{H}_{0}$")
ps.panel_label(ax, "C", dx=-0.30)

ax = axes[3]
draw(ax, "mu_H", "mean", "b")
ax.axhline(a.mu_h0, color=ps.C["ref"], lw=0.6, ls=":")
ps.tidy(ax, r"Bounded metabolites, $b$", r"mean post-evolution $\mu^{H}$")
ps.panel_label(ax, "D", dx=-0.30)

print("wrote", ps.save(fig, "A1_muH_verdict", FIG,
                       {"tables": ["A1_muH_curves.csv", "A1_muH_peaks.csv"],
                        "n_rxn": a.n_rxn}))
