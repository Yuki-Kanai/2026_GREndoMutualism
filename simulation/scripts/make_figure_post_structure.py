"""R4 post-evolution structure, with the zero-c^H convention made explicit.

Replaces the earlier R4_post_structure figure, whose panel A plotted `b` under
the default convention only.  That curve is dominated by metabolites the evolved
host does not offer at all, so it moves opposite to the pre-evolution `b`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from grhd import plots, plotstyle as ps

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="tables/A14_post_conventions.csv")
ap.add_argument("--pre", default="tables/R2_stats.csv")
ap.add_argument("--figures", default="figures")
a = ap.parse_args()

g = pd.read_csv(a.table)
pre = pd.read_csv(a.pre)
FIG = Path(a.figures)
ps.use()

meta = {"tables": [a.table, a.pre], "font": ps.SANS,
        "note": "panel A shows both zero-c^H conventions (01_model_spec.md A14); "
                "b under the default rule counts metabolites the host does not "
                "offer, because those have zero slack"}

fig, axes = plt.subplots(1, 4, figsize=(ps.COL2, 2.0))
fig.subplots_adjust(wspace=0.48)
rmax = float(g["r"].max())

ax = axes[0]
plots.mean_curve(ax, g["r"], g["b_default"], color=ps.C["scatter"], marker="o",
                 label=r"$b$, default rule")
plots.mean_curve(ax, g["r"], g["b_alt"], color=ps.C["mean"], marker="o",
                 label=r"$b$, alt. rule $=$ provisioned")
plots.mean_curve(ax, pre["r"], pre["n_bounded_mean"], color=ps.C["mean2"],
                 lw=1.0, label=r"$b$, before evolution")
plots.log_r_axis(ax, rmax=rmax)
ax.set_ylim(-0.3, 9.4)
ax.legend(loc="center left", fontsize=5.4)
ps.tidy(ax, "Symbiont genome size, $r$", "Number of metabolites")
ps.panel_label(ax, "A", dx=-0.34)

ax = axes[1]
plots.mean_curve(ax, g["r"], g["n_excreted"], color=ps.C["accent"], marker="o")
plots.log_r_axis(ax, rmax=rmax)
ps.tidy(ax, "Symbiont genome size, $r$", "Excreted metabolites")
ax.set_title("convention-free", loc="left", fontsize=6, color="#777777")
ps.panel_label(ax, "B", dx=-0.34)

ax = axes[2]
plots.mean_curve(ax, g["r"], g["ub_default"], color=ps.C["scatter"], marker="o",
                 label="default rule")
plots.mean_curve(ax, g["r"], g["ub_alt"], color=ps.C["green"], marker="o",
                 label="alt. rule")
plots.log_r_axis(ax, rmax=rmax)
ax.legend(loc="upper right", fontsize=5.6)
ps.tidy(ax, "Symbiont genome size, $r$", r"mean $u\,b$")
ps.panel_label(ax, "C", dx=-0.34)

ax = axes[3]
plots.mean_curve(ax, g["r"], g["s_default_med"], color=ps.C["scatter"],
                 marker="o", label="default rule")
plots.mean_curve(ax, g["r"], g["s_alt_med"], color=ps.C["purple"], marker="o",
                 label="alt. rule")
plots.log_r_axis(ax, rmax=rmax)
ax.legend(loc="upper right", fontsize=5.6)
ps.tidy(ax, "Symbiont genome size, $r$", r"median sensitivity $s$")
ps.panel_label(ax, "D", dx=-0.34)

print("wrote", ps.save(fig, "R4_post_structure", FIG, meta))
