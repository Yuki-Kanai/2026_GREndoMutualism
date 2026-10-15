"""Figures for A1-LP (gate G2a).  Run after a1_lp_peaks.py.

  A1_repertoire   the premise of the control: how the effective reaction
                  repertoire grows with r in each scheme
  A1_mechanism_r  s / n_excreted / ub / b against r, schemes overlaid
  A1_mechanism_b  the same quantities against b, which is the axis on which the
                  peak is supposed to be invariant
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grhd import plotstyle as ps
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--tables", default="tables")
ap.add_argument("--figures", default="figures")
a = ap.parse_args()

TAB, FIG = Path(a.tables), Path(a.figures)
rep = pd.read_csv(TAB / "A1_repertoire_size.csv")
cur = pd.read_csv(TAB / "A1_curves.csv")
pk = pd.read_csv(TAB / "A1_peaks.csv")
ps.use()

STYLE = {
    "paper":      dict(color=ps.C["mean"],   ls="-",            label=r"paper ($k=4$)"),
    "fixed_n12":  dict(color="#8FBADA",      ls=(0, (4, 1.5)),  label=r"fixed $n_{\rm rxn}=12$"),
    "fixed_n24":  dict(color="#C0392B",      ls=(0, (4, 1.5)),  label=r"fixed $n_{\rm rxn}=24$"),
    "fixed_n40":  dict(color="#7B241C",      ls=(0, (4, 1.5)),  label=r"fixed $n_{\rm rxn}=40$"),
    "ksweep_k2":  dict(color="#2E7D5B",      ls=(0, (1, 1.2)),  label=r"$k=2$"),
    "ksweep_k8":  dict(color="#E08214",      ls=(0, (1, 1.2)),  label=r"$k=8$"),
    "ksweep_k16": dict(color="#6A51A3",      ls=(0, (1, 1.2)),  label=r"$k=16$"),
}
FIXED = ["paper", "fixed_n12", "fixed_n24", "fixed_n40"]
KSW = ["ksweep_k2", "paper", "ksweep_k8", "ksweep_k16"]
QUANT = [("s", "median", r"median sensitivity $s$"),
         ("n_excreted", "mean", "mean # excreted metabolites"),
         ("ub", "mean", r"mean $ub$"),
         ("n_bounded", "mean", r"mean $b$")]


def draw(ax, arms, quantity, stat, axis, band=True):
    for arm in arms:
        g = cur[(cur.arm == arm) & (cur.quantity == quantity)
                & (cur.statistic == stat) & (cur.axis == axis)].sort_values("group_value")
        if g.empty:
            continue
        st = STYLE[arm]
        ax.plot(g["group_value"], g["value"], color=st["color"], ls=st["ls"],
                label=st["label"], lw=1.1)
        if band:
            ax.fill_between(g["group_value"], g["ci_lo"], g["ci_hi"],
                            color=st["color"], alpha=0.16, lw=0)
        p = pk[(pk.arm == arm) & (pk.quantity == quantity)
               & (pk.statistic == stat) & (pk.axis == axis)]
        if not p.empty:
            ax.plot([p["peak_at"].iloc[0]], [p["peak_value"].iloc[0]],
                    marker="v", ms=3.0, color=st["color"], mec="none")


# --------------------------------------------------------------------------- #
# 1. the premise: effective repertoire vs r
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(ps.COL15, 2.0))
fig.subplots_adjust(wspace=0.34)

ax = axes[0]
for arm in FIXED + ["ksweep_k2", "ksweep_k8", "ksweep_k16"]:
    g = rep[rep.arm == arm].sort_values("r")
    if g.empty:
        continue
    st = STYLE[arm]
    ax.plot(g["r"], g["rxn_eff_mean"], color=st["color"], ls=st["ls"],
            label=st["label"], lw=1.1)
ax.axhline(45, color=ps.C["ref"], lw=0.6, ls=":")
ax.text(52, 43.2, r"pool size $\binom{m}{2}=45$", fontsize=5.6,
        color=ps.C["ref"], va="top")
ax.set_xscale("log")
ax.legend(loc="lower right", fontsize=5.4, ncol=2)
ps.tidy(ax, r"Number of modes, $r$", "unique reactions in $R^{S}$")
ps.panel_label(ax, "A", dx=-0.19)

ax = axes[1]
for arm in FIXED:
    g = rep[rep.arm == arm].sort_values("r")
    st = STYLE[arm]
    ax.plot(g["r"], g["mode_size_mean"], color=st["color"], ls=st["ls"],
            label=st["label"], lw=1.1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.legend(loc="upper right", fontsize=5.4)
ps.tidy(ax, r"Number of modes, $r$", "mean reactions per mode")
ps.panel_label(ax, "B", dx=-0.19)
print("wrote", ps.save(fig, "A1_repertoire", FIG,
                       {"tables": ["A1_repertoire_size.csv"]}))

# --------------------------------------------------------------------------- #
# 2. mechanism quantities vs r, and vs b
# --------------------------------------------------------------------------- #
for axis, xlab, figid in (("r", r"Number of modes, $r$", "A1_mechanism_r"),
                          ("b", r"Bounded metabolites, $b$", "A1_mechanism_b")):
    fig, axes = plt.subplots(2, 4, figsize=(ps.COL2, 3.7))
    fig.subplots_adjust(wspace=0.40, hspace=0.46)
    for row, (arms, rowlab) in enumerate(
            [(FIXED, "fixed repertoire"), (KSW, "$k$ sweep")]):
        for col, (q, st, ylab) in enumerate(QUANT):
            ax = axes[row, col]
            draw(ax, arms, q, st, axis)
            if axis == "r":
                ax.set_xscale("log")
            if q == "n_bounded" and axis == "b":
                ax.set_visible(False)
                continue
            if axis == "b":
                ax.axvline(4.5, color=ps.C["ref"], lw=0.6, ls=":")
            ps.tidy(ax, xlab, ylab)
            if col == 0:
                ax.set_title(rowlab, loc="left", fontsize=7.0)
                ax.legend(loc="best", fontsize=5.2)
            ps.panel_label(ax, "ABCDEFGH"[row * 4 + col], dx=-0.34)
    print("wrote", ps.save(fig, figid, FIG,
                           {"tables": ["A1_curves.csv", "A1_peaks.csv"],
                            "axis": axis}))
