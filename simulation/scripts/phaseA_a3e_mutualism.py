"""A3-e — the mutualism fraction as a primary statistic.

07_run2_workorder.md Phase A.  Post-evolution mu_H is zero-inflated (its median
sits exactly at mu_H_0 for most r), so the mean alone hides the shape of the
distribution.  The fraction of pairs that become mutualistic is reported with
the same peak / CI machinery as the mean curve.

mutualistic  <=>  mu_H > mu_H_0 + eps  and  mu_S > theta
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.stats import bootstrap_curve_and_peak, loess_peak

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="data/raw/post_paper")
ap.add_argument("--tables", default="tables")
ap.add_argument("--figures", default="figures")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
B = int(cfg["stats"]["bootstrap_B"])
SEED = int(cfg["master_seed"])
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)

df = pd.concat([pd.read_parquet(f) for f in
                sorted(Path(a.cells).glob("cells_*.parquet"))], ignore_index=True)
mu_H_0 = float(df["mu_H_0"].iloc[0])
print(f"{len(df)} post cells, mu_H_0 = {mu_H_0:.6f}", flush=True)

EPS = [0.0, 1e-3, 1e-2]
THETA = [1e-6, 1e-3]

curves, peaks = [], []
for eps in EPS:
    for th in THETA:
        tag = f"eps{eps:g}_theta{th:g}"
        df[tag] = ((df["mu_H"] > mu_H_0 + eps) & (df["mu_S"] > th)).astype(float)
        c, pk = bootstrap_curve_and_peak(df, tag, group="r", unit="symbiont_id",
                                         stat=np.mean, B=B, master_seed=SEED,
                                         table_id=f"A3e_{tag}")
        c["eps"], c["theta"] = eps, th
        curves.append(c)
        pk.update({"eps": eps, "theta": th,
                   "loess_peak_at": loess_peak(c["r"], c["value"])})
        peaks.append(pk)

curves = pd.concat(curves, ignore_index=True)
curves.rename(columns={"value": "fraction"}).to_csv(
    TAB / "A3e_mutualism_curves.csv", index=False)
pk = pd.DataFrame(peaks)
pk.to_csv(TAB / "A3e_mutualism_peaks.csv", index=False)

# zero-inflation: how often does the host gain exactly nothing?
tolz = 1e-9
zero = df.assign(at_floor=(df["mu_H"] - mu_H_0).abs() <= tolz)
z = zero.groupby("r").agg(n=("at_floor", "size"),
                          frac_muH_eq_muH0=("at_floor", "mean"),
                          frac_muS_zero=("mu_S", lambda s: float((s <= 1e-9).mean())),
                          median_mu_H=("mu_H", "median")).reset_index()
z.to_csv(TAB / "A3e_zero_inflation.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== mutualism fraction: peak and endpoint contrasts ===")
print(pk[["eps", "theta", "peak_at", "peak_value", "peak_ci_lo", "peak_ci_hi",
          "loess_peak_at", "diff_peak_minus_first", "diff_peak_minus_first_lo",
          "diff_peak_minus_first_hi", "diff_peak_minus_last",
          "diff_peak_minus_last_lo", "diff_peak_minus_last_hi"]]
      .round(4).to_string(index=False))
print("\n=== zero inflation ===")
print(z.round(4).to_string(index=False))

# --------------------------------------------------------------------------- #
from grhd import plots, plotstyle as ps                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

ps.use()
fig, axes = plt.subplots(1, 2, figsize=(ps.COL15, 2.0))
fig.subplots_adjust(wspace=0.36)

ax = axes[0]
# the six (eps, theta) combinations are visually indistinguishable: draw the
# default bold with its CI and the other five as thin lines over it
for eps, th in [(e, t) for e in EPS for t in THETA]:
    if (eps, th) == (0.0, 1e-6):
        continue
    c = curves[(curves["eps"] == eps) & (curves["theta"] == th)].sort_values("r")
    ax.plot(c["r"], c["value"], color="#9aa3ab", lw=0.6, zorder=4)
c = curves[(curves["eps"] == 0.0) & (curves["theta"] == 1e-6)].sort_values("r")
plots.mean_curve(ax, c["r"], c["value"], c["ci_lo"], c["ci_hi"],
                 color=ps.C["mean"], marker="o",
                 label=r"$\varepsilon=0$, $\theta=10^{-6}$ (95% CI)")
ax.plot([], [], color="#9aa3ab", lw=0.6,
        label=r"5 other $(\varepsilon,\theta)$")
plots.log_r_axis(ax, rmax=float(curves["r"].max()))
ax.set_ylim(-0.03, 1.10)
ax.legend(loc="upper left", fontsize=5.6)
ps.tidy(ax, "Symbiont genome size, $r$", "fraction mutualistic")
ps.panel_label(ax, "A", dx=-0.22)

ax = axes[1]
plots.mean_curve(ax, z["r"], z["frac_muH_eq_muH0"], color=ps.C["mean2"],
                 marker="o", label=r"$\mu^{H}=\mu^{H}_{0}$ exactly")
plots.mean_curve(ax, z["r"], z["frac_muS_zero"], color=ps.C["mean"], lw=0.8,
                 label=r"$\mu^{S}=0$ (overlaps)")
plots.log_r_axis(ax, rmax=float(z["r"].max()))
ax.set_ylim(-0.03, 1.10)
ax.legend(loc="lower left", fontsize=5.8)
ps.tidy(ax, "Symbiont genome size, $r$", "fraction of pairs")
ps.panel_label(ax, "B", dx=-0.22)
print("\nwrote", ps.save(fig, "A3e_mutualism_fraction", Path(a.figures),
                         {"tables": ["A3e_mutualism_curves.csv",
                                     "A3e_mutualism_peaks.csv",
                                     "A3e_zero_inflation.csv"],
                          "mu_H_0": mu_H_0, "bootstrap_B": B,
                          "definition": "mu_H > mu_H_0 + eps and mu_S > theta"}))

json.dump({"mu_H_0": mu_H_0, "eps": EPS, "theta": THETA, "bootstrap_B": B},
          open(TAB / "A3e_meta.json", "w"), indent=2)
