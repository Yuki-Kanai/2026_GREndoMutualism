"""A3-a — how much does each post-hoc convention move the post-evolution curves?

07_run2_workorder.md Phase A.  The raw values are never overwritten; every
convention is reconstructed in the statistics layer from mu_S, mu_H and the
flag columns.

  C0_raw        nothing
  C1_muS0       mu_S := 0 where delta_mu_H < 1e-2
  C2_muHfloor   mu_H := max(mu_H, 1e-5)
  C3_infeas0    mu_H = mu_S = 0 where flag_infeasible
  C123_all      all three
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.stats import paired_group_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="data/raw/post_paper")
ap.add_argument("--tables", default="tables")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
B = int(cfg["stats"]["bootstrap_B"])
SEED = int(cfg["master_seed"])
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)

files = sorted(Path(a.cells).glob("cells_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
mu_H_0 = float(df["mu_H_0"].iloc[0])
print(f"{len(df)} post cells, mu_H_0 = {mu_H_0:.6f}", flush=True)

DELTA_THR = 1e-2
MU_H_FLOOR = 1e-5

small = df["delta_mu_H"] < DELTA_THR
infeas = df["flag_infeasible"].astype(bool)
print(f"flag_delta_small: {int(small.sum())}  flag_infeasible: {int(infeas.sum())}",
      flush=True)

V = {}
V["C0_raw"] = (df["mu_S"].to_numpy(float), df["mu_H"].to_numpy(float))
V["C1_muS0"] = (np.where(small, 0.0, df["mu_S"]), df["mu_H"].to_numpy(float))
V["C2_muHfloor"] = (df["mu_S"].to_numpy(float),
                    np.maximum(df["mu_H"].to_numpy(float), MU_H_FLOOR))
V["C3_infeas0"] = (np.where(infeas, 0.0, df["mu_S"]),
                   np.where(infeas, 0.0, df["mu_H"]))
V["C123_all"] = (np.where(small | infeas, 0.0, df["mu_S"]),
                 np.where(infeas, 0.0, np.maximum(df["mu_H"].to_numpy(float),
                                                  MU_H_FLOOR)))

work = df[["r", "symbiont_id"]].copy()
for name, (ms, mh) in V.items():
    work[f"mu_S__{name}"] = ms
    work[f"mu_H__{name}"] = mh

# --------------------------------------------------------------------------- #
curves = []
for qty, stat, label in (("mu_S", np.mean, "mean"), ("mu_S", np.median, "median"),
                         ("mu_H", np.mean, "mean"), ("mu_H", np.median, "median")):
    cols = [f"{qty}__{v}" for v in V]
    t = paired_group_bootstrap(work, "r", "symbiont_id", cols, stat=stat, B=B,
                               master_seed=SEED, table_id=f"A3a_{qty}_{label}",
                               baseline=f"{qty}__C0_raw")
    t["quantity"] = qty
    t["statistic"] = label
    t["version"] = t["column"].str.split("__").str[-1]
    curves.append(t.drop(columns="column"))
curves = pd.concat(curves, ignore_index=True)
curves.to_csv(TAB / "A3a_convention_curves.csv", index=False)

delta = curves[curves["version"] != "C0_raw"].dropna(subset=["diff_vs_baseline"])
delta[["r", "quantity", "statistic", "version", "n", "value",
       "diff_vs_baseline", "diff_lo", "diff_hi"]].to_csv(
    TAB / "A3a_convention_delta.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== mean shift vs C0_raw, pooled over r (max |shift| and where) ===")
for qty in ("mu_S", "mu_H"):
    d = delta[(delta["quantity"] == qty) & (delta["statistic"] == "mean")]
    for v, g in d.groupby("version"):
        j = g["diff_vs_baseline"].abs().idxmax()
        row = g.loc[j]
        # "moves the curve" means the CI is strictly on one side of zero;
        # an exactly-zero effect has CI [0, 0] and must not count
        sig = int(((g["diff_lo"] > 0) | (g["diff_hi"] < 0)).sum())
        print(f"  {qty:5s} {v:12s} max|Δ| = {row['diff_vs_baseline']:+.5f} at r={row['r']:>3}"
              f"   (CI [{row['diff_lo']:+.5f}, {row['diff_hi']:+.5f}])"
              f"   CI excludes 0 at {sig}/{len(g)} r values")

# --------------------------------------------------------------------------- #
# threshold sweep for C1_muS0
rows = []
thresholds = [("abs", t, t) for t in (1e-3, 1e-2, 1e-1)] + \
             [("rel", f, f * mu_H_0) for f in (0.01, 0.10, 0.20)]
for kind, nominal, thr in thresholds:
    mask = df["delta_mu_H"] < thr
    mus = np.where(mask, 0.0, df["mu_S"])
    tmp = df[["r"]].copy()
    tmp["mu_S_conv"] = mus
    g = tmp.groupby("r")["mu_S_conv"].agg(["mean", "median", "size"]).reset_index()
    g["threshold_kind"] = kind
    g["threshold_nominal"] = nominal
    g["threshold_value"] = thr
    g["frac_zeroed"] = df.assign(m=mask).groupby("r")["m"].mean().to_numpy()
    rows.append(g)
sweep = pd.concat(rows, ignore_index=True)
sweep.to_csv(TAB / "A3a_threshold_sweep.csv", index=False)
print("\n=== C1_muS0 threshold sweep: fraction of cells zeroed ===")
print(sweep.pivot_table(index="r", columns=["threshold_kind", "threshold_nominal"],
                        values="frac_zeroed").round(3).to_string())
print("\n=== and the resulting mean mu_S ===")
print(sweep.pivot_table(index="r", columns=["threshold_kind", "threshold_nominal"],
                        values="mean").round(4).to_string())

# --------------------------------------------------------------------------- #
# figure: raw vs full convention, mean mu_S and mu_H with CI
from grhd import plots, plotstyle as ps                       # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402

ps.use()
fig, axes = plt.subplots(1, 2, figsize=(ps.COL15, 2.0))
fig.subplots_adjust(wspace=0.38)
for ax, qty, ylab in ((axes[0], "mu_S", r"mean $\mu^{S}$ after evolution"),
                      (axes[1], "mu_H", r"mean $\mu^{H}$ after evolution")):
    d = curves[(curves["quantity"] == qty) & (curves["statistic"] == "mean")]
    for ver, col, lab in (("C0_raw", ps.C["mean"], "raw values"),
                          ("C123_all", ps.C["mean2"], "all three conventions")):
        g = d[d["version"] == ver].sort_values("r")
        plots.mean_curve(ax, g["r"], g["value"], g["ci_lo"], g["ci_hi"],
                         color=col, label=lab, marker="o")
    plots.log_r_axis(ax, rmax=float(d["r"].max()))
    ps.tidy(ax, "Symbiont genome size, $r$", ylab)
    if qty == "mu_H":
        plots.hline(ax, mu_H_0)
axes[0].legend(loc="best")
ps.panel_label(axes[0], "A", dx=-0.22)
ps.panel_label(axes[1], "B", dx=-0.22)
print("\nwrote", ps.save(fig, "A3a_fig4B_raw_vs_convention", Path("figures"),
                          {"tables": ["A3a_convention_curves.csv"],
                           "conventions": "C1 mu_S:=0 if delta<1e-2; "
                                          "C2 mu_H:=max(mu_H,1e-5); "
                                          "C3 infeasible -> 0",
                           "mu_H_0": mu_H_0, "bootstrap_B": B}))

json.dump({"mu_H_0": mu_H_0, "n_cells": int(len(df)),
           "n_delta_small": int(small.sum()), "n_infeasible": int(infeas.sum()),
           "delta_threshold": DELTA_THR, "mu_H_floor": MU_H_FLOOR,
           "bootstrap_B": B},
          open(TAB / "A3a_meta.json", "w"), indent=2)
print("\nwrote A3a_convention_curves.csv, A3a_convention_delta.csv, "
      "A3a_threshold_sweep.csv")
