"""2.3 — bound the detlimit cells instead of re-solving them.

07_run2_workorder.md Phase B 2.3.  154 / 4,500 Level 3 instances hit the
deterministic time limit.  Their incumbent is feasible but not optimal, so it
biases mu^H *downwards*, and it does so preferentially at large r.

Rather than paying ~250 core-hours to re-solve, evaluate the interval:

  lower  max(MILP incumbent, best mu^H over the pre-evolution sampled c^H)
         the second term is legitimate because any feasible c^H gives a lower
         bound on the Eq.3 optimum (01_model_spec.md section 10.1-9); it also
         repairs the 3 cells where the incumbent was below a sampled point.
  upper  min(CPLEX best bound, 1)      mu^H <= 1^T c^E = 1 (section 10-3)

Run 1 stored mip_gap but not the best bound, so the bound is reconstructed as
incumbent * (1 + gap)  [CPLEX relative gap = |bb - inc| / (1e-10 + |inc|)].
New runs store mip_best_bound directly.

The question is only whether the conclusion (peak location, endpoint contrasts)
is the same at both ends of the interval.
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
ap.add_argument("--post", default="data/raw/post_paper")
ap.add_argument("--join", default="data/derived/plotdata_post_join.parquet")
ap.add_argument("--tables", default="tables")
ap.add_argument("--figures", default="figures")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
B = int(cfg["stats"]["bootstrap_B"])
SEED = int(cfg["master_seed"])
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)

df = pd.concat([pd.read_parquet(f) for f in
                sorted(Path(a.post).glob("cells_*.parquet"))], ignore_index=True)
mu_H_0 = float(df["mu_H_0"].iloc[0])
j = pd.read_parquet(a.join)[["r", "symbiont_id", "mu_H_pre_max"]]
df = df.merge(j, on=["r", "symbiont_id"], how="left")
det = df["flag_detlimit"].astype(bool)
print(f"{len(df)} cells, {int(det.sum())} detlimit ({det.mean():.3%})", flush=True)

# ---- bounds ---------------------------------------------------------------- #
df["mu_H_incumbent"] = df["mu_H"]
df["mu_H_lower"] = np.maximum(df["mu_H"], df["mu_H_pre_max"].fillna(-np.inf))
if "mip_best_bound" in df.columns and df["mip_best_bound"].notna().any():
    bb = df["mip_best_bound"]
else:
    bb = df["mu_H"] * (1.0 + df["mip_gap"].fillna(0.0))
df["mu_H_upper"] = np.where(det, np.minimum(bb, 1.0), df["mu_H"])
df["mu_H_upper"] = np.maximum(df["mu_H_upper"], df["mu_H_lower"])
df["interval_width"] = df["mu_H_upper"] - df["mu_H_lower"]

repaired = df[det & (df["mu_H_lower"] > df["mu_H"] + 1e-12)]
repaired_any = df[df["mu_H_lower"] > df["mu_H"] + 1e-12]
print(f"lower bound raised above the incumbent in {len(repaired_any)} cells "
      f"({len(repaired)} of them detlimit); "
      f"max gain {float((df['mu_H_lower'] - df['mu_H']).max()):.3e}", flush=True)

per_r = df.assign(det=det).groupby("r").agg(
    n=("mu_H", "size"), n_detlimit=("det", "sum"),
    frac_detlimit=("det", "mean"),
    mean_incumbent=("mu_H_incumbent", "mean"),
    mean_lower=("mu_H_lower", "mean"),
    mean_upper=("mu_H_upper", "mean"),
    max_interval=("interval_width", "max"),
    mean_interval=("interval_width", "mean"),
    gap_median=("mip_gap", "median"), gap_max=("mip_gap", "max")).reset_index()
per_r.to_csv(TAB / "B23_detlimit_interval_per_r.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== per-r mean mu^H at both ends of the interval ===")
print(per_r.round(4).to_string(index=False))

# ---- does the conclusion change at either end? ----------------------------- #
summary = []
for col, label in (("mu_H_incumbent", "incumbent (as reported)"),
                   ("mu_H_lower", "lower bound (incumbent or sampled best)"),
                   ("mu_H_upper", "upper bound (best bound, capped at 1)")):
    curve, pk = bootstrap_curve_and_peak(df, col, group="r", unit="symbiont_id",
                                         B=B, master_seed=SEED,
                                         table_id=f"B23_{col}")
    pk.update({"variant": label, "column": col,
               "loess_peak_at": loess_peak(curve["r"], curve["value"])})
    summary.append(pk)
    curve.to_csv(TAB / f"B23_curve_{col}.csv", index=False)
s = pd.DataFrame(summary)
s.to_csv(TAB / "B23_detlimit_peaks.csv", index=False)
print("\n=== peak and endpoint contrasts at both ends ===")
print(s[["variant", "peak_at", "peak_ci_lo", "peak_ci_hi", "peak_value",
         "loess_peak_at", "diff_peak_minus_first", "diff_peak_minus_first_lo",
         "diff_peak_minus_last", "diff_peak_minus_last_lo"]]
      .round(4).to_string(index=False))

same_peak = len(set(s["peak_at"])) == 1
all_pos = bool((s["diff_peak_minus_first_lo"] > 0).all()
               and (s["diff_peak_minus_last_lo"] > 0).all())
print(f"\nsame peak location at both ends : {same_peak}")
print(f"both endpoint contrasts stay > 0 : {all_pos}")

cells_gap1 = df[det & (df["mip_gap"] > 1.0)]
cells_gap1[["r", "symbiont_id", "mu_H", "mu_H_lower", "mu_H_upper", "mip_gap",
            "solve_time_milp_s"]].to_csv(TAB / "B23_detlimit_gap_gt1.csv",
                                         index=False)
print(f"\ncells with mip_gap > 1: {len(cells_gap1)}")
print(cells_gap1.groupby("r").size().to_string())

json.dump({"n_cells": int(len(df)), "n_detlimit": int(det.sum()),
           "same_peak_at_both_ends": bool(same_peak),
           "endpoint_contrasts_positive_at_both_ends": all_pos,
           "n_lower_bound_repaired": int(len(repaired_any)),
           "best_bound_source": ("column" if "mip_best_bound" in df.columns
                                 and df["mip_best_bound"].notna().any()
                                 else "reconstructed from mip_gap"),
           "mu_H_0": mu_H_0},
          open(TAB / "B23_meta.json", "w"), indent=2)

# ---- figure ---------------------------------------------------------------- #
from grhd import plots, plotstyle as ps                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

ps.use()
fig, ax = plt.subplots(figsize=(ps.COL1, 1.95))
lo = pd.read_csv(TAB / "B23_curve_mu_H_lower.csv")
hi = pd.read_csv(TAB / "B23_curve_mu_H_upper.csv")
inc = pd.read_csv(TAB / "B23_curve_mu_H_incumbent.csv")
ax.fill_between(lo["r"], lo["value"], hi["value"], color=ps.C["band"],
                alpha=0.55, linewidth=0, zorder=2, label="bound interval")
plots.mean_curve(ax, inc["r"], inc["value"], color=ps.C["mean2"], marker="o",
                 label="incumbent (reported)")
plots.mean_curve(ax, hi["r"], hi["value"], color=ps.C["mean"], lw=0.9,
                 label="upper bound")
plots.hline(ax, mu_H_0)
plots.log_r_axis(ax, rmax=float(inc["r"].max()))
ax.legend(loc="upper right", fontsize=5.8)
ps.tidy(ax, "Symbiont genome size, $r$", r"mean $\mu^{H}$ after evolution")
print("\nwrote", ps.save(fig, "B23_detlimit_interval", Path(a.figures),
                         {"tables": ["B23_detlimit_interval_per_r.csv",
                                     "B23_detlimit_peaks.csv"],
                          "mu_H_0": mu_H_0, "bootstrap_B": B}))
