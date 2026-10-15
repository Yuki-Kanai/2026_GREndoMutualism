"""A1-MILP (gate G2b, the MAIN VERDICT).

07_run2_workorder.md section 3.2 / 03_additional_analyses.md A1.

The question is the one the whole control exists for: with the metabolic
repertoire held fixed, does post-evolution mu^H still peak at an intermediate
r?  Two statistics are primary and BOTH are reported (07 section 3.1 A3-e):

  * the mean of mu^H
  * the mutualism fraction, #{mu^H > mu^H_0 + eps and mu^S > theta} / N

The verdict is the bootstrap CI of (peak - endpoint), not the look of a curve.

Because fixed_repertoire cannot go past r = n_rxn, the two arms are also
compared on axes that do not depend on the r ceiling: b, ub and r/n_rxn.

Outputs
  tables/A1_muH_peaks.csv   peak + CI + both endpoint contrasts, per arm/statistic
  tables/A1_muH_curves.csv  the curves behind them (long format)
  tables/A1_milp_health.csv solver status, milp_path, detlimit, gap, per arm and r
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.stats import bootstrap_curve_and_peak

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--tables", default="tables")
ap.add_argument("--n-rxn", type=int, default=40)
ap.add_argument("--boot", type=int, default=4000)
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
SEED = int(cfg["master_seed"])
M = int(cfg["model"]["m"])
RAW, TAB = Path(a.raw), Path(a.tables)
TAB.mkdir(parents=True, exist_ok=True)

ARMS = [("paper", "paper", np.nan, "post_paper"),
        (f"fixed_n{a.n_rxn}", "fixed_repertoire", float(a.n_rxn),
         f"post_fixed_repertoire_n{a.n_rxn}")]

# mutualism thresholds (A3-e).  The first pair is the primary definition.
EPS_THETA = [(0.0, 1e-6), (0.0, 1e-3), (1e-3, 1e-6), (1e-2, 1e-6)]


def load(sub):
    files = sorted((RAW / sub).glob("cells_*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


frames, health = {}, []
for arm_id, scheme, nrxn, sub in ARMS:
    df = load(sub)
    if df is None:
        print(f"MISSING {sub}"); continue
    df["mutual"] = ((df["mu_H"] > df["mu_H_0"] + EPS_THETA[0][0])
                    & (df["mu_S"] > EPS_THETA[0][1])).astype(float)
    df["muH_over_muH0"] = df["mu_H"] / df["mu_H_0"]
    # mu^H == mu^H_0 exactly: the host gains nothing at all
    df["no_gain"] = (df["mu_H"] <= df["mu_H_0"] + 1e-12).astype(float)
    frames[arm_id] = df
    for r, g in df.groupby("r"):
        health.append({
            "arm": arm_id, "r": int(r), "n": len(g),
            "optimal": int((g.solver_status == "optimal").sum()),
            "detlimit": int(g.flag_detlimit.sum()),
            "infeasible": int(g.flag_infeasible.sum()),
            "cmax_active": int(g.flag_cmax_active.sum()),
            # the paper arm predates the milp_path column
            **{f"milp_path_{lab}": (int((g["milp_path"] == val).sum())
                                    if "milp_path" in g.columns else 0)
               for lab, val in (("default", "default"),
                                ("num_emph", "numerical_emphasis"),
                                ("presolve", "presolve_on"))},
            "max_gap": float(np.nanmax(g.mip_gap)) if g.mip_gap.notna().any() else np.nan,
            "viol_muH_lt_muH0": int((g.mu_H < g.mu_H_0 - 1e-9).sum()),
            "core_hours": float(g.solve_time_milp_s.sum() / 3600),
        })

pd.DataFrame(health).to_csv(TAB / "A1_milp_health.csv", index=False)

# --------------------------------------------------------------------------- #
QUANT = [("mu_H", "mean", np.mean), ("mu_H", "median", np.median),
         ("mutual", "fraction", np.mean),
         ("muH_over_muH0", "mean", np.mean),
         ("no_gain", "fraction", np.mean)]

curves, peaks = [], []
for arm_id, scheme, nrxn, _ in ARMS:
    if arm_id not in frames:
        continue
    df = frames[arm_id]
    bmap = df.groupby("r")["n_bounded"].mean()
    for col, sname, fn in QUANT:
        for axis, gcol in (("r", "r"), ("b", "n_bounded")):
            sub = df[np.isfinite(df[col])]
            if axis == "b":
                keep = sub.groupby(gcol)["symbiont_id"].transform("nunique") >= 20
                sub = sub[keep]
            if sub.empty or sub[gcol].nunique() < 3:
                continue
            cv, pk = bootstrap_curve_and_peak(
                sub, col, group=gcol, unit="symbiont_id", stat=fn, B=a.boot,
                master_seed=SEED, table_id=f"A1M_{arm_id}_{col}_{sname}_{axis}")
            cv = cv.rename(columns={gcol: "group_value"})
            for kk, vv in (("axis", axis), ("statistic", sname), ("quantity", col),
                           ("arm", arm_id)):
                cv.insert(0, kk, vv)
            curves.append(cv)
            pk.update({"arm": arm_id, "scheme": scheme, "n_rxn": nrxn,
                       "quantity": col, "statistic": sname, "axis": axis,
                       "support_min": int(cv["n"].min())})
            if axis == "r":
                pk["peak_at_mean_b"] = float(bmap.get(pk["peak_at"], np.nan))
                pk["peak_r_over_n_rxn"] = (pk["peak_at"] / nrxn
                                           if np.isfinite(nrxn) else np.nan)
            peaks.append(pk)

# the mutualism fraction under the other (eps, theta) pairs, r axis only
for arm_id, _, _, _ in ARMS:
    if arm_id not in frames:
        continue
    df = frames[arm_id]
    for eps, th in EPS_THETA[1:]:
        col = f"mutual_e{eps:g}_t{th:g}"
        df[col] = ((df["mu_H"] > df["mu_H_0"] + eps) & (df["mu_S"] > th)).astype(float)
        cv, pk = bootstrap_curve_and_peak(
            df, col, group="r", unit="symbiont_id", stat=np.mean, B=a.boot,
            master_seed=SEED, table_id=f"A1M_{arm_id}_{col}")
        pk.update({"arm": arm_id, "quantity": "mutual", "statistic": f"eps{eps:g}_th{th:g}",
                   "axis": "r", "support_min": int(cv["n"].min())})
        peaks.append(pk)

pd.concat(curves, ignore_index=True).to_csv(TAB / "A1_muH_curves.csv", index=False)
pk = pd.DataFrame(peaks)
pk.to_csv(TAB / "A1_muH_peaks.csv", index=False)

pd.set_option("display.width", 260)
show = ["arm", "quantity", "statistic", "axis", "peak_at", "peak_ci_lo", "peak_ci_hi",
        "peak_value", "diff_peak_minus_first", "diff_peak_minus_first_lo",
        "diff_peak_minus_first_hi", "diff_peak_minus_last",
        "diff_peak_minus_last_lo", "diff_peak_minus_last_hi", "last_r"]
print("\n=== MAIN VERDICT: peak vs endpoints ===")
print(pk[show].round(4).to_string(index=False))

print("\n=== curves (r axis) ===")
cv = pd.concat(curves, ignore_index=True)
for q, sname in [("mu_H", "mean"), ("mutual", "fraction"), ("no_gain", "fraction")]:
    print(f"\n-- {q} ({sname})")
    w = cv[(cv.quantity == q) & (cv.statistic == sname) & (cv.axis == "r")]
    print(w.pivot_table(index="group_value", columns="arm",
                        values="value").round(4).to_string())

print("\n=== solver health ===")
h = pd.DataFrame(health)
print(h.groupby("arm")[["n", "optimal", "detlimit", "infeasible", "cmax_active",
                        "milp_path_default", "milp_path_num_emph",
                        "milp_path_presolve", "viol_muH_lt_muH0",
                        "core_hours"]].sum().to_string())
