"""R4 tables and plot data from the post-evolution (Level 3) cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.rng import derive
from grhd.stats import (bootstrap_curve_and_peak, fisher_pool, group_summary,
                        loess_peak, partial_spearman, quartile_contrast,
                        spearman)

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="data/raw/post_paper")
ap.add_argument("--pre-per-symbiont", default="tables/pre_per_symbiont.csv")
ap.add_argument("--tables", default="tables")
ap.add_argument("--derived", default="data/derived")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
ap.add_argument("--tag", default="R4")
a = ap.parse_args()

cfg = load_config(*a.config)
B = int(cfg["stats"]["bootstrap_B"])
SEED = int(cfg["master_seed"])
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)
DER = Path(a.derived); DER.mkdir(parents=True, exist_ok=True)

files = sorted(Path(a.cells).glob("cells_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
mu_H_0 = float(df["mu_H_0"].iloc[0])
m = int(df["m"].iloc[0])
print(f"{len(df):,} post cells; mu_H_0={mu_H_0:.6f}; "
      f"status={df['solver_status'].value_counts().to_dict()}", flush=True)

df["detlimit"] = df["flag_detlimit"].astype(bool)
df["mutualistic_1e6"] = (df["mu_H"] > mu_H_0) & (df["mu_S"] > 1e-6)
df["mutualistic_1e3"] = (df["mu_H"] > mu_H_0) & (df["mu_S"] > 1e-3)

# --------------------------------------------------------------------------- #
# G1 sanity: mu_H >= mu_H_0 must hold by construction (c^H = 0 is feasible)
# --------------------------------------------------------------------------- #
viol = df[df["mu_H"] < mu_H_0 - 1e-9]
viol_summary = {
    "n_violations": int(len(viol)),
    "by_r": viol.groupby("r").size().to_dict() if len(viol) else {},
    "by_status": viol["solver_status"].value_counts().to_dict() if len(viol) else {},
    "worst_deficit": float((mu_H_0 - viol["mu_H"]).max()) if len(viol) else 0.0,
}
print("G1 mu_H >= mu_H_0 violations:", viol_summary, flush=True)

# --------------------------------------------------------------------------- #
# per-r summaries, with and without detlimit incumbents
# --------------------------------------------------------------------------- #
def _extra(g):
    return {
        "frac_detlimit": float(g["detlimit"].mean()),
        "frac_cmax_active": float(g["flag_cmax_active"].mean()),
        "frac_infeasible": float(g["flag_infeasible"].mean()),
        "frac_trivial_ub": float(g["trivial_ub"].mean()),
        "frac_mutualistic_1e6": float(g["mutualistic_1e6"].mean()),
        "frac_mutualistic_1e3": float(g["mutualistic_1e3"].mean()),
        "frac_muH_gt_muH0": float((g["mu_H"] > mu_H_0 + 1e-9).mean()),
        "mip_gap_median": float(np.nanmedian(g["mip_gap"])),
        "mip_gap_p95": float(np.nanquantile(g["mip_gap"], 0.95)),
        "mip_gap_max": float(np.nanmax(g["mip_gap"])),
        "solve_time_median_s": float(g["solve_time_milp_s"].median()),
        "solve_time_p95_s": float(g["solve_time_milp_s"].quantile(0.95)),
        "mu_H_0": mu_H_0,
    }

cols = ["mu_H", "mu_S", "delta_mu_H", "n_bounded", "n_unbounded", "ub",
        "n_excreted", "r_plus", "alpha", "s", "c_H_l1"]

frames = {}
for label, d in (("all", df), ("no_detlimit", df[~df["detlimit"]])):
    t = group_summary(d, "r", cols, extra=_extra)
    for col in ("mu_H", "mu_S"):
        curve, pk = bootstrap_curve_and_peak(
            d, col, group="r", unit="symbiont_id", B=B, master_seed=SEED,
            table_id=f"R4_{label}_{col}")
        t = t.merge(curve.rename(columns={"value": f"{col}_boot_mean",
                                          "ci_lo": f"{col}_ci_lo",
                                          "ci_hi": f"{col}_ci_hi",
                                          "n": f"{col}_n"}), on="r")
        if col == "mu_H":
            pk.update({"subset": label, "quantity": col, "statistic": "mean",
                       "loess_peak_at": loess_peak(curve["r"], curve["value"]),
                       "mu_H_0": mu_H_0})
            frames.setdefault("peaks", []).append(pk)
        # median peak too
    curve_md, pk_md = bootstrap_curve_and_peak(
        d, "mu_H", group="r", unit="symbiont_id", stat=np.median, B=B,
        master_seed=SEED, table_id=f"R4_{label}_muH_median")
    pk_md.update({"subset": label, "quantity": "mu_H", "statistic": "median",
                  "loess_peak_at": loess_peak(curve_md["r"], curve_md["value"]),
                  "mu_H_0": mu_H_0})
    frames.setdefault("peaks", []).append(pk_md)
    t["subset"] = label
    frames.setdefault("stats", []).append(t)

stats = pd.concat(frames["stats"], ignore_index=True)
stats.to_csv(TAB / "R4_stats.csv", index=False)
peaks = pd.DataFrame(frames["peaks"])
peaks.to_csv(TAB / "R4_peak.csv", index=False)
print(peaks.to_string(index=False), flush=True)

# curve used by the figure (all cells)
curve = stats[stats["subset"] == "all"][
    ["r", "mu_H_mean", "mu_H_ci_lo", "mu_H_ci_hi", "mu_H_median",
     "mu_S_mean", "mu_S_ci_lo", "mu_S_ci_hi", "mu_S_median",
     "n_bounded_mean", "n_excreted_mean", "s_median", "s_mean", "ub_mean",
     "frac_mutualistic_1e6", "frac_detlimit"]].copy()
curve.to_csv(TAB / "R4_curve.csv", index=False)

# --------------------------------------------------------------------------- #
# pre/post join for Fig.5A, Fig.S1 and Fig.S2A
# --------------------------------------------------------------------------- #
pre = pd.read_csv(a.pre_per_symbiont)
j = df.merge(pre, on=["r", "symbiont_id"], how="left", suffixes=("", "_pre"))
j = j.rename(columns={"mu_H": "mu_H_post", "mu_S": "mu_S_post"})
j["trivial_pre"] = j["ub_pre_mean"] < 1.0
keep = ["r", "symbiont_id", "mu_H_post", "mu_S_post", "mu_H_0",
        "mu_H_pre_max", "mu_H_pre_median", "mu_H_pre_mean", "ub_pre_mean",
        "s_mean", "alpha_mean", "b_mean", "n_excreted_mean", "trivial_pre",
        "parasitic_pre", "trivial_ub", "detlimit", "n_bounded", "n_excreted",
        "s", "ub"]
j[keep].to_parquet(DER / "plotdata_post_join.parquet", index=False)
df[["r", "symbiont_id", "mu_H", "mu_S", "delta_mu_H", "n_bounded",
    "n_excreted", "s", "ub", "trivial_ub", "detlimit"]].to_parquet(
        DER / "plotdata_post.parquet", index=False)

rows = []
for xcol, name in (("mu_H_pre_max", "5A_max"), ("mu_H_pre_median", "5A_median"),
                   ("ub_pre_mean", "S2A_ub")):
    for subset, sel in (("all", j["r"] > 0),
                        ("5<=r<=25", (j["r"] >= 5) & (j["r"] <= 25)),
                        ("non_trivial", ~j["trivial_pre"])):
        d = j[sel]
        rho, p, n = spearman(d[xcol], d["mu_H_post"])
        prho, pn = partial_spearman(d[xcol], d["mu_H_post"], d["r"])
        wr = [spearman(g[xcol], g["mu_H_post"]) for _, g in d.groupby("r")]
        rows.append({"figure": name, "x": xcol, "subset": subset,
                     "spearman_pooled": rho, "p": p, "n": n,
                     "partial_spearman_on_r": prho,
                     "within_r_fisher": fisher_pool([w[0] for w in wr],
                                                    [w[2] for w in wr]),
                     "within_r_median": float(np.nanmedian([w[0] for w in wr])),
                     **{f"qc_{k}": v for k, v in
                        quartile_contrast(d[xcol], d["mu_H_post"]).items()}})
pd.DataFrame(rows).to_csv(TAB / "R4_correlations.csv", index=False)

json.dump({"mu_H_0": mu_H_0, "m": m, "n_post_cells": int(len(df)),
           "r_grid": sorted(int(x) for x in df["r"].unique()),
           "N_S": int(df.groupby("r")["symbiont_id"].nunique().max()),
           "bootstrap_B": B, "G1_violations": viol_summary,
           "status_counts": df["solver_status"].value_counts().to_dict(),
           "frac_detlimit": float(df["detlimit"].mean()),
           "frac_cmax_active": float(df["flag_cmax_active"].mean()),
           "total_milp_core_hours": float(df["solve_time_milp_s"].sum() / 3600)},
          open(TAB / "R4_meta.json", "w"), indent=2)
print("wrote R4 tables", flush=True)
