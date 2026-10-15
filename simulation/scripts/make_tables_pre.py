"""R1 / R2 / R3 tables and plot data from the pre-evolution cells.

Figures never touch data/raw: everything a figure needs is written here into
tables/*.csv and data/derived/plotdata_*.parquet (05_conventions.md section 8).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import ROOT, load_config
from grhd.rng import derive
from grhd.stats import (bootstrap_curve_and_peak, fisher_pool, group_summary,
                        loess_peak, partial_spearman, pearson, spearman,
                        summarise)

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="data/raw/pre_paper")
ap.add_argument("--vectors", default=None, help="defaults to the cells directory")
ap.add_argument("--tables", default="tables")
ap.add_argument("--derived", default="data/derived")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
ap.add_argument("--tag", default="R1")
ap.add_argument("--skip-r3", action="store_true")
a = ap.parse_args()

cfg = load_config(*a.config)
B = int(cfg["stats"]["bootstrap_B"])
SEED = int(cfg["master_seed"])
NSUB = int(cfg["stats"]["scatter_subsample"])
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)
DER = Path(a.derived); DER.mkdir(parents=True, exist_ok=True)

src = Path(a.cells)
files = sorted(src.glob("cells_*.parquet"))
print(f"reading {len(files)} cell blocks from {src}", flush=True)
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"{len(df):,} cells, r in [{df.r.min()},{df.r.max()}]", flush=True)

mu_H_0 = float(df["mu_H_0"].iloc[0])
m = int(df["m"].iloc[0])

# --------------------------------------------------------------------------- #
# R1
# --------------------------------------------------------------------------- #
sat_tol = 1e-8 + 1e-6 * df["c_H_l1"]
df["saturated"] = (df["c_H_l1"] - df["mu_S"]).abs() <= sat_tol

def _extra(g):
    return {
        "frac_saturated": float(g["saturated"].mean()),
        "frac_infeasible": float(g["flag_infeasible"].mean()),
        "frac_qp_fallback": float((g["qp_path"] == "lp_vertex").mean())
        if "qp_path" in g else np.nan,
        "frac_simplex_fallback": float((g["lp_path"] == "simplex_fallback").mean())
        if "lp_path" in g else np.nan,
        "frac_trivial_ub": float(g["trivial_ub"].mean()),
        "frac_delta_small": float(g["flag_delta_small"].mean()),
        "mu_H_0": mu_H_0,
    }

r1 = group_summary(df, "r", ["mu_S", "mu_H", "delta_mu_H"], extra=_extra)

for col in ("mu_S", "mu_H"):
    curve, _ = bootstrap_curve_and_peak(df, col, group="r", unit="symbiont_id",
                                        B=B, master_seed=SEED, table_id=f"{a.tag}_{col}")
    r1 = r1.merge(curve.rename(columns={"value": f"{col}_boot_mean",
                                        "ci_lo": f"{col}_ci_lo",
                                        "ci_hi": f"{col}_ci_hi",
                                        "n": f"{col}_n_symbionts"}), on="r")

# Spearman: pooled and within r
rho_p, p_p, n_p = spearman(df["mu_H"], df["mu_S"])
pe_p, _, _ = pearson(df["mu_H"], df["mu_S"])
prho, pn = partial_spearman(df["mu_H"], df["mu_S"], df["r"])
within = []
for rr, g in df.groupby("r"):
    rho, p, n = spearman(g["mu_H"], g["mu_S"])
    pe, _, _ = pearson(g["mu_H"], g["mu_S"])
    within.append({"r": rr, "spearman_rho": rho, "spearman_p": p,
                   "pearson_r": pe, "n": n})
within = pd.DataFrame(within)
r1 = r1.merge(within, on="r", how="left")
r1.to_csv(TAB / f"{a.tag}_stats.csv", index=False)

corr = pd.DataFrame([{
    "scope": "pooled", "statistic": "spearman_rho_muH_muS", "value": rho_p,
    "p": p_p, "n": n_p},
    {"scope": "pooled", "statistic": "pearson_r_muH_muS", "value": pe_p,
     "p": np.nan, "n": n_p},
    {"scope": "partial_on_r", "statistic": "partial_spearman_muH_muS",
     "value": prho, "p": np.nan, "n": pn},
    {"scope": "within_r_fisher_pooled", "statistic": "spearman_rho_muH_muS",
     "value": fisher_pool(within["spearman_rho"], within["n"]), "p": np.nan,
     "n": int(within["n"].sum())},
    {"scope": "within_r_median", "statistic": "spearman_rho_muH_muS",
     "value": float(within["spearman_rho"].median()), "p": np.nan,
     "n": len(within)},
    {"scope": "within_r_iqr", "statistic": "spearman_rho_muH_muS",
     "value": float(within["spearman_rho"].quantile(.75)
                    - within["spearman_rho"].quantile(.25)),
     "p": np.nan, "n": len(within)},
])
corr.to_csv(TAB / f"{a.tag}_correlations.csv", index=False)
print("wrote", TAB / f"{a.tag}_stats.csv", flush=True)

# --------------------------------------------------------------------------- #
# R2
# --------------------------------------------------------------------------- #
r2cols = ["s", "alpha", "n_bounded", "n_unbounded", "ub", "n_excreted",
          "r_plus", "n_degenerate", "cond_b"]
r2 = group_summary(df, "r", r2cols)
r2["r_plus_over_r"] = r2["r_plus_mean"] / r2["r"]
r2.to_csv(TAB / "R2_stats.csv", index=False)

peaks = []
for col, statname, fn in [("s", "median", np.median), ("s", "mean", np.mean),
                          ("s", "trimmed_mean_5", None),
                          ("n_excreted", "mean", np.mean),
                          ("ub", "mean", np.mean),
                          ("n_bounded", "mean", np.mean)]:
    if fn is None:
        from scipy import stats as sps
        fn = lambda z, axis=None: sps.trim_mean(z, 0.05, axis=axis)
    curve, pk = bootstrap_curve_and_peak(df, col, group="r", unit="symbiont_id",
                                         stat=fn, B=B, master_seed=SEED,
                                         table_id=f"R2_{col}_{statname}")
    pk.update({"quantity": col, "statistic": statname,
               "loess_peak_at": loess_peak(curve["r"], curve["value"])})
    # b-axis: which mean b does the peak sit at?
    bmap = r2.set_index("r")["n_bounded_mean"]
    pk["peak_at_mean_b"] = float(bmap.get(pk["peak_at"], np.nan))
    pk["b_reference_(m-1)/2"] = (m - 1) / 2
    peaks.append(pk)
    curve.to_csv(TAB / f"R2_curve_{col}_{statname}.csv", index=False)
pd.DataFrame(peaks).to_csv(TAB / "R2_peaks.csv", index=False)
print("wrote", TAB / "R2_peaks.csv", flush=True)

# --------------------------------------------------------------------------- #
# plot data (thinned scatter layers)
# --------------------------------------------------------------------------- #
rng = derive(SEED, "subsample", fig_id="pre_scatter")
parts = []
for rr, g in df.groupby("r"):
    n = min(NSUB, len(g))
    parts.append(g.iloc[rng.choice(len(g), size=n, replace=False)])
plot = pd.concat(parts, ignore_index=True)[
    ["r", "symbiont_id", "c_id", "mu_S", "mu_H", "delta_mu_H", "s", "ub",
     "n_bounded", "n_excreted", "alpha", "trivial_ub"]]
plot.to_parquet(DER / "plotdata_pre.parquet", index=False)

# HDR panels use a separate, larger draw at the four r values
hdr_rs = [5, 10, 20, 40]
rng2 = derive(SEED, "subsample", fig_id="pre_hdr")
parts = []
for rr in hdr_rs:
    g = df[df["r"] == rr]
    n = min(300, len(g))
    parts.append(g.iloc[rng2.choice(len(g), size=n, replace=False)])
pd.concat(parts, ignore_index=True)[
    ["r", "mu_S", "mu_H", "delta_mu_H"]].to_parquet(
        DER / "plotdata_pre_hdr.parquet", index=False)
print("wrote plot data", flush=True)

# --------------------------------------------------------------------------- #
# per-symbiont table (needed by R4 fig 5A / S2A and by R3)
# --------------------------------------------------------------------------- #
per_sym = df.groupby(["r", "symbiont_id"]).agg(
    mu_H_pre_max=("mu_H", "max"), mu_H_pre_median=("mu_H", "median"),
    mu_H_pre_mean=("mu_H", "mean"), mu_S_pre_mean=("mu_S", "mean"),
    ub_pre_mean=("ub", "mean"), s_mean=("s", "mean"), s_median=("s", "median"),
    alpha_mean=("alpha", "mean"), b_mean=("n_bounded", "mean"),
    n_excreted_mean=("n_excreted", "mean"),
    trivial_pre=("trivial_ub", "mean"), n_cells=("mu_S", "size"),
    n_infeasible=("flag_infeasible", "sum"),
).reset_index()
per_sym["parasitic_pre"] = per_sym["mu_H_pre_max"] < mu_H_0
per_sym.to_csv(TAB / "pre_per_symbiont.csv", index=False)
print("wrote", TAB / "pre_per_symbiont.csv", flush=True)

# --------------------------------------------------------------------------- #
# R3 — SD_b / SD_u from the saved f_S* vectors
# --------------------------------------------------------------------------- #
if not a.skip_r3:
    vsrc = Path(a.vectors) if a.vectors else src
    vfiles = sorted(vsrc.glob("vectors_*.parquet"))
    print(f"R3: reading {len(vfiles)} vector blocks", flush=True)
    recs = []
    for f in vfiles:
        v = pd.read_parquet(f, columns=["r", "symbiont_id", "f_S_star",
                                        "bounded_mask"])
        for (rr, sid), g in v.groupby(["r", "symbiont_id"]):
            F = np.stack(g["f_S_star"].to_numpy()).astype(float)   # n_c x (m-1)
            Bm = np.stack(g["bounded_mask"].to_numpy()).astype(bool)
            if F.shape[0] < 2:
                continue
            # primary: majority vote over c^H
            maj = Bm.mean(axis=0) > 0.5
            var_all = F.var(axis=0, ddof=1)
            SD_b = float(np.sqrt(var_all[maj].sum())) if maj.any() else 0.0
            SD_u = float(np.sqrt(var_all[~maj].sum())) if (~maj).any() else 0.0
            # secondary: per-component variance over the cells where that
            # component was bounded / unbounded, weighted by how often it was
            seq_b, seq_u = 0.0, 0.0
            for j in range(F.shape[1]):
                mb = Bm[:, j]
                if mb.sum() > 1:
                    seq_b += (mb.mean()) * float(np.var(F[mb, j], ddof=1))
                if (~mb).sum() > 1:
                    seq_u += ((~mb).mean()) * float(np.var(F[~mb, j], ddof=1))
            # tertiary: pooled scalar SD and mean of per-component SD
            pooled_b = float(np.std(F[Bm], ddof=1)) if Bm.sum() > 1 else np.nan
            pooled_u = float(np.std(F[~Bm], ddof=1)) if (~Bm).sum() > 1 else np.nan
            recs.append({
                "r": rr, "symbiont_id": sid, "n_c": F.shape[0],
                "SD_b": SD_b, "SD_u": SD_u,
                "SD_b_sequential": float(np.sqrt(seq_b)),
                "SD_u_sequential": float(np.sqrt(seq_u)),
                "SD_b_pooled_scalar": pooled_b, "SD_u_pooled_scalar": pooled_u,
                "SD_b_mean_component": float(np.mean(np.sqrt(var_all[maj])))
                if maj.any() else 0.0,
                "SD_u_mean_component": float(np.mean(np.sqrt(var_all[~maj])))
                if (~maj).any() else 0.0,
                "b_majority": int(maj.sum()),
            })
    sd = pd.DataFrame(recs)
    sd = sd.merge(per_sym[["r", "symbiont_id", "alpha_mean", "s_mean", "b_mean"]],
                  on=["r", "symbiont_id"], how="left")
    sd.to_csv(TAB / "R3_per_symbiont.csv", index=False)

    def _ols(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                    "resid_sd": np.nan, "spearman": np.nan, "n": int(ok.sum())}
        X = np.column_stack([np.ones(ok.sum()), x[ok]])
        beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        pred = X @ beta
        ss_res = float(((y[ok] - pred) ** 2).sum())
        ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
        rho, _, _ = spearman(x[ok], y[ok])
        return {"slope": float(beta[1]), "intercept": float(beta[0]),
                "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                "resid_sd": float(np.sqrt(ss_res / max(ok.sum() - 2, 1))),
                "spearman": rho, "n": int(ok.sum())}

    rows = []
    for subset, sel in (("5<=r<=25", (sd["r"] >= 5) & (sd["r"] <= 25)),
                        ("all_r", sd["r"] > 0)):
        d = sd[sel]
        for xcol, ycol, which in (
                ("alpha_mean", "SD_b", "primary_rss"),
                ("s_mean", "SD_u", "primary_rss"),
                ("alpha_mean", "SD_b_sequential", "secondary_sequential"),
                ("s_mean", "SD_u_sequential", "secondary_sequential"),
                ("alpha_mean", "SD_b_pooled_scalar", "pooled_scalar"),
                ("s_mean", "SD_u_pooled_scalar", "pooled_scalar"),
                ("alpha_mean", "SD_b_mean_component", "mean_component"),
                ("s_mean", "SD_u_mean_component", "mean_component")):
            rec = {"subset": subset, "x": xcol, "y": ycol, "definition": which}
            rec.update(_ols(d[xcol], d[ycol]))
            rows.append(rec)
    pd.DataFrame(rows).to_csv(TAB / "R3_stats.csv", index=False)
    print("wrote", TAB / "R3_stats.csv", flush=True)

meta = {"n_cells": int(len(df)), "mu_H_0": mu_H_0, "m": m,
        "r_values": int(df["r"].nunique()),
        "N_S": int(df.groupby("r")["symbiont_id"].nunique().max()),
        "N_c": int(df.groupby(["r", "symbiont_id"]).size().max()),
        "bootstrap_B": B, "subsample_per_r": NSUB,
        "source": str(src)}
(Path(a.tables) / f"{a.tag}_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
