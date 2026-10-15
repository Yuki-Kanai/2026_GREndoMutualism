"""A2-D1..D4, D7 — decomposition diagnostics.

07_run2_workorder.md Phase B 2.1.  D1/D2/D3 are aggregations of the existing
pre-evolution cells; the rank test and the rcond sweep need R^S and the masks,
so those run on a subsample of cells whose vectors were saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.analyze import sensitivity_matrix
from grhd.cfgio import load_config
from grhd.pipeline import make_R_S
from grhd.rng import derive
from grhd.stats import spearman

ap = argparse.ArgumentParser()
ap.add_argument("--pre", default="data/raw/pre_paper")
ap.add_argument("--per-symbiont", default="tables/pre_per_symbiont.csv")
ap.add_argument("--tables", default="tables")
ap.add_argument("--figures", default="figures")
ap.add_argument("--rank-per-r", type=int, default=1500)
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
SEED = int(cfg["master_seed"])
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)
m = int(cfg["model"]["m"])

cols = ["r", "symbiont_id", "c_id", "s", "ub", "n_bounded", "n_unbounded",
        "r_plus", "n_degenerate", "cond_b", "alpha", "trivial_ub", "n_excreted"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in
                sorted(Path(a.pre).glob("cells_*.parquet"))], ignore_index=True)
print(f"{len(df):,} pre cells", flush=True)
df["sqrt_ub"] = np.sqrt(df["ub"].to_numpy(float))
pd.set_option("display.width", 230)


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                "resid_sd": np.nan, "n": int(x.size)}
    X = np.column_stack([np.ones(x.size), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"slope": float(beta[1]), "intercept": float(beta[0]),
            "r2": 1 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan,
            "resid_sd": float(np.std(res, ddof=2)) if x.size > 2 else np.nan,
            "n": int(x.size), "_resid": res, "_ok": ok}


# --------------------------------------------------------------------------- #
# D1 — is s a function of sqrt(ub)?
# --------------------------------------------------------------------------- #
rows = []
per_sym = pd.read_csv(a.per_symbiont)
per_sym["sqrt_ub_mean"] = np.sqrt(per_sym["ub_pre_mean"].to_numpy(float))

for level, d, xcol, ycol in (("cell", df, "sqrt_ub", "s"),
                             ("symbiont_mean", per_sym, "sqrt_ub_mean", "s_mean")):
    for subset, sel in (("all", np.ones(len(d), bool)),
                        ("non_trivial", (d["trivial_ub"] == False).to_numpy()
                         if "trivial_ub" in d else (d["ub_pre_mean"] >= 1).to_numpy())):
        dd = d[sel]
        f = ols(dd[xcol], dd[ycol])
        rho, _, n = spearman(dd[xcol], dd[ycol])
        rec = {"level": level, "subset": subset, "x": xcol, "y": ycol,
               "slope": f["slope"], "intercept": f["intercept"], "r2": f["r2"],
               "resid_sd": f["resid_sd"], "spearman": rho, "n": f["n"]}
        # does r or b explain the residual?
        if np.isfinite(f["r2"]):
            res = f["_resid"]
            sub = dd[f["_ok"]]
            bcol = "n_bounded" if "n_bounded" in sub else "b_mean"
            rec["resid_vs_r_r2"] = ols(sub["r"], res)["r2"]
            rec["resid_vs_b_r2"] = ols(sub[bcol], res)["r2"]
            rec["resid_vs_r_spearman"] = spearman(sub["r"], res)[0]
            rec["resid_vs_b_spearman"] = spearman(sub[bcol], res)[0]
        rows.append(rec)
d1 = pd.DataFrame(rows)
d1.to_csv(TAB / "A2_D1.csv", index=False)
print("\n=== D1: s vs sqrt(ub) ===")
print(d1[["level", "subset", "n", "slope", "intercept", "r2", "spearman",
          "resid_vs_r_r2", "resid_vs_b_r2", "resid_vs_r_spearman",
          "resid_vs_b_spearman"]].round(4).to_string(index=False))

# --------------------------------------------------------------------------- #
# D2 / D3 — activity and the b/u split
# --------------------------------------------------------------------------- #
d23 = df.groupby("r").agg(
    n=("s", "size"),
    r_plus_mean=("r_plus", "mean"), r_plus_over_r=("r_plus", lambda s: np.nan),
    b_mean=("n_bounded", "mean"), b_q25=("n_bounded", lambda s: s.quantile(.25)),
    b_q75=("n_bounded", lambda s: s.quantile(.75)),
    u_mean=("n_unbounded", "mean"), ub_mean=("ub", "mean"),
    ub_q25=("ub", lambda s: s.quantile(.25)), ub_q75=("ub", lambda s: s.quantile(.75)),
    n_degenerate_mean=("n_degenerate", "mean"),
    bplusu=("n_bounded", "size")).reset_index()
d23["r_plus_over_r"] = d23["r_plus_mean"] / d23["r"]
bad_partition = int(((df["n_bounded"] + df["n_unbounded"]) != m - 1).sum())
d23["b_plus_u_violations"] = bad_partition
# r_plus vs b
rho_rb, _, n_rb = spearman(df["r_plus"], df["n_bounded"])
d23.to_csv(TAB / "A2_D3.csv", index=False)
print(f"\n=== D3: b + u = m-1 violations: {bad_partition} / {len(df)} ===")
print(f"    spearman(r_plus, b) pooled = {rho_rb:.4f}  (n = {n_rb})")

# --------------------------------------------------------------------------- #
# D2 rank test + D7 rcond sweep, on a subsample with saved vectors
# --------------------------------------------------------------------------- #
RC = [1e-8, 1e-10, 1e-12]
rng = derive(SEED, "subsample", fig_id="A2_D2_rank")
rows = []
vfiles = sorted(Path(a.pre).glob("vectors_*.parquet"))
by_r: dict[int, list] = {}
for f in vfiles:
    r = int(f.stem.split("_")[1][1:])
    by_r.setdefault(r, []).append(f)

for r, fs in sorted(by_r.items()):
    take = max(1, a.rank_per_r // len(fs))
    parts = []
    for f in fs:
        v = pd.read_parquet(f, columns=["r", "symbiont_id", "c_id", "f_S_star",
                                        "w_S_star", "bounded_mask"])
        idx = rng.choice(len(v), size=min(take, len(v)), replace=False)
        parts.append(v.iloc[np.sort(idx)])
    v = pd.concat(parts, ignore_index=True)
    Rcache: dict[int, np.ndarray] = {}
    for _, rec in v.iterrows():
        sid = int(rec["symbiont_id"])
        if sid not in Rcache:
            Rcache[sid] = make_R_S(cfg, r, sid)
        R_S = Rcache[sid]
        bo = np.asarray(rec["bounded_mask"], dtype=bool)
        w = np.asarray(rec["w_S_star"], dtype=float)
        act = w > float(cfg["tolerances"]["tol_w"])
        b, u, rp = int(bo.sum()), int((~bo).sum()), int(act.sum())
        out = {"r": r, "symbiont_id": sid, "c_id": int(rec["c_id"]),
               "b": b, "u": u, "r_plus": rp, "trivial": bool(b == 0 or u == 0)}
        if b and rp:
            R_bp = R_S[:m - 1][bo][:, act]
            sv = np.linalg.svd(R_bp, compute_uv=False)
            rank = int((sv > sv[0] * 1e-10).sum()) if sv.size else 0
            out["rank_b_plus"] = rank
            out["rank_deficient"] = bool(rank < b)
            out["cond_b"] = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
            for rc in RC:
                out[f"s_rcond{rc:g}"] = (
                    float(np.linalg.norm(
                        sensitivity_matrix(R_S, bo, ~bo, act, rcond=rc), ord="fro"))
                    if u else 0.0)
        else:
            out["rank_b_plus"] = 0
            out["rank_deficient"] = False
            out["cond_b"] = np.nan
            for rc in RC:
                out[f"s_rcond{rc:g}"] = 0.0
        rows.append(out)
    print(f"  rank/rcond done r={r} (n={len(v)})", flush=True)

sub = pd.DataFrame(rows)
sub.to_parquet("data/derived/A2_D2D7_subsample.parquet", index=False)

d2 = sub.groupby("r").agg(
    n=("b", "size"),
    frac_rank_deficient=("rank_deficient", "mean"),
    frac_rplus_lt_b=("b", lambda s: np.nan),
    mean_rank=("rank_b_plus", "mean"), mean_b=("b", "mean"),
    mean_r_plus=("r_plus", "mean"),
    frac_trivial=("trivial", "mean")).reset_index()
d2["frac_rplus_lt_b"] = sub.assign(x=sub["r_plus"] < sub["b"]).groupby("r")["x"].mean().to_numpy()
d2.to_csv(TAB / "A2_D2.csv", index=False)
print("\n=== D2: rank(R^S_{b+}) < b, and r_+ < b (its necessary condition) ===")
print(d2.round(4).to_string(index=False))

nt = sub[~sub["trivial"]]
d7 = nt.groupby("r").agg(
    n=("cond_b", "size"),
    cond_b_median=("cond_b", "median"),
    cond_b_q90=("cond_b", lambda s: float(np.nanquantile(s, .90))),
    cond_b_max=("cond_b", "max"),
    **{f"s_rcond{rc:g}_median": (f"s_rcond{rc:g}", "median") for rc in RC},
    **{f"s_rcond{rc:g}_mean": (f"s_rcond{rc:g}", "mean") for rc in RC},
).reset_index()
d7.to_csv(TAB / "A2_D7.csv", index=False)
print("\n=== D7: cond_b and the rcond sweep (non-trivial cells) ===")
print(d7.round(4).to_string(index=False))

# who makes the tail of s?
nt2 = nt.dropna(subset=["cond_b"])
q = nt2["s_rcond1e-10"].quantile(0.99)
tail = nt2[nt2["s_rcond1e-10"] >= q]
print(f"\ntop 1% of s (>= {q:.3g}, n = {len(tail)}):")
print(f"  median cond_b in the tail : {tail['cond_b'].median():.3e}")
print(f"  median cond_b elsewhere   : {nt2[nt2['s_rcond1e-10'] < q]['cond_b'].median():.3e}")
print(f"  spearman(log cond_b, s)   : "
      f"{spearman(np.log10(np.clip(nt2['cond_b'], 1, None)), nt2['s_rcond1e-10'])[0]:.4f}")
print(f"  mean b in the tail        : {tail['b'].mean():.2f} vs "
      f"{nt2[nt2['s_rcond1e-10'] < q]['b'].mean():.2f}")

json.dump({"n_pre_cells": int(len(df)), "rank_subsample_per_r": a.rank_per_r,
           "rcond_values": RC, "b_plus_u_violations": bad_partition,
           "spearman_rplus_b": rho_rb},
          open(TAB / "A2_meta.json", "w"), indent=2)
print("\nwrote A2_D1.csv A2_D2.csv A2_D3.csv A2_D7.csv")
