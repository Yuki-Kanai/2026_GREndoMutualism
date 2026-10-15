#!/usr/bin/env python
"""Overlay the r x mu^H_evo curve as solved by MILP, by GA, and by max(both).

Purpose
-------
ga_curve.py recomputed the top layer with two GAs on exactly the symbionts the
MILP used (host 0, k=4, ids 0..n-1).  The MILP is proven optimal wherever it does
not truncate and demonstrably poor where it does (BENCH_ga_modes: on truncated
cells the MILP mean is 0.302 against the GA's 0.462, and in the worst cell the
MILP returns 6.4 % of the best value found).  The GA in turn falls ~1 % short
where the MILP is exact.

So neither method alone is the right curve.  Per cell:

    mu_H_best = max(MILP, ga_full, ga_ray)

takes the MILP's proven optimum where it exists and the GA's rescue where the
MILP truncated.  This needs no extra compute -- all three values already exist.

Question
--------
Does the peak of the mean post-evolution mu^H survive, and do its position r*
and height change, when the truncated cells are repaired?

Output: tables/GA_vs_MILP_curve.csv (tidy, one row per r per method) plus at
most 20 printed lines.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def boot_ci(x: np.ndarray, B: int, seed: int, q=(2.5, 97.5)):
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.percentile(means, q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ga-dir", default="tables/ga_curve")
    ap.add_argument("--milp-dir", default="data/raw/post_paper")
    ap.add_argument("--out", default="tables/GA_vs_MILP_curve.csv")
    ap.add_argument("--out-cells", default="tables/GA_vs_MILP_cells.csv")
    ap.add_argument("--n-sym", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    ga = pd.concat([pd.read_csv(f) for f in
                    sorted(glob.glob(os.path.join(args.ga_dir, "*.csv")))],
                   ignore_index=True)
    # blocks may be split by symbiont-id range across files; one row per cell
    n_raw = len(ga)
    ga = ga.drop_duplicates(["method", "r", "symbiont_id"], keep="last")
    if n_raw != len(ga):
        print(f"de-duplicated {n_raw - len(ga)} overlapping rows")
    rs = sorted(ga["r"].unique())
    wide = ga.pivot_table(index=["r", "symbiont_id"], columns="method",
                          values="mu_H")

    mp = pd.concat([pd.read_parquet(f, columns=["r", "symbiont_id", "mu_H",
                                                "mu_H_0", "flag_detlimit"])
                    for f in sorted(glob.glob(os.path.join(args.milp_dir,
                                                           "cells_*.parquet")))],
                   ignore_index=True).drop_duplicates(["r", "symbiont_id"])
    mp = mp[(mp["symbiont_id"] < args.n_sym) & (mp["r"].isin(rs))]
    mp = mp.set_index(["r", "symbiont_id"]).rename(columns={"mu_H": "milp"})

    j = wide.join(mp[["milp", "mu_H_0", "flag_detlimit"]], how="inner").reset_index()
    methods = [c for c in ("milp", "ga_full", "ga_ray") if c in j]
    j["best"] = j[methods].max(axis=1)
    j.to_csv(args.out_cells, index=False)

    rows = []
    for r, g in j.groupby("r"):
        for mth in methods + ["best"]:
            x = g[mth].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            lo, hi = boot_ci(x, args.bootstrap, args.seed + int(r))
            rows.append({"r": int(r), "method": mth, "n": len(x),
                         "mean_mu_H": x.mean(), "ci_lo": lo, "ci_hi": hi,
                         "trunc_frac": float(g["flag_detlimit"].mean()),
                         "mu_H_0": float(g["mu_H_0"].iloc[0])})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)

    piv = out.pivot(index="r", columns="method", values="mean_mu_H")
    piv["trunc"] = out.groupby("r")["trunc_frac"].first()
    print(piv.round(4).to_string())
    print("--- peak (argmax of the mean over this r grid)")
    for mth in methods + ["best"]:
        s = out[out["method"] == mth].set_index("r")["mean_mu_H"]
        rstar = int(s.idxmax())
        ci = out[(out["method"] == mth) & (out["r"] == rstar)].iloc[0]
        print(f"  {mth:8s} r* = {rstar:<4d} peak {s.max():.4f} "
              f"[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
    print(f"wrote {args.out} and {args.out_cells}")


if __name__ == "__main__":
    main()
