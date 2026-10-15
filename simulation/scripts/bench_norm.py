#!/usr/bin/env python
"""Does scaling the columns of R^S fix the det-limit truncation?

Purpose
-------
R^S columns are q_j = sum_{a in K_j} kappa_ja v_a with kappa ~ U(0,1) and k = 4,
so their L1 norms range over roughly 0.5 - 8: a 16x scale spread across the
columns of the matrix that appears in every KKT block.  The author's own Level 3
code normalises those columns (LP.jl: Rp = parasiteR ./ sum(abs.(parasiteR),
dims=1)) before solving, and his truncation rate at m=10, k=4 is near zero.

Two variants, deliberately separated:

  exact  change of variables w'_j = omega_j w_j, objective kept at the spec's
         unweighted min sum_j w_j^2.  Mathematically the same optimum, so mu^H
         must not move -- any speedup is pure numerical conditioning.
  model  the author's tie-break: min sum_j (omega_j w_j)^2.  mu^H may move; if
         it does, that is a modelling decision for the author, not a bug.

Question
--------
On cells that truncated under production settings: does either variant end the
truncation, and does mu^H stay put under `exact` (which it must, if the
substitution is implemented correctly -- this doubles as a correctness check)?

Output: one CSV plus at most 20 printed lines.
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from grhd.cfgio import ROOT, load_config          # noqa: E402
from grhd.milp import EvolutionSolver             # noqa: E402
from grhd.pipeline import get_host, make_R_S_meta  # noqa: E402

CCOLS = ["r", "symbiont_id", "host_id", "m", "k", "scheme", "n_rxn",
         "mu_H", "mu_S", "flag_detlimit", "solve_time_milp_s"]

# name -> (col_normalize, mu_S_floor, presolve, c_H_box)
# c_H_box = None keeps the config value (10.0).  The author's own Level 3 model
# bounds the host supply at 1 (LP.jl: 1 >= cp >= 0); since Level 1 is homogeneous
# in c^H, a larger box lets the host keep pushing supply upward, and an optimum
# sitting on the box is exactly the case SOS1 branching cannot close.  This is a
# model parameter, not a free knob: mu^H is expected to move.
VARIANTS = {
    "base":              ("none",  0.0,  0, None),
    "norm_exact":        ("exact", 0.0,  0, None),
    "norm_model":        ("model", 0.0,  0, None),
    "norm_model+floor":  ("model", 1e-5, 0, None),
    "norm_model+presol": ("model", 0.0,  1, None),
    # combinations of the two levers that worked on their own.  norm_exact keeps
    # the spec's optimum, presolve and the mu^S floor do not.
    "exact+presol":       ("exact", 0.0,  1, None),
    "exact+floor":        ("exact", 1e-5, 0, None),
    "exact+presol+floor": ("exact", 1e-5, 1, None),
    # the c^H box
    "cmax1":              ("none",  0.0,  0, 1.0),
    "exact+cmax1":        ("exact", 0.0,  0, 1.0),
    "exact+presol+cmax1": ("exact", 0.0,  1, 1.0),
    "all+cmax1":          ("exact", 1e-5, 1, 1.0),
}


def pick(d: str, n_trunc: int, n_opt: int, seed: int = 0) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(d, "cells_*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=CCOLS) for f in files],
                   ignore_index=True).drop_duplicates(["r", "symbiont_id"])
    df["trunc"] = df["flag_detlimit"].fillna(False).astype(bool)
    rng = np.random.default_rng(seed)
    t, o = df[df["trunc"]], df[~df["trunc"]]
    t = t.iloc[rng.choice(len(t), size=min(n_trunc, len(t)), replace=False)]
    o = o.iloc[rng.choice(len(o), size=min(n_opt, len(o)), replace=False)]
    return pd.concat([t, o], ignore_index=True)


def _one(job):
    row, name, cfg_paths = job
    colnorm, floor, pre, cmax = VARIANTS[name]
    cfg = copy.deepcopy(load_config(*cfg_paths))
    cfg["flags"]["col_normalize"] = colnorm
    cfg["flags"]["mu_S_floor"] = floor
    if cmax is not None:
        cfg["flags"]["c_H_box"] = float(cmax)
    m, k = int(row["m"]), int(row["k"])
    hd, cE = get_host(cfg, int(row["host_id"]), m)
    R_S, _ = make_R_S_meta(cfg, int(row["r"]), int(row["symbiont_id"]),
                           str(row["scheme"]), m, k,
                           None if pd.isna(row["n_rxn"]) else int(row["n_rxn"]))
    w1 = np.abs(R_S).sum(axis=0)
    t0 = time.perf_counter()
    sol = EvolutionSolver(R_S, hd.R_H, cE, cfg, presolve=pre).solve()
    dt = time.perf_counter() - t0
    return {"variant": name, "r": int(row["r"]),
            "symbiont_id": int(row["symbiont_id"]),
            "was_trunc": bool(row["trunc"]), "mu_H": sol.mu_H,
            "mu_H_ref": float(row["mu_H"]), "mu_S": sol.mu_S,
            "status": sol.status, "detlimit": "detlimit" in str(sol.status),
            "mip_gap": sol.mip_gap, "seconds": dt,
            "box_active": bool(sol.flag_box_active),
            "col_l1_min": float(w1.min()), "col_l1_max": float(w1.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/post_paper")
    ap.add_argument("--out", default="tables/BENCH_norm.csv")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-trunc", type=int, default=30)
    ap.add_argument("--n-opt", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", nargs="+",
                    default=[str(ROOT / "config" / "base.yaml")])
    args = ap.parse_args()

    cells = pick(args.dir, args.n_trunc, args.n_opt)
    names = [v for v in args.variants.split(",") if v in VARIANTS]
    jobs = [(row, nm, args.config) for nm in names for _, row in cells.iterrows()]
    print(f"{len(cells)} cells x {len(names)} variants = {len(jobs)} solves")

    if args.workers <= 1:
        out = [_one(j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(_one)(j) for j in jobs)
    res = pd.DataFrame(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False)

    w = res[res["variant"] == "base"]
    print(f"column L1 norms: min {w['col_l1_min'].min():.3f} "
          f"max {w['col_l1_max'].max():.3f}")
    for trunc, lab in ((True, "was truncated"), (False, "was optimal")):
        g = res[res["was_trunc"] == trunc]
        if not len(g):
            continue
        base = g[g["variant"] == "base"].set_index(["r", "symbiont_id"])["mu_H"]
        rows = []
        for nm in names:
            s = g[g["variant"] == nm]
            si = s.set_index(["r", "symbiont_id"])["mu_H"]
            j = pd.concat([si.rename("x"), base.rename("b")], axis=1).dropna()
            d = (j["x"] - j["b"]).abs()
            rows.append({"variant": nm, "detlim": s["detlimit"].mean(),
                         "med_s": s["seconds"].median(),
                         "tot_s": s["seconds"].sum(),
                         "box": s["box_active"].mean(),
                         "dmuH_absmax": d.max() if len(d) else np.nan})
        print(f"--- {lab} ({g['variant'].value_counts().iloc[0]} cells)")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
