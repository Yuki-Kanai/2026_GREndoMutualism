#!/usr/bin/env python
"""Which setting actually removes the det-limit truncation?

Purpose
-------
BENCH_boxes ruled out the artificial boxes: shrinking W_max / Lambda_max from
1e4 to 1e1 left the truncation rate at 0.80-0.93 and the median solve time at
~180 s, while moving mu^H by up to 0.75 (so it is not even a safe change).
The remaining candidates, cheapest first:

  presolve     the spec fixes MIP presolve OFF (01_model_spec.md section 8.2) for
               reproducibility.  On an SOS1 model that removes most of CPLEX's
               strength.
  mipgap       we ask for 1e-9 relative / 1e-12 absolute.  The observed gaps are
               ~4e2, so this should not be the binding stopping rule -- but it
               also stops CPLEX ever declaring victory early.
  mu_S floor   the author's own model constrains mu^S >= 1e-5, which deletes the
               whole degenerate face w^S = 0 / mu^S = 0.  Ours allows mu^S = 0
               (spec section 3) and 79 % of the solved cells sit there.
  numerical    emphasis.numerical = 1.

Question
--------
On cells that truncated under the production settings: which of these, alone or
combined, ends the truncation, and does mu^H move when it does?

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

# name -> (presolve, mipgap, absmipgap, mu_S_floor, numerical_emphasis)
VARIANTS = {
    "base":              (0, 1e-9, 1e-12, 0.0,  0),
    "presolve":          (1, 1e-9, 1e-12, 0.0,  0),
    "gap1e-4":           (0, 1e-4, 1e-12, 0.0,  0),
    "presolve+gap":      (1, 1e-4, 1e-12, 0.0,  0),
    "muSfloor":          (0, 1e-9, 1e-12, 1e-5, 0),
    "presolve+gap+floor": (1, 1e-4, 1e-12, 1e-5, 0),
    "numemph":           (0, 1e-9, 1e-12, 0.0,  1),
}


def pick(d: str, n_trunc: int, n_opt: int, seed: int = 0) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(d, "cells_*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=CCOLS) for f in files],
                   ignore_index=True)
    df = df.drop_duplicates(["r", "symbiont_id"])
    df["trunc"] = df["flag_detlimit"].fillna(False).astype(bool)
    rng = np.random.default_rng(seed)
    t, o = df[df["trunc"]], df[~df["trunc"]]
    t = t.iloc[rng.choice(len(t), size=min(n_trunc, len(t)), replace=False)]
    o = o.iloc[rng.choice(len(o), size=min(n_opt, len(o)), replace=False)]
    return pd.concat([t, o], ignore_index=True)


def _one(job):
    row, name, cfg_paths = job
    pre, gap, absgap, floor, nemph = VARIANTS[name]
    cfg = copy.deepcopy(load_config(*cfg_paths))
    cfg["solver"]["milp"]["mipgap"] = gap
    cfg["solver"]["milp"]["absmipgap"] = absgap
    cfg["flags"]["mu_S_floor"] = floor
    m, k = int(row["m"]), int(row["k"])
    hd, cE = get_host(cfg, int(row["host_id"]), m)
    R_S, _ = make_R_S_meta(cfg, int(row["r"]), int(row["symbiont_id"]),
                           str(row["scheme"]), m, k,
                           None if pd.isna(row["n_rxn"]) else int(row["n_rxn"]))
    t0 = time.perf_counter()
    sol = EvolutionSolver(R_S, hd.R_H, cE, cfg, numerical_emphasis=nemph,
                          presolve=pre).solve()
    dt = time.perf_counter() - t0
    return {"variant": name, "r": int(row["r"]),
            "symbiont_id": int(row["symbiont_id"]),
            "was_trunc": bool(row["trunc"]), "mu_H": sol.mu_H,
            "mu_H_ref": float(row["mu_H"]), "mu_S": sol.mu_S,
            "status": sol.status, "detlimit": "detlimit" in str(sol.status),
            "mip_gap": sol.mip_gap, "bound": sol.mip_best_bound,
            "seconds": dt, "seconds_ref": float(row["solve_time_milp_s"]),
            "box_active": bool(sol.flag_box_active)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/post_paper")
    ap.add_argument("--out", default="tables/BENCH_solver.csv")
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
            d = j["x"] - j["b"]
            rows.append({"variant": nm, "detlim": s["detlimit"].mean(),
                         "med_s": s["seconds"].median(),
                         "tot_s": s["seconds"].sum(),
                         "dmuH_min": d.min() if len(d) else np.nan,
                         "dmuH_max": d.max() if len(d) else np.nan})
        print(f"--- {lab} ({g['variant'].value_counts().iloc[0]} cells)")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
