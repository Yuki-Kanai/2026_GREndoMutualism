#!/usr/bin/env python
"""MILP vs GA-full vs GA-ray vs seeded GA, on the same cells.

Purpose
-------
GA_validate compared full and ray against the MILP only where the MILP proves
optimality (r in {8,13,20}).  The truncated region -- the one that matters, since
87 % of the solve time and all of the uncertainty live there -- was covered by a
single cell (r=30, symbiont 0).  Meanwhile BENCH_boxes / BENCH_norm showed that a
truncated cell's mu^H moves by up to 0.76 depending on solver settings alone, so
"what the MILP returns there" is not a fixed quantity to compare against.

This runs every method on the *same* 50 cells (30 that truncated under production
settings, 20 that solved to proven optimality), so the comparison is like for
like.  `ga_full_seeded` starts from the MILP incumbent, so by elitism it can
never be worse than the MILP value it replaces.

Question
--------
1. On proven-optimal cells: how close does each GA mode get?  (quality check)
2. On truncated cells: which method returns the largest mu^H, and at what cost?
3. Does evolving only the direction (ray) beat evolving c^H itself, once the
   comparison is made where the MILP is weak rather than where it is strong?

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
from grhd.rng import derive_seed                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_top_layer import Evaluator, ga_full, ga_ray  # noqa: E402

CCOLS = ["r", "symbiont_id", "host_id", "m", "k", "scheme", "n_rxn",
         "mu_H", "mu_S", "flag_detlimit", "solve_time_milp_s"]

METHODS = ["milp", "ga_full", "ga_ray", "ga_full_seeded"]


def pick(d: str, n_trunc: int, n_opt: int, seed: int = 0) -> pd.DataFrame:
    """Identical sample to bench_solver.py / bench_norm.py (same seed, same rule)."""
    files = sorted(glob.glob(os.path.join(d, "cells_*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=CCOLS) for f in files],
                   ignore_index=True).drop_duplicates(["r", "symbiont_id"])
    df["trunc"] = df["flag_detlimit"].fillna(False).astype(bool)
    rng = np.random.default_rng(seed)
    t, o = df[df["trunc"]], df[~df["trunc"]]
    t = t.iloc[rng.choice(len(t), size=min(n_trunc, len(t)), replace=False)]
    o = o.iloc[rng.choice(len(o), size=min(n_opt, len(o)), replace=False)]
    return pd.concat([t, o], ignore_index=True)


def load_seeds(d: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(d, "vectors_*.parquet")))
    out = []
    for f in files:
        try:
            out.append(pd.read_parquet(f, columns=["r", "symbiont_id", "c_H"]))
        except Exception as e:                                   # noqa: BLE001
            print(f"SKIP {os.path.basename(f)}: {e}", file=sys.stderr)
    if not out:
        return pd.DataFrame(columns=["r", "symbiont_id", "c_H"])
    return pd.concat(out, ignore_index=True).drop_duplicates(
        ["r", "symbiont_id"], keep="last")


def _one(job):
    row, method, pop, gens, ray_gens, cfg_paths = job
    cfg = copy.deepcopy(load_config(*cfg_paths))
    m, k = int(row["m"]), int(row["k"])
    hd, cE = get_host(cfg, int(row["host_id"]), m)
    R_S, _ = make_R_S_meta(cfg, int(row["r"]), int(row["symbiont_id"]),
                           str(row["scheme"]), m, k,
                           None if pd.isna(row["n_rxn"]) else int(row["n_rxn"]))
    c_max = float(cfg["flags"]["c_H_box"])
    rec = {"method": method, "r": int(row["r"]),
           "symbiont_id": int(row["symbiont_id"]),
           "was_trunc": bool(row["trunc"]), "mu_H_ref": float(row["mu_H"]),
           "n_eval": np.nan, "status": ""}

    t0 = time.perf_counter()
    if method == "milp":
        sol = EvolutionSolver(R_S, hd.R_H, cE, cfg).solve()
        rec["mu_H"] = sol.mu_H
        rec["status"] = sol.status
    else:
        ev = Evaluator(R_S, hd.R_H, cE, cfg)
        seed = derive_seed(cfg["master_seed"], "ga_bench", method=method,
                           r=int(row["r"]), symbiont_id=int(row["symbiont_id"]),
                           host_id=int(row["host_id"]))
        rng = np.random.default_rng(seed)
        if method == "ga_ray":
            best, _ = ga_ray(ev, m, rng, pop_size=pop, generations=ray_gens,
                             c_max=c_max)
        elif method == "ga_full":
            best, _ = ga_full(ev, m, rng, pop_size=pop, generations=gens,
                              c_max=c_max)
        else:                                       # ga_full_seeded
            s = row.get("c_H")
            s = np.asarray(s, dtype=float) if isinstance(
                s, (list, np.ndarray)) else None
            best, _ = ga_full(ev, m, rng, pop_size=pop, generations=gens,
                              c_max=c_max, seed_c=s)
            rec["status"] = "seeded" if s is not None else "no_seed"
        rec["mu_H"] = float(best)
        rec["n_eval"] = ev.n_eval
        ev.close()
    rec["seconds"] = time.perf_counter() - t0
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/post_paper")
    ap.add_argument("--out", default="tables/BENCH_ga_modes.csv")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n-trunc", type=int, default=30)
    ap.add_argument("--n-opt", type=int, default=20)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--ray-generations", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", nargs="+",
                    default=[str(ROOT / "config" / "base.yaml")])
    args = ap.parse_args()

    cells = pick(args.dir, args.n_trunc, args.n_opt).merge(
        load_seeds(args.dir), on=["r", "symbiont_id"], how="left")
    methods = [m for m in args.methods.split(",") if m in METHODS]
    jobs = [(row, mth, args.pop, args.generations, args.ray_generations,
             args.config) for mth in methods for _, row in cells.iterrows()]
    print(f"{len(cells)} cells x {len(methods)} methods = {len(jobs)} runs")

    if args.workers <= 1:
        out = [_one(j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(_one)(j) for j in jobs)
    res = pd.DataFrame(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False)

    wide = res.pivot_table(index=["r", "symbiont_id", "was_trunc"],
                           columns="method", values="mu_H").reset_index()
    best = wide[methods].max(axis=1)
    for trunc, lab in ((False, "proven optimal"), (True, "truncated")):
        w = wide[wide["was_trunc"] == trunc]
        b = best[wide["was_trunc"] == trunc]
        if not len(w):
            continue
        print(f"--- {lab} ({len(w)} cells)")
        rows = []
        for mth in methods:
            s = res[(res["method"] == mth) & (res["was_trunc"] == trunc)]
            ratio = (w[mth] / b.replace(0, np.nan))
            rows.append({"method": mth, "med_s": s["seconds"].median(),
                         "tot_s": s["seconds"].sum(),
                         "mean_muH": w[mth].mean(),
                         "ratio_to_best_med": ratio.median(),
                         "ratio_to_best_min": ratio.min(),
                         "is_best": float((w[mth] >= b - 1e-6).mean())})
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
