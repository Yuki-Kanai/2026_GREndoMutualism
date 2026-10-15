#!/usr/bin/env python
"""Does shrinking the artificial boxes remove the det-limit truncation?

Purpose
-------
The boxes W_max = Lambda_max = 1e4 exist only so SOS1 branching has finite
bounds (01_model_spec.md section 5.1).  Diagnostics (DIAG_detlimit*) showed the
attained values are 2-3 orders below those boxes for w^S, lam1, lam2, g -- while
the MIP gap on truncated cells has median 3.92, i.e. the relaxation is so loose
that the proven bound never leaves the trivial mu^H <= 1.  A box that is 1000x
larger than anything attained is exactly what makes an SOS1 relaxation useless.

Question
--------
For cells that truncated at the current boxes: does W_max = Lambda_max = 1e3 /
1e2 / 1e1 (a) remove the truncation, (b) change mu^H, (c) make any box bind?

Method
------
Each variant is solved with EvolutionSolver *directly*, bypassing the widening
retry chain in solve_host_evolution, so that a binding box is reported instead of
silently repaired.  A smaller box can only shrink the feasible set, so
mu^H(tight) <= mu^H(loose) unless the loose run was itself truncated -- in which
case mu^H(tight) > mu^H(loose) is a *better incumbent*, which is also the answer
we are looking for.  det_time_limit is held fixed across variants.

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


def pick(d: str, n_trunc: int, n_opt: int, seed: int = 0) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(d, "cells_*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=CCOLS) for f in files],
                   ignore_index=True)
    df["trunc"] = df["flag_detlimit"].fillna(False).astype(bool)
    rng = np.random.default_rng(seed)
    t = df[df["trunc"]]
    o = df[~df["trunc"]]
    t = t.iloc[rng.choice(len(t), size=min(n_trunc, len(t)), replace=False)]
    o = o.iloc[rng.choice(len(o), size=min(n_opt, len(o)), replace=False)]
    return pd.concat([t, o], ignore_index=True)


def _one(job):
    (row, box, cfg_paths) = job
    cfg = load_config(*cfg_paths)
    cfg = copy.deepcopy(cfg)
    if box is not None:
        cfg["flags"]["w_S_box"] = float(box)
        cfg["flags"]["lambda_box"] = float(box)
    m, k = int(row["m"]), int(row["k"])
    hd, cE = get_host(cfg, int(row["host_id"]), m)
    R_S, _ = make_R_S_meta(cfg, int(row["r"]), int(row["symbiont_id"]),
                           str(row["scheme"]), m, k,
                           None if pd.isna(row["n_rxn"]) else int(row["n_rxn"]))
    t0 = time.perf_counter()
    sol = EvolutionSolver(R_S, hd.R_H, cE, cfg).solve()
    dt = time.perf_counter() - t0
    rep = sol.box_report or {}
    return {"r": int(row["r"]), "symbiont_id": int(row["symbiont_id"]),
            "was_trunc": bool(row["trunc"]), "box": box,
            "mu_H": sol.mu_H, "mu_H_ref": float(row["mu_H"]),
            "status": sol.status, "detlimit": "detlimit" in str(sol.status),
            "mip_gap": sol.mip_gap, "bound": sol.mip_best_bound,
            "seconds": dt, "seconds_ref": float(row["solve_time_milp_s"]),
            "box_active": bool(sol.flag_box_active),
            "worst_box": max(rep.values()) if rep else np.nan,
            "worst_box_which": (max(rep, key=rep.get) if rep else ""),
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/post_paper")
    ap.add_argument("--out", default="tables/BENCH_boxes.csv")
    ap.add_argument("--boxes", default="1e4,1e3,1e2,1e1")
    ap.add_argument("--n-trunc", type=int, default=30)
    ap.add_argument("--n-opt", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", nargs="+",
                    default=[str(ROOT / "config" / "base.yaml")])
    args = ap.parse_args()

    cells = pick(args.dir, args.n_trunc, args.n_opt)
    boxes = [float(x) for x in args.boxes.split(",")]
    jobs = [(row, b, args.config) for b in boxes
            for _, row in cells.iterrows()]
    print(f"{len(cells)} cells x {len(boxes)} boxes = {len(jobs)} solves")

    if args.workers <= 1:
        out = [_one(j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(_one)(j) for j in jobs)
    res = pd.DataFrame(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False)

    for grp, lab in ((res[res["was_trunc"]], "was truncated"),
                     (res[~res["was_trunc"]], "was optimal  ")):
        print(f"--- {lab} ({grp['r'].count() // len(boxes)} cells)")
        g = grp.groupby("box").agg(
            detlimit=("detlimit", "mean"),
            med_s=("seconds", "median"),
            tot_s=("seconds", "sum"),
            box_hit=("box_active", "mean"),
            d_muH_max=("mu_H", lambda s: np.nan),
        )
        g = g.drop(columns=["d_muH_max"])
        base = grp[grp["box"] == max(boxes)].set_index(["r", "symbiont_id"])["mu_H"]
        d = []
        for b in boxes:
            s = grp[grp["box"] == b].set_index(["r", "symbiont_id"])["mu_H"]
            diff = (s - base).reindex(base.index)
            d.append((b, float(diff.min()), float(diff.max())))
        g["dmuH_min"] = [x[1] for x in d]
        g["dmuH_max"] = [x[2] for x in d]
        print(g.round(4).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
