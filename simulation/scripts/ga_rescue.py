#!/usr/bin/env python
"""GA rescue of det-limit-truncated MILP cells (NEXT section 2.4, option (A) variant).

Purpose
-------
`post_paper` spends 87 % of its MILP time on the 6 % of cells that hit the
deterministic time limit, and those truncated values are exactly the ones the GA
beats (3.1x at m=10, r=30, in 1/33 of the time -- NEXT section 2.3).  This script
re-solves those cells with the GA, seeded with the MILP incumbent so the result
is >= the value it replaces by construction, and writes one row per cell with
both values plus `mu_H_final = max(MILP, GA)`.

It never overwrites the raw parquet.  Analysis code joins this table on
(host_id, r, symbiont_id) and takes `mu_H_final`, so the definition of the final
value is the same for every cell: max over {MILP incumbent, seeded GA}.

Question it answers
-------------------
"How much of the post-evolution mu^H curve is an artefact of the det limit?"

Usage
-----
  PYTHONPATH=src ./.venv/bin/python scripts/ga_rescue.py \
      --dir data/raw/post_paper --out tables/GA_rescue_post_paper.csv \
      --workers 40 --restarts 2
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from grhd.cfgio import ROOT, load_config          # noqa: E402
from grhd.lp import HostSolver, SymbiontSolver    # noqa: E402
from grhd.pipeline import get_host, make_R_S_meta  # noqa: E402
from grhd.rng import derive_seed                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_top_layer import Evaluator, ga_full       # noqa: E402

CELL_COLS = ["r", "symbiont_id", "host_id", "m", "k", "scheme", "n_rxn",
             "mu_H", "mu_H_0", "flag_detlimit", "solver_status", "milp_path"]


def load_cells(d: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(d, "cells_*.parquet")))
    if not files:
        sys.exit(f"no cells_*.parquet under {d}")
    out = []
    for f in files:
        try:
            out.append(pd.read_parquet(f, columns=CELL_COLS))
        except Exception as e:                                  # noqa: BLE001
            print(f"SKIP unreadable {os.path.basename(f)}: {e}", file=sys.stderr)
    return pd.concat(out, ignore_index=True)


def load_vectors(d: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(d, "vectors_*.parquet")))
    out = []
    for f in files:
        try:
            out.append(pd.read_parquet(f, columns=["r", "symbiont_id", "c_H"]))
        except Exception as e:                                  # noqa: BLE001
            print(f"SKIP unreadable {os.path.basename(f)}: {e}", file=sys.stderr)
    if not out:
        return pd.DataFrame(columns=["r", "symbiont_id", "c_H"])
    v = pd.concat(out, ignore_index=True)
    return v.drop_duplicates(subset=["r", "symbiont_id"], keep="last")


def _one(job):
    (r, sid, host_id, m, k, scheme, n_rxn, mu_H_milp, seed_c,
     pop, gens, restarts, cfg_paths, master_seed) = job
    cfg = load_config(*cfg_paths)
    c_max = float(cfg["flags"]["c_H_box"])
    hd, cE = get_host(cfg, host_id, m)
    R_S, _meta = make_R_S_meta(cfg, r, sid, scheme, m, k, n_rxn)
    ev = Evaluator(R_S, hd.R_H, cE, cfg)

    t0 = time.perf_counter()
    best, best_c = -np.inf, None
    for j in range(restarts):
        seed = derive_seed(master_seed, "ga_rescue", r=r, symbiont_id=sid,
                           host_id=host_id, m=m, k=k, restart=j)
        rng = np.random.default_rng(seed)
        b, c = ga_full(ev, m, rng, pop_size=pop, generations=gens, c_max=c_max,
                       seed_c=seed_c)
        if b > best:
            best, best_c = float(b), c
    dt = time.perf_counter() - t0

    # The seed itself, re-evaluated here: this is the MILP incumbent scored by the
    # GA's own evaluator, so mu_H_seed vs mu_H_milp also checks that the rebuilt
    # R^S and host match the ones the MILP saw (they should agree to LP accuracy).
    mu_seed = ev.mu_H(seed_c)[0] if seed_c is not None else np.nan
    ev.close()
    final = max(best, float(mu_H_milp) if np.isfinite(mu_H_milp) else -np.inf)
    return {"host_id": host_id, "r": r, "symbiont_id": sid, "m": m, "k": k,
            "mu_H_0": hd.mu_H_0, "mu_H_milp": mu_H_milp, "mu_H_seed_recheck": mu_seed,
            "mu_H_ga": best, "mu_H_final": final,
            "gain": final - mu_H_milp,
            "ratio": (best / mu_H_milp) if mu_H_milp > 0 else np.nan,
            "seconds_ga": dt, "n_eval": ev.n_eval,
            "restarts": restarts, "generations": gens, "pop": pop}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="data/raw/<run> to rescue")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--r-min", type=int, default=None)
    ap.add_argument("--r-max", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N cells only")
    ap.add_argument("--all-cells", action="store_true",
                    help="rescue every cell, not only the truncated ones")
    ap.add_argument("--config", nargs="+",
                    default=[str(ROOT / "config" / "base.yaml")])
    args = ap.parse_args()

    cfg = load_config(*args.config)
    cells = load_cells(args.dir)
    n_all = len(cells)
    sel = cells if args.all_cells else cells[cells["flag_detlimit"].fillna(False)]
    if args.r_min is not None:
        sel = sel[sel["r"] >= args.r_min]
    if args.r_max is not None:
        sel = sel[sel["r"] <= args.r_max]
    sel = sel.sort_values(["host_id", "r", "symbiont_id"])
    if args.limit:
        sel = sel.head(args.limit)

    vec = load_vectors(args.dir)
    sel = sel.merge(vec, on=["r", "symbiont_id"], how="left")
    n_noseed = int(sel["c_H"].isna().sum())
    print(f"{n_all} cells, {len(sel)} to rescue, {n_noseed} without a stored c_H seed")
    if len(sel) == 0:
        return

    jobs = []
    for row in sel.itertuples(index=False):
        seed_c = None if not isinstance(row.c_H, (list, np.ndarray)) else np.asarray(row.c_H, dtype=float)
        jobs.append((int(row.r), int(row.symbiont_id), int(row.host_id), int(row.m),
                     int(row.k), str(row.scheme),
                     None if pd.isna(row.n_rxn) else int(row.n_rxn),
                     float(row.mu_H), seed_c, args.pop, args.generations,
                     args.restarts, args.config, cfg["master_seed"]))

    t0 = time.perf_counter()
    if args.workers <= 1:
        out = [_one(j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(_one)(j) for j in jobs)
    res = pd.DataFrame(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False)

    # 20 lines at most on the terminal (START_HERE section 1)
    improved = res["mu_H_ga"] > res["mu_H_milp"] * (1 + 1e-6)
    drift = (res["mu_H_seed_recheck"] - res["mu_H_milp"]).abs()
    print(f"wrote {args.out}  ({len(res)} rows, {time.perf_counter() - t0:.0f} s wall, "
          f"{res['seconds_ga'].sum() / 3600:.2f} core-hours)")
    print(f"GA > MILP: {int(improved.sum())}/{len(res)} cells; "
          f"median ratio {res.loc[improved, 'ratio'].median():.3f}, "
          f"max ratio {res['ratio'].max():.3f}")
    print(f"seed re-check |mu_seed - mu_milp|: median {drift.median():.2e}, "
          f"max {drift.max():.2e}")
    by_r = (res.groupby("r")
               .agg(n=("r", "size"), mean_gain=("gain", "mean"),
                    mean_ratio=("ratio", "mean"))
               .sort_values("mean_gain", ascending=False).head(10))
    print(by_r.round(4).to_string())


if __name__ == "__main__":
    main()
