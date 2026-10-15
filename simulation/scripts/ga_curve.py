#!/usr/bin/env python
"""The r x mu^H_evo curve, solved by GA instead of MILP.

Purpose
-------
The main result is a peak of the post-evolution mean mu^H at intermediate r.  It
was produced by the Level 3 MILP, which truncates on 5.95 % of cells -- all at
r >= 22, i.e. on the downhill side of the peak -- and BENCH_boxes / BENCH_norm
showed a truncated cell's mu^H moves by up to 0.76 depending on solver settings
alone.  So the downhill side of the curve rests on values that are not pinned
down.  GA_validate compared GA against the MILP only at r in {8,13,20}, where the
MILP proves optimality, and the ray mode was never compared where the MILP is
weak.

This recomputes the whole curve with the GA, on the *same* symbionts the MILP
used (host 0, k=4, symbiont_id 0..n-1 of post_paper), so the two curves can be
overlaid and compared pairwise rather than in distribution.

  ga_full  evolves c^H itself -- the pure GA, no structure assumed
  ga_ray   evolves only the direction d of c^H and line-searches the magnitude t,
           using f^{S*}(t d) = t f^{S*}(d) (Level 1 is homogeneous in c^H)

Question
--------
Does the peak survive, and does its position or height change, when the top
layer is solved by a method that does not truncate?

One CSV per (method, r) under --outdir; existing blocks are skipped, so the run
is resumable and the grid can be extended later.  At most 20 printed lines.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from grhd.cfgio import ROOT, load_config          # noqa: E402
from grhd.pipeline import get_host, make_R_S_meta  # noqa: E402
from grhd.rng import derive_seed                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_top_layer import Evaluator, ga_full, ga_ray  # noqa: E402

R_GRID = "1,2,3,4,5,6,8,10,12,13,14,16,18,20,25,30,40,50,70,100"


def _one(job):
    (method, r, sid, host_id, m, k, pop, gens, cfg_paths) = job
    cfg = load_config(*cfg_paths)
    hd, cE = get_host(cfg, host_id, m)
    R_S, meta = make_R_S_meta(cfg, r, sid, "paper", m, k)
    ev = Evaluator(R_S, hd.R_H, cE, cfg)
    c_max = float(cfg["flags"]["c_H_box"])
    seed = derive_seed(cfg["master_seed"], "ga_curve", method=method, r=r,
                       symbiont_id=sid, host_id=host_id, m=m, k=k)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if method == "ga_ray":
        best, c = ga_ray(ev, m, rng, pop_size=pop, generations=gens, c_max=c_max)
    else:
        best, c = ga_full(ev, m, rng, pop_size=pop, generations=gens, c_max=c_max)
    dt = time.perf_counter() - t0
    n_eval = ev.n_eval
    ev.close()
    return {"method": method, "host_id": host_id, "m": m, "k": k, "r": r,
            "symbiont_id": sid, "mu_H_0": hd.mu_H_0, "mu_H": float(best),
            "delta_mu_H": float(best) - hd.mu_H_0,
            "c_H_l1": float(np.sum(np.abs(c))) if c is not None else np.nan,
            "seconds": dt, "n_eval": n_eval, "pop": pop, "generations": gens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "tables" / "ga_curve"))
    ap.add_argument("--methods", default="ga_full,ga_ray")
    ap.add_argument("--r-grid", default=R_GRID)
    ap.add_argument("--n-sym", type=int, default=100,
                    help="symbiont ids 0..n-1, the same ones post_paper used")
    ap.add_argument("--sid-lo", type=int, default=None,
                    help="run ids [sid_lo, n_sym) instead of [0, n_sym); the "
                         "block file name carries the range, so extending an "
                         "existing sweep cannot silently skip the new ids "
                         "(START_HERE section 8e)")
    ap.add_argument("--host-id", type=int, default=0)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--ray-generations", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", nargs="+",
                    default=[str(ROOT / "config" / "base.yaml")])
    args = ap.parse_args()

    cfg = load_config(*args.config)
    m, k = int(cfg["model"]["m"]), int(cfg["model"]["k"])
    os.makedirs(args.outdir, exist_ok=True)
    rs = [int(x) for x in args.r_grid.split(",")]
    methods = [x for x in args.methods.split(",") if x in ("ga_full", "ga_ray")]

    lo = 0 if args.sid_lo is None else int(args.sid_lo)
    # ids [0, n) keep the original name so the first sweep stays valid; any other
    # range gets its own file, and the summary de-duplicates on (method, r, sid)
    def blockname(mth, r):
        return (f"{mth}_r{r}.csv" if lo == 0
                else f"{mth}_r{r}_s{lo}-{args.n_sym - 1}.csv")

    blocks = [(mth, r) for r in rs for mth in methods]
    todo = [(mth, r) for mth, r in blocks
            if not os.path.exists(os.path.join(args.outdir, blockname(mth, r)))]
    print(f"host {args.host_id}, m={m}, k={k}, symbionts {lo}..{args.n_sym - 1}")
    print(f"{len(blocks)} blocks, {len(todo)} to run "
          f"({len(todo) * args.n_sym} cells)")

    par = None
    if args.workers > 1:
        from joblib import Parallel, delayed
        par = Parallel(n_jobs=args.workers, backend="loky")

    t_start = time.perf_counter()
    for mth, r in todo:
        gens = args.ray_generations if mth == "ga_ray" else args.generations
        jobs = [(mth, r, sid, args.host_id, m, k, args.pop, gens, args.config)
                for sid in range(lo, args.n_sym)]
        if par is None:
            out = [_one(j) for j in jobs]
        else:
            from joblib import delayed
            out = par(delayed(_one)(j) for j in jobs)
        df = pd.DataFrame(out)
        path = os.path.join(args.outdir, blockname(mth, r))
        df.to_csv(path, index=False)
        print(f"{mth} r={r:<4d} mean mu_H {df['mu_H'].mean():.4f}  "
              f"core-h {df['seconds'].sum() / 3600:.2f}  "
              f"elapsed {(time.perf_counter() - t_start) / 3600:.2f} h",
              flush=True)
    print(f"done: {args.outdir}")


if __name__ == "__main__":
    main()
