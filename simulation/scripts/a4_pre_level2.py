"""A4 / gate G7 -- pre-evolution (Level 2) for many hosts, reusing the Level 1 cache.

Purpose
-------
Level 1 (the symbiont's own optimisation) does not depend on the host, so the
f^{S*} vectors already stored in data/raw/pre_paper/vectors_*.parquet are valid
for every host.  Only Level 2 -- the host LP with right-hand side c^E - f^{S*} --
has to be re-solved per host (02_reproduction_tasks.md section 0.4,
07_run2_workorder.md section 4.2).  That turns a 15 core-hour re-run into minutes
and, crucially, gives the per-host **Level 2 infeasible rate**, which is required
output for G7 and which was exactly 0 for the single run-1 host.

Question
--------
Do the pre-evolution mu^H curves, and the rate at which Level 2 is infeasible,
depend on which host was drawn?

Output
------
data/A4_pre_level2_cells.parquet  one row per (host_id, r, symbiont_id, c_id)
tables/A4_pre_level2_by_r.csv     per (host_id, r) aggregate, both conventions
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grhd.cfgio import ROOT, load_config, parse_r_grid  # noqa: E402
from grhd.lp import HostSolver  # noqa: E402
from grhd.pipeline import get_host  # noqa: E402

EPS_MUT = 1e-3     # same thresholds as A3-e
THETA_MUT = 1e-6


def _one_r(r: int, src: Path, cfg: dict, n_hosts: int, n_s: int, n_c: int):
    """All hosts at one r.  Hosts are re-derived in the worker (~70 cheap LPs)."""
    files = sorted(src.glob(f"vectors_r{r:03d}_s*.parquet"))
    if not files:
        return None
    v = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    v = v[(v["symbiont_id"] < n_s) & (v["c_id"] < n_c)]
    if v.empty:
        return None
    F = np.stack([np.asarray(x, dtype=np.float64) for x in v["f_S_star"]])
    sid = v["symbiont_id"].to_numpy()
    cid = v["c_id"].to_numpy()
    # mu_S is host-independent: take it from the cached Level 1 cell table so the
    # mutualism criterion is the one A3-e used.
    cf = pd.concat([pd.read_parquet(f, columns=["symbiont_id", "c_id", "mu_S"])
                    for f in sorted(src.glob(f"cells_r{r:03d}_s*.parquet"))],
                   ignore_index=True)
    mu_S = pd.DataFrame({"symbiont_id": sid, "c_id": cid}).merge(
        cf, on=["symbiont_id", "c_id"], how="left")["mu_S"].to_numpy()

    out = []
    for h in range(n_hosts):
        hd, cE = get_host(cfg, host_id=h)
        hs = HostSolver(hd.R_H, cfg)
        mu = np.empty(len(F))
        bad = np.zeros(len(F), dtype=bool)
        for i in range(len(F)):
            sol = hs.solve(F[i], cE)
            if sol.status == "infeasible":
                mu[i], bad[i] = np.nan, True
            else:
                mu[i] = sol.mu_H
        out.append(pd.DataFrame({
            "host_id": h, "r": r, "symbiont_id": sid, "c_id": cid,
            "mu_H": mu, "mu_H_0": hd.mu_H_0, "flag_infeasible": bad,
            "mu_S": mu_S,
        }))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "raw" / "pre_paper"))
    ap.add_argument("--n-hosts", type=int, default=20)
    ap.add_argument("--r-grid", default="1..100")
    ap.add_argument("--n-s", type=int, default=100, help="symbiont_id < n_s")
    ap.add_argument("--n-c", type=int, default=33, help="c_id < n_c")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "config" / "base.yaml")])
    ap.add_argument("--tables", default=str(ROOT / "tables"))
    ap.add_argument("--cells-out",
                    default=str(ROOT / "data" / "A4_pre_level2_cells.parquet"))
    args = ap.parse_args()

    cfg = load_config(*args.config)
    # parse_r_grid iterates a bare string character-wise, so always hand it a list
    # unless it is a ".." range.
    g = args.r_grid
    r_grid = parse_r_grid(g if ".." in g else g.split(","))
    src = Path(args.src)
    tab = Path(args.tables)
    tab.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    jobs = [(r, src, cfg, args.n_hosts, args.n_s, args.n_c) for r in r_grid]
    if args.workers <= 1:
        parts = [_one_r(*j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(_one_r)(*j) for j in jobs)
    cells = pd.concat([p for p in parts if p is not None], ignore_index=True)
    print(f"{len(cells)} host-cells in {time.perf_counter() - t0:.0f}s", flush=True)

    cells["delta_mu_H"] = cells["mu_H"] - cells["mu_H_0"]
    cells["mu_H_paper"] = cells["mu_H"].fillna(0.0)       # 01 section 4.1 convention
    cells["mu_S_paper"] = np.where(cells["flag_infeasible"], 0.0, cells["mu_S"])
    cells["mutualistic"] = ((cells["mu_H"] > cells["mu_H_0"] + EPS_MUT)
                            & (cells["mu_S"] > THETA_MUT))
    out = Path(args.cells_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out, index=False)

    agg = cells.groupby(["host_id", "r"], as_index=False).agg(
        n=("mu_H", "size"),
        n_infeasible=("flag_infeasible", "sum"),
        mu_H_mean=("mu_H", "mean"),                 # default: infeasible excluded
        mu_H_mean_paper=("mu_H_paper", "mean"),     # paper convention: 0 substituted
        mu_H_median=("mu_H", "median"),
        delta_mean=("delta_mu_H", "mean"),
        frac_mutualistic=("mutualistic", "mean"),
        mu_H_0=("mu_H_0", "first"),
    )
    agg["frac_infeasible"] = agg["n_infeasible"] / agg["n"]
    agg["mu_H_rel"] = agg["mu_H_mean"] / agg["mu_H_0"]
    agg.to_csv(tab / "A4_pre_level2_by_r.csv", index=False)

    tot = cells.groupby("host_id")["flag_infeasible"].mean()
    print(f"cells -> {out}\nby_r  -> {tab / 'A4_pre_level2_by_r.csv'}")
    print(f"Level 2 infeasible rate per host: min={tot.min():.4g} "
          f"median={tot.median():.4g} max={tot.max():.4g}; "
          f"hosts with any infeasible = {int((tot > 0).sum())}/{len(tot)}")


if __name__ == "__main__":
    main()
