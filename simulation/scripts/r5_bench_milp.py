"""R5 / gate G9 -- MILP benchmark per m, for both designs (08_r5_two_designs.md section 3).

Purpose
-------
The budget for R5 cannot be taken from G3.  G3 measured m = 10 only, and its
extrapolation to 88,000 MILP is a straight-line guess: the variable count and the
number of SOS1 sets both grow like O(m + r), and in design R5-A the host grows too
(r^H = 2.4 m), so nothing about the m = 10 timing carries over on its own.
08 section 3 therefore requires an explicit benchmark before the production run,
and requires checking that the model stays inside the CPLEX Community Edition
limits of 1000 variables and 1000 constraints.

Question
--------
How long does one Level 3 MILP take as m grows, in each design, and does the model
still fit inside CPLEX CE?

Method
------
Both designs x m in {5, 10, 15, 20, 25} x r in {2, 10, 25, 60}, 20 instances each.
One host per (design, m), drawn with the same rejection sampler and the same
acceptance window as everywhere else.  Records solve time, status, and the model
size actually built.

Output
------
tables/R5_milp_bench.csv        one row per solved instance
tables/R5_milp_bench_summary.csv  median / p95 time, detlimit rate, sizes per cell
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

from grhd.cfgio import ROOT, load_config  # noqa: E402
from grhd.milp import solve_host_evolution  # noqa: E402
from grhd.pipeline import get_host, make_R_S  # noqa: E402

CE_VAR_LIMIT = 1000
CE_ROW_LIMIT = 1000


def r_H_of(design: str, m: int) -> int:
    """R5-A keeps r^H/m fixed at the paper's m=10, r^H=24 choice; R5-B fixes r^H."""
    return int(round(2.4 * m)) if design == "A" else 24


def _one_cell(design: str, m: int, r: int, sid_lo: int, sid_hi: int, cfg_path,
              det_limit, part_dir: str | None = None, label: str = "bench"):
    """One (design, m, r) cell.  The result is written to its own CSV as soon as it
    is done, so progress is visible from outside and a killed run resumes by
    skipping finished parts -- the mistake this script made on its first run was
    printing only at the end, which is indistinguishable from a hang."""
    part = None
    if part_dir is not None:
        part = (Path(part_dir) /
                f"{label}_{design}_m{m:02d}_r{r:03d}_s{sid_lo:04d}.csv")
        if part.exists():
            return pd.read_csv(part)
    cfg = load_config(*cfg_path)
    if det_limit is not None:
        cfg["solver"]["milp"]["det_time_limit"] = det_limit
    cfg["model"]["r_H"] = r_H_of(design, m)
    hd, cE = get_host(cfg, host_id=0, m=m)
    rows = []
    for sid in range(sid_lo, sid_hi):
        R_S = make_R_S(cfg, r, sid, "paper", m, cfg["model"]["k"])
        t0 = time.perf_counter()
        sol = solve_host_evolution(R_S, hd.R_H, cE, cfg)
        dt = time.perf_counter() - t0
        rows.append({
            "design": design, "m": m, "r": r, "r_H": cfg["model"]["r_H"],
            "symbiont_id": sid, "seconds": dt, "status": sol.status,
            "milp_path": sol.detail.get("milp_path"),
            "n_vars": sol.detail.get("n_vars"),
            "n_rows": sol.detail.get("n_rows"),
            "n_sos": sol.detail.get("n_sos"),
            "mu_H": sol.mu_H, "mu_H_0": hd.mu_H_0, "mu_S": sol.mu_S,
            "mip_gap": sol.mip_gap,
            "mip_best_bound": sol.mip_best_bound,
            "det_time_limit": cfg["solver"]["milp"]["det_time_limit"],
            "host_accepted": bool(hd.accepted), "host_attempts": hd.n_attempts,
        })
    out = pd.DataFrame(rows)
    if part is not None:
        part.parent.mkdir(parents=True, exist_ok=True)
        tmp = part.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, part)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="A,B")
    ap.add_argument("--m-grid", default="5,10,15,20,25")
    ap.add_argument("--r-grid", default="2,10,25,60")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--det-limit", type=float, default=None)
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "config" / "base.yaml")])
    ap.add_argument("--tables", default=str(ROOT / "tables"))
    ap.add_argument("--parts", default=str(ROOT / "data" / "r5_bench_parts"),
                    help="per-cell CSVs, written as each cell finishes (resumable)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="symbionts per job; smaller = more parallelism on few cells")
    ap.add_argument("--label", default="bench", help="prefix for part files")
    ap.add_argument("--out-prefix", default="R5_milp_bench")
    args = ap.parse_args()

    designs = args.designs.split(",")
    ms = [int(x) for x in args.m_grid.split(",")]
    rs = [int(x) for x in args.r_grid.split(",")]
    parts_dir = Path(args.parts)
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk or args.n
    jobs = [(d, m, r, lo, min(lo + chunk, args.n), args.config, args.det_limit,
             str(parts_dir), args.label)
            for d in designs for m in ms for r in rs
            for lo in range(0, args.n, chunk)]

    t0 = time.perf_counter()
    if args.workers <= 1:
        parts = [_one_cell(*j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(_one_cell)(*j) for j in jobs)
    df = pd.concat(parts, ignore_index=True)

    tab = Path(args.tables); tab.mkdir(parents=True, exist_ok=True)
    df.to_csv(tab / f"{args.out_prefix}.csv", index=False)

    g = df.groupby(["design", "m", "r"], as_index=False).agg(
        n=("seconds", "size"),
        t_median=("seconds", "median"),
        t_p95=("seconds", lambda x: float(np.quantile(x, 0.95))),
        t_max=("seconds", "max"),
        frac_detlimit=("status", lambda x: float((x.str.contains("detlimit")).mean())),
        frac_infeasible=("status", lambda x: float((x == "infeasible").mean())),
        n_vars=("n_vars", "max"), n_rows=("n_rows", "max"), n_sos=("n_sos", "max"),
        r_H=("r_H", "first"), mu_H_0=("mu_H_0", "first"),
    )
    g["ce_vars_ok"] = g["n_vars"] <= CE_VAR_LIMIT
    g["ce_rows_ok"] = g["n_rows"] <= CE_ROW_LIMIT
    g.to_csv(tab / f"{args.out_prefix}_summary.csv", index=False)

    print(f"{len(df)} instances in {time.perf_counter() - t0:.0f}s wall "
          f"({df.seconds.sum() / 3600:.2f} core-hours)")
    print(g[["design", "m", "r", "r_H", "t_median", "t_p95", "frac_detlimit",
             "n_vars", "n_rows", "n_sos"]].to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))
    bad = g[~(g["ce_vars_ok"] & g["ce_rows_ok"])]
    print("\nCPLEX CE limit exceeded:", "none" if bad.empty else
          bad[["design", "m", "r", "n_vars", "n_rows"]].to_string(index=False))
    print("statuses:", df["status"].value_counts().to_dict())
    print("milp_path:", df["milp_path"].fillna("none").value_counts().to_dict())


if __name__ == "__main__":
    main()
