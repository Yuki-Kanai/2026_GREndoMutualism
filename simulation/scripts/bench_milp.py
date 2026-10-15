"""G3 benchmark: MILP solve time and status per r (02_reproduction_tasks.md section 0.2).

Writes one CSV row per solved instance, flushed as it goes so progress is visible.
"""
import argparse
import csv
import json
import sys
import time

import numpy as np

from grhd.cfgio import load_config
from grhd.generate import c_E
from grhd.lp import HostSolver, SymbiontSolver
from grhd.milp import solve_host_evolution
from grhd.pipeline import get_host, make_R_S

ap = argparse.ArgumentParser()
ap.add_argument("--r-grid", default="2,5,10,25,50,100")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--det-limit", type=float, default=None)
ap.add_argument("--out", default="tables/G3_milp_bench.csv")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
if a.det_limit is not None:
    cfg["solver"]["milp"]["det_time_limit"] = a.det_limit
m = cfg["model"]["m"]
hd, cE = get_host(cfg, 0, m)
hs = HostSolver(hd.R_H, cfg)
print("mu_H_0 = %.6f (attempts %d)" % (hd.mu_H_0, hd.n_attempts), flush=True)

fields = ["r", "symbiont_id", "seconds", "status", "status1", "status2", "n_vars",
          "mu_H", "mu_H_0", "mu_S", "mip_gap", "l1_step1", "l1_step2",
          "cmax_active", "box_active", "box_report", "box_retries",
          "muH_lp_recheck", "muS_lp_recheck",
          "b_post", "u_post", "s_post"]
with open(a.out, "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=fields)
    wr.writeheader()
    for r in [int(x) for x in a.r_grid.split(",")]:
        for sid in range(a.n):
            R_S = make_R_S(cfg, r, sid)
            t0 = time.time()
            sol = solve_host_evolution(R_S, hd.R_H, cE, cfg)
            dt = time.time() - t0
            ss = SymbiontSolver(R_S, cfg)
            lp = ss.solve(sol.c_H) if np.all(np.isfinite(sol.c_H)) else None
            h2 = hs.solve(lp.f_S, cE) if lp is not None else None
            from grhd.analyze import analyze_cell
            an = (analyze_cell(R_S, sol.c_H, sol.f_S, sol.w_S, None, cfg)
                  if np.isfinite(sol.mu_H) else None)
            wr.writerow({
                "r": r, "symbiont_id": sid, "seconds": round(dt, 4),
                "status": sol.status, "status1": sol.detail.get("status_step1"),
                "status2": sol.detail.get("status_step2"),
                "n_vars": sol.detail.get("n_vars"),
                "mu_H": sol.mu_H, "mu_H_0": hd.mu_H_0, "mu_S": sol.mu_S,
                "mip_gap": sol.mip_gap, "l1_step1": sol.l1_step1,
                "l1_step2": sol.l1_step2, "cmax_active": sol.flag_cmax_active,
                "box_active": sol.flag_box_active,
                "box_report": json.dumps({k: round(v, 4) for k, v in sol.box_report.items()}),
                "box_retries": sol.detail.get("box_retries", 0),
                "muH_lp_recheck": (h2.mu_H if h2 is not None else None),
                "muS_lp_recheck": (lp.mu_S if lp is not None else None),
                "b_post": (an.n_bounded if an else None),
                "u_post": (an.n_unbounded if an else None),
                "s_post": (an.s if an else None),
            })
            fh.flush()
            print("r=%3d sid=%3d %8.2fs %-12s muH=%.6f gap=%.2g" %
                  (r, sid, dt, sol.status, sol.mu_H, sol.mip_gap), flush=True)
