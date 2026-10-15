"""Shared-tournament check (SW) -- re-solve, with SCIP, the post cells CPLEX left infeasible.

The Level-3 model is always feasible, so an "infeasible" that survives the CPLEX
retry chain (default -> numerical_emphasis -> presolve_on) is a solver failure
(START_HERE section 6-1).  Same procedure as scripts/a1_milp_rescue.py: SCIP on the
same KKT+SOS1 formulation, then an independent LP re-solve at SCIP's c^H.

Writes tables/SW_milp_rescue.csv.  Nothing is written back into data/raw.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.lp import HostSolver, SymbiontSolver
from grhd.pipeline import get_host, make_R_S_meta

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="data/raw/post_shared_world*_w*")
ap.add_argument("--time-limit", type=float, default=900.0)
ap.add_argument("--tables", default="tables")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
fl = cfg["flags"]
C_MAX, W_MAX, L_MAX = (float(fl["c_H_box"]), float(fl["w_S_box"]),
                       float(fl["lambda_box"]))


def solve_scip(R_S, R_H, cE, time_limit):
    from pyscipopt import Model, quicksum
    r = R_S.shape[1]; rH = R_H.shape[1]; mm = R_S.shape[0]
    mo = Model("level3"); mo.hideOutput(); mo.setParam("limits/time", time_limit)
    w = [mo.addVar(lb=0, ub=W_MAX, name=f"w{j}") for j in range(r)]
    cH = [mo.addVar(lb=0, ub=C_MAX, name=f"c{i}") for i in range(mm - 1)]
    muS = mo.addVar(lb=0, ub=1, name="muS")
    g = [mo.addVar(lb=0, ub=(0.0 if i == mm - 1 else G_MAX), name=f"g{i}")
         for i in range(mm)]
    l1 = [mo.addVar(lb=(1.0 if i == mm - 1 else 0.0),
                    ub=(1.0 if i == mm - 1 else L_MAX), name=f"l1_{i}")
          for i in range(mm)]
    l2 = [mo.addVar(lb=0, ub=L_MAX, name=f"l2_{j}") for j in range(r)]
    l3 = [mo.addVar(lb=0, ub=L_MAX, name=f"l3_{i}") for i in range(mm)]
    l4 = [mo.addVar(lb=0, ub=L_MAX, name=f"l4_{j}") for j in range(r)]
    wH = [mo.addVar(lb=0, ub=W_MAX, name=f"wH{l}") for l in range(rH)]
    muH = mo.addVar(lb=0, ub=1, name="muH")
    for i in range(mm):
        e = g[i] - quicksum(R_S[i, j] * w[j] for j in range(r))
        e = e - cH[i] if i < mm - 1 else e + muS
        mo.addCons(e == 0)
    for j in range(r):
        mo.addCons(quicksum(R_S[i, j] * l1[i] for i in range(mm)) + l2[j] == 0)
    for j in range(r):
        mo.addCons(-2 * w[j]
                   + quicksum(R_S[i, j] * l3[i] for i in range(mm)) + l4[j] == 0)
    for i in range(mm - 1):
        mo.addCons(-quicksum(R_S[i, j] * w[j] for j in range(r))
                   - quicksum(R_H[i, l] * wH[l] for l in range(rH)) <= cE[i])
    mo.addCons(muH - quicksum(R_H[mm - 1, l] * wH[l] for l in range(rH)) == 0)
    for i in range(mm - 1):
        mo.addConsSOS1([l1[i], g[i]]); mo.addConsSOS1([l3[i], g[i]])
    for j in range(r):
        mo.addConsSOS1([l2[j], w[j]]); mo.addConsSOS1([l4[j], w[j]])
    mo.setObjective(muH, "maximize")
    try:
        mo.optimize()
    except Exception as exc:                                     # noqa: BLE001
        return None, f"error:{type(exc).__name__}", np.nan, np.nan
    st = mo.getStatus()
    if mo.getNSols() == 0:
        return None, st, np.nan, np.nan
    sol = mo.getBestSol()
    return (np.array([mo.getSolVal(sol, v) for v in cH]), st,
            float(mo.getSolVal(sol, muH)), float(mo.getDualbound()))


rows = []
for d in sorted(glob.glob(a.glob)):
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(str(Path(d) / "cells_*.parquet"))],
                   ignore_index=True)
    bad = df[df.flag_infeasible]
    if bad.empty:
        continue
    m, world = int(bad.m.iloc[0]), int(bad.host_id.iloc[0])
    G_MAX = m * C_MAX
    hd, cE = get_host(cfg, world, m, "shared_world")
    hs = HostSolver(hd.R_H, cfg)
    for rec in bad.itertuples():
        R_S, _ = make_R_S_meta(cfg, int(rec.r), int(rec.symbiont_id), "shared_world",
                               m, int(rec.k), None, world_id=world)
        t0 = time.time()
        c_scip, st, mu_scip, db = solve_scip(R_S, hd.R_H, cE, a.time_limit)
        dt = time.time() - t0
        recheck = np.nan
        if c_scip is not None:
            ss = SymbiontSolver(R_S, cfg)
            lp = ss.solve(c_scip)
            h = hs.solve(lp.f_S, cE)
            ss.close()
            recheck = h.mu_H if h.status == "optimal" else np.nan
        rows.append({"run": Path(d).name, "m": m, "world_id": world,
                     "r": int(rec.r), "symbiont_id": int(rec.symbiont_id),
                     "cplex_status": rec.solver_status, "cplex_milp_path": rec.milp_path,
                     "scip_status": st, "scip_mu_H": mu_scip, "scip_dualbound": db,
                     "scip_lp_recheck_mu_H": recheck, "scip_seconds": dt,
                     "mu_H_0": float(hd.mu_H_0)})
        print(rows[-1], flush=True)

out = pd.DataFrame(rows)
if len(out):
    out["recheck_abs_diff"] = (out.scip_mu_H - out.scip_lp_recheck_mu_H).abs()
Path(a.tables).mkdir(parents=True, exist_ok=True)
out.to_csv(Path(a.tables) / "SW_milp_rescue.csv", index=False)
print(f"{len(out)} cells rescued -> tables/SW_milp_rescue.csv")
