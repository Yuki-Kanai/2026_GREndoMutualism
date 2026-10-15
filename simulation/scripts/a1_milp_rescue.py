"""Resolve the A1-MILP cells CPLEX could not solve, with SCIP.

G2a section 6 documented that CPLEX declares some fixed_repertoire instances
"integer infeasible" although the model is always feasible.  The retry chain
(numerical emphasis, then presolve) recovers most of them; this script settles
the remainder with an independent solver, and also re-solves the detlimit cells
so that their interval [incumbent, best bound] can be reported honestly.

Writes tables/A1_milp_rescue.csv.  Nothing is written back into data/raw: the
raw values stay as the production run produced them (05_conventions.md).
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

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_scipmod", Path(__file__).with_name("phaseB_scip_crosscheck.py"))

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw/post_fixed_repertoire_n40")
ap.add_argument("--n-rxn", type=int, default=40)
ap.add_argument("--time-limit", type=float, default=900.0)
ap.add_argument("--tables", default="tables")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
m = int(cfg["model"]["m"])
hd, cE = get_host(cfg, 0, m)
hs = HostSolver(hd.R_H, cfg)
fl = cfg["flags"]
C_MAX, W_MAX, L_MAX = (float(fl["c_H_box"]), float(fl["w_S_box"]),
                       float(fl["lambda_box"]))
G_MAX = m * C_MAX


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


files = sorted(glob.glob(str(Path(a.raw) / "cells_*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
bad = df[df.flag_infeasible | df.flag_detlimit][
    ["r", "symbiont_id", "solver_status", "mu_H", "mip_gap",
     "mip_best_bound", "milp_path"]]
print(f"{len(bad)} cells to rescue\n", bad.to_string(index=False), flush=True)

rows = []
for rec in bad.itertuples():
    R_S, _ = make_R_S_meta(cfg, int(rec.r), int(rec.symbiont_id),
                           "fixed_repertoire", m, None, a.n_rxn)
    t0 = time.time()
    c_scip, st, mu_scip, db = solve_scip(R_S, hd.R_H, cE, a.time_limit)
    dt = time.time() - t0
    # independent LP re-solve at SCIP's c^H (the V11 check): does the reported
    # mu^H actually follow from that c^H through Level 1 and Level 2?
    recheck = np.nan
    if c_scip is not None:
        ss = SymbiontSolver(R_S, cfg)
        lp = ss.solve(c_scip)
        h = hs.solve(lp.f_S, cE)
        ss.close()
        recheck = h.mu_H if h.status == "optimal" else np.nan
    rows.append({"r": int(rec.r), "symbiont_id": int(rec.symbiont_id),
                 "cplex_status": rec.solver_status, "cplex_milp_path": rec.milp_path,
                 "cplex_mu_H": rec.mu_H, "cplex_gap": rec.mip_gap,
                 "cplex_best_bound": rec.mip_best_bound,
                 "scip_status": st, "scip_mu_H": mu_scip, "scip_dualbound": db,
                 "scip_lp_recheck_mu_H": recheck, "scip_seconds": dt,
                 "mu_H_0": float(hd.mu_H_0)})
    print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out["recheck_abs_diff"] = (out.scip_mu_H - out.scip_lp_recheck_mu_H).abs()
Path(a.tables).mkdir(parents=True, exist_ok=True)
out.to_csv(Path(a.tables) / "A1_milp_rescue.csv", index=False)
pd.set_option("display.width", 250)
print("\n=== rescue ===")
print(out.round(8).to_string(index=False))
print("\nmax |scip_mu_H - LP recheck| =", np.nanmax(out.recheck_abs_diff))
