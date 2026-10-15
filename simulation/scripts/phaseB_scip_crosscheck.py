"""2.5 (optional) — cross-validate Level 3 with SCIP.

07_run2_workorder.md Phase B 2.5.  V15 showed Level 1 is solver independent, but
Level 3 has only ever been solved by CPLEX CE.  This rebuilds the same KKT+SOS1
formulation in SCIP (Apache-2.0) and compares mu^H.  No big-M anywhere: SCIP
supports SOS1 natively.

Only step 1 (maximise mu^H) is compared.  The ||c^H||_1 tie-break is a second
MILP with the same feasible set, so agreeing on the optimal value is the check
that matters for the formulation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.lp import HostSolver, SymbiontSolver
from grhd.milp import EvolutionSolver
from grhd.pipeline import get_host, make_R_S

ap = argparse.ArgumentParser()
ap.add_argument("--r-grid", default="2,5,10,20")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--time-limit", type=float, default=300.0)
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

    r = R_S.shape[1]
    rH = R_H.shape[1]
    mm = R_S.shape[0]
    mo = Model("level3")
    mo.hideOutput()
    mo.setParam("limits/time", time_limit)
    # SCIP's default feastol (1e-6); forcing 1e-9 makes SoPlex fail with
    # "error in LP solver" on some of these degenerate instances

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

    for i in range(mm):                                   # (C1) slack definition
        e = g[i] - quicksum(R_S[i, j] * w[j] for j in range(r))
        e = e - cH[i] if i < mm - 1 else e + muS
        mo.addCons(e == 0)
    for j in range(r):                                    # (C2) LP stationarity
        mo.addCons(quicksum(R_S[i, j] * l1[i] for i in range(mm)) + l2[j] == 0)
    for j in range(r):                                    # (C4) QP stationarity
        mo.addCons(-2 * w[j]
                   + quicksum(R_S[i, j] * l3[i] for i in range(mm)) + l4[j] == 0)
    for i in range(mm - 1):                               # (C5) host balance
        mo.addCons(-quicksum(R_S[i, j] * w[j] for j in range(r))
                   - quicksum(R_H[i, l] * wH[l] for l in range(rH)) <= cE[i])
    mo.addCons(muH - quicksum(R_H[mm - 1, l] * wH[l] for l in range(rH)) == 0)

    for i in range(mm - 1):                               # SOS1 complementarity
        mo.addConsSOS1([l1[i], g[i]])
        mo.addConsSOS1([l3[i], g[i]])
    for j in range(r):
        mo.addConsSOS1([l2[j], w[j]])
        mo.addConsSOS1([l4[j], w[j]])

    mo.setObjective(muH, "maximize")
    try:
        mo.optimize()
    except Exception as exc:                            # noqa: BLE001
        return None, f"error:{type(exc).__name__}", np.nan, np.nan
    st = mo.getStatus()
    if mo.getNSols() == 0:
        return None, st, np.nan, np.nan
    sol = mo.getBestSol()
    return (np.array([mo.getSolVal(sol, v) for v in cH]), st,
            float(mo.getSolVal(sol, muH)), float(mo.getDualbound()))


rows = []
for r in [int(x) for x in a.r_grid.split(",")]:
    for sid in range(a.n):
        R_S = make_R_S(cfg, r, sid)
        t0 = time.time()
        cx = EvolutionSolver(R_S, hd.R_H, cE, cfg).solve()
        t_cplex = time.time() - t0
        t0 = time.time()
        c_scip, st_scip, mu_scip, db_scip = solve_scip(R_S, hd.R_H, cE,
                                                       a.time_limit)
        t_scip = time.time() - t0
        rec = {"r": r, "symbiont_id": sid,
               "mu_H_cplex": cx.mu_H, "mu_H_cplex_step1": cx.detail.get("mu_H_step1"),
               "status_cplex": cx.status, "seconds_cplex": t_cplex,
               "mu_H_scip": mu_scip, "status_scip": st_scip,
               "dualbound_scip": db_scip, "seconds_scip": t_scip}
        # independent LP re-solve at SCIP's c^H (same check as V11)
        if c_scip is not None:
            ss = SymbiontSolver(R_S, cfg)
            lp = ss.solve(c_scip)
            h = hs.solve(lp.f_S, cE)
            ss.close()
            rec["mu_H_scip_lp_recheck"] = h.mu_H if h.status == "optimal" else np.nan
        both_opt = (cx.status == "optimal" and st_scip == "optimal")
        rec["both_optimal"] = both_opt
        rec["abs_diff"] = (abs(cx.detail.get("mu_H_step1", cx.mu_H) - mu_scip)
                           if np.isfinite(mu_scip) else np.nan)
        rows.append(rec)
        print(f"r={r:3d} sid={sid:3d} cplex={cx.mu_H:.6f}({cx.status},{t_cplex:.1f}s) "
              f"scip={mu_scip:.6f}({st_scip},{t_scip:.1f}s) "
              f"|d|={rec['abs_diff']:.2e}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(Path(a.tables) / "B25_scip_crosscheck.csv", index=False)
pd.set_option("display.width", 220)
ok = df[df["both_optimal"]]
print(f"\n{len(df)} instances, both solvers optimal in {len(ok)}")
if len(ok):
    print(f"  max |mu_H(CPLEX step1) - mu_H(SCIP)| = {ok['abs_diff'].max():.3e}")
    print(f"  median                                = {ok['abs_diff'].median():.3e}")
    d2 = (ok["mu_H_scip"] - ok["mu_H_scip_lp_recheck"]).abs()
    print(f"  max |SCIP mu_H - LP re-solve at its c^H| = {d2.max():.3e}")
print("\nstatus pairs:")
print(df.groupby(["status_cplex", "status_scip"]).size().to_string())
print("\nsolve time (s), median:")
print(df.groupby("r")[["seconds_cplex", "seconds_scip"]].median().round(3).to_string())

json.dump({"n": int(len(df)), "n_both_optimal": int(len(ok)),
           "max_abs_diff": float(ok["abs_diff"].max()) if len(ok) else None,
           "r_grid": a.r_grid, "time_limit_s": a.time_limit},
          open(Path(a.tables) / "B25_meta.json", "w"), indent=2)
