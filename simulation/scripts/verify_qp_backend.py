"""Does swapping the min-norm QP backend change any reported quantity?

Runs the same cells through the quadprog and the CPLEX chain and compares the
cell-level outputs that end up in the tables.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd

from grhd.analyze import analyze_cell
from grhd.cfgio import load_config
from grhd.lp import HostSolver, SymbiontSolver
from grhd.pipeline import c_H_samples, get_host, make_R_S

ap = argparse.ArgumentParser()
ap.add_argument("--r-grid", default="1,2,3,5,8,10,12,15,20,25,30,40,60,80,100")
ap.add_argument("--n-sym", type=int, default=25)
ap.add_argument("--n-c", type=int, default=20)
ap.add_argument("--out", default="tables/QP_backend_equivalence.csv")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

base = load_config(*a.config)
m = base["model"]["m"]
hd, cE = get_host(base, 0, m)
hs = HostSolver(hd.R_H, base)

CHAINS = {"quadprog": ["quadprog"], "cplex": ["cplex"]}
cfgs = {}
for name, chain in CHAINS.items():
    c = copy.deepcopy(base)
    c["solver"].setdefault("qp", {})["backends"] = chain
    cfgs[name] = c

rows = []
timing = {k: 0.0 for k in CHAINS}
for r in [int(x) for x in a.r_grid.split(",")]:
    for sid in range(a.n_sym):
        R_S = make_R_S(base, r, sid)
        CH = c_H_samples(base, sid, a.n_c)
        out = {}
        for name, cfg in cfgs.items():
            ss = SymbiontSolver(R_S, cfg)
            t0 = time.perf_counter()
            recs = []
            for c_H in CH:
                sol = ss.solve(c_H)
                an = analyze_cell(R_S, c_H, sol.f_S, sol.w_S, sol.lam2, cfg)
                h = hs.solve(sol.f_S, cE)
                recs.append((sol.mu_S, h.mu_H if h.status == "optimal" else np.nan,
                             an.n_bounded, an.r_plus, an.s, an.alpha,
                             an.n_excreted, an.ub, sol.qp_path))
            timing[name] += time.perf_counter() - t0
            ss.close()
            out[name] = recs
        for i, (q, c) in enumerate(zip(out["quadprog"], out["cplex"])):
            rows.append({
                "r": r, "sid": sid, "cid": i,
                "b_same": q[2] == c[2], "rplus_same": q[3] == c[3],
                "nexc_same": q[6] == c[6], "ub_same": q[7] == c[7],
                "d_mu_S": abs(q[0] - c[0]),
                "d_mu_H": abs(q[1] - c[1]) if np.isfinite(q[1]) and np.isfinite(c[1]) else 0.0,
                "d_s": abs(q[4] - c[4]),
                "rel_s": abs(q[4] - c[4]) / max(abs(c[4]), 1e-12),
                "qp_path_quadprog": q[8], "qp_path_cplex": c[8],
            })
    print("done r =", r, flush=True)

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
n = len(df)
print(f"\n{n} cells compared (quadprog chain vs cplex chain)")
print(f"  b identical        : {df.b_same.mean() * 100:.4f} %")
print(f"  r_plus identical   : {df.rplus_same.mean() * 100:.4f} %")
print(f"  n_excreted identical: {df.nexc_same.mean() * 100:.4f} %")
print(f"  ub identical       : {df.ub_same.mean() * 100:.4f} %")
print(f"  max |d mu_S|       : {df.d_mu_S.max():.3e}")
print(f"  max |d mu_H|       : {df.d_mu_H.max():.3e}")
print(f"  max |d s|          : {df.d_s.max():.3e}   median {df.d_s.median():.3e}")
print(f"  max rel |d s|      : {df.rel_s.max():.3e}")
print(f"  fallbacks used     : quadprog {df.qp_path_quadprog.value_counts().to_dict()}"
      f" / cplex {df.qp_path_cplex.value_counts().to_dict()}")
print(f"\nwall time (LP+QP+analyse+host LP): "
      + ", ".join(f"{k} {v:.1f}s" for k, v in timing.items()))
bad = df[~df.b_same | ~df.rplus_same | ~df.nexc_same]
if len(bad):
    print(f"\n{len(bad)} cells with a discrete difference:")
    print(bad.head(20).to_string(index=False))
