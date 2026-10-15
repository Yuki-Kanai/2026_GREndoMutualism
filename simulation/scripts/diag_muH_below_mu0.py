"""Why does the Level-3 MILP return mu^H < mu^H_0 (or 'infeasible')?

The point c^H = 0, w^S = 0, mu^S = 0 should always be feasible and give
mu^H = mu^H_0.  For a given (host, r, symbiont) cell this script
  1. re-solves the MILP exactly as the pipeline does (per retry setting),
  2. re-solves with c^H fixed to 0 (is the fallback point feasible in the
     boxed model at all?),
  3. computes the smallest dual lam1 that certifies mu^S = 0 at c^H = 0
     (does it fit inside lambda_box?).

    PYTHONPATH=src python scripts/diag_muH_below_mu0.py --host 7 --r 6 --sid 177
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grhd.cfgio import ROOT, load_config  # noqa: E402
from grhd.milp import EvolutionSolver, _status_name  # noqa: E402
from grhd.pipeline import get_host, make_R_S_meta  # noqa: E402


def step1(solver, fix_cH0=False):
    prob, idx = solver._build()
    if fix_cH0:
        prob.variables.set_upper_bounds([(v, 0.0) for v in idx["c_H"]])
    prob.objective.set_sense(prob.objective.sense.maximize)
    prob.objective.set_linear([(idx["mu_H"], 1.0)])
    prob.solve()
    st = _status_name(solver.cplex, prob)
    out = {"status": st, "cplex_status": prob.solution.get_status_string()}
    if prob.solution.get_solution_type() != prob.solution.type.none:
        x = np.array(prob.solution.get_values())
        out.update(mu_H=x[idx["mu_H"]], cH_l1=np.abs(x[idx["c_H"]]).sum(),
                   wS_l1=np.abs(x[idx["w_S"]]).sum(), wH_l1=np.abs(x[idx["w_H"]]).sum(),
                   mu_S=x[idx["mu_S"]], lam1_max=x[idx["lam1"]].max())
        try:
            out["bound"] = prob.solution.MIP.get_best_objective()
        except Exception:
            pass
    return out


def min_dual(R_S, L_max):
    """min max_i lam1_i  s.t. lam1 >= 0, lam1_m = 1, (R^S)^T lam1 <= 0."""
    import highspy
    m, r = R_S.shape
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    # vars: lam1 (m), t
    inf = highspy.kHighsInf
    for i in range(m):
        lo = 1.0 if i == m - 1 else 0.0
        h.addVar(lo, 1.0 if i == m - 1 else inf)
    h.addVar(0.0, inf)
    h.changeColCost(m, 1.0)
    for j in range(r):
        h.addRow(-inf, 0.0, m, np.arange(m, dtype=np.int32), R_S[:, j])
    for i in range(m - 1):
        h.addRow(-inf, 0.0, 2, np.array([i, m], dtype=np.int32), np.array([1.0, -1.0]))
    h.run()
    ms = h.modelStatusToString(h.getModelStatus())
    return ms, (h.getInfo().objective_function_value if ms == "Optimal" else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--scheme", default="paper")
    a = ap.parse_args()

    cfg = load_config(str(ROOT / "config" / "base.yaml"))
    m = a.m or cfg["model"]["m"]
    k = a.k or cfg["model"]["k"]
    hd, cE = get_host(cfg, a.host, m, a.scheme)
    R_S, _ = make_R_S_meta(cfg, a.r, a.sid, a.scheme, m, k, None, world_id=a.host)
    print(f"host={a.host} r={a.r} sid={a.sid} m={m} k={k} mu_H_0={hd.mu_H_0:.6g} "
          f"L_max={float(cfg['flags']['lambda_box']):g}")
    print("R^S col nnz:", (np.abs(R_S) > 0).sum(0).tolist())
    print("R^S last row (biomass):", np.round(R_S[-1], 4).tolist())

    ms, t = min_dual(R_S, float(cfg["flags"]["lambda_box"]))
    print(f"min max lam1 certifying mu^S=0 at c^H=0: {ms} {t:.6g}")

    for name, kw in [("default", {}), ("numerical_emphasis", dict(numerical_emphasis=1)),
                     ("presolve_on", dict(numerical_emphasis=1, presolve=1))]:
        for fix in (False, True):
            s = EvolutionSolver(R_S, hd.R_H, cE, cfg, **kw)
            o = step1(s, fix_cH0=fix)
            print(f"{name:20s} fix_cH0={fix!s:5s} " +
                  " ".join(f"{kk}={v:.6g}" if isinstance(v, float) else f"{kk}={v}"
                           for kk, v in o.items()))


if __name__ == "__main__":
    main()
