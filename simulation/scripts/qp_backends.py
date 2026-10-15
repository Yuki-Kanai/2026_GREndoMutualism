"""Compare min-norm QP backends.

Question 1: does the HiGHS QP still fail once Level 1 is solved at unit scale?
Question 2: can the tie-break be done deterministically without CPLEX?

The min-norm problem is, per cell,

    min 1/2 ||w||^2   s.t.  G w <= h,  w >= 0
    G = [ -R_{-m} ; -R_m ],   h = [ c_hat ; -mu* ]

Its Wolfe dual has only m variables regardless of r:

    max_{lam >= 0}  -1/2 || [ -G^T lam ]_+ ||^2  -  h^T lam
    w*(lam) = [ -G^T lam ]_+ ,    grad = G w - h

m = 10 here, so the dual is a 10-variable concave piecewise-quadratic programme.
`dual_active_set` solves it with L-BFGS-B and then polishes by solving the KKT
equalities on the identified active sets, which costs one small lstsq and makes
the answer exact to machine precision.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from grhd.cfgio import load_config
from grhd.lp import SymbiontSolver
from grhd.pipeline import c_H_samples, make_R_S


# --------------------------------------------------------------------------- #
def dual_active_set(G, h, w0=None, polish=True):
    """Deterministic min-norm solve through the m-dimensional dual."""
    m, r = G.shape
    Gt = G.T

    def negg(lam):
        u = -(Gt @ lam)
        w = np.maximum(u, 0.0)
        val = 0.5 * w @ w + h @ lam            # = -g(lam)
        grad = h - G @ w                       # = -grad g
        return val, grad

    lam0 = np.zeros(m)
    res = minimize(negg, lam0, jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * m,
                   options={"maxiter": 500, "ftol": 1e-18, "gtol": 1e-14})
    lam = np.maximum(res.x, 0.0)
    w = np.maximum(-(Gt @ lam), 0.0)
    if not polish:
        return w, lam, res.nit

    # polish: fix the active sets, solve the KKT equalities exactly
    scale = max(1.0, float(np.max(np.abs(h))))
    I = lam > 1e-12 * scale                    # active constraints
    J = w > 1e-12 * max(1.0, float(np.max(w)))  # active (positive) w
    if I.any() and J.any():
        Gij = G[np.ix_(I, J)]
        M = Gij @ Gij.T
        try:
            lam_I = np.linalg.lstsq(M, -h[I], rcond=None)[0]
            if np.all(lam_I >= -1e-10):
                w_new = np.zeros(r)
                w_new[J] = -(Gij.T @ lam_I)
                if np.all(w_new >= -1e-10):
                    lam_full = np.zeros(m)
                    lam_full[I] = np.maximum(lam_I, 0.0)
                    w = np.maximum(w_new, 0.0)
                    lam = lam_full
        except np.linalg.LinAlgError:
            pass
    return w, lam, res.nit


def build_Gh(R_S, c_hat, mu_star):
    m = R_S.shape[0]
    G = np.vstack([-R_S[: m - 1, :], -R_S[m - 1, :][None, :]])
    h = np.concatenate([c_hat, [-mu_star]])
    return G, h


# --------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("--r-grid", default="2,5,10,20,40,60,100")
ap.add_argument("--n-sym", type=int, default=40)
ap.add_argument("--n-c", type=int, default=10)
ap.add_argument("--out", default="tables/QP_backends.csv")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
m = cfg["model"]["m"]
R_GRID = [int(x) for x in a.r_grid.split(",")]

# optional open-source backends
backends = {}
try:
    import piqp
    backends["piqp"] = piqp
except Exception:
    pass
try:
    import clarabel
    backends["clarabel"] = clarabel
except Exception:
    pass
try:
    import quadprog
    backends["quadprog"] = quadprog
except Exception:
    pass
try:
    import osqp
    backends["osqp"] = osqp
except Exception:
    pass
print("extra backends available:", sorted(backends) or "(none)", flush=True)

rows = []
for r in R_GRID:
    for sid in range(a.n_sym):
        R_S = make_R_S(cfg, r, sid)
        ss = SymbiontSolver(R_S, cfg)          # CPLEX QP, normalised
        for cid, c_H in enumerate(c_H_samples(cfg, sid, a.n_c)):
            scale = float(np.sum(np.abs(c_H)))
            c_hat = c_H / scale

            t0 = time.perf_counter()
            ref = ss.solve(c_H)
            t_ref = time.perf_counter() - t0
            if ref.qp_path != "cplex":
                rows.append({"r": r, "sid": sid, "cid": cid,
                             "backend": "reference_failed", "ok": False})
                continue
            w_ref = ref.w_S / scale            # back to unit scale
            mu_star = float(R_S[m - 1] @ w_ref)
            G, h = build_Gh(R_S, c_hat, mu_star)

            rec = {"r": r, "sid": sid, "cid": cid,
                   "norm_ref": float(np.linalg.norm(w_ref)),
                   "t_cplex_lp_plus_qp": t_ref}

            # (1) HiGHS QP on the normalised problem
            t0 = time.perf_counter()
            w_h = ss._solve_qp_highs(c_hat, mu_star - 1e-9)
            rec["t_highs"] = time.perf_counter() - t0
            rec["highs_ok"] = w_h is not None
            if w_h is not None:
                rec["highs_dw"] = float(np.max(np.abs(w_h - w_ref)))
                rec["highs_dnorm"] = float(np.linalg.norm(w_h)
                                           - np.linalg.norm(w_ref))

            # (2) dual active-set, pure numpy/scipy
            t0 = time.perf_counter()
            w_d, lam_d, nit = dual_active_set(G, h)
            rec["t_dual"] = time.perf_counter() - t0
            rec["dual_iters"] = nit
            rec["dual_dw"] = float(np.max(np.abs(w_d - w_ref)))
            rec["dual_dnorm"] = float(np.linalg.norm(w_d) - np.linalg.norm(w_ref))
            rec["dual_feas"] = float(np.max(np.maximum(G @ w_d - h, 0.0)))
            rows.append(rec)
    print("done r =", r, flush=True)
    ss.close()

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)

pd.set_option("display.width", 220)
ok = df[df.get("highs_ok").notna()] if "highs_ok" in df else df
print("\n=== HiGHS QP on the normalised problem ===")
print(ok.groupby("r").agg(n=("highs_ok", "size"),
                          fail_rate=("highs_ok", lambda s: 1 - s.mean()),
                          max_dw=("highs_dw", "max"),
                          ms=("t_highs", lambda s: 1000 * s.median())
                          ).round(6).to_string())
print("\n=== dual active-set (numpy/scipy only) vs CPLEX ===")
print(df.groupby("r").agg(n=("dual_dw", "size"),
                          max_dw=("dual_dw", "max"),
                          med_dw=("dual_dw", "median"),
                          max_dnorm=("dual_dnorm", "max"),
                          max_infeas=("dual_feas", "max"),
                          ms=("t_dual", lambda s: 1000 * s.median()),
                          iters=("dual_iters", "median")).round(12).to_string())
print(f"\nwrote {a.out}")
