"""Open-source, deterministic replacements for the CPLEX min-norm QP.

Per cell (at unit scale):  min 1/2 ||w||^2  s.t.  G w <= h,  w >= 0
with G = [-R_{-m}; -R_m], h = [c_hat; -mu*].

Backends compared against the CPLEX solution:
  quadprog   Goldfarb-Idnani dual active set.  Finite, exact, deterministic;
             needs a positive-definite Hessian, which ours is (identity).
  piqp       proximal interior point (BSD-2).
  clarabel   interior point (Apache-2.0).
  highs      HiGHS's own active-set QP, for reference.
  dual_as    the m-dimensional Wolfe dual + KKT polish, iterated to a fixed point
             (pure numpy/scipy, no external solver).
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


def build_Gh(R_S, c_hat, mu_star):
    m = R_S.shape[0]
    G = np.vstack([-R_S[: m - 1, :], -R_S[m - 1, :][None, :]])
    h = np.concatenate([c_hat, [-mu_star]])
    return G, h


# --------------------------------------------------------------------------- #
def solve_quadprog(G, h):
    import quadprog
    m, r = G.shape
    P = np.eye(r)
    q = np.zeros(r)
    # quadprog: min 1/2 x'Px - q'x  s.t.  C' x >= b
    C = np.hstack([-G.T, np.eye(r)])            # r x (m + r)
    b = np.concatenate([-h, np.zeros(r)])
    sol = quadprog.solve_qp(P, q, C, b, 0)
    return np.asarray(sol[0])


def solve_piqp(G, h):
    import piqp
    from scipy import sparse
    m, r = G.shape
    P = sparse.identity(r, format="csc")
    c = np.zeros(r)
    # piqp: min 1/2 x'Px + c'x s.t. Ax = b, h_l <= Gx <= h_u, x_l <= x <= x_u
    solver = piqp.SparseSolver()
    solver.settings.verbose = False
    solver.settings.eps_abs = 1e-12
    solver.settings.eps_rel = 1e-12
    solver.setup(P, c, None, None, sparse.csc_matrix(G),
                 np.full(m, -np.inf), h, np.zeros(r), np.full(r, np.inf))
    solver.solve()
    return np.asarray(solver.result.x)


def solve_clarabel(G, h):
    import clarabel
    from scipy import sparse
    m, r = G.shape
    P = sparse.identity(r, format="csc")
    q = np.zeros(r)
    A = sparse.vstack([sparse.csc_matrix(G), -sparse.identity(r)], format="csc")
    b = np.concatenate([h, np.zeros(r)])
    cones = [clarabel.NonnegativeConeT(m + r)]
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = 1e-13
    st.tol_gap_rel = 1e-13
    st.tol_feas = 1e-13
    sol = clarabel.DefaultSolver(P, q, A, b, cones, st).solve()
    return np.asarray(sol.x)


def solve_dual_as(G, h, max_polish=6):
    """Wolfe dual (m variables) + repeated KKT polish on the identified sets."""
    m, r = G.shape
    Gt = G.T

    def negg(lam):
        w = np.maximum(-(Gt @ lam), 0.0)
        return 0.5 * w @ w + h @ lam, h - G @ w

    res = minimize(negg, np.zeros(m), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * m,
                   options={"maxiter": 800, "ftol": 1e-18, "gtol": 1e-14,
                            "maxcor": 20})
    lam = np.maximum(res.x, 0.0)
    w = np.maximum(-(Gt @ lam), 0.0)
    best = w
    best_err = np.inf
    for _ in range(max_polish):
        I = np.flatnonzero(lam > 0)
        J = np.flatnonzero(w > 0)
        if I.size == 0 or J.size == 0:
            break
        Gij = G[np.ix_(I, J)]
        try:
            lam_I = np.linalg.lstsq(Gij @ Gij.T, -h[I], rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        w_new = np.zeros(r)
        w_new[J] = -(Gij.T @ lam_I)
        if np.any(w_new < -1e-11) or np.any(lam_I < -1e-11):
            break
        w_new = np.maximum(w_new, 0.0)
        err = float(np.max(np.maximum(G @ w_new - h, 0.0)))
        if err < best_err:
            best_err, best = err, w_new
        lam_new = np.zeros(m)
        lam_new[I] = np.maximum(lam_I, 0.0)
        if np.allclose(lam_new, lam, rtol=0, atol=1e-14):
            break
        lam, w = lam_new, w_new
    return best if best_err < np.inf else w


# --------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("--r-grid", default="2,5,10,20,40,60,100")
ap.add_argument("--n-sym", type=int, default=30)
ap.add_argument("--n-c", type=int, default=10)
ap.add_argument("--out", default="tables/QP_backends2.csv")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
m = cfg["model"]["m"]

CANDIDATES = [("quadprog", solve_quadprog), ("piqp", solve_piqp),
              ("clarabel", solve_clarabel), ("dual_as", solve_dual_as)]
AVAILABLE = []
for name, fn in CANDIDATES:
    try:
        g = np.array([[-1.0, 0.5], [0.3, -0.8]])
        fn(g, np.array([0.4, 0.2]))
        AVAILABLE.append((name, fn))
    except Exception as exc:                        # noqa: BLE001
        print(f"backend {name} unavailable: {exc.__class__.__name__}: {exc}")
print("testing:", [n for n, _ in AVAILABLE], flush=True)

rows = []
for r in [int(x) for x in a.r_grid.split(",")]:
    for sid in range(a.n_sym):
        R_S = make_R_S(cfg, r, sid)
        ss = SymbiontSolver(R_S, cfg)
        for cid, c_H in enumerate(c_H_samples(cfg, sid, a.n_c)):
            scale = float(np.sum(np.abs(c_H)))
            ref = ss.solve(c_H)
            if ref.qp_path != "cplex":
                continue
            w_ref = ref.w_S / scale
            mu_star = float(R_S[m - 1] @ w_ref)
            G, h = build_Gh(R_S, c_H / scale, mu_star)
            nref = float(np.linalg.norm(w_ref))

            for name, fn in AVAILABLE:
                t0 = time.perf_counter()
                try:
                    w = fn(G, h)
                    ok = True
                except Exception:                    # noqa: BLE001
                    w, ok = None, False
                dt = time.perf_counter() - t0
                rec = {"r": r, "sid": sid, "cid": cid, "backend": name,
                       "ok": ok, "seconds": dt, "norm_ref": nref}
                if ok:
                    w = np.maximum(np.asarray(w, dtype=float), 0.0)
                    rec["max_dw"] = float(np.max(np.abs(w - w_ref)))
                    rec["dnorm"] = float(np.linalg.norm(w) - nref)
                    rec["infeas"] = float(np.max(np.maximum(G @ w - h, 0.0)))
                rows.append(rec)

            t0 = time.perf_counter()
            wh = ss._solve_qp_highs(c_H / scale, mu_star - 1e-9)
            rec = {"r": r, "sid": sid, "cid": cid, "backend": "highs",
                   "ok": wh is not None, "seconds": time.perf_counter() - t0,
                   "norm_ref": nref}
            if wh is not None:
                rec["max_dw"] = float(np.max(np.abs(wh - w_ref)))
                rec["dnorm"] = float(np.linalg.norm(wh) - nref)
                rec["infeas"] = float(np.max(np.maximum(G @ wh - h, 0.0)))
            rows.append(rec)
        ss.close()
    print("done r =", r, flush=True)

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
pd.set_option("display.width", 220)
print("\n=== vs CPLEX, pooled over r ===")
print(df.groupby("backend").agg(
    n=("ok", "size"), fail_rate=("ok", lambda s: 1 - s.mean()),
    max_dw=("max_dw", "max"), p999_dw=("max_dw", lambda s: s.quantile(0.999)),
    med_dw=("max_dw", "median"), max_infeas=("infeas", "max"),
    max_dnorm=("dnorm", "max"), ms_med=("seconds", lambda s: 1000 * s.median()),
    ms_p95=("seconds", lambda s: 1000 * s.quantile(0.95)),
).to_string(float_format=lambda v: f"{v:.3g}"))
print("\n=== failure rate by r ===")
print(df.pivot_table(index="r", columns="backend", values="ok",
                     aggfunc=lambda s: 1 - s.mean()).round(4).to_string())
print("\n=== max |w - w_cplex| by r ===")
print(df.pivot_table(index="r", columns="backend", values="max_dw",
                     aggfunc="max").to_string(float_format=lambda v: f"{v:.2e}"))
print(f"\nwrote {a.out}")
