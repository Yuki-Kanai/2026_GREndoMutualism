"""Reproduce the V2/V4/V5/V6/V7/V8 failures and print the actual residuals."""
import numpy as np

from grhd.analyze import analyze_cell, bounded_predicates
from grhd.cfgio import load_config
from grhd.generate import generate_R, reaction_pool, sample_c_H
from grhd.lp import SymbiontSolver

cfg = load_config("config/base.yaml")
m = cfg["model"]["m"]


def mk(r, seed, k=None):
    rng = np.random.default_rng(seed)
    V = reaction_pool(m, rng)
    return generate_R(m, r, k or cfg["model"]["k"], V, rng)


print("== V6 scale invariance ==")
for seed in range(4):
    r = int(np.random.default_rng(seed).integers(2, 51))
    R_S = mk(r, 23_000 + seed)
    ss = SymbiontSolver(R_S, cfg)
    c_H = sample_c_H(m, 1, np.random.default_rng(25_000 + seed))[0]
    a = ss.solve(c_H)
    for lam in (0.1, 1.0, 10.0, 100.0):
        b = ss.solve(lam * c_H)
        # fresh solver object, to separate warm-start effects from real ones
        c = SymbiontSolver(R_S, cfg).solve(lam * c_H)
        print(f" seed={seed} r={r:3d} lam={lam:6g} "
              f"dmu_reuse={b.mu_S - lam * a.mu_S:+.3e} "
              f"dmu_fresh={c.mu_S - lam * a.mu_S:+.3e} "
              f"dw_reuse={np.max(np.abs(b.w_S - lam * a.w_S)):.3e} "
              f"dw_fresh={np.max(np.abs(c.w_S - lam * a.w_S)):.3e} "
              f"st={b.status}/{c.status}")

print("== V2 KKT residuals ==")
worst = {}
for seed in range(10):
    r = int(np.random.default_rng(seed).integers(1, 41))
    R_S = mk(r, 7_000 + seed)
    ss = SymbiontSolver(R_S, cfg)
    for c_H in sample_c_H(m, 4, np.random.default_rng(9_000 + seed)):
        sol = ss.solve(c_H)
        if sol.status != "optimal":
            continue
        lam1, lam2, w = sol.lam1, sol.lam2, sol.w_S
        g = np.concatenate([c_H, [0.0]]) + R_S @ w
        g[m - 1] -= sol.mu_S
        vals = {
            "lam1_neg": -min(0.0, float(lam1.min())),
            "lam2_neg": -min(0.0, float(lam2.min())),
            "g_neg": -min(0.0, float(g.min())),
            "stationarity": float(np.max(np.abs(R_S.T @ lam1 + lam2))),
            "compl_1g": float(np.max(np.abs(lam1 * g))),
            "compl_2w": float(np.max(np.abs(lam2 * w))),
        }
        for kk, vv in vals.items():
            worst[kk] = max(worst.get(kk, 0.0), vv)
for kk, vv in worst.items():
    print(f"  worst {kk:14s} {vv:.3e}")

print("== V4 monotonicity ==")
worst = 0.0
for seed in range(10):
    r = int(np.random.default_rng(seed).integers(1, 41))
    R_S = mk(r, 15_000 + seed)
    ss = SymbiontSolver(R_S, cfg)
    c_H = sample_c_H(m, 1, np.random.default_rng(17_000 + seed))[0]
    base = ss.solve(c_H).mu_S
    for i in range(m - 1):
        c2 = c_H.copy()
        c2[i] += 0.003
        worst = min(worst, ss.solve(c2).mu_S - base)
print(f"  most negative increment: {worst:.3e}")

print("== V5 unbounded insensitivity ==")
tol = cfg["tolerances"]
wm, wf = 0.0, 0.0
for seed in range(10):
    r = int(np.random.default_rng(seed).integers(2, 41))
    R_S = mk(r, 19_000 + seed)
    ss = SymbiontSolver(R_S, cfg)
    c_H = sample_c_H(m, 1, np.random.default_rng(21_000 + seed))[0]
    sol = ss.solve(c_H)
    _, un = bounded_predicates(c_H, sol.f_S, tol["tol_abs"], tol["tol_rel"])
    for i in np.flatnonzero(un):
        c2 = c_H.copy()
        c2[i] += 1e-4 * float(np.sum(c_H))
        s2 = ss.solve(c2)
        wm = max(wm, abs(s2.mu_S - sol.mu_S))
        wf = max(wf, float(np.max(np.abs(s2.f_S - sol.f_S))))
print(f"  worst |dmu|={wm:.3e}  worst |df|={wf:.3e}")
