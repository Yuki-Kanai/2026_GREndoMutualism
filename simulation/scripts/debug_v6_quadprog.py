"""Locate the V6 scale-invariance mismatch under the quadprog backend."""
import copy

import numpy as np

from grhd.cfgio import load_config
from grhd.generate import generate_R, reaction_pool, sample_c_H
from grhd.lp import SymbiontSolver

cfg = load_config("config/base.yaml")
m = cfg["model"]["m"]
alt = copy.deepcopy(cfg)
alt["solver"]["qp"]["backends"] = ["cplex"]


def mk(r, seed):
    rng = np.random.default_rng(seed)
    return generate_R(m, r, cfg["model"]["k"], reaction_pool(m, rng), rng)


for lam in (0.1, 10.0, 100.0):
    print(f"--- lam = {lam} ---")
    for seed in range(15):
        r = int(np.random.default_rng(seed).integers(2, 51))
        R_S = mk(r, 23_000 + seed)
        c_H = sample_c_H(m, 1, np.random.default_rng(25_000 + seed))[0]
        for tag, conf in (("quadprog", cfg), ("cplex", alt)):
            ss = SymbiontSolver(R_S, conf)
            a = ss.solve(c_H)
            b = ss.solve(lam * c_H)
            ss.close()
            d = np.abs(b.w_S - lam * a.w_S)
            ref = np.abs(lam * a.w_S)
            j = int(np.argmax(d - 1e-6 * ref))
            worst_rel = float(np.max(d / np.maximum(ref, 1e-300)))
            ok = np.allclose(b.w_S, lam * a.w_S, rtol=1e-6, atol=1e-11)
            if not ok or tag == "quadprog" and seed < 2:
                # also: is the normalised right-hand side bit-identical?
                s1 = float(np.sum(np.abs(c_H)))
                s2 = float(np.sum(np.abs(lam * c_H)))
                same_chat = np.array_equal(c_H / s1, (lam * c_H) / s2)
                print(f"  seed={seed:2d} r={r:3d} {tag:8s} allclose={ok} "
                      f"max|d|={d.max():.3e} at j={j} "
                      f"(w={lam * a.w_S[j]:.6e} vs {b.w_S[j]:.6e}) "
                      f"worst_rel_on_nonzero={worst_rel:.3e} "
                      f"c_hat bit-identical={same_chat}")
