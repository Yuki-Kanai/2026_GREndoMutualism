"""Min-norm QP robustness / timing check across scales and solver variants."""
import copy
import time

import numpy as np

from grhd.cfgio import load_config
from grhd.generate import generate_R, reaction_pool, sample_c_H
from grhd.lp import SymbiontSolver

cfg = load_config("config/base.yaml")
m = cfg["model"]["m"]


def mk(r, seed):
    rng = np.random.default_rng(seed)
    V = reaction_pool(m, rng)
    return generate_R(m, r, cfg["model"]["k"], V, rng)


CASES = []
for seed in range(12):
    r = int(np.random.default_rng(seed).integers(2, 101))
    CASES.append((seed, r, mk(r, 23_000 + seed),
                  sample_c_H(m, 8, np.random.default_rng(25_000 + seed))))

LAMS = (0.01, 0.1, 1.0, 10.0, 100.0)

VARIANTS = {
    "cplex+normalize (default)": {},
    "cplex, no normalize": {"flags": {"normalize_cH": False}},
}
for label, patch in VARIANTS.items():
    c2 = copy.deepcopy(cfg)
    for kk, vv in patch.items():
        c2[kk].update(vv)
    paths = {}
    n = 0
    t0 = time.time()
    worst_hom = 0.0
    for seed, r, R_S, CH in CASES:
        ss = SymbiontSolver(R_S, c2)
        for c0 in CH:
            base = ss.solve(c0)
            for lam in LAMS:
                s = ss.solve(lam * c0)
                paths[s.qp_path] = paths.get(s.qp_path, 0) + 1
                n += 1
                if base.mu_S > 0:
                    worst_hom = max(worst_hom,
                                    abs(s.mu_S - lam * base.mu_S) / (lam * base.mu_S))
    dt = time.time() - t0
    print(f"{label:28s} n={n:5d}  {dt / n * 1000:5.2f} ms/cell  paths={paths}  "
          f"worst rel homogeneity error={worst_hom:.2e}", flush=True)
