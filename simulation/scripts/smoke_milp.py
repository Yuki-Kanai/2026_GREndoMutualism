"""Quick MILP sanity run: V9/V11-style checks plus a random multistart lower bound."""
import sys
import time

import numpy as np

from grhd.cfgio import load_config
from grhd.generate import c_E, generate_R, reaction_pool, sample_host
from grhd.lp import HostSolver, SymbiontSolver, solve_host_once
from grhd.milp import EvolutionSolver
from grhd.rng import derive

cfg = load_config("config/base.yaml")
m, k, rH = cfg["model"]["m"], cfg["model"]["k"], cfg["model"]["r_H"]
cE = c_E(m)
rng = derive(cfg["master_seed"], "R_H", m=m, k=k, r_H=rH, host_id=0)
hd = sample_host(m, rH, k, rng, lambda R, f, c: solve_host_once(R, f, c, cfg), cE, 0.05, 0.10)
print("mu_H_0 =", hd.mu_H_0, "attempts", hd.n_attempts, flush=True)
hs = HostSolver(hd.R_H, cfg)

n_rand = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
for r in [2, 5, 10, 25, 50, 100]:
    rngS = derive(cfg["master_seed"], "R_S", m=m, k=k, r=r, scheme="paper", symbiont_id=0)
    V = reaction_pool(m, rngS)
    R_S = generate_R(m, r, k, V, rngS)
    es = EvolutionSolver(R_S, hd.R_H, cE, cfg)
    t0 = time.time()
    sol = es.solve()
    dt = time.time() - t0

    ss = SymbiontSolver(R_S, cfg)
    lp = ss.solve(sol.c_H)
    h2 = hs.solve(lp.f_S, cE)

    best = -1.0
    Cs = 4 * max(float(np.max(np.abs(sol.c_H))), hd.mu_H_0)
    rr = np.random.default_rng(0)
    t1 = time.time()
    for _ in range(n_rand):
        c = rr.random(m - 1) * Cs if rr.random() < 0.5 else 10 ** rr.uniform(-4, 0, m - 1) * Cs
        l = ss.solve(c)
        hh = hs.solve(l.f_S, cE)
        if hh.status == "optimal" and hh.mu_H > best:
            best = hh.mu_H
    nvars = sol.detail["n_vars"]
    print(
        "r=%3d  %6.2fs  status=%-10s nvar=%3d  muH=%.6f (LP re-solve %.6f)  "
        "muS=%.6f (LP %.6f)  rand_best=%.6f  l1 %.4g->%.4g  cmax=%s box=%s gap=%.2g  rand %.0fs"
        % (r, dt, sol.status, nvars, sol.mu_H, h2.mu_H, sol.mu_S, lp.mu_S, best,
           sol.l1_step1, sol.l1_step2, sol.flag_cmax_active, sol.flag_box_active,
           sol.mip_gap, time.time() - t1),
        flush=True,
    )
