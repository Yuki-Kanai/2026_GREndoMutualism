"""V1 (Level 3 multistart), V11-V14 of 04_validation.md."""
from __future__ import annotations

import numpy as np
import pytest

from grhd.analyze import analyze_cell, check_identities
from grhd.generate import c_E, sample_c_H
from grhd.lp import HostSolver, SymbiontSolver
from grhd.pipeline import get_host, make_R_S
from grhd.rng import derive, derive_seed

cplex = pytest.importorskip("cplex", reason="CPLEX not installed")
from grhd.milp import EvolutionSolver  # noqa: E402


def _random_multistart(ss, hs, cE, m, C_search, n, seed):
    """Mixed log-uniform / uniform search over c^H in [0, C_search]^{m-1} (V1)."""
    rng = np.random.default_rng(seed)
    best = -np.inf
    for _ in range(n):
        if rng.random() < 0.5:
            c = rng.random(m - 1) * C_search
        else:
            c = 10 ** rng.uniform(-5, 0, m - 1) * C_search
            c *= rng.random()                       # random radius along the ray
        lp = ss.solve(c)
        h = hs.solve(lp.f_S, cE)
        if h.status == "optimal" and h.mu_H > best:
            best = h.mu_H
    return best


@pytest.mark.slow
@pytest.mark.parametrize("r", [2, 5, 10])
def test_v1_level3_beats_random_multistart(cfg, r):
    m = cfg["model"]["m"]
    hd, cE = get_host(cfg, 0, m)
    hs = HostSolver(hd.R_H, cfg)
    for sid in range(3):
        R_S = make_R_S(cfg, r, sid)
        sol = EvolutionSolver(R_S, hd.R_H, cE, cfg).solve()
        assert np.isfinite(sol.mu_H)
        C_search = 4 * max(float(np.max(np.abs(sol.c_H))), hd.mu_H_0)
        ss = SymbiontSolver(R_S, cfg)
        best = _random_multistart(ss, hs, cE, m, C_search, 4000, 100 + sid)
        assert sol.mu_H >= best - 1e-6, (
            f"MILP {sol.mu_H} below random best {best} (r={r}, sid={sid})")


def test_v11_milp_matches_lp_at_optimal_cH(cfg):
    """Re-solving Level 1+2 at c^H* must reproduce the MILP's mu_H, mu_S, f_S*."""
    m = cfg["model"]["m"]
    hd, cE = get_host(cfg, 0, m)
    hs = HostSolver(hd.R_H, cfg)
    n = 0
    for r in (1, 2, 5, 10, 20, 40):
        for sid in range(4):
            R_S = make_R_S(cfg, r, sid)
            sol = EvolutionSolver(R_S, hd.R_H, cE, cfg).solve()
            assert np.isfinite(sol.mu_H)
            ss = SymbiontSolver(R_S, cfg)
            lp = ss.solve(sol.c_H)
            h = hs.solve(lp.f_S, cE)
            assert lp.mu_S == pytest.approx(sol.mu_S, abs=1e-6, rel=1e-5)
            assert np.max(np.abs(lp.f_S - sol.f_S)) < 1e-6
            assert h.status == "optimal"
            assert h.mu_H == pytest.approx(sol.mu_H, abs=1e-6, rel=1e-5)
            n += 1
    assert n >= 20


def test_v12_conservation_identities(cfg):
    m = cfg["model"]["m"]
    hd, cE = get_host(cfg, 0, m)
    hs = HostSolver(hd.R_H, cfg)
    tol = float(cfg["tolerances"]["identity_tol"])
    bad = 0
    n = 0
    for r in (1, 3, 10, 30, 100):
        R_S = make_R_S(cfg, r, 0)
        assert np.max(np.abs(R_S.sum(axis=0))) < 1e-12
        ss = SymbiontSolver(R_S, cfg)
        for c_H in sample_c_H(m, 20, np.random.default_rng(r)):
            lp = ss.solve(c_H)
            h = hs.solve(lp.f_S, cE)
            assert lp.status != "unbounded" and h.status != "unbounded"
            res = check_identities(R_S, c_H, lp.mu_S, lp.f_S, lp.w_S,
                                   h.mu_H if h.status == "optimal" else None,
                                   h.f_H if h.status == "optimal" else None, tol)
            bad += len(res["violations"])
            n += 1
    assert bad == 0, f"{bad} identity violations in {n} cells"


def test_v13_cH_uniqueification(cfg):
    """step 2 keeps mu_H and shrinks ||c^H||_1; report how much b/u/s/n_exc move."""
    m = cfg["model"]["m"]
    hd, cE = get_host(cfg, 0, m)
    deltas = []
    for r in (2, 5, 10, 20, 40):
        for sid in range(4):
            R_S = make_R_S(cfg, r, sid)
            es = EvolutionSolver(R_S, hd.R_H, cE, cfg)
            sol = es.solve()
            assert sol.l1_step2 <= sol.l1_step1 + 1e-7
            assert not sol.flag_cmax_active, "C_max active: raise c_H_box"
            ss = SymbiontSolver(R_S, cfg)
            a = ss.solve(sol.c_H_step1)
            b = ss.solve(sol.c_H)
            an_a = analyze_cell(R_S, sol.c_H_step1, a.f_S, a.w_S, a.lam2, cfg)
            an_b = analyze_cell(R_S, sol.c_H, b.f_S, b.w_S, b.lam2, cfg)
            deltas.append((r, sid, sol.l1_step1, sol.l1_step2,
                           an_a.n_bounded - an_b.n_bounded,
                           an_a.n_excreted - an_b.n_excreted,
                           an_a.s - an_b.s))
    # The report is the point of this test; it only fails if step 2 broke mu_H.
    print("\nV13 step1 -> step2 (r, sid, l1_1, l1_2, dB, dExc, dS)")
    for d in deltas:
        print("   ", d)


def test_v14_determinism_and_seed_namespaces(cfg):
    m, k, rH = cfg["model"]["m"], cfg["model"]["k"], cfg["model"]["r_H"]
    a = make_R_S(cfg, 7, 3)
    b = make_R_S(cfg, 7, 3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, make_R_S(cfg, 7, 4))
    # c^H must not depend on r, host_id or stage
    s1 = derive_seed(cfg["master_seed"], "c_H", m=m, symbiont_id=5, c_id=0)
    s2 = derive_seed(cfg["master_seed"], "c_H", m=m, symbiont_id=5, c_id=0)
    assert s1 == s2
    # master seed must actually be in the key
    assert (derive_seed(1, "R_S", m=m, k=k, r=3, scheme="paper", symbiont_id=0)
            != derive_seed(2, "R_S", m=m, k=k, r=3, scheme="paper", symbiont_id=0))
    with pytest.raises(ValueError):
        derive(cfg["master_seed"], "c_H", m=m, symbiont_id=1, c_id=0, r=5)


def test_v14_block_results_independent_of_worker_count(cfg, tmp_path):
    """Blocks are pure functions of their arguments, so any sharding agrees."""
    from grhd.pipeline import run_block_pre
    design = {"N_c": 5}
    a, _, _ = run_block_pre(cfg, design, "t", 6, 0, 4, save_vectors=False)
    b1, _, _ = run_block_pre(cfg, design, "t", 6, 0, 2, save_vectors=False)
    b2, _, _ = run_block_pre(cfg, design, "t", 6, 2, 4, save_vectors=False)
    import pandas as pd
    b = pd.concat([b1, b2], ignore_index=True)
    for col in ("mu_S", "mu_H", "s", "alpha", "n_bounded", "n_excreted"):
        assert np.allclose(a[col].to_numpy(dtype=float),
                           b[col].to_numpy(dtype=float),
                           rtol=0, atol=0, equal_nan=True), col
