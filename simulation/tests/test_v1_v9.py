"""V1-V9 of 04_validation.md."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from grhd.analyze import analyze_cell, bounded_predicates, check_identities
from grhd.generate import c_E, generate_R, reaction_pool, sample_c_H, sample_host
from grhd.lp import HostSolver, SymbiontSolver, solve_host_once
from grhd.rng import derive


def _mk(cfg, r, seed, m=None, k=None):
    m = m or cfg["model"]["m"]
    k = k or cfg["model"]["k"]
    rng = np.random.default_rng(seed)
    V = reaction_pool(m, rng)
    return generate_R(m, r, k, V, rng)


# --------------------------------------------------------------------------- #
# V1 — tiny systems against exhaustive / brute-force reference
# --------------------------------------------------------------------------- #
def _brute_force_level1(R_S, c_H, n_grid=60):
    """Vertex enumeration of max R_m w s.t. -R_{-m} w <= c_H, w >= 0.

    With r variables and (m-1)+r constraints the optimum sits at a basic
    feasible solution: pick r active constraints, solve, keep feasible ones.
    """
    m, r = R_S.shape
    A = np.vstack([-R_S[:m - 1, :], -np.eye(r)])
    b = np.concatenate([c_H, np.zeros(r)])
    obj = R_S[m - 1, :]
    best = 0.0                                    # w = 0 is always feasible
    n = A.shape[0]
    for rows in itertools.combinations(range(n), r):
        M = A[list(rows), :]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        try:
            w = np.linalg.solve(M, b[list(rows)])
        except np.linalg.LinAlgError:
            continue
        if np.all(w >= -1e-9) and np.all(A @ w <= b + 1e-9):
            best = max(best, float(obj @ w))
    return best


@pytest.mark.parametrize("m,r,k", [(4, 2, 2), (4, 3, 3), (5, 2, 2), (5, 3, 3)])
def test_v1_level1_matches_vertex_enumeration(cfg, m, r, k):
    for seed in range(12):
        R_S = _mk(cfg, r, 1000 + seed, m=m, k=k)
        rng = np.random.default_rng(5000 + seed)
        c_H = sample_c_H(m, 1, rng)[0]
        got = SymbiontSolver(R_S, cfg).solve(c_H)
        ref = _brute_force_level1(R_S, c_H)
        assert got.status in ("optimal", "qp_fallback")
        # IPM with crossover off leaves a ~1e-9 interior residual, so the LP
        # value sits just *below* the true vertex optimum; both directions are
        # checked, the upper one tightly (an over-estimate would be a bug).
        assert got.mu_S <= ref + 1e-9
        assert got.mu_S == pytest.approx(ref, abs=1e-8, rel=1e-6)


def test_v1_level2_infeasible_is_detected(cfg):
    """A symbiont excreting into a metabolite the environment cannot supply can
    make Level 2 infeasible; make sure we see at least one and never crash."""
    m = cfg["model"]["m"]
    cE = c_E(m)
    seen_infeasible = 0
    seen_optimal = 0
    for seed in range(40):
        R_H = _mk(cfg, 3, 20_000 + seed)           # deliberately weak host
        hs = HostSolver(R_H, cfg)
        rng = np.random.default_rng(30_000 + seed)
        f_S = rng.normal(scale=0.05, size=m - 1)   # arbitrary net uptake vector
        res = hs.solve(f_S, cE)
        assert res.status in ("optimal", "infeasible")
        seen_infeasible += res.status == "infeasible"
        seen_optimal += res.status == "optimal"
    assert seen_infeasible > 0, "no infeasible Level 2 encountered"
    assert seen_optimal > 0


# --------------------------------------------------------------------------- #
# V2 — KKT of Level 1
# --------------------------------------------------------------------------- #
def test_v2_level1_kkt(cfg):
    m = cfg["model"]["m"]
    n_checked = 0
    for seed in range(60):
        r = int(np.random.default_rng(seed).integers(1, 41))
        R_S = _mk(cfg, r, 7_000 + seed)
        ss = SymbiontSolver(R_S, cfg)
        rng = np.random.default_rng(9_000 + seed)
        for c_H in sample_c_H(m, 4, rng):
            sol = ss.solve(c_H)
            if sol.status != "optimal":
                continue
            lam1, lam2, w = sol.lam1, sol.lam2, sol.w_S
            g = np.concatenate([c_H, [0.0]]) + R_S @ w
            g[m - 1] -= sol.mu_S
            assert lam1[m - 1] == pytest.approx(1.0)
            assert np.all(lam1 >= -1e-7) and np.all(lam2 >= -1e-7)
            assert np.all(g >= -1e-7) and np.all(w >= -1e-9)
            stat = R_S.T @ lam1 + lam2
            assert np.max(np.abs(stat)) < 1e-6
            assert np.max(np.abs(lam1 * g)) < 1e-7
            assert np.max(np.abs(lam2 * w)) < 1e-7
            n_checked += 1
    assert n_checked >= 200


# --------------------------------------------------------------------------- #
# V3 — balance identities and the b + u partition
# --------------------------------------------------------------------------- #
def test_v3_balances_and_partition(cfg):
    m = cfg["model"]["m"]
    cE = c_E(m)
    R_H = _mk(cfg, cfg["model"]["r_H"], 42)
    hs = HostSolver(R_H, cfg)
    tol = cfg["tolerances"]
    for seed in range(25):
        r = int(np.random.default_rng(seed).integers(1, 61))
        R_S = _mk(cfg, r, 11_000 + seed)
        ss = SymbiontSolver(R_S, cfg)
        for c_H in sample_c_H(m, 4, np.random.default_rng(13_000 + seed)):
            sol = ss.solve(c_H)
            assert np.allclose(sol.f_S, -R_S[:m - 1] @ sol.w_S, atol=1e-10)
            assert sol.mu_S == pytest.approx(float(R_S[m - 1] @ sol.w_S), abs=1e-12)
            bo, un = bounded_predicates(c_H, sol.f_S, tol["tol_abs"], tol["tol_rel"])
            assert not np.any(bo & un) and not np.any(~bo & ~un)
            assert int(bo.sum()) + int(un.sum()) == m - 1
            h = hs.solve(sol.f_S, cE)
            if h.status == "optimal":
                assert np.allclose(h.f_H, sol.f_S - R_H[:m - 1] @ h.w_H, atol=1e-9)
                assert np.all(h.f_H <= cE + 1e-7)


# --------------------------------------------------------------------------- #
# V4 — monotonicity in c^H
# --------------------------------------------------------------------------- #
def test_v4_monotone_in_cH(cfg):
    m = cfg["model"]["m"]
    n = 0
    for seed in range(25):
        r = int(np.random.default_rng(seed).integers(1, 41))
        R_S = _mk(cfg, r, 15_000 + seed)
        ss = SymbiontSolver(R_S, cfg)
        c_H = sample_c_H(m, 1, np.random.default_rng(17_000 + seed))[0]
        base = ss.solve(c_H).mu_S
        for i in range(m - 1):
            c2 = c_H.copy()
            c2[i] += 0.003
            assert ss.solve(c2).mu_S >= base - 1e-9
            n += 1
    assert n >= 200


# --------------------------------------------------------------------------- #
# V5 — insensitivity along unbounded directions
# --------------------------------------------------------------------------- #
def test_v5_unbounded_insensitivity(cfg):
    m = cfg["model"]["m"]
    tol = cfg["tolerances"]
    n = 0
    for seed in range(40):
        r = int(np.random.default_rng(seed).integers(2, 41))
        R_S = _mk(cfg, r, 19_000 + seed)
        ss = SymbiontSolver(R_S, cfg)
        c_H = sample_c_H(m, 1, np.random.default_rng(21_000 + seed))[0]
        sol = ss.solve(c_H)
        _, un = bounded_predicates(c_H, sol.f_S, tol["tol_abs"], tol["tol_rel"])
        for i in np.flatnonzero(un):
            c2 = c_H.copy()
            c2[i] += 1e-4 * float(np.sum(c_H))
            s2 = ss.solve(c2)
            assert s2.mu_S == pytest.approx(sol.mu_S, abs=1e-9)
            assert np.max(np.abs(s2.f_S - sol.f_S)) < 1e-8
            n += 1
    assert n >= 50


# --------------------------------------------------------------------------- #
# V6 — positive homogeneity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam", [0.1, 1.0, 10.0, 100.0])
def test_v6_scale_invariance(cfg, lam):
    m = cfg["model"]["m"]
    for seed in range(15):
        r = int(np.random.default_rng(seed).integers(2, 51))
        R_S = _mk(cfg, r, 23_000 + seed)
        ss = SymbiontSolver(R_S, cfg)
        c_H = sample_c_H(m, 1, np.random.default_rng(25_000 + seed))[0]
        a = ss.solve(c_H)
        b = ss.solve(lam * c_H)
        assert b.mu_S == pytest.approx(lam * a.mu_S, rel=1e-7, abs=1e-12)
        # absolute tolerances have to follow the solution scale: w is determined
        # to ~1e-10 relative, so comparing components at a fixed 1e-11 would be
        # asking for more precision than mu^S itself carries.  This is backend
        # independent (quadprog and CPLEX disagree by the same amount here).
        atol_f = 1e-9 * max(1.0, float(np.max(np.abs(lam * a.f_S))))
        atol_w = 1e-9 * max(1.0, float(np.max(np.abs(lam * a.w_S))))
        assert np.allclose(b.f_S, lam * a.f_S, rtol=1e-6, atol=atol_f)
        assert np.allclose(b.w_S, lam * a.w_S, rtol=1e-6, atol=atol_w)
        an_a = analyze_cell(R_S, c_H, a.f_S, a.w_S, a.lam2, cfg)
        an_b = analyze_cell(R_S, lam * c_H, b.f_S, b.w_S, b.lam2, cfg)
        assert (an_a.n_bounded, an_a.r_plus) == (an_b.n_bounded, an_b.r_plus)
        assert an_b.s == pytest.approx(an_a.s, rel=1e-6, abs=1e-9)


# --------------------------------------------------------------------------- #
# V7 — min-norm tie-break
# --------------------------------------------------------------------------- #
def test_v7_min_norm(cfg):
    m = cfg["model"]["m"]
    for seed in range(20):
        r = int(np.random.default_rng(seed).integers(5, 61))
        R_S = _mk(cfg, r, 27_000 + seed)
        ss = SymbiontSolver(R_S, cfg)
        c_H = sample_c_H(m, 1, np.random.default_rng(29_000 + seed))[0]
        sol = ss.solve(c_H)
        assert float(R_S[m - 1] @ sol.w_S) == pytest.approx(sol.mu_S, abs=1e-12)
        # any other optimal-growth feasible point must have >= norm
        rng = np.random.default_rng(31_000 + seed)
        import highspy
        for _ in range(3):
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("threads", 1)
            wvars = h.addVariables(r, lb=0.0, ub=1e4)
            A = -R_S[:m - 1, :]
            for i in range(m - 1):
                h.addConstr(sum(A[i, j] * wvars[j] for j in range(r)) <= c_H[i])
            # exactly the growth our QP achieved: w_S itself is feasible here,
            # so any slack would let the reference LP buy a smaller norm by
            # giving up growth, which is not what V7 is about
            h.addConstr(sum(R_S[m - 1, j] * wvars[j] for j in range(r))
                        >= sol.mu_S)
            cvec = rng.normal(size=r)
            h.minimize(sum(cvec[j] * wvars[j] for j in range(r)))
            if h.modelStatusToString(h.getModelStatus()) != "Optimal":
                continue
            wp = np.array(h.getSolution().col_value)
            assert np.linalg.norm(sol.w_S) <= np.linalg.norm(wp) * (1 + 1e-6) + 1e-9


def test_v7_uniqueness_across_paths(cfg):
    """Same cell solved through different solver settings gives the same w."""
    import copy
    m = cfg["model"]["m"]
    alt = copy.deepcopy(cfg)
    alt["solver"]["lp"] = {"solver": "simplex", "presolve": "on"}
    for seed in range(10):
        r = int(np.random.default_rng(seed).integers(5, 41))
        R_S = _mk(cfg, r, 33_000 + seed)
        c_H = sample_c_H(m, 1, np.random.default_rng(35_000 + seed))[0]
        a = SymbiontSolver(R_S, cfg).solve(c_H)
        b = SymbiontSolver(R_S, alt).solve(c_H)
        assert np.max(np.abs(a.w_S - b.w_S)) < 1e-7


# --------------------------------------------------------------------------- #
# V8 — permutation invariance (metabolite m fixed; c^E permuted too)
# --------------------------------------------------------------------------- #
def test_v8_permutation_invariance(cfg):
    m = cfg["model"]["m"]
    r_H = cfg["model"]["r_H"]
    R_H = _mk(cfg, r_H, 77)
    for seed in range(12):
        r = int(np.random.default_rng(seed).integers(2, 41))
        R_S = _mk(cfg, r, 37_000 + seed)
        c_H = sample_c_H(m, 1, np.random.default_rng(39_000 + seed))[0]
        cE = c_E(m)
        rng = np.random.default_rng(41_000 + seed)
        perm = rng.permutation(m - 1)
        pm = np.concatenate([perm, [m - 1]])
        colp = rng.permutation(r)

        a_s = SymbiontSolver(R_S, cfg).solve(c_H)
        a_h = HostSolver(R_H, cfg).solve(a_s.f_S, cE)
        an_a = analyze_cell(R_S, c_H, a_s.f_S, a_s.w_S, a_s.lam2, cfg)

        b_s = SymbiontSolver(R_S[pm][:, colp], cfg).solve(c_H[perm])
        b_h = HostSolver(R_H[pm], cfg).solve(b_s.f_S, cE[perm])
        an_b = analyze_cell(R_S[pm][:, colp], c_H[perm], b_s.f_S, b_s.w_S,
                            b_s.lam2, cfg)

        assert b_s.mu_S == pytest.approx(a_s.mu_S, abs=1e-10)
        assert np.allclose(b_s.w_S, a_s.w_S[colp], atol=1e-8)
        assert np.allclose(b_s.f_S, a_s.f_S[perm], atol=1e-9)
        assert (an_a.n_bounded, an_a.r_plus) == (an_b.n_bounded, an_b.r_plus)
        assert an_b.s == pytest.approx(an_a.s, rel=1e-6, abs=1e-9)
        if a_h.status == "optimal":
            assert b_h.status == "optimal"
            assert b_h.mu_H == pytest.approx(a_h.mu_H, abs=1e-9)


# --------------------------------------------------------------------------- #
# V9 — Level 3 lower bounds (needs CPLEX)
# --------------------------------------------------------------------------- #
cplex = pytest.importorskip("cplex", reason="CPLEX not installed")


def test_v9_level3_lower_bounds(cfg):
    from grhd.milp import EvolutionSolver
    from grhd.pipeline import get_host, make_R_S
    m = cfg["model"]["m"]
    hd, cE = get_host(cfg, 0, m)
    hs = HostSolver(hd.R_H, cfg)
    for r in (2, 5, 10, 20):
        for sid in range(3):
            R_S = make_R_S(cfg, r, sid)
            sol = EvolutionSolver(R_S, hd.R_H, cE, cfg).solve()
            assert np.isfinite(sol.mu_H)
            assert sol.mu_H >= hd.mu_H_0 - 1e-7
            ss = SymbiontSolver(R_S, cfg)
            best = -np.inf
            for c_H in sample_c_H(m, 40, np.random.default_rng(sid)):
                lp = ss.solve(c_H)
                h = hs.solve(lp.f_S, cE)
                if h.status == "optimal":
                    best = max(best, h.mu_H)
            assert sol.mu_H >= best - 1e-6
