"""V16 — the A1 generation schemes (03_additional_analyses.md A1).

The fixed-repertoire arm is only a valid control if the reaction set, the
weights and the order are genuinely held fixed while r varies.  These tests
check that invariance directly rather than trusting the seed derivation.
"""
from __future__ import annotations

import numpy as np
import pytest

from grhd.generate import draw_replicate, generate_R_fixed, n_pool
from grhd.pipeline import make_R_S, make_R_S_meta
from grhd.rng import derive

N_RXN = [12, 24, 40]


# --- V16-a: the repertoire and the weights do not move when r moves --------- #
@pytest.mark.parametrize("n_rxn", N_RXN)
def test_matched_design_sum_of_modes_is_r_invariant(cfg, n_rxn):
    """sum_j q_j = sum_{a in Omega} kappa_a v_a, which cannot depend on r.

    This is the single strongest check: if any of the tournament, Omega, kappa
    or the permutation were redrawn per r, this sum would change.  It is exact
    in exact arithmetic; the residual here is only the summation order (r = 1
    adds the n_rxn terms in one dot product, larger r adds partial sums).
    """
    ref = None
    for r in range(1, n_rxn + 1):
        R, _ = make_R_S_meta(cfg, r, 7, "fixed_repertoire", n_rxn=n_rxn)
        tot = R.sum(axis=1)
        if ref is None:
            ref = tot
        else:
            assert np.allclose(tot, ref, atol=1e-12, rtol=0), f"r={r}"


def test_fixed_repertoire_key_cannot_contain_r(cfg):
    """05_conventions.md section 3: namespace R_S_fixed has fields (m, n_rxn, rep_id).

    The strict field check in rng.derive is what actually enforces the matched
    design, so make sure it is really strict.
    """
    with pytest.raises(ValueError):
        derive(cfg["master_seed"], "R_S_fixed", m=10, n_rxn=12, rep_id=0, r=3)


@pytest.mark.parametrize("n_rxn", N_RXN)
def test_matched_design_repertoire_is_exactly_n_rxn(cfg, n_rxn):
    """Every reaction of Omega appears in exactly one mode, for every r."""
    for r in range(1, n_rxn + 1):
        _, meta = make_R_S_meta(cfg, r, 3, "fixed_repertoire", n_rxn=n_rxn)
        assert meta["n_rxn_effective"] == n_rxn
        assert meta["n_rxn"] == n_rxn
        assert meta["k"] is None                      # k is undefined here (A1 §5)
        assert meta["mode_size_max"] - meta["mode_size_min"] <= 1
        assert abs(meta["mode_size_mean"] * r - n_rxn) < 1e-9


@pytest.mark.parametrize("n_rxn", N_RXN)
def test_matched_design_partition_is_a_refinement_chain(cfg, n_rxn):
    """r = n_rxn puts one reaction in each mode: R is then kappa_a v_a column-wise.

    Combined with the sum test above this pins the construction down: the
    finest level is the bare weighted reactions, and coarser levels are sums of
    those same columns.
    """
    rng = derive(cfg["master_seed"], "R_S_fixed", m=cfg["model"]["m"],
                 n_rxn=n_rxn, rep_id=3)
    rep = draw_replicate(cfg["model"]["m"], n_rxn, rng)
    finest, _ = generate_R_fixed(cfg["model"]["m"], n_rxn, rep)
    for r in (1, 2, 5, n_rxn // 2, n_rxn):
        R, blocks = generate_R_fixed(cfg["model"]["m"], r, rep)
        pos = 0
        for j, blk in enumerate(blocks):
            want = finest[:, pos:pos + len(blk)].sum(axis=1)
            assert np.allclose(R[:, j], want, atol=1e-12)
            pos += len(blk)
        assert pos == n_rxn


def test_matched_design_replicates_differ(cfg):
    a = make_R_S(cfg, 6, 0, "fixed_repertoire", n_rxn=24)
    b = make_R_S(cfg, 6, 1, "fixed_repertoire", n_rxn=24)
    assert not np.allclose(a, b)


def test_fixed_repertoire_rejects_r_above_n_rxn(cfg):
    with pytest.raises(ValueError):
        make_R_S(cfg, 13, 0, "fixed_repertoire", n_rxn=12)


# --- V16-b: k_sweep is the paper generator with a different k --------------- #
def test_k_sweep_at_k4_reproduces_the_paper_arm(cfg):
    """So the k = 4 arm of the sweep does not have to be recomputed."""
    for r in (1, 3, 10, 40):
        assert np.array_equal(make_R_S(cfg, r, 5, "k_sweep", k=4),
                              make_R_S(cfg, r, 5, "paper"))


@pytest.mark.parametrize("k", [2, 4, 8, 16])
def test_k_sweep_mode_sizes_and_repertoire(cfg, k):
    r = 10
    _, meta = make_R_S_meta(cfg, r, 2, "k_sweep", k=k)
    assert meta["k"] == k
    assert meta["n_rxn"] is None
    assert meta["mode_size_min"] == meta["mode_size_max"] == k
    # modes draw without replacement inside a mode but independently across
    # modes, so the union is at most k*r and at most the whole pool
    assert k <= meta["n_rxn_effective"] <= min(k * r, n_pool(cfg["model"]["m"]))


def test_k_sweep_rejects_k_above_pool(cfg):
    with pytest.raises(ValueError):
        make_R_S(cfg, 3, 0, "k_sweep", k=n_pool(cfg["model"]["m"]) + 1)


# --- V16-c: the paper arm is untouched -------------------------------------- #
def test_paper_scheme_unchanged_by_the_refactor(cfg):
    """make_R_S must still be reproducible and independent of the new args."""
    a = make_R_S(cfg, 7, 3)
    b = make_R_S(cfg, 7, 3, "paper", cfg["model"]["m"], cfg["model"]["k"])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, make_R_S(cfg, 7, 4))
