"""V17 — the shared-tournament scheme (scheme="shared_world", tournament_scope="shared").

The robustness check only means something if the host and every symbiont of a
world really draw their modes from that world's single tournament.

A cone test (is every mode a non-negative combination of world reactions?) does
NOT discriminate: a strongly connected tournament reaches j -> i through a path,
so its edge cone is the whole plane 1^T x = 0.  Instead the tests use k = 1,
where each mode is kappa * v_a for one reaction and its direction is read off
directly (substrate = the negative entry, product = the positive entry).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from grhd.pipeline import get_host, make_R_S, world_pool


@pytest.fixture(scope="module")
def k1_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["model"]["k"] = 1
    # accept the first host draw: this test is about where modes come from,
    # not about the mu_H_0 window
    c["model"]["host_accept_lo"] = 0.0
    c["model"]["host_accept_hi"] = 1e9
    return c


def _world_edges(V: np.ndarray) -> set:
    return {(int(np.argmin(v)), int(np.argmax(v))) for v in V}


def _mode_edges(R: np.ndarray) -> list:
    assert ((R != 0).sum(axis=0) == 2).all()          # k = 1: one reaction per mode
    return [(int(np.argmin(q)), int(np.argmax(q))) for q in R.T]


def test_symbiont_modes_use_world_directions(k1_cfg):
    for world in (0, 1):
        E = _world_edges(world_pool(k1_cfg, world))
        for r in (1, 12, 60):
            R = make_R_S(k1_cfg, r, 3, scheme="shared_world", world_id=world)
            assert all(e in E for e in _mode_edges(R))


def test_paper_scheme_breaks_world_directions(k1_cfg):
    """Discrimination: per-individual symbionts reverse about half the pairs."""
    E = _world_edges(world_pool(k1_cfg, 0))
    R = make_R_S(k1_cfg, 60, 3, scheme="paper")
    frac = np.mean([e in E for e in _mode_edges(R)])
    assert 0.25 < frac < 0.75


def test_host_modes_use_world_directions(k1_cfg):
    hd, _ = get_host(k1_cfg, host_id=0, scheme="shared_world")
    E = _world_edges(world_pool(k1_cfg, 0))
    assert all(e in E for e in _mode_edges(hd.R_H))


def test_shared_world_host_accepted_at_production_settings(cfg):
    hd, _ = get_host(cfg, host_id=0, scheme="shared_world")
    lo, hi = cfg["model"]["host_accept_lo"], cfg["model"]["host_accept_hi"]
    assert hd.accepted and lo <= hd.mu_H_0 <= hi


def test_shared_world_is_deterministic_and_world_specific(cfg):
    a = make_R_S(cfg, 12, 5, scheme="shared_world", world_id=2)
    b = make_R_S(cfg, 12, 5, scheme="shared_world", world_id=2)
    c = make_R_S(cfg, 12, 5, scheme="shared_world", world_id=3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
