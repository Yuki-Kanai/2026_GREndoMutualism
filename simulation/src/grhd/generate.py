"""Reaction pools, control matrices and host rejection sampling (01_model_spec.md §1, §2, §6)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def n_pool(m: int) -> int:
    """Number of possible one-to-one conversions between m metabolites, mC2."""
    return m * (m - 1) // 2


def reaction_pool(m: int, rng: np.random.Generator) -> np.ndarray:
    """One tournament: for each unordered pair {i,j} pick a direction uniformly.

    Returns V of shape (mC2, m) with -1 at the substrate and +1 at the product,
    so that 1^T v_a = 0 for every reaction (01_model_spec.md §10).
    """
    pairs = np.array(list(combinations(range(m), 2)), dtype=np.int64)
    flip = rng.random(len(pairs)) < 0.5
    src = np.where(flip, pairs[:, 1], pairs[:, 0])
    dst = np.where(flip, pairs[:, 0], pairs[:, 1])
    V = np.zeros((len(pairs), m), dtype=np.float64)
    V[np.arange(len(pairs)), src] = -1.0
    V[np.arange(len(pairs)), dst] = +1.0
    return V


def generate_R(m: int, r: int, k: int, V: np.ndarray, rng: np.random.Generator,
               return_idx: bool = False):
    """R = [q_1 ... q_r], q_j = sum_{a in K_j} kappa_ja v_a.

    K_j: k reactions drawn without replacement inside a mode, independently
    across modes (flag sample_within_mode="without_replacement").

    With return_idx the per-mode reaction index arrays are returned as well,
    so callers can count the effective repertoire (A1 / 03 A1).  The draw order
    is unchanged, so R is bit-identical either way.
    """
    P = V.shape[0]
    if k > P:
        raise ValueError(f"k={k} exceeds the reaction pool size mC2={P} (m={m})")
    R = np.zeros((m, r), dtype=np.float64)
    modes = []
    for j in range(r):
        idx = rng.choice(P, size=k, replace=False)
        kappa = rng.random(k)
        R[:, j] = kappa @ V[idx]
        if return_idx:
            modes.append(idx)
    return (R, modes) if return_idx else R


# --------------------------------------------------------------------------- #
# A1 fixed-repertoire control (03_additional_analyses.md A1)
# --------------------------------------------------------------------------- #
@dataclass
class Replicate:
    """Everything a fixed-repertoire replicate keeps fixed as r varies.

    The whole point of the matched design is that the tournament, the reaction
    subset Omega, the weight of each reaction and the order of Omega are drawn
    ONCE per replicate and never redrawn when r changes.  Only the partition of
    the (fixed) ordered list into r blocks changes.  Hence the RNG key must NOT
    contain r -- see make_R_S_meta() in pipeline.py.
    """
    V: np.ndarray        # (mC2, m) tournament
    omega: np.ndarray    # (n_rxn,) reaction indices into V, in the fixed order
    kappa: np.ndarray    # (n_rxn,) one weight per reaction, aligned with omega


def draw_replicate(m: int, n_rxn: int, rng: np.random.Generator) -> Replicate:
    """Draw one fixed-repertoire replicate: a tournament, n_rxn of its reactions
    with one weight each, and a fixed order (Fig. S2D).  generate_R_fixed then
    cuts it into r modes."""
    P = n_pool(m)
    if n_rxn > P:
        raise ValueError(f"n_rxn={n_rxn} exceeds the reaction pool size mC2={P} (m={m})")
    V = reaction_pool(m, rng)
    omega = rng.choice(P, size=n_rxn, replace=False)   # the fixed reaction set
    kappa = rng.random(n_rxn)                          # one weight per reaction
    perm = rng.permutation(n_rxn)                      # the fixed order
    return Replicate(V=V, omega=omega[perm], kappa=kappa[perm])


def generate_R_fixed(m: int, r: int, rep: Replicate):
    """Split the replicate's ordered reaction list into r near-equal blocks.

    Block sizes differ by at most 1 (np.array_split).  q_j = sum_{a in block_j}
    kappa_a v_a, so summing the r modes gives the same vector for every r: the
    repertoire and the weights are identical, only the granularity changes.
    """
    n_rxn = rep.omega.size
    if not 1 <= r <= n_rxn:
        raise ValueError(f"fixed_repertoire needs 1 <= r <= n_rxn ({r}, {n_rxn})")
    R = np.zeros((m, r), dtype=np.float64)
    blocks = np.array_split(np.arange(n_rxn), r)
    for j, blk in enumerate(blocks):
        R[:, j] = rep.kappa[blk] @ rep.V[rep.omega[blk]]
    return R, [rep.omega[blk] for blk in blocks]


def sample_c_H(m: int, n: int, rng: np.random.Generator, l1: float = 0.01) -> np.ndarray:
    """c^H = l1 * x / sum(x), x_i ~ U(0,1).  NOT Dirichlet(1,..,1) — see §6.3."""
    x = rng.random((n, m - 1))
    return l1 * x / x.sum(axis=1, keepdims=True)


def c_E(m: int, first: float = 1.0) -> np.ndarray:
    """The external environment c^E = (first, 0, ..., 0), length m-1 (SI S2)."""
    v = np.zeros(m - 1, dtype=np.float64)
    v[0] = first
    return v


@dataclass
class HostDraw:
    """Result of sample_host."""
    R_H: np.ndarray      # (m, r_H) host regulatory matrix
    mu_H_0: float        # host growth without a symbiont, in c^E
    n_attempts: int      # draws needed until mu_H_0 fell in [lo, hi]
    accepted: bool       # False if max_attempts ran out (R_H is then the last draw)


def sample_host(
    m: int,
    r_H: int,
    k: int,
    rng: np.random.Generator,
    solve_host_fn,
    cE: np.ndarray,
    lo: float = 0.05,
    hi: float = 0.10,
    max_attempts: int = 200_000,
    shared_pool: np.ndarray | None = None,
) -> HostDraw:
    """Rejection sampling on mu_H_0 in [lo, hi] (§6.1).

    A fresh tournament is drawn for every attempt (flag
    tournament_scope="per_individual", the default).  Pass `shared_pool` to use
    one fixed tournament instead (tournament_scope="shared", §9-A1); this is a
    diagnostic switch used by A2-D6, not the production setting.

    Never raises on exhaustion: returns accepted=False so the caller can report
    host_sampling_failed.
    """
    zero_f_S = np.zeros(m - 1)
    last = None
    for attempt in range(1, max_attempts + 1):
        V = reaction_pool(m, rng) if shared_pool is None else shared_pool
        R_H = generate_R(m, r_H, k, V, rng)
        res = solve_host_fn(R_H, zero_f_S, cE)
        mu0 = res.mu_H
        last = (R_H, mu0)
        if res.status == "optimal" and lo <= mu0 <= hi:
            return HostDraw(R_H=R_H, mu_H_0=float(mu0), n_attempts=attempt, accepted=True)
    R_H, mu0 = last
    return HostDraw(R_H=R_H, mu_H_0=float(mu0), n_attempts=max_attempts, accepted=False)
