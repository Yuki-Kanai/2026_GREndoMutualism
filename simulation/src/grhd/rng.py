"""Deterministic seed derivation (05_conventions.md §3).

Every generator is derived from the master seed through a namespaced, sorted
key so that results never depend on execution order or worker count (V14).
"""
from __future__ import annotations

import hashlib

import numpy as np

# Namespaces fixed by 05_conventions.md §3. Keep this table in sync with the doc.
NAMESPACE_FIELDS = {
    "R_S": ("m", "k", "r", "scheme", "symbiont_id"),
    "R_S_fixed": ("m", "n_rxn", "rep_id"),
    "R_S_fixed_split": ("m", "n_rxn", "rep_id"),
    "R_H": ("m", "k", "r_H", "host_id"),
    # shared-tournament robustness check (scheme="shared_world"): one tournament per
    # world, the host and every symbiont of that world draw their modes from it.
    "world": ("m", "world_id"),
    "R_H_world": ("m", "k", "r_H", "world_id"),
    "R_S_world": ("m", "k", "r", "world_id", "symbiont_id"),
    "c_H": ("m", "symbiont_id", "c_id"),
    "boot": ("table_id", "statistic"),
    "subsample": ("fig_id",),
}


def derive_seed(master_seed: int, namespace: str, **fields) -> int:
    """A 64-bit seed from the master seed, a namespace and the object's identity.

    E.g. namespace "R_S" with m=10, k=4, r=13, scheme="paper", symbiont_id=170
    hashes the text "20260916|R_S|k=4|m=10|r=13|scheme=paper|symbiont_id=170".
    The same object therefore always gets the same random numbers, whichever
    process, block or machine generates it.
    """
    key = f"{master_seed}|{namespace}|" + "|".join(
        f"{k}={v}" for k, v in sorted(fields.items())
    )
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


def derive(master_seed: int, namespace: str, **fields) -> np.random.Generator:
    """Return a Generator for one object. Fields must match NAMESPACE_FIELDS."""
    expected = NAMESPACE_FIELDS.get(namespace)
    if expected is not None and tuple(sorted(fields)) != tuple(sorted(expected)):
        raise ValueError(
            f"namespace {namespace!r} expects fields {sorted(expected)}, got {sorted(fields)}"
        )
    return np.random.default_rng(derive_seed(master_seed, namespace, **fields))
