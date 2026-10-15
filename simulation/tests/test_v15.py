"""V15 — min-norm QP backend equivalence (04_validation.md).

The min-norm solution is the solution of a strictly convex QP, so it is unique:
every correct backend must return the same answer.  This is a purely
mathematical check — it depends on neither the MILP nor the LP settings.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from grhd.analyze import analyze_cell
from grhd.lp import SymbiontSolver
from grhd.pipeline import c_H_samples, make_R_S

R_GRID = [2, 5, 10, 20, 40, 100]
N_SYM = 4
N_C = 5


def _available(cfg):
    out = []
    for name in ("quadprog", "cplex", "highs"):
        try:
            c = copy.deepcopy(cfg)
            c["solver"].setdefault("qp", {})["backends"] = [name]
            R = make_R_S(c, 3, 0)
            s = SymbiontSolver(R, c).solve(c_H_samples(c, 0, 1)[0])
            if s.qp_path == name:
                out.append(name)
        except Exception:                               # noqa: BLE001
            pass
    return out


def test_v15_backend_equivalence(cfg, tmp_path):
    backends = _available(cfg)
    assert "quadprog" in backends, (
        "quadprog is the default min-norm backend (01_model_spec.md A17) "
        "and must be installed")
    if len(backends) < 2:
        pytest.skip(f"only one backend available: {backends}")

    cfgs = {}
    for name in backends:
        c = copy.deepcopy(cfg)
        c["solver"].setdefault("qp", {})["backends"] = [name]
        cfgs[name] = c

    rows = []
    for r in R_GRID:
        for sid in range(N_SYM):
            R_S = make_R_S(cfg, r, sid)
            CH = c_H_samples(cfg, sid, N_C)
            per = {}
            for name, c in cfgs.items():
                ss = SymbiontSolver(R_S, c)
                recs = []
                for c_H in CH:
                    sol = ss.solve(c_H)
                    an = analyze_cell(R_S, c_H, sol.f_S, sol.w_S, sol.lam2, c)
                    recs.append(dict(
                        b=an.n_bounded, r_plus=an.r_plus, ub=an.ub,
                        n_excreted=an.n_excreted, s=an.s, mu_S=sol.mu_S,
                        norm_w=float(np.linalg.norm(sol.w_S)),
                        qp_path=sol.qp_path))
                ss.close()
                per[name] = recs
            for i in range(N_C):
                base = per["quadprog"][i]
                for name in backends:
                    if name == "quadprog":
                        continue
                    o = per[name][i]
                    rows.append({
                        "r": r, "sid": sid, "cid": i, "backend": name,
                        "b_same": base["b"] == o["b"],
                        "rplus_same": base["r_plus"] == o["r_plus"],
                        "ub_same": base["ub"] == o["ub"],
                        "nexc_same": base["n_excreted"] == o["n_excreted"],
                        "d_s": abs(base["s"] - o["s"]),
                        "d_mu_S": abs(base["mu_S"] - o["mu_S"]),
                        "d_norm_w": abs(base["norm_w"] - o["norm_w"]),
                        "qp_path": o["qp_path"],
                    })

    df = pd.DataFrame(rows)
    # never silently fall back: lp_vertex would mean the tie-break is broken
    assert not (df["qp_path"] == "lp_vertex").any(), (
        "a backend fell through to the LP vertex; the min-norm tie-break is broken")

    for col in ("b_same", "rplus_same", "ub_same", "nexc_same"):
        bad = df[~df[col]]
        assert bad.empty, (
            f"{col}: {len(bad)} cells disagree between backends\n"
            f"{bad.head(10).to_string(index=False)}")

    # s is a function of the discrete sets and R^S only, so it must match to
    # round-off; mu_S and ||w|| may differ by the solvers' own accuracy
    assert df["d_s"].max() < 1e-9, df.nlargest(5, "d_s").to_string(index=False)
    assert df["d_mu_S"].max() < 1e-9
    assert df["d_norm_w"].max() < 1e-8

    print(f"\nV15: {len(df)} comparisons over backends {backends}")
    print(df.groupby("backend").agg(
        n=("d_s", "size"), max_d_s=("d_s", "max"),
        max_d_mu_S=("d_mu_S", "max"), max_d_norm_w=("d_norm_w", "max"),
    ).to_string())
