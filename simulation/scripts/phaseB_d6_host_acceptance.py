"""A2-D6 — host rejection-sampling acceptance rate.

07_run2_workorder.md Phase B 2.4.  Run 1 accepted a single host in 9 attempts,
so n = 1 and the acceptance rate was unknown.  A4 needs 20 hosts, so the rate
has to be pinned down first.  Level 2 LP only, so this is cheap.

Reports, per (m, k):
  acceptance rate for mu_H_0 in [0.05, 0.10]
  where the rejected hosts fall (below vs above the window)
  structural differences between accepted and rejected R^H
and, at m = 10 / k = 4, how the rate depends on two 01_model_spec.md section 9
flags (tournament_scope, normalize_cH).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.generate import c_E, generate_R, n_pool, reaction_pool
from grhd.lp import HostSolver
from grhd.rng import derive

ap = argparse.ArgumentParser()
ap.add_argument("--m-grid", default="5,10,15,20,25")
ap.add_argument("--k-grid", default="2,4,8")
ap.add_argument("--n-try", type=int, default=200)
ap.add_argument("--tables", default="tables")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
LO = float(cfg["model"]["host_accept_lo"])
HI = float(cfg["model"]["host_accept_hi"])
R_H_FIXED = int(cfg["model"]["r_H"])
SEED = int(cfg["master_seed"])
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)


def reachable_from_cE(V_rows_used, m):
    """Metabolites reachable from M_1 through the reactions the host actually has."""
    adj = {i: set() for i in range(m)}
    for src, dst in V_rows_used:
        adj[src].add(dst)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in adj[u] - seen:
            seen.add(v)
            stack.append(v)
    return len(seen)


def host_attempts(m, k, n_try, conf, scope="per_individual", tag=""):
    cE = c_E(m, conf["model"]["c_E_first"])
    rows = []
    # stable across interpreter runs: Python's hash() is salted per process
    host_id = int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=4).digest(),
                             "little")
    rng = derive(SEED, "R_H", m=m, k=k, r_H=R_H_FIXED, host_id=host_id)
    shared_V = reaction_pool(m, rng) if scope == "shared" else None
    zero_f = np.zeros(m - 1)
    for t in range(n_try):
        V = shared_V if shared_V is not None else reaction_pool(m, rng)
        R_H = generate_R(m, R_H_FIXED, k, V, rng)
        res = HostSolver(R_H, conf).solve(zero_f, cE)
        mu0 = float(res.mu_H) if res.status == "optimal" else np.nan
        # directed reactions available in this individual's tournament
        pairs = [(int(np.argmin(v)), int(np.argmax(v))) for v in V]
        rows.append({
            "m": m, "k": k, "scope": scope, "tag": tag, "attempt": t,
            "mu_H_0": mu0, "status": res.status,
            "accepted": bool(np.isfinite(mu0) and LO <= mu0 <= HI),
            "below_window": bool(np.isfinite(mu0) and mu0 < LO),
            "above_window": bool(np.isfinite(mu0) and mu0 > HI),
            "row_rank": int(np.linalg.matrix_rank(R_H, tol=1e-10)),
            "n_reachable": reachable_from_cE(pairs, m),
            "nnz_biomass_row": int(np.sum(np.abs(R_H[m - 1, :]) > 1e-12)),
            "max_biomass_coef": float(np.max(R_H[m - 1, :])),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# main m x k grid
# --------------------------------------------------------------------------- #
frames = []
for m in [int(x) for x in a.m_grid.split(",")]:
    for k in [int(x) for x in a.k_grid.split(",")]:
        if k > n_pool(m):
            print(f"  skip m={m} k={k}: k > mC2 = {n_pool(m)}", flush=True)
            continue
        c = copy.deepcopy(cfg)
        c["model"]["m"] = m
        frames.append(host_attempts(m, k, a.n_try, c, tag=f"grid_m{m}_k{k}"))
        print(f"  done m={m} k={k}", flush=True)
att = pd.concat(frames, ignore_index=True)
att.to_parquet("data/derived/A2D6_host_attempts.parquet", index=False)

g = att.groupby(["m", "k"]).agg(
    n=("accepted", "size"),
    acceptance_rate=("accepted", "mean"),
    frac_below=("below_window", "mean"),
    frac_above=("above_window", "mean"),
    frac_infeasible=("status", lambda s: float((s != "optimal").mean())),
    mu_H_0_median=("mu_H_0", "median"),
    mu_H_0_q10=("mu_H_0", lambda s: float(np.nanquantile(s, .10))),
    mu_H_0_q90=("mu_H_0", lambda s: float(np.nanquantile(s, .90))),
    mu_H_0_max=("mu_H_0", "max"),
).reset_index()
g["expected_attempts_per_host"] = np.where(g["acceptance_rate"] > 0,
                                           1.0 / g["acceptance_rate"], np.inf)
g.to_csv(TAB / "A2_D6_acceptance.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== acceptance rate by (m, k), window [0.05, 0.10], r_H = 24 ===")
print(g.round(4).to_string(index=False))
print("\n=== acceptance rate matrix ===")
print(g.pivot(index="m", columns="k", values="acceptance_rate").round(3).to_string())
print("\n=== fraction BELOW the window ===")
print(g.pivot(index="m", columns="k", values="frac_below").round(3).to_string())
print("\n=== fraction ABOVE the window ===")
print(g.pivot(index="m", columns="k", values="frac_above").round(3).to_string())

# accepted vs rejected structure
struct = att.groupby(["m", "k", "accepted"]).agg(
    n=("row_rank", "size"), row_rank=("row_rank", "mean"),
    n_reachable=("n_reachable", "mean"),
    nnz_biomass_row=("nnz_biomass_row", "mean"),
    max_biomass_coef=("max_biomass_coef", "mean"),
    mu_H_0_mean=("mu_H_0", "mean")).reset_index()
struct.to_csv(TAB / "A2_D6_accepted_vs_rejected.csv", index=False)
print("\n=== accepted vs rejected structure (m=10) ===")
print(struct[struct.m == 10].round(3).to_string(index=False))

# --------------------------------------------------------------------------- #
# flag dependence at m = 10, k = 4
# --------------------------------------------------------------------------- #
flag_rows = []
base = copy.deepcopy(cfg); base["model"]["m"] = 10
variants = [
    ("tournament_scope=per_individual (default)", base, "per_individual"),
    ("tournament_scope=shared", base, "shared"),
]
alt = copy.deepcopy(base); alt["flags"]["normalize_cH"] = False
variants.append(("normalize_cH=false", alt, "per_individual"))

for label, conf, scope in variants:
    reps = []
    for rep in range(5):                        # 5 independent worlds x 200 tries
        d = host_attempts(10, 4, a.n_try, conf, scope=scope,
                          tag=f"flag_{label}_{rep}")
        reps.append(float(d["accepted"].mean()))
    flag_rows.append({"variant": label, "scope": scope,
                      "n_worlds": len(reps), "n_try_per_world": a.n_try,
                      "acceptance_mean": float(np.mean(reps)),
                      "acceptance_sd_between_worlds": float(np.std(reps, ddof=1)),
                      "acceptance_min": float(np.min(reps)),
                      "acceptance_max": float(np.max(reps))})
    print(f"  flag variant done: {label}", flush=True)
fl = pd.DataFrame(flag_rows)
fl.to_csv(TAB / "A2_D6_flag_dependence.csv", index=False)
print("\n=== flag dependence at m=10, k=4 (5 worlds x 200 attempts) ===")
print(fl.round(4).to_string(index=False))

json.dump({"window": [LO, HI], "r_H": R_H_FIXED, "n_try": a.n_try,
           "m_grid": a.m_grid, "k_grid": a.k_grid},
          open(TAB / "A2_D6_meta.json", "w"), indent=2)
print("\nwrote tables/A2_D6_*.csv")
