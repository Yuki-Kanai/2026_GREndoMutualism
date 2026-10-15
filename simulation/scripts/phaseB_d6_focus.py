"""A2-D6, focused run at m = 10, k = 4 — the configuration A4 will actually use.

The grid run gave only 5 accepted hosts out of 200, too few to characterise the
accepted subset.  This draws many more attempts (Level 2 LP only, ~1 ms each)
to pin down the acceptance rate, the shape of the mu_H_0 distribution, where in
the window the accepted hosts land, and how accepted hosts differ structurally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.generate import c_E, generate_R, reaction_pool
from grhd.lp import HostSolver
from grhd.rng import derive

ap = argparse.ArgumentParser()
ap.add_argument("--n-try", type=int, default=6000)
ap.add_argument("--m", type=int, default=10)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--tables", default="tables")
ap.add_argument("--figures", default="figures")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
LO, HI = float(cfg["model"]["host_accept_lo"]), float(cfg["model"]["host_accept_hi"])
r_H = int(cfg["model"]["r_H"])
m, k = a.m, a.k
cE = c_E(m, cfg["model"]["c_E_first"])
TAB = Path(a.tables)

host_id = int.from_bytes(hashlib.blake2b(f"D6focus_m{m}_k{k}".encode(),
                                         digest_size=4).digest(), "little")
rng = derive(int(cfg["master_seed"]), "R_H", m=m, k=k, r_H=r_H, host_id=host_id)
zero_f = np.zeros(m - 1)

rows = []
for t in range(a.n_try):
    V = reaction_pool(m, rng)
    R_H = generate_R(m, r_H, k, V, rng)
    res = HostSolver(R_H, cfg).solve(zero_f, cE)
    mu0 = max(float(res.mu_H), 0.0) if res.status == "optimal" else np.nan
    pairs = [(int(np.argmin(v)), int(np.argmax(v))) for v in V]
    adj = {i: set() for i in range(m)}
    for s_, d_ in pairs:
        adj[s_].add(d_)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in adj[u] - seen:
            seen.add(v); stack.append(v)
    rows.append({
        "attempt": t, "mu_H_0": mu0, "status": res.status,
        "accepted": bool(np.isfinite(mu0) and LO <= mu0 <= HI),
        "biomass_reachable": bool((m - 1) in seen),
        "n_reachable": len(seen),
        "nnz_biomass_row": int(np.sum(np.abs(R_H[m - 1, :]) > 1e-12)),
        "sum_biomass_pos": float(np.sum(np.clip(R_H[m - 1, :], 0, None))),
        "max_biomass_coef": float(np.max(R_H[m - 1, :])),
        "row_rank": int(np.linalg.matrix_rank(R_H, tol=1e-10)),
    })
d = pd.DataFrame(rows)
d.to_parquet("data/derived/A2D6_focus_attempts.parquet", index=False)

n = len(d)
acc = d[d["accepted"]]
p = len(acc) / n
se = float(np.sqrt(p * (1 - p) / n))
print(f"=== m={m}, k={k}, r_H={r_H}, window [{LO}, {HI}] ===")
print(f"attempts {n}, accepted {len(acc)}  ->  rate {p:.4f} "
      f"(Wald 95% CI [{max(p - 1.96 * se, 0):.4f}, {p + 1.96 * se:.4f}]), "
      f"expected attempts per host {1 / p if p else np.inf:.1f}")

edges = [-1e-12, 1e-9, 1e-3, 0.01, LO, HI, 0.2, 0.5, 0.9, 1 - 1e-9, 1.0 + 1e-9]
lab = ["= 0", "(0, 1e-3)", "[1e-3, 0.01)", "[0.01, 0.05)", "[0.05, 0.10] ACCEPT",
       "(0.10, 0.2]", "(0.2, 0.5]", "(0.5, 0.9]", "(0.9, 1)", "= 1"]
cut = pd.cut(d["mu_H_0"], bins=edges, labels=lab, include_lowest=True)
dist = cut.value_counts().reindex(lab).to_frame("n")
dist["frac"] = dist["n"] / n
print("\n=== distribution of mu_H_0 ===")
print(dist.round(4).to_string())

print("\n=== where inside the window do accepted hosts land? ===")
if len(acc):
    q = acc["mu_H_0"].describe(percentiles=[.1, .25, .5, .75, .9])
    print(q.round(5).to_string())
    print(f"  lower half [0.05, 0.075) : {int((acc['mu_H_0'] < 0.075).sum())}"
          f" / {len(acc)}")

struct = d.groupby("accepted").agg(
    n=("mu_H_0", "size"), mu_H_0_mean=("mu_H_0", "mean"),
    biomass_reachable=("biomass_reachable", "mean"),
    n_reachable=("n_reachable", "mean"),
    nnz_biomass_row=("nnz_biomass_row", "mean"),
    sum_biomass_pos=("sum_biomass_pos", "mean"),
    max_biomass_coef=("max_biomass_coef", "mean"),
    row_rank=("row_rank", "mean")).reset_index()
print("\n=== accepted vs rejected structure ===")
print(struct.round(4).to_string(index=False))

# split the rejected side so the comparison is not dominated by mu_H_0 = 0
d["group"] = np.where(d["accepted"], "accepted",
                      np.where(d["mu_H_0"] < LO, "below", "above"))
g3 = d.groupby("group").agg(
    n=("mu_H_0", "size"), mu_H_0_median=("mu_H_0", "median"),
    biomass_reachable=("biomass_reachable", "mean"),
    nnz_biomass_row=("nnz_biomass_row", "mean"),
    sum_biomass_pos=("sum_biomass_pos", "mean"),
    max_biomass_coef=("max_biomass_coef", "mean")).reset_index()
print("\n=== below / accepted / above ===")
print(g3.round(4).to_string(index=False))

dist.to_csv(TAB / "A2_D6_focus_distribution.csv")
struct.to_csv(TAB / "A2_D6_focus_structure.csv", index=False)
g3.to_csv(TAB / "A2_D6_focus_three_groups.csv", index=False)
json.dump({"m": m, "k": k, "r_H": r_H, "window": [LO, HI], "n_try": n,
           "n_accepted": int(len(acc)), "acceptance_rate": p,
           "acceptance_ci": [max(p - 1.96 * se, 0), p + 1.96 * se],
           "expected_attempts_per_host": (1 / p if p else None),
           "frac_mu_H_0_zero": float((d["mu_H_0"] <= 1e-9).mean()),
           "frac_mu_H_0_one": float((d["mu_H_0"] >= 1 - 1e-9).mean())},
          open(TAB / "A2_D6_focus_meta.json", "w"), indent=2)

# --------------------------------------------------------------------------- #
from grhd import plotstyle as ps                                # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

ps.use()
fig, axes = plt.subplots(1, 2, figsize=(ps.COL15, 2.0))
fig.subplots_adjust(wspace=0.34)

ax = axes[0]
ax.hist(d["mu_H_0"].dropna(), bins=np.linspace(0, 1, 51), color=ps.C["scatter"],
        edgecolor="none")
ax.axvspan(LO, HI, color=ps.C["mean2"], alpha=0.30, lw=0)
ax.set_yscale("log")
ps.tidy(ax, r"$\mu^{H}_{0}$ of a randomly generated host",
        f"count (of {n} attempts)")
ax.annotate("acceptance\nwindow", xy=(HI, ax.get_ylim()[1] * 0.25),
            xytext=(0.20, ax.get_ylim()[1] * 0.25), fontsize=6,
            color=ps.C["mean2"], va="center",
            arrowprops=dict(arrowstyle="->", color=ps.C["mean2"], lw=0.7))
ps.panel_label(ax, "A", dx=-0.20)

ax = axes[1]
if len(acc):
    ax.hist(acc["mu_H_0"], bins=np.linspace(LO, HI, 21), color=ps.C["mean2"],
            edgecolor="none")
ax.axvline(0.051711, color=ps.C["ref"], lw=0.9, ls=(0, (3.5, 2)))
ax.annotate("run-1 host", xy=(0.0518, ax.get_ylim()[1] * 0.9),
            xytext=(0.058, ax.get_ylim()[1] * 0.9), fontsize=6,
            color=ps.C["ref"], va="center",
            arrowprops=dict(arrowstyle="->", color=ps.C["ref"], lw=0.7))
ps.tidy(ax, r"$\mu^{H}_{0}$ inside the window", f"count (of {len(acc)} accepted)")
ps.panel_label(ax, "B", dx=-0.20)
print("\nwrote", ps.save(fig, "A2_D6_host_acceptance", Path(a.figures),
                         {"tables": ["A2_D6_acceptance.csv",
                                     "A2_D6_focus_distribution.csv"],
                          "m": m, "k": k, "r_H": r_H, "n_try": n,
                          "acceptance_rate": p}))
