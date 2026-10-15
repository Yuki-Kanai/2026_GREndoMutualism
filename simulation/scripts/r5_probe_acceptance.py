"""R5 §2 — host acceptance-rate probe for the two designs (LP only).

08_r5_two_designs.md §2.  This runs BEFORE any R5 MILP.  Its purpose is not to
decide whether a design is feasible but to size the budget: how many attempts
does one accepted host cost, in each design and at each m?

  R5-A  r^H = round(2.4 m)   ratio fixed at the value used at m = 10 (r^H = 24)
  R5-B  r^H = 24             the original setting

Acceptance window mu_H_0 in [0.05, 0.10] is the same in both designs, and k = 4.
Cells with zero acceptances at n_try are re-run at n_extend (§2-2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.generate import c_E, generate_R, n_pool, reaction_pool
from grhd.lp import HostSolver
from grhd.rng import derive

ap = argparse.ArgumentParser()
ap.add_argument("--m-grid", default="5,8,10,12,15,18,20,22,25")
ap.add_argument("--n-try", type=int, default=20000)
ap.add_argument("--n-extend", type=int, default=200000)
ap.add_argument("--workers", type=int, default=18)
ap.add_argument("--tables", default="tables")
ap.add_argument("--figures", default="figures")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
LO = float(cfg["model"]["host_accept_lo"])
HI = float(cfg["model"]["host_accept_hi"])
K = int(cfg["model"]["k"])
SEED = int(cfg["master_seed"])
RATIO = 24.0 / 10.0                     # = 2.4, fixed by m = 10, r^H = 24
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)


def r_H_of(design: str, m: int) -> int:
    """R5-A keeps r^H/m fixed at the m = 10 value; R5-B keeps r^H itself fixed."""
    return int(round(RATIO * m)) if design == "A" else 24


def probe(design: str, m: int, n_try: int):
    """Generate n_try hosts and record mu_H_0.  Level 2 LP only."""
    r_H = r_H_of(design, m)
    cE = c_E(m, cfg["model"]["c_E_first"])
    zero_f = np.zeros(m - 1)
    # host_id offset keeps the probe stream disjoint from the production hosts;
    # the same seed is reused when a cell is extended, so the first n_try draws
    # of the extended run are identical to the short run
    rng = derive(SEED, "R_H", m=m, k=K, r_H=r_H, host_id=900_000)
    mu = np.empty(n_try)
    for t in range(n_try):
        V = reaction_pool(m, rng)
        R_H = generate_R(m, r_H, K, V, rng)
        res = HostSolver(R_H, cfg).solve(zero_f, cE)
        mu[t] = max(float(res.mu_H), 0.0) if res.status == "optimal" else np.nan
    return {"design": design, "m": m, "r_H": r_H, "n_try": n_try, "mu": mu}


def summarise(rec):
    mu = rec["mu"]
    n = mu.size
    fin = np.isfinite(mu)
    acc = fin & (mu >= LO) & (mu <= HI)
    p = float(acc.sum()) / n
    se = np.sqrt(max(p * (1 - p), 1e-12) / n)
    # Wilson interval: the Wald interval is useless when p is 0 or tiny
    z = 1.96
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    inw = mu[acc]
    return {
        "design": rec["design"], "m": rec["m"], "r_H": rec["r_H"], "n_try": n,
        "n_accepted": int(acc.sum()), "acceptance_rate": p,
        "ci_lo": max(centre - half, 0.0), "ci_hi": centre + half,
        "expected_attempts_per_host": (1.0 / p) if p > 0 else np.inf,
        "attempts_for_10_hosts": (10.0 / p) if p > 0 else np.inf,
        "frac_exactly_0": float(np.mean(fin & (mu <= 1e-9))),
        "frac_exactly_1": float(np.mean(fin & (mu >= 1 - 1e-9))),
        "frac_middle": float(np.mean(fin & (mu > 1e-9) & (mu < 1 - 1e-9))),
        "frac_below_window": float(np.mean(fin & (mu < LO))),
        "frac_above_window": float(np.mean(fin & (mu > HI))),
        "frac_positive": float(np.mean(fin & (mu > 1e-9))),
        "mu_H_0_max": float(np.nanmax(mu)),
        "mu_H_0_p99": float(np.nanquantile(mu, 0.99)),
        "mu_H_0_p999": float(np.nanquantile(mu, 0.999)),
        "in_window_mean": float(inw.mean()) if inw.size else np.nan,
        "in_window_median": float(np.median(inw)) if inw.size else np.nan,
        "in_window_frac_lower_half": float(np.mean(inw < 0.075)) if inw.size else np.nan,
        "n_infeasible": int((~fin).sum()),
    }


CELLS = [(d, m) for d in ("A", "B") for m in [int(x) for x in a.m_grid.split(",")]]
CELLS = [(d, m) for d, m in CELLS if K <= n_pool(m)]
print(f"{len(CELLS)} cells x {a.n_try} attempts", flush=True)

from joblib import Parallel, delayed                            # noqa: E402

recs = Parallel(n_jobs=a.workers, backend="loky")(
    delayed(probe)(d, m, a.n_try) for d, m in CELLS)
rows = [summarise(r) for r in recs]
mu_store = {(r["design"], r["m"]): r["mu"] for r in recs}

# extend the cells that accepted nothing (section 2-2)
zero_cells = [(r["design"], r["m"]) for r in rows if r["n_accepted"] == 0]
if zero_cells:
    print(f"extending {len(zero_cells)} zero-acceptance cells to {a.n_extend}: "
          f"{zero_cells}", flush=True)
    ext = Parallel(n_jobs=min(a.workers, len(zero_cells)), backend="loky")(
        delayed(probe)(d, m, a.n_extend) for d, m in zero_cells)
    by = {(r["design"], r["m"]): r for r in ext}
    rows = [summarise(by[(r["design"], r["m"])])
            if (r["design"], r["m"]) in by else r for r in rows]
    mu_store.update({(r["design"], r["m"]): r["mu"] for r in ext})

df = pd.DataFrame(rows).sort_values(["design", "m"]).reset_index(drop=True)
df.to_csv(TAB / "R5_probe_acceptance.csv", index=False)

long = pd.concat([pd.DataFrame({"design": d, "m": m, "mu_H_0": v})
                  for (d, m), v in mu_store.items()], ignore_index=True)
long.to_parquet("data/derived/R5_probe_mu_H_0.parquet", index=False)

pd.set_option("display.width", 250)
print("\n=== acceptance probe ===")
print(df[["design", "m", "r_H", "n_try", "n_accepted", "acceptance_rate",
          "ci_lo", "ci_hi", "attempts_for_10_hosts", "frac_below_window",
          "frac_above_window", "mu_H_0_max"]].round(6).to_string(index=False))
print("\n=== shape of the mu_H_0 distribution ===")
print(df[["design", "m", "r_H", "frac_exactly_0", "frac_middle", "frac_exactly_1",
          "frac_positive", "mu_H_0_p99"]].round(5).to_string(index=False))
print("\n=== where accepted hosts sit inside the window ===")
print(df[["design", "m", "n_accepted", "in_window_mean", "in_window_median",
          "in_window_frac_lower_half"]].round(5).to_string(index=False))

A = df[df.design == "A"]
print(f"\nR5-A acceptance across m: min {A.acceptance_rate.min():.5f}, "
      f"max {A.acceptance_rate.max():.5f}, "
      f"ratio max/min {A.acceptance_rate.max() / max(A.acceptance_rate.min(), 1e-12):.2f}")
B = df[df.design == "B"]
print(f"R5-B acceptance across m: min {B.acceptance_rate.min():.5f}, "
      f"max {B.acceptance_rate.max():.5f}")
print("\ncells still at zero acceptance:",
      [(r.design, int(r.m), int(r.n_try)) for r in df.itertuples()
       if r.n_accepted == 0] or "none")

json.dump({"window": [LO, HI], "k": K, "ratio": RATIO, "n_try": a.n_try,
           "n_extend": a.n_extend,
           "m_grid": [int(x) for x in a.m_grid.split(",")],
           "r_H_rule_A": "round(2.4 * m)", "r_H_B": 24},
          open(TAB / "R5_probe_meta.json", "w"), indent=2)

# --------------------------------------------------------------------------- #
from grhd import plots, plotstyle as ps                          # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402

ps.use()
fig, axes = plt.subplots(1, 3, figsize=(ps.COL2, 2.05))
fig.subplots_adjust(wspace=0.42)
COL = {"A": ps.C["mean"], "B": ps.C["mean2"]}
LAB = {"A": r"R5-A  $r^{H}=2.4\,m$", "B": r"R5-B  $r^{H}=24$"}

ax = axes[0]
for d in ("A", "B"):
    g = df[df.design == d].sort_values("m")
    ax.errorbar(g["m"], g["acceptance_rate"],
                yerr=[g["acceptance_rate"] - g["ci_lo"],
                      g["ci_hi"] - g["acceptance_rate"]],
                color=COL[d], marker="o", markersize=2.6, lw=1.1, capsize=1.6,
                elinewidth=0.7, label=LAB[d])
ax.set_yscale("symlog", linthresh=1e-5)
ax.axhspan(0, 1e-5, color="#eeeeee", zorder=0)
ax.legend(loc="best", fontsize=5.8)
ps.tidy(ax, "Number of metabolites, $m$", "host acceptance rate")
ps.panel_label(ax, "A", dx=-0.30)

ax = axes[1]
for d in ("A", "B"):
    g = df[df.design == d].sort_values("m")
    ax.plot(g["m"], g["frac_exactly_0"], color=COL[d], marker="o", markersize=2.4,
            lw=1.1, label=f"{d}: $\\mu^{{H}}_{{0}}=0$")
    ax.plot(g["m"], g["frac_exactly_1"], color=COL[d], marker="s", markersize=2.4,
            lw=1.1, ls=(0, (3, 1.5)), label=f"{d}: $\\mu^{{H}}_{{0}}=1$")
ax.set_ylim(-0.03, 1.03)
ax.legend(loc="center right", fontsize=5.2)
ps.tidy(ax, "Number of metabolites, $m$", "fraction of generated hosts")
ps.panel_label(ax, "B", dx=-0.30)

ax = axes[2]
for d in ("A", "B"):
    g = df[df.design == d].sort_values("m")
    ax.plot(g["m"], g["frac_below_window"], color=COL[d], marker="o",
            markersize=2.4, lw=1.1, label=f"{d}: below window")
    ax.plot(g["m"], g["frac_above_window"], color=COL[d], marker="^",
            markersize=2.4, lw=1.1, ls=(0, (3, 1.5)), label=f"{d}: above window")
ax.set_ylim(-0.03, 1.03)
ax.legend(loc="center right", fontsize=5.2)
ps.tidy(ax, "Number of metabolites, $m$", "fraction of generated hosts")
ps.panel_label(ax, "C", dx=-0.30)
print("\nwrote", ps.save(fig, "R5_probe_acceptance", Path(a.figures),
                         {"tables": ["R5_probe_acceptance.csv"],
                          "n_try": a.n_try, "n_extend": a.n_extend,
                          "window": [LO, HI], "k": K,
                          "r_H_rule_A": "round(2.4 m)", "r_H_B": 24}))
