"""Does the mutualism fraction at small r depend on the host's mu_H_0, or on the threshold?

Question
--------
Yuki's earlier implementation saw almost no mutualism at r = 5 with a host whose
mu_H_0 was about 0.09; here the Fig.4 host has mu_H_0 = 0.052 and r = 5 looks
strongly mutualistic.  Two candidate explanations, both testable on data we have:
  (1) mu_H_0: the 20 A4 hosts span the whole acceptance window [0.05, 0.10], so
      the fraction at r = 5 can be regressed on mu_H_0 directly;
  (2) the threshold: "mutualistic" here is mu_H > mu_H_0 and mu_S > 1e-6, which
      counts arbitrarily small gains.  Stricter absolute and relative thresholds
      show how much of the fraction is made of tiny gains.
Pre-evolution (Level 2, random c^H) fractions are taken from
tables/A4_pre_level2_by_r.csv (criterion mu_H > mu_H_0 + 1e-3, mu_S > 1e-6).

Output
------
tables/MUT_vs_mu0_by_host.csv   one row per (scheme, unit, r)
tables/MUT_vs_mu0_spearman.csv  Spearman rho of each fraction against mu_H_0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CONS = ROOT / "data" / "consolidated"
R_SHOW = [3, 5, 8, 10, 15]
N_S = 100
COLS = ["r", "symbiont_id", "mu_H_0", "mu_H", "mu_S"]

CRITERIA = {
    "strict":    lambda d: (d.mu_H > d.mu_H_0) & (d.mu_S > 1e-6),
    "abs_1e-3":  lambda d: (d.mu_H > d.mu_H_0 + 1e-3) & (d.mu_S > 1e-6),
    "abs_1e-2":  lambda d: (d.mu_H > d.mu_H_0 + 1e-2) & (d.mu_S > 1e-6),
    "rel_10pct": lambda d: (d.mu_H > 1.1 * d.mu_H_0) & (d.mu_S > 1e-6),
    "rel_50pct": lambda d: (d.mu_H > 1.5 * d.mu_H_0) & (d.mu_S > 1e-6),
    "rel_100pct": lambda d: (d.mu_H > 2.0 * d.mu_H_0) & (d.mu_S > 1e-6),
}


def load(run: str) -> pd.DataFrame:
    d = pd.read_parquet(CONS / f"{run}_cells.parquet", columns=COLS)
    return d[(d.symbiont_id < N_S) & d.r.isin(R_SHOW)]


def main() -> None:
    parts = []
    for h in range(20):
        parts.append(load("post_paper" if h == 0 else f"post_paper_h{h}")
                     .assign(scheme="per_individual", unit=h))
    for w in range(20):
        parts.append(load(f"post_shared_world_w{w}").assign(scheme="shared", unit=w))
    d = pd.concat(parts, ignore_index=True)

    rows = []
    for (sch, u, r), g in d.groupby(["scheme", "unit", "r"]):
        rec = {"scheme": sch, "unit": u, "r": r, "mu_H_0": g.mu_H_0.iloc[0],
               "n": len(g), "mean_mu_H": g.mu_H.mean(),
               "median_gain_rel": float(np.median(g.mu_H / g.mu_H_0))}
        for k, f in CRITERIA.items():
            rec[f"frac_{k}"] = float(f(g).mean())
        rows.append(rec)
    t = pd.DataFrame(rows)

    pre = pd.read_csv(ROOT / "tables" / "A4_pre_level2_by_r.csv")
    pre = pre[pre.r.isin(R_SHOW)][["host_id", "r", "frac_mutualistic"]]
    t = t.merge(pre.rename(columns={"host_id": "unit", "frac_mutualistic": "frac_pre_1e-3"})
                .assign(scheme="per_individual"), on=["scheme", "unit", "r"], how="left")
    t.to_csv(ROOT / "tables" / "MUT_vs_mu0_by_host.csv", index=False)

    fr = [c for c in t.columns if c.startswith("frac_")] + ["mean_mu_H"]
    sp = []
    for (sch, r), g in t.groupby(["scheme", "r"]):
        for c in fr:
            x = g[["mu_H_0", c]].dropna()
            if len(x) < 5:
                continue
            rho, p = spearmanr(x.mu_H_0, x[c])
            sp.append({"scheme": sch, "r": r, "quantity": c, "rho": rho, "p": p,
                       "n_units": len(x), "mean_over_units": x[c].mean()})
    sp = pd.DataFrame(sp)
    sp.to_csv(ROOT / "tables" / "MUT_vs_mu0_spearman.csv", index=False)

    pd.set_option("display.width", 200)
    print("== per-individual hosts, r = 5, sorted by mu_H_0 (fractions) ==")
    show = ["unit", "mu_H_0", "frac_strict", "frac_abs_1e-2", "frac_rel_50pct",
            "frac_rel_100pct", "frac_pre_1e-3", "median_gain_rel"]
    print(t[(t.scheme == "per_individual") & (t.r == 5)].sort_values("mu_H_0")[show]
          .round(3).to_string(index=False))
    print("== Spearman rho vs mu_H_0 at r = 5 ==")
    print(sp[sp.r == 5][["scheme", "quantity", "rho", "p", "mean_over_units"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
