"""How often do host AND symbiont both grow well after evolution, with mu_H + mu_S ~ 1?

Question
--------
Yuki's earlier implementation rarely produced evolved pairs in which mu_H and
mu_S are both large (e.g. both >= 0.2) with mu_H + mu_S pinned near 1.  Here
they look common at r = 5.  Is that tied to the host's mu_H_0 (the Fig.4 host
has mu_H_0 = 0.052; Yuki's earlier host had about 0.09)?  The 20 A4 hosts span
the window [0.05, 0.10], so the fraction can be set against mu_H_0 directly.

Output
------
tables/JOINT_vs_mu0_by_host.csv   one row per (scheme, unit, r)
tables/JOINT_vs_mu0_spearman.csv  Spearman rho of each quantity against mu_H_0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CONS = ROOT / "data" / "consolidated"
R_SHOW = [1, 3, 5, 8, 10, 15, 25, 40]
N_S = 100
COLS = ["r", "symbiont_id", "mu_H_0", "mu_H", "mu_S"]


def load(run: str) -> pd.DataFrame:
    d = pd.read_parquet(CONS / f"{run}_cells.parquet", columns=COLS)
    return d[(d.symbiont_id < N_S) & d.r.isin(R_SHOW)]


def summarise(g: pd.DataFrame) -> dict:
    tot = g.mu_H + g.mu_S
    both = lambda t: float(((g.mu_H >= t) & (g.mu_S >= t)).mean())
    return {"mu_H_0": g.mu_H_0.iloc[0], "n": len(g),
            "frac_both_ge_0.1": both(0.1), "frac_both_ge_0.2": both(0.2),
            "frac_both_ge_0.3": both(0.3),
            "frac_sum_ge_0.9": float((tot >= 0.9).mean()),
            "frac_sum_ge_0.99": float((tot >= 0.99).mean()),
            "frac_both_ge_0.2_and_sum_ge_0.9": float(((g.mu_H >= 0.2) & (g.mu_S >= 0.2)
                                                      & (tot >= 0.9)).mean()),
            "mean_sum": float(tot.mean()), "max_sum": float(tot.max()),
            "mean_mu_S": float(g.mu_S.mean()),
            "frac_mu_S_ge_0.2": float((g.mu_S >= 0.2).mean())}


def main() -> None:
    parts = [load("post_paper" if h == 0 else f"post_paper_h{h}")
             .assign(scheme="per_individual", unit=h) for h in range(20)]
    parts += [load(f"post_shared_world_w{w}").assign(scheme="shared", unit=w)
              for w in range(20)]
    d = pd.concat(parts, ignore_index=True)

    t = pd.DataFrame([{"scheme": s, "unit": u, "r": r, **summarise(g)}
                      for (s, u, r), g in d.groupby(["scheme", "unit", "r"])])
    t.to_csv(ROOT / "tables" / "JOINT_vs_mu0_by_host.csv", index=False)

    qs = [c for c in t.columns if c.startswith(("frac_", "mean_"))]
    sp = []
    for (s, r), g in t.groupby(["scheme", "r"]):
        for c in qs:
            rho, p = spearmanr(g.mu_H_0, g[c]) if g[c].nunique() > 1 else (np.nan, np.nan)
            sp.append({"scheme": s, "r": r, "quantity": c, "rho": rho, "p": p,
                       "mean_over_units": g[c].mean(), "min": g[c].min(),
                       "max": g[c].max()})
    sp = pd.DataFrame(sp)
    sp.to_csv(ROOT / "tables" / "JOINT_vs_mu0_spearman.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"max(mu_H + mu_S) over all cells: {float((d.mu_H + d.mu_S).max()):.6f}")
    show = ["unit", "mu_H_0", "frac_both_ge_0.2", "frac_both_ge_0.2_and_sum_ge_0.9",
            "frac_sum_ge_0.9", "mean_mu_S"]
    print("== per-individual hosts, r = 5, sorted by mu_H_0 ==")
    print(t[(t.scheme == "per_individual") & (t.r == 5)].sort_values("mu_H_0")[show]
          .round(3).to_string(index=False))
    key = ["frac_both_ge_0.2", "frac_both_ge_0.2_and_sum_ge_0.9", "frac_sum_ge_0.9"]
    print("== by r: mean over units, and Spearman rho vs mu_H_0 ==")
    print(sp[sp.quantity.isin(key)].pivot_table(
        index=["scheme", "r"], columns="quantity", values=["mean_over_units", "rho"])
        .round(2).to_string())


if __name__ == "__main__":
    main()
