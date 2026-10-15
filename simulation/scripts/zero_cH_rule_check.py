"""A14: how much of the post-evolution `b` curve is the zero-c^H convention?

After host evolution, step 2 minimises ||c^H||_1, so components the symbiont does
not consume are driven to exactly zero.  A zero component has slack
c^H_i - f^S*_i = 0 and therefore counts as *bounded* under the default rule
`bounded_if_zero_flux`.  This script separates the three kinds of component and
recomputes b under both conventions, pre and post.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.analyze import bounded_predicates
from grhd.cfgio import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--post", default="data/raw/post_paper")
ap.add_argument("--pre", default="data/raw/pre_paper")
ap.add_argument("--out", default="tables/A14_zero_cH_rule.csv")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
tol = cfg["tolerances"]
TA, TR = float(tol["tol_abs"]), float(tol["tol_rel"])


def classify(files, stage, limit_per_r=None):
    rows = []
    for f in sorted(files):
        v = pd.read_parquet(f, columns=["r", "symbiont_id", "c_H", "f_S_star"])
        if limit_per_r is not None:
            v = v.groupby("r", group_keys=False).head(limit_per_r)
        for _, rec in v.iterrows():
            c = np.asarray(rec["c_H"], dtype=float)
            fs = np.asarray(rec["f_S_star"], dtype=float)
            b_def, u_def = bounded_predicates(c, fs, TA, TR, "bounded_if_zero_flux")
            b_alt, u_alt = bounded_predicates(c, fs, TA, TR, "always_unbounded")
            zero = np.abs(c) <= TA
            excreting = zero & (fs < -TA)
            rows.append({
                "stage": stage, "r": int(rec["r"]),
                "symbiont_id": int(rec["symbiont_id"]),
                "n_offered": int((~zero).sum()),          # c^H_i > 0
                "n_zero_offer": int(zero.sum()),          # c^H_i == 0
                "n_zero_and_excreting": int(excreting.sum()),
                "n_zero_and_no_flux": int((zero & ~excreting).sum()),
                # b split into "really saturated" vs "counted bounded because the
                # host offers nothing and the symbiont emits nothing"
                "b_default": int(b_def.sum()),
                "b_alt": int(b_alt.sum()),
                "b_saturated_positive_offer": int((b_def & ~zero).sum()),
                "u_default": int(u_def.sum()),
                "u_alt": int(u_alt.sum()),
            })
    return pd.DataFrame(rows)


post = classify(Path(a.post).glob("vectors_*.parquet"), "post")
pre = classify(Path(a.pre).glob("vectors_*.parquet"), "pre", limit_per_r=300)
df = pd.concat([post, pre], ignore_index=True)

g = (df.groupby(["stage", "r"])
       .agg(n=("b_default", "size"),
            b_default=("b_default", "mean"),
            b_alt=("b_alt", "mean"),
            b_saturated_positive_offer=("b_saturated_positive_offer", "mean"),
            u_default=("u_default", "mean"),
            u_alt=("u_alt", "mean"),
            n_offered=("n_offered", "mean"),
            n_zero_offer=("n_zero_offer", "mean"),
            n_zero_and_excreting=("n_zero_and_excreting", "mean"),
            n_zero_and_no_flux=("n_zero_and_no_flux", "mean"))
       .reset_index())
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
g.round(4).to_csv(a.out, index=False)

pd.set_option("display.width", 220)
for stage in ("post", "pre"):
    s = g[g["stage"] == stage]
    if not len(s):
        continue
    print(f"\n=== {stage} ===")
    print(s.drop(columns="stage").round(3).to_string(index=False))
print(f"\nwrote {a.out}")
