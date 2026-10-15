"""A14 — the post-evolution structural quantities under both zero-c^H rules.

After host evolution, step 2 minimises ||c^H||_1, so any metabolite the symbiont
does not consume is offered at exactly zero.  A zero component has slack
c^H_i - f^S*_i = 0, so under the default rule `bounded_if_zero_flux` it is
counted as *bounded*.  That makes `b` large precisely where the host provisions
nothing, which is the opposite of what `b` means before evolution (where every
component is offered and `bounded` means "fully consumed").

This script recomputes b, u, ub, alpha, s and n_excreted for every post cell
under both conventions and adds `n_offered` = #{i : c^H_i > 0}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.analyze import bounded_predicates, sensitivity_matrix
from grhd.cfgio import load_config
from grhd.pipeline import make_R_S

ap = argparse.ArgumentParser()
ap.add_argument("--post", default="data/raw/post_paper")
ap.add_argument("--out-cells", default="data/derived/post_conventions.parquet")
ap.add_argument("--out", default="tables/A14_post_conventions.csv")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
tol = cfg["tolerances"]
TA, TR = float(tol["tol_abs"]), float(tol["tol_rel"])
TW, TE = float(tol["tol_w"]), float(tol["tol_exc"])
RC = float(tol["rcond"])
m = cfg["model"]["m"]

R_CACHE: dict = {}


def R_of(r, sid):
    key = (r, sid)
    if key not in R_CACHE:
        R_CACHE[key] = make_R_S(cfg, r, sid)
    return R_CACHE[key]


rows = []
for f in sorted(Path(a.post).glob("vectors_*.parquet")):
    v = pd.read_parquet(f, columns=["r", "symbiont_id", "c_H", "f_S_star",
                                    "w_S_star"])
    for _, rec in v.iterrows():
        r, sid = int(rec["r"]), int(rec["symbiont_id"])
        c = np.asarray(rec["c_H"], dtype=float)
        fs = np.asarray(rec["f_S_star"], dtype=float)
        w = np.asarray(rec["w_S_star"], dtype=float)
        R_S = R_of(r, sid)
        active = w > TW
        out = {"r": r, "symbiont_id": sid,
               "n_offered": int((np.abs(c) > TA).sum()),
               "n_excreted": int((fs < -TE).sum()),
               "r_plus": int(active.sum()),
               "c_H_l1": float(np.abs(c).sum())}
        for rule, tag in (("bounded_if_zero_flux", "default"),
                          ("always_unbounded", "alt")):
            bo, un = bounded_predicates(c, fs, TA, TR, rule)
            b, u = int(bo.sum()), int(un.sum())
            if b == 0 or u == 0 or out["r_plus"] == 0:
                s = 0.0
            else:
                s = float(np.linalg.norm(
                    sensitivity_matrix(R_S, bo, un, active, RC), ord="fro"))
            out[f"b_{tag}"] = b
            out[f"u_{tag}"] = u
            out[f"ub_{tag}"] = b * u
            out[f"alpha_{tag}"] = float(np.sqrt(b))
            out[f"s_{tag}"] = s
            out[f"trivial_{tag}"] = bool(b == 0 or u == 0)
        rows.append(out)
    R_CACHE.clear()

df = pd.DataFrame(rows)
Path(a.out_cells).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(a.out_cells, index=False)

g = (df.groupby("r")
       .agg(n=("b_default", "size"),
            n_offered=("n_offered", "mean"),
            n_excreted=("n_excreted", "mean"),
            b_default=("b_default", "mean"),
            b_alt=("b_alt", "mean"),
            u_default=("u_default", "mean"),
            u_alt=("u_alt", "mean"),
            ub_default=("ub_default", "mean"),
            ub_alt=("ub_alt", "mean"),
            s_default_med=("s_default", "median"),
            s_alt_med=("s_alt", "median"),
            s_default_mean=("s_default", "mean"),
            s_alt_mean=("s_alt", "mean"),
            trivial_default=("trivial_default", "mean"),
            trivial_alt=("trivial_alt", "mean"))
       .reset_index())
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
g.round(4).to_csv(a.out, index=False)

pd.set_option("display.width", 240)
print(g.round(3).to_string(index=False))
print(f"\nwrote {a.out} and {a.out_cells}")
