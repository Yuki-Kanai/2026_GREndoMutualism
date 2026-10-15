#!/usr/bin/env python
"""Who are the det-limit-truncated cells?  (user question, 2026-09-21)

Purpose
-------
The author's own implementation almost never hits the det limit at m=10, k=4,
while ours truncates ~6 % of cells.  Before spending any more time on GA rescue
we need to know *which* cells truncate.  The hypothesis to test: they are cells
whose symbiont barely grows (mu^S ~ 0), i.e. degenerate instances that a
mu^S < 1e-2 filter would remove anyway.

Question
--------
1. Truncation rate by r.
2. For truncated cells: the distribution of mu^S, mu^H, the MIP gap and the
   proven upper bound.
3. What fraction of truncated cells would a `mu^S < 1e-2 -> drop` rule catch,
   and how many *non*-truncated cells would the same rule drop?

Output: one CSV plus at most 20 printed lines.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

COLS = ["r", "symbiont_id", "mu_H_0", "mu_S", "mu_H", "flag_detlimit",
        "solver_status", "mip_gap", "mip_best_bound", "solve_time_milp_s",
        "milp_path", "trivial_ub", "b", "n_excreted"]


def load(d: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(d, "cells_*.parquet")))
    if not files:
        sys.exit(f"no cells_*.parquet under {d}")
    got = None
    out = []
    for f in files:
        try:
            if got is None:
                import pyarrow.parquet as pq
                have = set(pq.ParquetFile(f).schema.names)
                got = [c for c in COLS if c in have]
            out.append(pd.read_parquet(f, columns=got))
        except Exception as e:                                   # noqa: BLE001
            print(f"SKIP {os.path.basename(f)}: {e}", file=sys.stderr)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/post_paper")
    ap.add_argument("--out", default="tables/DIAG_detlimit_post_paper.csv")
    ap.add_argument("--mu-s-cut", type=float, default=1e-2)
    args = ap.parse_args()

    df = load(args.dir)
    df["trunc"] = df["flag_detlimit"].fillna(False).astype(bool)
    n, nt = len(df), int(df["trunc"].sum())
    print(f"{args.dir}: {n} cells, truncated {nt} ({100 * nt / n:.2f} %)")

    t = df[df["trunc"]]
    o = df[~df["trunc"]]
    cut = args.mu_s_cut

    def q(s):
        s = pd.Series(s).dropna()
        if len(s) == 0:
            return "n/a"
        return " ".join(f"{np.quantile(s, p):.3g}" for p in (0.0, 0.25, 0.5, 0.75, 1.0))

    print(f"mu_S  quantiles (min q25 med q75 max)  truncated: {q(t['mu_S'])}")
    print(f"mu_S  quantiles                        optimal  : {q(o['mu_S'])}")
    print(f"mu_H  quantiles                        truncated: {q(t['mu_H'])}")
    print(f"mu_H  quantiles                        optimal  : {q(o['mu_H'])}")
    if "mip_gap" in df:
        print(f"mip_gap quantiles                      truncated: {q(t['mip_gap'])}")
    if "mip_best_bound" in df:
        rr = (t["mip_best_bound"] / t["mu_H"].replace(0, np.nan))
        print(f"bound/incumbent quantiles              truncated: {q(rr)}")

    # the user's proposed rule
    for lab, s in (("truncated", t), ("optimal", o)):
        k = int((s["mu_S"] < cut).sum())
        print(f"mu_S < {cut:g}: {k}/{len(s)} ({100 * k / max(len(s), 1):.1f} %) of {lab} cells")
    tiny = df["mu_S"] < cut
    print(f"among mu_S < {cut:g} cells, truncation rate "
          f"{100 * df.loc[tiny, 'trunc'].mean():.1f} % "
          f"(vs {100 * df.loc[~tiny, 'trunc'].mean():.1f} % otherwise)")
    if "solve_time_milp_s" in df:
        share = t.loc[t["mu_S"] < cut, "solve_time_milp_s"].sum() / max(
            t["solve_time_milp_s"].sum(), 1e-12)
        print(f"share of truncated-cell solve time held by mu_S < {cut:g}: {100 * share:.1f} %")

    by_r = (df.groupby("r")
              .agg(n=("r", "size"), trunc=("trunc", "mean"),
                   t_sum_h=("solve_time_milp_s", lambda s: s.sum() / 3600.0),
                   mu_S_med=("mu_S", "median"),
                   frac_tiny_muS=("mu_S", lambda s: float((s < cut).mean())))
              .reset_index())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    by_r.to_csv(args.out, index=False)
    top = by_r.sort_values("t_sum_h", ascending=False).head(8)
    print("--- worst r by solve time (r, n, trunc frac, core-h, median mu_S, frac mu_S<cut)")
    print(top.round(4).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
