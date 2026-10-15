#!/usr/bin/env python
"""Where exactly does the det limit bite?  (user question, 2026-09-21)

Purpose
-------
diag_detlimit.py ruled out "truncated cells are the mu^S ~ 0 ones".  The MIP gap
on truncated cells has median 3.92, i.e. the proven upper bound is still near the
trivial mu^H <= 1.  That points at the *formulation*, not at the instances.  Two
candidate causes, both cheap to check from what is already stored:

  (a) It is step 2 (minimise ||c^H||_1 under a mu^H floor) that truncates, not
      step 1.  Then mu^H itself is proven optimal and only the tie-break is
      inexact -- and an implementation that does step 1 only would almost never
      truncate, which is what the author reports.
  (b) The artificial boxes (W_max = Lambda_max = 1e4, G_max = m*C_max = 100) are
      orders of magnitude larger than the values actually attained, so the LP
      relaxation is hopelessly loose and SOS1 branching gets no traction.

Question
--------
1. Of the truncated cells, how many truncated in step 1 / step 2 / both?
2. Does mu_H_step1 differ from the reported mu_H (i.e. does step 2 move it)?
3. How close do w^S, lambda, g actually get to their boxes?  (box_report stores
   attained/box, so the max over cells says how far the boxes could be tightened.)

Output: one CSV plus at most 20 printed lines.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

VCOLS = ["r", "symbiont_id", "l1_step1", "l1_step2", "flag_box_active",
         "box_report", "box_retries", "mu_H_step1", "status_step1", "status_step2"]
CCOLS = ["r", "symbiont_id", "mu_H", "mu_S", "flag_detlimit", "solve_time_milp_s"]


def _read(files, cols):
    out = []
    for f in files:
        try:
            out.append(pd.read_parquet(f, columns=cols))
        except Exception as e:                                   # noqa: BLE001
            print(f"SKIP {os.path.basename(f)}: {e}", file=sys.stderr)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/post_paper")
    ap.add_argument("--out", default="tables/DIAG_detlimit_steps.csv")
    args = ap.parse_args()

    v = _read(sorted(glob.glob(os.path.join(args.dir, "vectors_*.parquet"))), VCOLS)
    c = _read(sorted(glob.glob(os.path.join(args.dir, "cells_*.parquet"))), CCOLS)
    if len(v) == 0:
        sys.exit("no vectors_*.parquet -- rerun with --save-vectors all")
    v = v.drop_duplicates(["r", "symbiont_id"], keep="last")
    df = c.merge(v, on=["r", "symbiont_id"], how="left")
    df["trunc"] = df["flag_detlimit"].fillna(False).astype(bool)

    s1 = df["status_step1"].fillna("").str.contains("detlimit")
    s2 = df["status_step2"].fillna("").str.contains("detlimit")
    print(f"{len(df)} cells, truncated {int(df['trunc'].sum())}")
    print(f"  step 1 only : {int((s1 & ~s2).sum())}")
    print(f"  step 2 only : {int((~s1 & s2).sum())}   <- mu^H is proven optimal here")
    print(f"  both        : {int((s1 & s2).sum())}")
    tt = df[df["trunc"]]
    if "solve_time_milp_s" in df:
        tot = tt["solve_time_milp_s"].sum()
        s2only = tt.loc[~s1[tt.index] & s2[tt.index], "solve_time_milp_s"].sum()
        print(f"  solve time held by step-2-only truncations: "
              f"{100 * s2only / max(tot, 1e-12):.1f} % of {tot / 3600:.1f} core-h")

    d = (df["mu_H"] - df["mu_H_step1"]).abs()
    print(f"|mu_H - mu_H_step1|: median {d.median():.2e}, max {d.max():.2e} "
          f"(eps_H floor is 1e-9)")

    # how much headroom do the artificial boxes have?
    rep = df["box_report"].dropna().head(20000).apply(
        lambda s: json.loads(s) if isinstance(s, str) else {})
    if len(rep):
        keys = sorted({k for d_ in rep for k in d_})
        print("--- attained / box  (max over cells; 1.0 means the box binds)")
        for k in keys:
            vals = np.array([d_.get(k, np.nan) for d_ in rep], dtype=float)
            print(f"  {k:6s} max {np.nanmax(vals):.3e}  q99 {np.nanquantile(vals, 0.99):.3e}")
    print(f"box_retries > 0: {int((df['box_retries'].fillna(0) > 0).sum())} cells")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = df.assign(trunc_step1=s1, trunc_step2=s2)[
        ["r", "symbiont_id", "mu_H", "mu_H_step1", "mu_S", "trunc",
         "trunc_step1", "trunc_step2", "solve_time_milp_s", "l1_step1", "l1_step2"]]
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
