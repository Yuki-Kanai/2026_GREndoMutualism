"""R5 / gate G9 -- block-level integrity check.  Run this BEFORE r5_analyze.py.

Purpose
-------
Blocks are named cells_r<r>_s<sid_lo>.parquet and a block is skipped when its file
already exists, so the block size (`--chunk`) is baked into the file names.  If a
directory is written once with chunk 10 (sid_lo = 0,10,...,90) and again with
chunk 50 (sid_lo = 0,50), the two layouts collide:

  * chunk 50 sees the chunk-10 file s00000 (symbionts 0-9 only), treats that r as
    done for symbionts 0-49, and symbionts 10-49 are never computed -- a silent
    gap that re-running does NOT repair;
  * or chunk 50 writes s00000 (symbionts 0-49) next to a chunk-10 s00010, and the
    same symbionts appear twice -- silent double counting in every mean.

This happened during the R5 run when the chunk was changed mid-stream for m <= 14.
Three (m, host) directories were affected and had to be deleted and recomputed.

Question
--------
Does every (design, m, host, r) cell hold exactly N_S distinct symbionts, once?

Output
------
tables/R5_integrity.csv -- one row per offending (dir, r); empty means clean.
Exit code 1 if anything is wrong, so it can gate the analysis.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--pattern", default="post_paper_m*")
    ap.add_argument("--n-s", type=int, default=100)
    ap.add_argument("--out", default="tables/R5_integrity.csv")
    ap.add_argument("--include-running", action="store_true",
                    help="report short cells too; by default a cell that is merely "
                         "incomplete is still reported, since only you know whether "
                         "the queue is still working on it")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.raw, args.pattern))):
        files = sorted(glob.glob(os.path.join(d, "cells_r*_s*.parquet")))
        if not files:
            continue
        # block layout per r, straight from the file names
        layout: dict[int, set[int]] = {}
        for f in files:
            mm = re.match(r"cells_r(\d+)_s(\d+)", os.path.basename(f))
            layout.setdefault(int(mm.group(1)), set()).add(int(mm.group(2)))
        t = pd.concat([pd.read_parquet(f, columns=["r", "symbiont_id"])
                       for f in files], ignore_index=True)
        g = t.groupby("r")["symbiont_id"].agg(["size", "nunique"])
        for r, row in g.iterrows():
            n, u = int(row["size"]), int(row["nunique"])
            strides = sorted(layout[r])
            step = (min(b - a for a, b in zip(strides, strides[1:]))
                    if len(strides) > 1 else None)
            problem = []
            if n != u:
                problem.append("duplicate")
            if u < args.n_s:
                problem.append("short")
            if problem:
                rows.append({"dir": os.path.basename(d), "r": r,
                             "n_rows": n, "n_unique": u,
                             "block_starts": ",".join(map(str, strides)),
                             "min_block_step": step,
                             "problem": "+".join(problem)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if df.empty:
        print(f"integrity OK: every cell holds {args.n_s} distinct symbionts once")
        return 0
    dup = df[df["problem"].str.contains("duplicate")]
    short = df[df["problem"].str.contains("short")]
    print(f"{len(df)} offending cells -> {out}")
    print(f"  duplicate: {len(dup)}  (these need the r's block files deleted "
          f"and the task re-run)")
    print(f"  short    : {len(short)}  (in-flight cells look like this too)")
    print(df.head(20).to_string(index=False))
    return 1


if __name__ == "__main__":
    sys.exit(main())
