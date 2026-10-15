#!/usr/bin/env python
"""Merge each run's thousands of 32 KB parquet blocks into one file per run.

Purpose
-------
data/raw holds 157,496 parquet files averaging 32 KB across 251 run directories.
That layout exists because a block is the unit of resumption (a block is skipped
when its file already exists), which was the right choice while the runs were in
flight.  Now that they have finished it is pure overhead: rsync spends all its
time on per-file round trips, and every analysis reopens up to 3,000 files per
run through `pd.concat`.

This writes data/consolidated/<run_id>_cells.parquet (and _vectors.parquet), one
file of 16-20 MB per run, and verifies each one before reporting success.  It
NEVER deletes the sources -- deleting is a separate, explicit decision, and only
after the verification below has passed.

Verification per output file
----------------------------
  * row count equals the sum over the source blocks
  * the multiset of key columns (r, symbiont_id, c_id) is identical
  * the sum of mu_H matches to 1e-9 relative

Question it answers
-------------------
"Can the block files be replaced by one file per run without losing or
duplicating a single cell?"

Output: data/consolidated/, tables/CONSOLIDATE_report.csv, <=20 printed lines.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time

import numpy as np
import pandas as pd

KEYS = ["r", "symbiont_id", "c_id"]


def _keyframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEYS if c in df.columns]
    k = df[cols].copy()
    for c in cols:
        k[c] = k[c].astype("float64")          # c_id is nullable in post runs
    return k.fillna(-1).sort_values(cols, kind="mergesort").reset_index(drop=True)


def consolidate(src_dir: str, kind: str, out_path: str, verify: bool) -> dict:
    import glob
    files = sorted(glob.glob(os.path.join(src_dir, f"{kind}_*.parquet")))
    if not files:
        return {}
    t0 = time.perf_counter()
    bytes_in = sum(os.path.getsize(f) for f in files)
    parts, bad = [], []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception as e:                                   # noqa: BLE001
            bad.append(f"{os.path.basename(f)}: {e}")
    if bad:
        return {"error": f"{len(bad)} unreadable block(s): {bad[0]}"}
    df = pd.concat(parts, ignore_index=True)
    del parts

    tmp = out_path + ".tmp"
    df.to_parquet(tmp, index=False, compression="snappy")
    os.replace(tmp, out_path)

    rec = {"n_files": len(files), "n_rows": len(df),
           "bytes_in": bytes_in, "bytes_out": os.path.getsize(out_path),
           "seconds": time.perf_counter() - t0, "verified": False, "error": ""}
    if verify:
        back = pd.read_parquet(out_path)
        ok = len(back) == len(df)
        if ok:
            ok = _keyframe(back).equals(_keyframe(df))
        if ok and "mu_H" in df.columns:
            a, b = np.nansum(df["mu_H"]), np.nansum(back["mu_H"])
            ok = bool(np.isclose(a, b, rtol=1e-9, atol=1e-12))
        rec["verified"] = bool(ok)
        if not ok:
            rec["error"] = "VERIFY FAILED"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/consolidated")
    ap.add_argument("--report", default="tables/CONSOLIDATE_report.csv")
    ap.add_argument("--pattern", default="*", help="run directories to include")
    ap.add_argument("--exclude", default="", help="comma-separated glob(s) to skip")
    ap.add_argument("--kinds", default="cells,vectors")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    dirs = sorted(d for d in os.listdir(args.raw)
                  if os.path.isdir(os.path.join(args.raw, d))
                  and fnmatch.fnmatch(d, args.pattern))
    for pat in [p for p in args.exclude.split(",") if p]:
        dirs = [d for d in dirs if not fnmatch.fnmatch(d, pat)]
    os.makedirs(args.out, exist_ok=True)
    kinds = args.kinds.split(",")
    print(f"{len(dirs)} run directories, kinds={kinds}")

    rows, failed = [], []
    for i, d in enumerate(dirs, 1):
        for kind in kinds:
            out_path = os.path.join(args.out, f"{d}_{kind}.parquet")
            if os.path.exists(out_path):
                continue
            rec = consolidate(os.path.join(args.raw, d), kind, out_path,
                              not args.no_verify)
            if not rec:
                continue
            rec.update({"run": d, "kind": kind})
            rows.append(rec)
            if rec.get("error"):
                failed.append(f"{d}/{kind}: {rec['error']}")
                print(f"  !! {d}/{kind}: {rec['error']}", flush=True)
        if i % 25 == 0:
            print(f"  ... {i}/{len(dirs)} directories", flush=True)

    if not rows:
        print("nothing to do (all outputs already exist)")
        return 0
    rep = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    rep.to_csv(args.report, index=False)
    print(f"files  {rep['n_files'].sum():,} -> {len(rep):,}")
    print(f"bytes  {rep['bytes_in'].sum() / 2**30:.2f} GB -> "
          f"{rep['bytes_out'].sum() / 2**30:.2f} GB")
    print(f"rows   {rep['n_rows'].sum():,}")
    print(f"verified {int(rep['verified'].sum())}/{len(rep)}  "
          f"in {rep['seconds'].sum() / 60:.1f} min")
    if failed:
        print(f"FAILURES ({len(failed)}) -- sources must NOT be deleted:")
        for f in failed[:10]:
            print("  " + f)
        return 1
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
