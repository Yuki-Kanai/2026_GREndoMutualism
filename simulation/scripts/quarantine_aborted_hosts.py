#!/usr/bin/env python
"""Restore hosts 10-19 to the A4 design by moving the aborted sweep's leftovers.

Purpose
-------
data/raw/post_paper_h10 .. h19 hold two layers on top of each other:

  * the A4 / gate G7 run (config/design_a4.yaml: 15 r values, N_S = 100, chunk
    10, so 10 blocks per r, symbiont ids 0-99), and
  * an N_S = 300 full-r sweep that was started for these hosts and then stopped
    when hosts 10-19 were declared unnecessary.  It left blocks at r values that
    are not in the A4 grid, extra blocks (sid_lo >= 100) at r values that are,
    two cells holding only 20 symbionts, and three *.parquet.tmp write stubs.

Hosts 0-9 are NOT touched: their A4 and 300-symbiont layers used the same chunk
(10), so the block names line up exactly and every r holds 300 symbionts once.

Leaving the leftovers in place does not corrupt anything today -- a4_analyze.py
cuts to symbiont_id < n_s -- but it does give hosts 10-19 a denser r grid than
the other hosts, so re-running the A4 analysis would no longer reproduce G7.

Nothing is deleted: files are MOVED to data/quarantine_aborted/<dir>/, so the
change is reversible.

Question it answers
-------------------
"Does every hosts-10-19 directory hold exactly the A4 design, once?"

Output: <= 20 printed lines.  Exit 1 if a directory does not end up at exactly
15 r values x 10 blocks.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

BLOCK = re.compile(r"^(cells|vectors)_r(\d+)_s(\d+)\.parquet$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--quarantine", default="data/quarantine_aborted")
    ap.add_argument("--design", default="config/design_a4.yaml")
    ap.add_argument("--hosts", default="10,11,12,13,14,15,16,17,18,19")
    ap.add_argument("--n-s", type=int, default=100)
    ap.add_argument("--apply", action="store_true",
                    help="without this the script only reports what it would move")
    args = ap.parse_args()

    import yaml
    with open(args.design, encoding="utf-8") as fh:
        design = yaml.safe_load(fh)
    grid = set(int(x) for x in design["r_grid"])
    n_s = int(design.get("N_S", args.n_s))
    print(f"A4 grid: {sorted(grid)}  N_S={n_s}  "
          f"({'APPLY' if args.apply else 'dry run'})")

    bad = []
    for h in args.hosts.split(","):
        d = os.path.join(args.raw, f"post_paper_h{h}")
        if not os.path.isdir(d):
            continue
        qd = os.path.join(args.quarantine, f"post_paper_h{h}")
        moved, kept_r = 0, {}
        for name in sorted(os.listdir(d)):
            src = os.path.join(d, name)
            if name.endswith(".tmp"):
                reason = "tmp"
            else:
                m = BLOCK.match(name)
                if not m:
                    continue
                r, sid_lo = int(m.group(2)), int(m.group(3))
                if r not in grid:
                    reason = "r off grid"
                elif sid_lo >= n_s:
                    reason = "sid >= N_S"
                else:
                    if m.group(1) == "cells":
                        kept_r[r] = kept_r.get(r, 0) + 1
                    continue
            moved += 1
            if args.apply:
                os.makedirs(qd, exist_ok=True)
                shutil.move(src, os.path.join(qd, name))
        exp = n_s // 10
        wrong = {r: c for r, c in kept_r.items() if c != exp}
        missing = sorted(grid - set(kept_r))
        status = "OK" if (not wrong and not missing) else "CHECK"
        if status != "OK":
            bad.append(f"h{h}: wrong={wrong} missing={missing}")
        print(f"  h{h:<3s} moved {moved:4d}  kept {len(kept_r):2d} r x "
              f"{exp} blocks  {status}")

    if not args.apply:
        print("dry run: nothing moved.  re-run with --apply")
        return 0
    if bad:
        print("NOT the A4 design after the move:")
        for b in bad:
            print("  " + b)
        return 1
    print("every host 10-19 directory now holds exactly the A4 design")
    return 0


if __name__ == "__main__":
    sys.exit(main())
