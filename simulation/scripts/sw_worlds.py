"""Shared-tournament check (SW) -- inventory of the worlds.

Purpose
-------
In the production model every R^S and every R^H draws its own tournament, so a
symbiont can carry the reverse of a host reaction.  In scheme="shared_world"
one tournament (the "world") is drawn and the host and all symbionts of that
world take their modes from it.  Before running Level 1-3, this script writes
down which worlds are used: whether a host in [lo, hi] could be found in each,
how many attempts that took, and how reachable the biomass metabolite is in the
world's tournament.  The first --n-use accepted worlds are the ones to run.

Output
------
tables/SW_worlds.csv -- one row per world_id (use = True for the worlds to run).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grhd.cfgio import ROOT, load_config  # noqa: E402
from grhd.pipeline import get_host, world_pool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=30)
    ap.add_argument("--n-use", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=20_000)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--config", default=str(ROOT / "config" / "base.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.m is not None:          # r_H stays at the base value (24), as in R5-B
        cfg["model"]["m"] = args.m
    if args.out is None:
        args.out = str(ROOT / "tables" / ("SW_worlds.csv" if args.m is None
                                          else f"SW_worlds_m{args.m}.csv"))
    # A world that cannot host an accepted R_H is abandoned after max_attempts.
    # For accepted worlds the draw is identical to the production max_attempts,
    # because the rejection sequence is the same up to the accepted attempt.
    cfg["model"]["max_attempts"] = min(args.max_attempts, cfg["model"]["max_attempts"])
    lo = float(cfg["model"]["host_accept_lo"])
    hi = float(cfg["model"]["host_accept_hi"])
    m = int(cfg["model"]["m"])

    rows, n_ok = [], 0
    for w in range(args.n_worlds):
        t0 = time.perf_counter()
        V = world_pool(cfg, w)
        hd, _ = get_host(cfg, host_id=w, scheme="shared_world")
        ok = bool(hd.accepted)
        rows.append({
            "world_id": w,
            "accepted": ok,
            "use": ok and n_ok < args.n_use,
            "mu_H_0": hd.mu_H_0 if ok else np.nan,
            "window_pos": (hd.mu_H_0 - lo) / (hi - lo) if ok else np.nan,
            "n_attempts": hd.n_attempts,
            "biomass_in_degree": int((V[:, m - 1] > 0).sum()),
            "biomass_row_nnz": int((np.abs(hd.R_H[-1, :]) > 0).sum()) if ok else -1,
            "seconds": time.perf_counter() - t0,
        })
        n_ok += ok
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    use = df.loc[df["use"], "world_id"].tolist()
    print(f"accepted {int(df['accepted'].sum())}/{len(df)}; use = {use}")
    print(f"attempts median {df['n_attempts'].median():.0f}; "
          f"mu_H_0 range {df['mu_H_0'].min():.4f}-{df['mu_H_0'].max():.4f}")


if __name__ == "__main__":
    main()
