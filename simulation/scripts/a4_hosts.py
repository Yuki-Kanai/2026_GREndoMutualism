"""A4 / gate G7 -- inventory of the 20 hosts.

Purpose
-------
Every result up to G2b rests on one host (host_id = 0, mu_H_0 = 0.051711), which
G6 showed sits at the very bottom of the acceptance window [0.05, 0.10] and is
not typical of the population (acceptance rate 1.5 %).  Before spending 90 core
hours of MILP, this script writes down *which* hosts A4 will use: their mu_H_0,
where that falls inside the window, and how many rejection-sampling attempts each
one took.  07_run2_workorder.md section 4.1 makes this table a required output.

Output
------
tables/A4_hosts.csv -- one row per host_id.
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
from grhd.pipeline import get_host  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hosts", type=int, default=20)
    ap.add_argument("--config", default=str(ROOT / "config" / "base.yaml"))
    ap.add_argument("--out", default=str(ROOT / "tables" / "A4_hosts.csv"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    lo = float(cfg["model"]["host_accept_lo"])
    hi = float(cfg["model"]["host_accept_hi"])

    rows = []
    for h in range(args.n_hosts):
        t0 = time.perf_counter()
        hd, _ = get_host(cfg, host_id=h)
        rows.append({
            "host_id": h,
            "mu_H_0": hd.mu_H_0,
            "n_attempts": hd.n_attempts,
            "accepted": bool(hd.accepted),
            # 0 = bottom of the acceptance window, 1 = top
            "window_pos": (hd.mu_H_0 - lo) / (hi - lo),
            "biomass_row_l1": float(np.abs(hd.R_H[-1, :]).sum()),
            "biomass_row_nnz": int((np.abs(hd.R_H[-1, :]) > 0).sum()),
            "seconds": time.perf_counter() - t0,
        })
        print(f"host {h:2d}  mu_H_0={hd.mu_H_0:.6f}  attempts={hd.n_attempts}"
              f"  accepted={hd.accepted}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} hosts -> {out}")
    print(f"mu_H_0: min={df.mu_H_0.min():.6f} median={df.mu_H_0.median():.6f} "
          f"max={df.mu_H_0.max():.6f}; attempts median={df.n_attempts.median():.0f}"
          f" total={df.n_attempts.sum()}")


if __name__ == "__main__":
    main()
