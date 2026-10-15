"""D7 follow-up — what actually makes the peak of `s` move between statistics?

Run 1 found the peak of mean s at r = 8 but the peak of median s at r = 13, and
the run-2 work order expects D7 (conditioning) to explain it.  It does not:
rcond has no effect at all and R^S_{b+} is essentially never rank deficient.
The candidate explanation tested here is zero inflation: cells with b = 0 or
u = 0 have s = 0 by construction, and they are the majority at both ends of r,
which clips the median to exactly 0 there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from grhd.cfgio import load_config
from grhd.stats import bootstrap_curve_and_peak, loess_peak

cfg = load_config("config/base.yaml")
B = int(cfg["stats"]["bootstrap_B"])
SEED = int(cfg["master_seed"])
TAB = Path("tables")

df = pd.concat([pd.read_parquet(f, columns=["r", "symbiont_id", "s", "ub",
                                            "trivial_ub", "n_bounded", "cond_b"])
                for f in sorted(Path("data/raw/pre_paper").glob("cells_*.parquet"))],
               ignore_index=True)
print(f"{len(df):,} cells; trivial overall {df['trivial_ub'].mean():.3%}", flush=True)

# how many symbionts still contribute once trivial cells are dropped?  the
# clustered estimator averages per-symbiont means, so a group with a handful of
# symbionts is dominated by single symbionts and must not be read as a curve.
support = (df[~df["trivial_ub"]].groupby("r")["symbiont_id"].nunique()
           .reset_index(name="n_symbionts_non_trivial"))
support["n_cells_non_trivial"] = (df[~df["trivial_ub"]].groupby("r").size()
                                  .to_numpy())
support.to_csv(TAB / "A2_D7_support.csv", index=False)
MIN_SYM = 50
r_ok = set(support.loc[support["n_symbionts_non_trivial"] >= MIN_SYM, "r"])
print(f"non-trivial symbiont support >= {MIN_SYM} at r in "
      f"[{min(r_ok)}, {max(r_ok)}] ({len(r_ok)} of {len(support)} r values)",
      flush=True)

trim = lambda z, axis=None: sps.trim_mean(z, 0.05, axis=axis)   # noqa: E731
rows = []
nt = df[~df["trivial_ub"]]
for subset, d in (("all_cells", df), ("non_trivial", nt),
                  (f"non_trivial_support{MIN_SYM}", nt[nt["r"].isin(r_ok)])):
    for label, stat in (("mean", np.mean), ("median", np.median),
                        ("trimmed_mean_5", trim)):
        curve, pk = bootstrap_curve_and_peak(d, "s", group="r",
                                             unit="symbiont_id", stat=stat,
                                             B=B, master_seed=SEED,
                                             table_id=f"D7_{subset}_{label}")
        pk.update({"subset": subset, "statistic": label,
                   "loess_peak_at": loess_peak(curve["r"], curve["value"]),
                   "min_n_units": int(curve["n"].min()),
                   "n_units_at_peak": int(
                       curve.loc[curve["r"] == pk["peak_at"], "n"].iloc[0])})
        rows.append(pk)
        curve.to_csv(TAB / f"A2_D7_curve_s_{subset}_{label}.csv", index=False)
        print(f"  {subset:12s} {label:15s} peak r = {pk['peak_at']:>3} "
              f"CI [{pk['peak_ci_lo']:.0f}, {pk['peak_ci_hi']:.0f}]  "
              f"loess {pk['loess_peak_at']}", flush=True)
pk = pd.DataFrame(rows)
pk.to_csv(TAB / "A2_D7_s_peaks.csv", index=False)

triv = df.groupby("r")["trivial_ub"].mean().reset_index(name="frac_trivial")
triv["median_is_zero"] = triv["frac_trivial"] > 0.5
med = df.groupby("r")["s"].median().reset_index(name="median_s")
t = triv.merge(med, on="r")
t.to_csv(TAB / "A2_D7_trivial_fraction.csv", index=False)

clipped = t[t["median_is_zero"]]
print(f"\nr values where >50% of cells are trivial: "
      f"{clipped['r'].min()}..{clipped['r'].max()} "
      f"({len(clipped)} of {len(t)} values)")
print("median s is exactly 0 at exactly those r:",
      bool((t["median_is_zero"] == (t["median_s"] <= 0)).all()))

pd.set_option("display.width", 220)
print("\n=== peak of s by subset and statistic ===")
print(pk[["subset", "statistic", "peak_at", "peak_ci_lo", "peak_ci_hi",
          "peak_value", "loess_peak_at", "min_n_units",
          "n_units_at_peak"]].round(3).to_string(index=False))
print("\n=== non-trivial symbiont support by r ===")
print(support[support["r"].isin([1, 5, 13, 20, 30, 40, 50, 60, 70, 77, 80, 90, 100])]
      .to_string(index=False))

json.dump({"n_cells": int(len(df)),
           "frac_trivial_overall": float(df["trivial_ub"].mean()),
           "r_with_majority_trivial": [int(x) for x in clipped["r"]],
           "median_zero_iff_majority_trivial":
               bool((t["median_is_zero"] == (t["median_s"] <= 0)).all()),
           "min_symbiont_support": MIN_SYM,
           "r_with_support": sorted(int(x) for x in r_ok)},
          open(TAB / "A2_D7_s_tail_meta.json", "w"), indent=2)
