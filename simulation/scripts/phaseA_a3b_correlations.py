"""A3-b — the full 3 x 2 x 4 factorial of correlations, with matched conditions.

07_run2_workorder.md Phase A.  Run 1 reported `5<=r<=25` and `non_trivial` as
separate rows and never crossed them; this computes every combination so that
"the relation weakened" can be told apart from "the sample was cut differently".

Exclusion conditions (stated explicitly because both are about "trivial" cells
but they are different things, 01_model_spec.md A12):
  exclude_ub_lt_1       drop symbionts whose PRE-evolution mean ub < 1
  exclude_delta_small   drop cells whose POST-evolution delta_mu_H < 1e-2
  exclude_both          drop a cell if either applies
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.stats import (fisher_pool, partial_spearman, quartile_contrast_ci,
                        spearman)

ap = argparse.ArgumentParser()
ap.add_argument("--join", default="data/derived/plotdata_post_join.parquet")
ap.add_argument("--tables", default="tables")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
B = int(cfg["stats"]["bootstrap_B"])
SEED = int(cfg["master_seed"])
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)

j = pd.read_parquet(a.join)
mu_H_0 = float(j["mu_H_0"].iloc[0])
j["delta_mu_H_post"] = j["mu_H_post"] - mu_H_0
j["is_ub_lt_1"] = j["ub_pre_mean"] < 1.0
j["is_delta_small"] = j["delta_mu_H_post"] < 1e-2
print(f"{len(j)} cells; ub<1: {int(j.is_ub_lt_1.sum())}; "
      f"delta<1e-2: {int(j.is_delta_small.sum())}", flush=True)

FIGS = {"5A_max": "mu_H_pre_max", "5A_median": "mu_H_pre_median",
        "S2A_ub": "ub_pre_mean"}
RRANGES = {"all_r": lambda d: d["r"] > 0,
           "5<=r<=25": lambda d: (d["r"] >= 5) & (d["r"] <= 25)}
EXCL = {"none": lambda d: np.ones(len(d), bool),
        "exclude_ub_lt_1": lambda d: ~d["is_ub_lt_1"].to_numpy(),
        "exclude_delta_small": lambda d: ~d["is_delta_small"].to_numpy(),
        "exclude_both": lambda d: ~(d["is_ub_lt_1"] | d["is_delta_small"]).to_numpy()}

rows, prof = [], []
for fig, xcol in FIGS.items():
    for rname, rsel in RRANGES.items():
        base = j[rsel(j)]
        for ename, esel in EXCL.items():
            d = base[esel(base)]
            rec = {"figure": fig, "x": xcol, "r_range": rname,
                   "exclusion": ename, "n": int(len(d))}
            if len(d) < 20:
                rows.append(rec)
                continue
            rho, p, n = spearman(d[xcol], d["mu_H_post"])
            prho, _ = partial_spearman(d[xcol], d["mu_H_post"], d["r"])
            wr = [(rr,) + spearman(g[xcol], g["mu_H_post"])
                  for rr, g in d.groupby("r")]
            wrho = np.array([w[1] for w in wr], float)
            wn = np.array([w[3] for w in wr], float)
            rec.update({
                "spearman_pooled": rho, "spearman_p": p,
                "partial_spearman_on_r": prho,
                "within_r_fisher": fisher_pool(wrho, wn),
                "within_r_median": float(np.nanmedian(wrho)),
                "within_r_iqr": float(np.nanquantile(wrho, .75)
                                      - np.nanquantile(wrho, .25)),
                "within_r_n_groups": int(np.isfinite(wrho).sum()),
            })
            qc = quartile_contrast_ci(d[xcol].to_numpy(float),
                                      d["mu_H_post"].to_numpy(float), B=B,
                                      master_seed=SEED,
                                      table_id=f"A3b_{fig}_{rname}_{ename}",
                                      mu_ref=mu_H_0)
            rec.update({
                "qc_q1_median": qc["q1_median"], "qc_q4_median": qc["q4_median"],
                "qc_contrast": qc["contrast"],
                "qc_contrast_ci_lo": qc["contrast_ci_lo"],
                "qc_contrast_ci_hi": qc["contrast_ci_hi"],
                "qc_contrast_over_muH0": qc.get("contrast_over_muH0"),
                "cliffs_delta": qc["cliffs_delta"],
                "cliffs_delta_ci_lo": qc["cliffs_delta_ci_lo"],
                "cliffs_delta_ci_hi": qc["cliffs_delta_ci_hi"],
                "hl_shift": qc["hl_shift"],
                "hl_shift_ci_lo": qc["hl_shift_ci_lo"],
                "hl_shift_ci_hi": qc["hl_shift_ci_hi"],
            })
            rows.append(rec)

            dropped = base[~esel(base)]
            prof.append({"figure": fig, "r_range": rname, "exclusion": ename,
                         "n_kept": int(len(d)), "n_dropped": int(len(dropped)),
                         "frac_dropped": float(len(dropped) / max(len(base), 1)),
                         "dropped_r_median": float(dropped["r"].median())
                         if len(dropped) else np.nan,
                         "dropped_r_q25": float(dropped["r"].quantile(.25))
                         if len(dropped) else np.nan,
                         "dropped_r_q75": float(dropped["r"].quantile(.75))
                         if len(dropped) else np.nan})
        # per-r drop profile, once per (figure, r_range)
        for ename, esel in EXCL.items():
            if ename == "none":
                continue
            drop_by_r = base.assign(dropped=~esel(base)).groupby("r")["dropped"].mean()
            for rr, v in drop_by_r.items():
                prof.append({"figure": fig, "r_range": rname,
                             "exclusion": f"{ename}@r={rr}", "n_kept": np.nan,
                             "n_dropped": np.nan, "frac_dropped": float(v),
                             "dropped_r_median": rr, "dropped_r_q25": np.nan,
                             "dropped_r_q75": np.nan})
        print(f"  done {fig} / {rname}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(TAB / "A3b_correlations_full.csv", index=False)
pd.DataFrame(prof).to_csv(TAB / "A3b_exclusion_profile.csv", index=False)

pd.set_option("display.width", 250)
print("\n=== A3b: 24-row factorial (correlations) ===")
print(out[["figure", "r_range", "exclusion", "n", "spearman_pooled",
           "partial_spearman_on_r", "within_r_fisher", "within_r_median"]]
      .round(4).to_string(index=False))
print("\n=== effect sizes ===")
print(out[["figure", "r_range", "exclusion", "n", "qc_contrast",
           "qc_contrast_ci_lo", "qc_contrast_ci_hi", "qc_contrast_over_muH0",
           "cliffs_delta", "cliffs_delta_ci_lo", "cliffs_delta_ci_hi"]]
      .round(4).to_string(index=False))
print("\n=== the row run 1 never computed: 5<=r<=25 x exclude_both ===")
print(out[(out.r_range == "5<=r<=25") & (out.exclusion == "exclude_both")]
      .round(4).to_string(index=False))

json.dump({"mu_H_0": mu_H_0, "n_cells": int(len(j)), "bootstrap_B": B,
           "definitions": {
               "exclude_ub_lt_1": "pre-evolution mean ub < 1 (trivial_ub)",
               "exclude_delta_small": "post-evolution mu_H - mu_H_0 < 1e-2",
               "exclude_both": "either of the above"}},
          open(TAB / "A3b_meta.json", "w"), indent=2)
