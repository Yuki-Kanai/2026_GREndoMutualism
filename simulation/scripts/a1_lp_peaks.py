"""A1-LP (gate G2a) — do the mechanism peaks survive a fixed reaction repertoire?

03_additional_analyses.md A1 / 07_run2_workorder.md section 3.1.

The paper's generator ties two things together: shrinking r shrinks both the
control granularity AND the metabolic repertoire.  The fixed_repertoire arm
breaks that tie (same reactions, same weights, only the granularity changes),
and the k_sweep arm moves the repertoire without moving r.  Here we ask, for
each arm, whether the peaks in s, n_excreted, ub and b survive, and whether
the peak sits at a fixed r or at a fixed b.

Outputs
  tables/A1_repertoire_size.csv  effective repertoire vs r, per arm  (the direct
                                 evidence for reviewer point M2)
  tables/A1_peaks.csv            peak position + bootstrap CI, r axis and b axis
  tables/A1_curves.csv           the curves behind those peaks (long format)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.cfgio import load_config
from grhd.pipeline import make_R_S_meta
from grhd.stats import bootstrap_curve_and_peak, loess_peak

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--tables", default="tables")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
SEED = int(cfg["master_seed"])
M = int(cfg["model"]["m"])
RAW = Path(a.raw)
TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)

COLS = ["r", "symbiont_id", "s", "ub", "n_excreted", "n_bounded"]
SMALL = {"r": "int16", "symbiont_id": "int16", "ub": "int16",
         "n_excreted": "int16", "n_bounded": "int16", "s": "float32"}
# The repertoire columns are recomputed from the generator rather than read from
# the parquets: the run-1 paper data predates those columns, and the repertoire
# is a property of the generator alone (no LP involved), so this is exact.

# arm_id -> (scheme, param label, param value, directory)
# k_sweep at k = 4 IS the paper run: same RNG key, so it is aliased, not re-run.
ARMS = [("paper",       "paper",            "k",     4,  "pre_paper"),
        ("fixed_n12",   "fixed_repertoire", "n_rxn", 12, "pre_fixed_repertoire_n12"),
        ("fixed_n24",   "fixed_repertoire", "n_rxn", 24, "pre_fixed_repertoire_n24"),
        ("fixed_n40",   "fixed_repertoire", "n_rxn", 40, "pre_fixed_repertoire_n40"),
        ("ksweep_k2",   "k_sweep",          "k",     2,  "pre_k_sweep_k2"),
        ("ksweep_k4",   "k_sweep",          "k",     4,  "pre_paper"),
        ("ksweep_k8",   "k_sweep",          "k",     8,  "pre_k_sweep_k8"),
        ("ksweep_k16",  "k_sweep",          "k",     16, "pre_k_sweep_k16")]

# quantity, statistic name, callable.  s is reported on the median because its
# tail is heavy (02 R2, and G6 section 3).
QUANT = [("s", "median", np.median), ("s", "mean", np.mean),
         ("s_norm", "median", np.median),
         ("n_excreted", "mean", np.mean), ("ub", "mean", np.mean),
         ("n_bounded", "mean", np.mean)]


def load_arm(arm):
    arm_id, scheme, pname, pval, sub = arm
    d = RAW / sub
    files = sorted(d.glob("cells_*.parquet"))
    if not files:
        return arm_id, None
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files],
                   ignore_index=True)
    df = df.astype(SMALL)                # ~17M rows across the arms: keep it small
    # ub = b*u is the ELEMENT COUNT of the sensitivity matrix S (u x b), and it is
    # maximised at b = 4 or 5 by pure algebra (b(m-1-b)).  So a peak of s at
    # b ~ 4 could be inherited from that count rather than from the structure of
    # R^S.  s/sqrt(ub) divides the count out; if it still peaks, the peak is real.
    ub = df["ub"].to_numpy(dtype=np.float32)
    df["s_norm"] = np.where(ub > 0, df["s"].to_numpy(np.float32) / np.sqrt(
        np.where(ub > 0, ub, 1.0)), np.nan).astype(np.float32)
    return arm_id, df


def repertoire_of(arm, r_values, n_sym):
    """Unique reactions present in R^S, straight from the generator."""
    arm_id, scheme, pname, pval, _ = arm
    kk = pval if pname == "k" else None
    nr = pval if pname == "n_rxn" else None
    out = []
    for r in r_values:
        eff, smin, smax, smean = [], [], [], []
        for sid in range(n_sym):
            _, meta = make_R_S_meta(cfg, int(r), sid, scheme, M, kk, nr)
            eff.append(meta["n_rxn_effective"]); smean.append(meta["mode_size_mean"])
            smin.append(meta["mode_size_min"]); smax.append(meta["mode_size_max"])
        out.append({"arm": arm_id, "r": int(r),
                    "rxn_eff_mean": float(np.mean(eff)),
                    "rxn_eff_min": int(np.min(eff)), "rxn_eff_max": int(np.max(eff)),
                    "mode_size_mean": float(np.mean(smean)),
                    "mode_size_min": int(np.min(smin)),
                    "mode_size_max": int(np.max(smax))})
    return pd.DataFrame(out)


def analyse(arm, df):
    arm_id, scheme, pname, pval, _ = arm
    tag = {"arm": arm_id, "scheme": scheme, "param_name": pname, "param": pval}

    # --- effective repertoire vs r (the M2 evidence) ----------------------- #
    rep = df.groupby("r").agg(
        n_cells=("s", "size"),
        n_symbionts=("symbiont_id", "nunique"),
        b_mean=("n_bounded", "mean"), ub_mean=("ub", "mean"),
    ).reset_index()
    rep = rep.merge(repertoire_of(arm, rep["r"].tolist(),
                                  int(df["symbiont_id"].nunique())),
                    on="r", how="left").drop(columns=["arm"])
    for kk, vv in tag.items():
        rep.insert(0, kk, vv)
    rep["r_over_n_rxn"] = rep["r"] / pval if pname == "n_rxn" else np.nan

    bmap = rep.set_index("r")["b_mean"]
    ubmap = rep.set_index("r")["ub_mean"]

    curves, peaks = [], []
    for col, sname, fn in QUANT:
        for axis, gcol in (("r", "r"), ("b", "n_bounded")):
            sub = df[np.isfinite(df[col])]
            if axis == "b":
                # b = 0 carries no sensitivity information (S is empty), and a
                # group with a handful of symbionts cannot support a bootstrap
                keep = sub.groupby(gcol)["symbiont_id"].transform("nunique") >= 20
                sub = sub[keep]
            if sub.empty or sub[gcol].nunique() < 3:
                continue
            cv, pk = bootstrap_curve_and_peak(
                sub, col, group=gcol, unit="symbiont_id", stat=fn, B=a.boot,
                master_seed=SEED, table_id=f"A1_{arm_id}_{col}_{sname}_{axis}")
            cv = cv.rename(columns={gcol: "group_value"})
            cv.insert(0, "axis", axis); cv.insert(0, "statistic", sname)
            cv.insert(0, "quantity", col)
            for kk, vv in tag.items():
                cv.insert(0, kk, vv)
            curves.append(cv)

            pk.update(tag)
            pk.update({"quantity": col, "statistic": sname, "axis": axis,
                       "loess_peak_at": loess_peak(cv["group_value"], cv["value"]),
                       "n_groups": int(cv.shape[0]),
                       "min_support": int(cv["n"].min()),
                       "support_at_peak": int(
                           cv.loc[cv["group_value"] == pk["peak_at"], "n"].iloc[0])})
            if axis == "r":
                pk["peak_at_mean_b"] = float(bmap.get(pk["peak_at"], np.nan))
                pk["peak_at_mean_ub"] = float(ubmap.get(pk["peak_at"], np.nan))
                pk["peak_r_over_n_rxn"] = (pk["peak_at"] / pval
                                           if pname == "n_rxn" else np.nan)
            pk["b_reference_(m-1)/2"] = (M - 1) / 2
            peaks.append(pk)
    return rep, pd.concat(curves, ignore_index=True), pd.DataFrame(peaks)


from joblib import Parallel, delayed                                # noqa: E402

present = []
for arm in ARMS:
    if (RAW / arm[4]).exists():
        present.append(arm)
    else:
        print(f"MISSING arm {arm[0]}: {RAW / arm[4]}", flush=True)

loaded = Parallel(n_jobs=min(a.workers, len(present)), backend="loky")(
    delayed(load_arm)(arm) for arm in present)
by_id = dict(loaded)

reps, curves, peaks = [], [], []
for arm in present:
    df = by_id[arm[0]]
    if df is None or df.empty:
        print(f"EMPTY arm {arm[0]}", flush=True)
        continue
    print(f"{arm[0]}: {len(df)} cells, r {df.r.min()}..{df.r.max()}", flush=True)
    rp, cv, pk = analyse(arm, df)
    reps.append(rp); curves.append(cv); peaks.append(pk)

rep = pd.concat(reps, ignore_index=True)
rep.to_csv(TAB / "A1_repertoire_size.csv", index=False)
pd.concat(curves, ignore_index=True).to_csv(TAB / "A1_curves.csv", index=False)
pk = pd.concat(peaks, ignore_index=True)
pk.to_csv(TAB / "A1_peaks.csv", index=False)

pd.set_option("display.width", 260)
show = ["arm", "quantity", "statistic", "axis", "peak_at", "peak_ci_lo",
        "peak_ci_hi", "diff_peak_minus_first", "diff_peak_minus_first_lo",
        "diff_peak_minus_first_hi", "diff_peak_minus_last",
        "diff_peak_minus_last_lo", "diff_peak_minus_last_hi", "min_support"]
for q, sname in [("s", "median"), ("n_excreted", "mean"), ("ub", "mean"),
                 ("n_bounded", "mean")]:
    print(f"\n=== {q} ({sname}) ===")
    sel = pk[(pk["quantity"] == q) & (pk["statistic"] == sname)]
    print(sel[show].round(4).to_string(index=False))

print("\n=== effective repertoire vs r (a few r per arm) ===")
print(rep[rep.r.isin([1, 2, 3, 5, 8, 10, 12, 20, 24, 40, 60, 100])][
    ["arm", "r", "rxn_eff_mean", "mode_size_mean", "b_mean", "ub_mean",
     "n_symbionts"]].round(3).to_string(index=False))
print("\nwrote", TAB / "A1_peaks.csv")
