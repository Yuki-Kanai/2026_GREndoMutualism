"""A4 / gate G7 -- does the mid-r peak survive changing the host?

Purpose
-------
Every result from G1 through G2b rests on a single host (host_id = 0), which G6
showed is atypical: the host acceptance rate is 1.5 %, accepted hosts have
systematically weak biomass rows, and this particular one sits at the very bottom
of the acceptance window (mu_H_0 = 0.0517 in [0.05, 0.10]).  So the peak in mean
post-evolution mu^H at intermediate r might be a property of that one host rather
than of genome reduction.

Question
--------
Does the mid-r peak seen with one host survive when the host is replaced?
For how many of the 20 hosts does it fail?

Method
------
07_run2_workorder.md section 4.3.  For five responses -- post mu^H, mu^H/mu^H_0,
the mutualism fraction, s, n_excreted -- fit

    y ~ 1 + log_r + I(log_r^2) + (1 + log_r | host_id)

and call the peak present when the log_r^2 coefficient is significantly negative
and the vertex exp(-b/2c) lies strictly inside the observed r range.  In addition
each host gets its own bootstrap peak r*, which is then regressed on mu_H_0, and
a second mixed model adds centred mu_H_0 and its interaction with log_r.

Outputs
-------
tables/A4_host_summary.csv     per host: mu_H_0, r*, peak value, solver paths
tables/A4_curves.csv           per (host, r): mean of each response
tables/A4_mixed_models.csv     fixed effects, vertex, verdict, per response
tables/A4_rstar_vs_mu0.csv     regression of r* on mu_H_0
tables/A4_milp_paths.csv       milp_path / solver_status distribution per host
figures/A4_mu_H_by_host*       thin line per host + thick normalised mean
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import grhd.plotstyle as ps  # noqa: E402
from grhd.cfgio import ROOT, load_config  # noqa: E402
from grhd.stats import bootstrap_curve_and_peak, loess_peak  # noqa: E402

EPS_MUT = 1e-3      # same mutualism thresholds as A3-e / G5
THETA_MUT = 1e-6
RESPONSES = [
    ("mu_H", "post-evolution $\\mu^H$"),
    ("mu_H_rel", "$\\mu^H/\\mu^H_0$"),
    ("mutualistic", "fraction mutualistic"),
    ("s", "sensitivity $s$"),
    ("n_excreted", "number of excreted metabolites"),
]


def load_cells(raw: Path, n_hosts: int, n_s: int) -> pd.DataFrame:
    """Host 0 lives in post_paper (run-1 blocks, reused); hosts >=1 in _h<id>.

    post_paper carries N_S = 300 from run 1, the A4 hosts N_S = 100, so every host
    is cut to the same first n_s symbionts.  Those are the same symbionts for every
    host: the R^S RNG key does not contain host_id (05_conventions.md section 3).
    """
    parts = []
    for h in range(n_hosts):
        d = raw / ("post_paper" if h == 0 else f"post_paper_h{h}")
        files = sorted(d.glob("cells_*.parquet"))
        if not files:
            print(f"  WARNING: no blocks for host {h} under {d}")
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df[df["symbiont_id"] < n_s].copy()
        df["host_id"] = h            # host 0's blocks were written before --host-id
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def fit_mixed(df: pd.DataFrame, y: str, with_mu0: bool):
    """LMM with random intercept and random log_r slope by host."""
    import statsmodels.formula.api as smf

    d = df[["host_id", "log_r", "log_r2", "mu0_c", y]].dropna().copy()
    d = d.rename(columns={y: "y"})
    if d["y"].dtype == bool:
        d["y"] = d["y"].astype(float)
    formula = "y ~ log_r + log_r2"
    if with_mu0:
        formula += " + mu0_c + log_r:mu0_c"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(formula, d, groups=d["host_id"], re_formula="1 + log_r")
        res = md.fit(method="lbfgs", maxiter=2000)
    fe, se, pv = res.fe_params, res.bse_fe, res.pvalues
    b, c = float(fe["log_r"]), float(fe["log_r2"])
    vertex = float(np.exp(-b / (2 * c))) if c != 0 else np.nan
    lo, hi = float(df["r"].min()), float(df["r"].max())
    out = {
        "response": y, "with_mu0": with_mu0, "n": len(d),
        "b_log_r": b, "b_log_r_se": float(se["log_r"]), "b_log_r_p": float(pv["log_r"]),
        "c_log_r2": c, "c_log_r2_se": float(se["log_r2"]),
        "c_log_r2_p": float(pv["log_r2"]),
        "vertex_r": vertex,
        "vertex_inside": bool(np.isfinite(vertex) and lo < vertex < hi),
        "c_negative_sig": bool(c < 0 and float(pv["log_r2"]) < 0.05),
        "converged": bool(res.converged),
    }
    out["peak_present"] = bool(out["c_negative_sig"] and out["vertex_inside"])
    if with_mu0:
        out["b_mu0_c"] = float(fe["mu0_c"])
        out["b_mu0_c_p"] = float(pv["mu0_c"])
        key = "log_r:mu0_c" if "log_r:mu0_c" in fe.index else "mu0_c:log_r"
        out["b_logr_x_mu0"] = float(fe[key])
        out["b_logr_x_mu0_p"] = float(pv[key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--n-hosts", type=int, default=20)
    ap.add_argument("--n-s", type=int, default=100)
    ap.add_argument("--tables", default=str(ROOT / "tables"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "config" / "base.yaml")])
    args = ap.parse_args()

    cfg = load_config(*args.config)
    B = int(cfg["stats"]["bootstrap_B"])
    SEED = int(cfg["master_seed"])
    TAB = Path(args.tables); TAB.mkdir(parents=True, exist_ok=True)
    FIG = Path(args.figures); FIG.mkdir(parents=True, exist_ok=True)

    df = load_cells(Path(args.raw), args.n_hosts, args.n_s)
    df["mu_H_rel"] = df["mu_H"] / df["mu_H_0"]
    df["mutualistic"] = ((df["mu_H"] > df["mu_H_0"] + EPS_MUT)
                         & (df["mu_S"] > THETA_MUT)).astype(float)
    df["log_r"] = np.log(df["r"])
    df["log_r2"] = df["log_r"] ** 2
    mu0 = df.groupby("host_id")["mu_H_0"].first()
    df["mu0_c"] = df["host_id"].map(mu0 - mu0.mean())
    print(f"{len(df)} post cells, {df.host_id.nunique()} hosts, "
          f"r={df.r.min()}..{df.r.max()}", flush=True)

    # --- solver bookkeeping (START_HERE section 6-1) --------------------------
    paths = (df.groupby(["host_id", "solver_status"]).size()
             .rename("n").reset_index())
    mp = (df.assign(milp_path=df["milp_path"].fillna("none"))
          .groupby(["host_id", "milp_path"]).size().rename("n").reset_index())
    pd.concat([paths.assign(kind="solver_status").rename(
        columns={"solver_status": "value"}),
        mp.assign(kind="milp_path").rename(columns={"milp_path": "value"})],
        ignore_index=True).to_csv(TAB / "A4_milp_paths.csv", index=False)
    print("solver_status:", json.dumps(df["solver_status"].value_counts().to_dict()))
    print("milp_path    :", json.dumps(
        df["milp_path"].fillna("none").value_counts().to_dict()))
    print("mu_H < mu_H_0 violations:", int((df["mu_H"] < df["mu_H_0"] - 1e-9).sum()))

    # --- per (host, r) curves and per-host peaks ------------------------------
    curves = df.groupby(["host_id", "r"], as_index=False).agg(
        **{c: (c, "mean") for c, _ in RESPONSES},
        n=("mu_H", "size"), mu_H_0=("mu_H_0", "first"))
    curves.to_csv(TAB / "A4_curves.csv", index=False)

    rows = []
    for h, g in df.groupby("host_id"):
        c, pk = bootstrap_curve_and_peak(g, "mu_H", group="r", unit="symbiont_id",
                                         stat=np.mean, B=B, master_seed=SEED,
                                         table_id=f"A4_h{h}")
        _, pkm = bootstrap_curve_and_peak(g, "mutualistic", group="r",
                                          unit="symbiont_id", stat=np.mean, B=B,
                                          master_seed=SEED, table_id=f"A4m_h{h}")
        end = c.set_index("r")["value"]
        rows.append({
            "host_id": h, "mu_H_0": float(g["mu_H_0"].iloc[0]),
            "n_cells": len(g),
            "r_star": pk["peak_at"], "r_star_ci_lo": pk["peak_ci_lo"],
            "r_star_ci_hi": pk["peak_ci_hi"], "peak_mu_H": pk["peak_value"],
            "r_star_loess": loess_peak(c["r"], c["value"]),
            "peak_minus_first": pk["diff_peak_minus_first"],
            "peak_minus_first_lo": pk["diff_peak_minus_first_lo"],
            "peak_minus_last": pk["diff_peak_minus_last"],
            "peak_minus_last_lo": pk["diff_peak_minus_last_lo"],
            # the peak counts as real for this host when it beats BOTH endpoints
            # with a bootstrap CI that excludes zero, and is not at an endpoint
            "peak_beats_both_ends": bool(pk["diff_peak_minus_first_lo"] > 0
                                         and pk["diff_peak_minus_last_lo"] > 0),
            "r_star_interior": bool(pk["peak_at"] not in (c["r"].min(), c["r"].max())),
            "mu_H_first": float(end.iloc[0]), "mu_H_last": float(end.iloc[-1]),
            "r_star_mutual": pkm["peak_at"], "peak_mutual": pkm["peak_value"],
            "frac_detlimit": float((g["flag_detlimit"]).mean()),
        })
    hs = pd.DataFrame(rows)
    hs["peak_survives"] = hs["peak_beats_both_ends"] & hs["r_star_interior"]
    hs.to_csv(TAB / "A4_host_summary.csv", index=False)
    print(f"\nhosts whose peak survives: {int(hs.peak_survives.sum())}/{len(hs)}")
    print(f"r* range {hs.r_star.min()}..{hs.r_star.max()}, "
          f"median {hs.r_star.median()}")

    # --- mixed models ---------------------------------------------------------
    mm = []
    for y, _ in RESPONSES:
        for with_mu0 in (False, True):
            try:
                mm.append(fit_mixed(df, y, with_mu0))
            except Exception as exc:                        # noqa: BLE001
                mm.append({"response": y, "with_mu0": with_mu0,
                           "error": f"{exc.__class__.__name__}: {exc}"})
    mm = pd.DataFrame(mm)
    mm.to_csv(TAB / "A4_mixed_models.csv", index=False)
    print("\n=== mixed models (no mu_H_0 term) ===")
    print(mm[~mm["with_mu0"].astype(bool)][
        ["response", "c_log_r2", "c_log_r2_p", "vertex_r", "peak_present"]
    ].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # --- r* against mu_H_0 ----------------------------------------------------
    import statsmodels.api as sm
    X = sm.add_constant(hs["mu_H_0"].to_numpy())
    reg = []
    for yname in ("r_star", "r_star_loess", "peak_mu_H"):
        yv = hs[yname].astype(float).to_numpy()
        ok = np.isfinite(yv)
        fit = sm.OLS(yv[ok], X[ok]).fit()
        reg.append({"y": yname, "slope": float(fit.params[1]),
                    "slope_se": float(fit.bse[1]), "p": float(fit.pvalues[1]),
                    "r2": float(fit.rsquared), "n": int(ok.sum())})
    reg = pd.DataFrame(reg)
    reg.to_csv(TAB / "A4_rstar_vs_mu0.csv", index=False)
    print("\n=== r* vs mu_H_0 ===")
    print(reg.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # --- figures --------------------------------------------------------------
    import matplotlib.pyplot as plt
    ps.use()
    for norm, lab, fid in (
            ("raw", "mean post-evolution $\\mu^H$", "A4_mu_H_by_host"),
            ("mu0", "$\\langle\\mu^H\\rangle / \\mu^H_0$", "A4_mu_H_by_host_rel_mu0"),
            ("max", "$\\langle\\mu^H\\rangle$ / host maximum",
             "A4_mu_H_by_host_rel_max")):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        acc = []
        for h, g in curves.groupby("host_id"):
            g = g.sort_values("r")
            y = g["mu_H"].to_numpy()
            if norm == "mu0":
                y = y / g["mu_H_0"].to_numpy()
            elif norm == "max":
                y = y / np.nanmax(y)
            ax.plot(g["r"], y, lw=0.5, alpha=0.45, color="0.45")
            acc.append(pd.Series(y, index=g["r"].to_numpy()))
        mean = pd.concat(acc, axis=1).mean(axis=1)
        ax.plot(mean.index, mean.to_numpy(), lw=2.0, color="C0", label="mean")
        ax.set_xscale("log")
        ps.tidy(ax, "Symbiont genome size, $r$", lab)
        print("wrote", ps.save(fig, fid, FIG,
                               {"tables": ["A4_curves.csv", "A4_host_summary.csv"],
                                "normalisation": norm}))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter(hs["mu_H_0"], hs["r_star"], s=14, color="C0")
    xs = np.linspace(hs["mu_H_0"].min(), hs["mu_H_0"].max(), 50)
    sl = reg.loc[reg["y"] == "r_star"].iloc[0]
    fit = sm.OLS(hs["r_star"].astype(float).to_numpy(), X).fit()
    ax.plot(xs, fit.params[0] + fit.params[1] * xs, lw=1.2, color="C1",
            label=f"slope {sl['slope']:.1f} (p={sl['p']:.2g})")
    ax.legend(frameon=False, fontsize=6)
    ps.tidy(ax, "$\\mu^H_0$ (host without symbiont)", "peak location $r^*$")
    print("wrote", ps.save(fig, "A4_rstar_vs_mu0", FIG,
                           {"tables": ["A4_rstar_vs_mu0.csv"]}))
    plt.close(fig)


if __name__ == "__main__":
    main()
