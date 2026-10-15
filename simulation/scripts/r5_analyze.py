"""R5 / gate G9 -- does r* ~ m-1 hold in BOTH designs?

Purpose
-------
Fig.6C of the paper reports that the optimal symbiont genome size tracks the number
of metabolites, r* ~ m-1.  Reproducing that with r^H held fixed at 24 confounds two
things: as m grows, a 24-mode host becomes relatively weaker, and G8 measured that
the host population itself changes quality with m (the continuous part falls from
28.1 % to 0.3 %).  08_r5_two_designs.md therefore runs two designs and takes as a
result only what holds in both:

  R5-A  r^H = round(2.4 m)   host ability held fixed relative to m (primary)
  R5-B  r^H = 24             the paper's setting (kept for continuity)

The ratio 2.4 is not a new free parameter: it is fixed by the paper's own m = 10,
r^H = 24, so the two designs coincide at m = 10.

Question
--------
Is r* ~ m-1 reproduced, and is it reproduced in both designs?

Method
------
08 section 4.  Per host: bootstrap curve and argmax of mean mu^H (ties to the
smallest r, independent resampling within each r, LOESS peak alongside).  Then r*
against m by OLS, with the slope tested against 1.  The same is done for the
mutualism fraction, and for the b-axis version (mean b at the maximising r,
against the (m-1)/2 reference).

Outputs
-------
tables/R5_host_peaks.csv      per (design, m, host): r*, CI, peak value, mean b
tables/R5_curves.csv          per (design, m, host, r): means and support counts
tables/R5_rstar_vs_m.csv      OLS of r* on m, slope tested against 1
tables/R5_health.csv          per (design, m): mu_H_0, detlimit, infeasible, paths
figures/R5_rstar_vs_m.*       both designs overlaid, r = m-1 reference
figures/R5_b_vs_m.*           b-axis version, (m-1)/2 reference
figures/R5_muH_curves.*       mean mu^H vs r, one panel per m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import grhd.plotstyle as ps  # noqa: E402
from grhd.cfgio import ROOT, load_config  # noqa: E402
from grhd.stats import bootstrap_curve_and_peak, loess_peak  # noqa: E402

EPS_MUT = 1e-3
THETA_MUT = 1e-6


def r_H_of(design: str, m: int) -> int:
    return int(round(2.4 * m)) if design == "A" else 24


def block_dir(raw: Path, design: str, m: int, host: int) -> Path:
    """Where run_r5.sh put this cell.  m = 10 is the run-1 / A4 setting, reused."""
    if m == 10 and r_H_of(design, m) == 24:
        return raw / ("post_paper" if host == 0 else f"post_paper_h{host}")
    rH = r_H_of(design, m)
    tag = f"post_paper_m{m}_rH{rH}" + (f"_h{host}" if host else "")
    return raw / tag


def load_cells(raw: Path, designs, ms, n_hosts: dict, n_s: int) -> pd.DataFrame:
    """n_hosts is per design: R5-B carries 10 hosts, R5-A 5 (the control arm)."""
    parts, missing = [], []
    for d in designs:
        for m in ms:
            for h in range(n_hosts[d]):
                p = block_dir(raw, d, m, h)
                files = sorted(p.glob("cells_*.parquet"))
                if not files:
                    missing.append(f"{d}/m{m}/h{h}")
                    continue
                df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                df = df[df["symbiont_id"] < n_s].copy()
                df["design"], df["host_id"] = d, h
                parts.append(df)
    if missing:
        print(f"  WARNING: {len(missing)} missing cells, e.g. {missing[:5]}")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--designs", default="A,B")
    ap.add_argument("--m-grid", default="5,10,15,20,25")
    ap.add_argument("--n-hosts", default="A=5,B=10",
                    help="hosts per design, e.g. A=5,B=10")
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
    designs = args.designs.split(",")
    ms = [int(x) for x in args.m_grid.split(",")]

    n_hosts = {kv.split("=")[0]: int(kv.split("=")[1])
               for kv in args.n_hosts.split(",")}
    df = load_cells(Path(args.raw), designs, ms, n_hosts, args.n_s)
    df["mutualistic"] = ((df["mu_H"] > df["mu_H_0"] + EPS_MUT)
                         & (df["mu_S"] > THETA_MUT)).astype(float)
    print(f"{len(df)} post cells; designs={designs} m={ms} "
          f"hosts={df.host_id.nunique()}", flush=True)

    # --- health, per (design, m) ---------------------------------------------
    health = df.groupby(["design", "m"], as_index=False).agg(
        n=("mu_H", "size"), r_H=("r_H", "first"),
        n_hosts=("host_id", "nunique"),
        mu_H_0_min=("mu_H_0", "min"), mu_H_0_median=("mu_H_0", "median"),
        mu_H_0_max=("mu_H_0", "max"),
        frac_detlimit=("flag_detlimit", "mean"),
        frac_infeasible=("flag_infeasible", "mean"),
        frac_violation=("delta_mu_H", lambda x: float((x < -1e-9).mean())),
    )
    paths = (df.assign(milp_path=df["milp_path"].fillna("none"))
             .groupby(["design", "m", "milp_path"]).size().rename("n").reset_index())
    paths.to_csv(TAB / "R5_milp_paths.csv", index=False)
    health.to_csv(TAB / "R5_health.csv", index=False)
    print(health.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("milp_path:", df["milp_path"].fillna("none").value_counts().to_dict())

    # --- curves and per-host peaks -------------------------------------------
    curves = df.groupby(["design", "m", "host_id", "r"], as_index=False).agg(
        mu_H=("mu_H", "mean"), mutualistic=("mutualistic", "mean"),
        b=("n_bounded", "mean"), s=("s", "median"),
        n_excreted=("n_excreted", "mean"),
        n_support=("mu_H", "size"))      # support count, 08 section 5
    curves["r_H"] = [r_H_of(d, m) for d, m in zip(curves["design"], curves["m"])]
    curves.to_csv(TAB / "R5_curves.csv", index=False)

    rows = []
    for (d, m, h), g in df.groupby(["design", "m", "host_id"]):
        c, pk = bootstrap_curve_and_peak(g, "mu_H", group="r", unit="symbiont_id",
                                         stat=np.mean, B=B, master_seed=SEED,
                                         table_id=f"R5_{d}_m{m}_h{h}")
        _, pkm = bootstrap_curve_and_peak(g, "mutualistic", group="r",
                                          unit="symbiont_id", stat=np.mean, B=B,
                                          master_seed=SEED,
                                          table_id=f"R5m_{d}_m{m}_h{h}")
        at = pk["peak_at"]
        rows.append({
            "design": d, "m": m, "host_id": h, "r_H": r_H_of(d, m),
            "mu_H_0": float(g["mu_H_0"].iloc[0]),
            "r_star": at, "r_star_ci_lo": pk["peak_ci_lo"],
            "r_star_ci_hi": pk["peak_ci_hi"], "peak_mu_H": pk["peak_value"],
            "r_star_loess": loess_peak(c["r"], c["value"]),
            # b-axis version: mean b at the maximising r (08 section 4)
            "b_at_peak": float(g.loc[g["r"] == at, "n_bounded"].mean()),
            "n_at_peak": int((g["r"] == at).sum()),
            "peak_minus_first_lo": pk["diff_peak_minus_first_lo"],
            "peak_minus_last_lo": pk["diff_peak_minus_last_lo"],
            "peak_beats_both_ends": bool(pk["diff_peak_minus_first_lo"] > 0
                                         and pk["diff_peak_minus_last_lo"] > 0),
            "r_star_mutual": pkm["peak_at"], "peak_mutual": pkm["peak_value"],
        })
    hp = pd.DataFrame(rows)
    hp.to_csv(TAB / "R5_host_peaks.csv", index=False)

    per_m = hp.groupby(["design", "m"], as_index=False).agg(
        r_star_mean=("r_star", "mean"), r_star_median=("r_star", "median"),
        r_star_sd=("r_star", "std"), r_star_min=("r_star", "min"),
        r_star_max=("r_star", "max"),
        b_at_peak_mean=("b_at_peak", "mean"),
        r_star_mutual_median=("r_star_mutual", "median"),
        n_hosts=("host_id", "size"),
        n_peak_survives=("peak_beats_both_ends", "sum"))
    per_m["m_minus_1"] = per_m["m"] - 1
    per_m["half_m_minus_1"] = (per_m["m"] - 1) / 2
    per_m.to_csv(TAB / "R5_rstar_by_m.csv", index=False)
    print("\n=== r* by m ===")
    print(per_m.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # --- r* ~ m, slope tested against 1 --------------------------------------
    import statsmodels.api as sm
    from scipy import stats as sps
    reg = []
    for d in designs:
        for yname, ref in (("r_star", 1.0), ("r_star_loess", 1.0),
                           ("r_star_mutual", 1.0), ("b_at_peak", 0.5)):
            sub = hp[hp["design"] == d]
            y = sub[yname].astype(float).to_numpy()
            X = sm.add_constant(sub["m"].astype(float).to_numpy())
            ok = np.isfinite(y)
            if ok.sum() < 3:
                continue
            fit = sm.OLS(y[ok], X[ok]).fit()
            t = (fit.params[1] - ref) / fit.bse[1]
            reg.append({
                "design": d, "y": yname, "reference_slope": ref,
                "slope": float(fit.params[1]), "slope_se": float(fit.bse[1]),
                "slope_ci_lo": float(fit.conf_int()[1][0]),
                "slope_ci_hi": float(fit.conf_int()[1][1]),
                "intercept": float(fit.params[0]), "r2": float(fit.rsquared),
                "t_vs_reference": float(t),
                "p_vs_reference": float(2 * sps.t.sf(abs(t), fit.df_resid)),
                "n": int(ok.sum()),
            })
    reg = pd.DataFrame(reg)
    reg.to_csv(TAB / "R5_rstar_vs_m.csv", index=False)
    print("\n=== r* ~ m (slope tested against the reference) ===")
    print(reg[["design", "y", "reference_slope", "slope", "slope_ci_lo",
               "slope_ci_hi", "p_vs_reference", "r2"]].to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))

    # --- figures --------------------------------------------------------------
    import matplotlib.pyplot as plt
    ps.use()
    colors = {"A": "C0", "B": "C1"}
    labels = {"A": "R5-A  $r^H = 2.4\\,m$", "B": "R5-B  $r^H = 24$"}

    for yname, ref_fn, ref_lab, ylab, fid in (
            ("r_star", lambda mm: mm - 1, "$r = m-1$", "peak location $r^*$",
             "R5_rstar_vs_m"),
            ("b_at_peak", lambda mm: (mm - 1) / 2, "$(m-1)/2$",
             "$\\langle b\\rangle$ at $r^*$", "R5_b_vs_m")):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        mm = np.array(sorted(hp["m"].unique()), dtype=float)
        ax.plot(mm, ref_fn(mm), lw=1.0, ls="--", color="0.4", label=ref_lab)
        for d in designs:
            sub = hp[hp["design"] == d]
            jitter = (0.15 if d == "A" else -0.15)
            ax.scatter(sub["m"] + jitter, sub[yname], s=8, alpha=0.5,
                       color=colors[d])
            mean = sub.groupby("m")[yname].mean()
            ax.plot(mean.index, mean.to_numpy(), lw=1.6, color=colors[d],
                    marker="o", ms=3, label=labels[d])
        ax.legend(frameon=False, fontsize=6)
        ps.tidy(ax, "Number of metabolites, $m$", ylab)
        print("wrote", ps.save(fig, fid, FIG,
                               {"tables": ["R5_host_peaks.csv",
                                           "R5_rstar_vs_m.csv"]}))
        plt.close(fig)

    n = len(ms)
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n + 0.6, 2.1), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), ms):
        for d in designs:
            sub = curves[(curves["design"] == d) & (curves["m"] == m)]
            if sub.empty:
                continue
            mean = sub.groupby("r")["mu_H"].mean()
            ax.plot(mean.index, mean.to_numpy(), lw=1.3, color=colors[d],
                    label=labels[d] if m == ms[0] else None)
        ax.axvline(m - 1, lw=0.8, ls="--", color="0.4")
        ax.set_xscale("log")
        ax.set_title(f"$m={m}$", fontsize=7)
        ps.tidy(ax, "$r$", "mean $\\mu^H$" if m == ms[0] else None)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=5)
    fig.tight_layout()
    print("wrote", ps.save(fig, "R5_muH_curves", FIG,
                           {"tables": ["R5_curves.csv"]}))
    plt.close(fig)


if __name__ == "__main__":
    main()
