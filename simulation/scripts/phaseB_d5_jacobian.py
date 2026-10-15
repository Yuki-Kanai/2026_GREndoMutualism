"""A2-D5 — numerical vs analytical Jacobian of f^S* with respect to c^H.

07_run2_workorder.md Phase B 2.2 / 03_additional_analyses.md A2-D5.

The paper's local linearisation is

    d f_b / d c_b ~ I_b ,   d f_u / d c_b ~ S = R^S_{u+} (R^S_{b+})^+ ,
    d f / d c_u   ~ O

Which of these is a real check matters:

  ||J_bb - I_b||_F   identically 0 while the active set does not change
  ||J_bu||_F         identically 0 for the same reason
  ||J_ub - S||_F     THE verification; the content of the approximation is here
  ||J_uu||_F         also a real check ("no response along unbounded directions")

Cells whose active set switches under the perturbation are flagged and excluded
from the aggregate; the switching rate itself is reported per r because it
measures how rough the fitness landscape is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.analyze import bounded_predicates, sensitivity_matrix
from grhd.cfgio import load_config
from grhd.lp import SymbiontSolver
from grhd.pipeline import c_H_samples, make_R_S
from grhd.rng import derive
from grhd.stats import spearman

ap = argparse.ArgumentParser()
ap.add_argument("--r-grid", default="2,5,10,15,20,25,40,60,100")
ap.add_argument("--n-cells", type=int, default=200)
ap.add_argument("--deltas", default="1e-3,1e-4,1e-5,1e-6")
ap.add_argument("--tables", default="tables")
ap.add_argument("--figures", default="figures")
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
m = int(cfg["model"]["m"])
tol = cfg["tolerances"]
TA, TR, TW = float(tol["tol_abs"]), float(tol["tol_rel"]), float(tol["tol_w"])
RC = float(tol["rcond"])
N_C = 100
DELTAS = [float(x) for x in a.deltas.split(",")]
TAB = Path(a.tables)

rows = []
for r in [int(x) for x in a.r_grid.split(",")]:
    rng = derive(int(cfg["master_seed"]), "subsample", fig_id=f"A2D5_r{r}")
    picks = rng.choice(300 * N_C, size=a.n_cells, replace=False)
    by_sym: dict[int, list[int]] = {}
    for p in picks:
        by_sym.setdefault(int(p // N_C), []).append(int(p % N_C))
    for sid, cids in sorted(by_sym.items()):
        R_S = make_R_S(cfg, r, sid)
        ss = SymbiontSolver(R_S, cfg)
        CH = c_H_samples(cfg, sid, N_C)
        for cid in cids:
            c_H = CH[cid]
            base = ss.solve(c_H)
            bo, un = bounded_predicates(c_H, base.f_S, TA, TR)
            act = base.w_S > TW
            b, u = int(bo.sum()), int(un.sum())
            l1 = float(np.sum(np.abs(c_H)))
            S = sensitivity_matrix(R_S, bo, un, act, RC) if (b and u and act.any()) \
                else np.zeros((u, b))
            for drel in DELTAS:
                d = drel * l1
                J = np.zeros((m - 1, m - 1))
                switched = False
                for i in range(m - 1):
                    cp, cm = c_H.copy(), c_H.copy()
                    cp[i] += d
                    cm[i] = max(cm[i] - d, 0.0)
                    sp, sm = ss.solve(cp), ss.solve(cm)
                    step = cp[i] - cm[i]
                    J[:, i] = (sp.f_S - sm.f_S) / step
                    for cc, sol in ((cp, sp), (cm, sm)):
                        b2, u2 = bounded_predicates(cc, sol.f_S, TA, TR)
                        if not np.array_equal(b2, bo):
                            switched = True
                J_bb = J[np.ix_(bo, bo)]
                J_bu = J[np.ix_(bo, un)]
                J_ub = J[np.ix_(un, bo)]
                J_uu = J[np.ix_(un, un)]
                rows.append({
                    "r": r, "symbiont_id": sid, "c_id": cid, "delta_rel": drel,
                    "b": b, "u": u, "r_plus": int(act.sum()),
                    "active_set_switched": switched,
                    "s_analytic": float(np.linalg.norm(S, ord="fro")),
                    "norm_J_ub": float(np.linalg.norm(J_ub, ord="fro")),
                    "err_J_ub_minus_S": float(np.linalg.norm(J_ub - S, ord="fro"))
                    if S.shape == J_ub.shape else np.nan,
                    "norm_J_uu": float(np.linalg.norm(J_uu, ord="fro")),
                    "err_J_bb_minus_I": float(np.linalg.norm(
                        J_bb - np.eye(b), ord="fro")) if b else 0.0,
                    "norm_J_bu": float(np.linalg.norm(J_bu, ord="fro")),
                })
        ss.close()
    print(f"  done r = {r}", flush=True)

d5 = pd.DataFrame(rows)
d5.to_parquet("data/derived/A2_D5_jacobian.parquet", index=False)
pd.set_option("display.width", 240)

sw = d5.groupby(["r", "delta_rel"])["active_set_switched"].mean().unstack()
print("\n=== fraction of cells whose active set switched under the perturbation ===")
print(sw.round(3).to_string())

ok = d5[~d5["active_set_switched"] & (d5["b"] > 0) & (d5["u"] > 0)]
agg = ok.groupby(["r", "delta_rel"]).agg(
    n=("norm_J_ub", "size"),
    identity_err_J_bb=("err_J_bb_minus_I", "max"),
    identity_norm_J_bu=("norm_J_bu", "max"),
    err_J_ub_minus_S_median=("err_J_ub_minus_S", "median"),
    err_J_ub_minus_S_max=("err_J_ub_minus_S", "max"),
    norm_J_uu_median=("norm_J_uu", "median"),
    norm_J_uu_max=("norm_J_uu", "max")).reset_index()
agg.to_csv(TAB / "A2_D5.csv", index=False)
print("\n=== aggregate over cells whose active set did NOT switch ===")
print(agg.round(6).to_string(index=False))

# the actual verification: numerical ||J_ub|| against analytical s
fits = []
for drel, g in ok.groupby("delta_rel"):
    x = g["s_analytic"].to_numpy(float)
    y = g["norm_J_ub"].to_numpy(float)
    X = np.column_stack([np.ones(x.size), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    fits.append({"delta_rel": drel, "n": int(x.size), "slope": float(beta[1]),
                 "intercept": float(beta[0]),
                 "r2": 1 - float((res ** 2).sum()) / ss_tot if ss_tot else np.nan,
                 "spearman": spearman(x, y)[0],
                 "max_abs_diff": float(np.max(np.abs(y - x))),
                 "median_abs_diff": float(np.median(np.abs(y - x)))})
fit = pd.DataFrame(fits)
fit.to_csv(TAB / "A2_D5_fit.csv", index=False)
print("\n=== ||J_ub||_F (numerical) vs s = ||S||_F (analytical) ===")
print(fit.round(6).to_string(index=False))

json.dump({"r_grid": a.r_grid, "n_cells_per_r": a.n_cells, "deltas": DELTAS,
           "n_rows": int(len(d5)),
           "frac_switched_overall": float(d5["active_set_switched"].mean())},
          open(TAB / "A2_D5_meta.json", "w"), indent=2)

# --------------------------------------------------------------------------- #
from grhd import plotstyle as ps                                # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

ps.use()
fig, axes = plt.subplots(1, 3, figsize=(ps.COL2, 2.0))
fig.subplots_adjust(wspace=0.40)

ax = axes[0]
ref = ok[ok["delta_rel"] == 1e-4]
ax.scatter(ref["s_analytic"], ref["norm_J_ub"], s=2.0, c=ps.C["mean"],
           alpha=0.35, linewidths=0, rasterized=True)
lim = [0, float(np.nanquantile(ref["s_analytic"], 0.995)) * 1.05]
ax.plot(lim, lim, color=ps.C["ref"], lw=0.8, ls=(0, (3.5, 2)))
ax.set_xlim(lim); ax.set_ylim(lim)
f = fit[fit["delta_rel"] == 1e-4].iloc[0]
ax.text(0.04, 0.95, f"slope = {f['slope']:.4f}\n$R^2$ = {f['r2']:.5f}\n"
                    f"$n$ = {int(f['n'])}", transform=ax.transAxes, va="top",
        fontsize=6)
ps.tidy(ax, r"analytical $s=\|S\|_{F}$", r"numerical $\|J_{ub}\|_{F}$")
ax.set_title(r"$\delta/|c^H|_1 = 10^{-4}$", loc="left", fontsize=6.5,
             color="#666666")
ps.panel_label(ax, "A", dx=-0.30)

ax = axes[1]
for i, drel in enumerate(DELTAS):
    g = agg[agg["delta_rel"] == drel]
    ax.plot(g["r"], g["err_J_ub_minus_S_median"], marker="o", markersize=2.2,
            lw=1.0, color=ps.SERIES[i % len(ps.SERIES)],
            label=rf"$\delta/|c^H|_1=${drel:g}")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
ax.minorticks_off()
ax.legend(loc="best", fontsize=5.4)
ps.tidy(ax, "Symbiont genome size, $r$", r"median $\|J_{ub}-S\|_{F}$")
ps.panel_label(ax, "B", dx=-0.34)

ax = axes[2]
for i, drel in enumerate(DELTAS):
    g = sw[drel]
    ax.plot(g.index, g.to_numpy(), marker="o", markersize=2.2, lw=1.0,
            color=ps.SERIES[i % len(ps.SERIES)],
            label=rf"$\delta/|c^H|_1=${drel:g}")
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
ax.minorticks_off()
top = float(np.nanmax(sw.to_numpy())) if sw.size else 0.05
ax.set_ylim(-0.02 * max(top, 0.01), max(top, 0.01) * 1.45)
ax.legend(loc="best", fontsize=5.4)
ps.tidy(ax, "Symbiont genome size, $r$", "fraction with active-set switch")
ps.panel_label(ax, "C", dx=-0.30)
print("\nwrote", ps.save(fig, "A2_D5_jacobian", Path(a.figures),
                         {"tables": ["A2_D5.csv", "A2_D5_fit.csv"],
                          "r_grid": a.r_grid, "n_cells_per_r": a.n_cells,
                          "deltas": DELTAS}))
