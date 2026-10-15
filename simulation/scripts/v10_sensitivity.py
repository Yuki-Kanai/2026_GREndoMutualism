"""V10 — tolerance and solver-setting sensitivity (04_validation.md).

This is a measure-and-report test, not a pass/fail one.  Writes
tables/V10_tolerance_sensitivity.csv.

Expectation stated in 01_model_spec.md section 8: the min-norm QP is strictly
convex, so w^S* is unique and b / r+ / alpha / s must not depend on the LP's
crossover or presolve settings, and the MIP presolve must not change the Level 3
optimum.  Any dependence found here is an implementation or tolerance problem,
not a property of the model.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grhd.analyze import analyze_cell
from grhd.cfgio import load_config
from grhd.lp import HostSolver, SymbiontSolver
from grhd.pipeline import c_H_samples, get_host, make_R_S

ap = argparse.ArgumentParser()
ap.add_argument("--r-grid", default="2,5,10,20,40,100")
ap.add_argument("--n-sym", type=int, default=20)
ap.add_argument("--n-c", type=int, default=5)
ap.add_argument("--out", default="tables/V10_tolerance_sensitivity.csv")
ap.add_argument("--milp-r", default="2,5,10")
ap.add_argument("--milp-n", type=int, default=5)
ap.add_argument("--config", nargs="+", default=["config/base.yaml"])
a = ap.parse_args()

cfg = load_config(*a.config)
m = cfg["model"]["m"]
hd, cE = get_host(cfg, 0, m)
hs = HostSolver(hd.R_H, cfg)
R_GRID = [int(x) for x in a.r_grid.split(",")]

CELLS = []
for r in R_GRID:
    for sid in range(a.n_sym):
        R_S = make_R_S(cfg, r, sid)
        CH = c_H_samples(cfg, sid, a.n_c)
        CELLS.append((r, sid, R_S, CH))
print(f"{len(CELLS)} (R_S, c^H batch) pairs, {len(CELLS) * a.n_c} cells", flush=True)


def run(conf):
    """Solve every cell under `conf`; return a tidy frame of the reported quantities."""
    rows = []
    for r, sid, R_S, CH in CELLS:
        ss = SymbiontSolver(R_S, conf)
        for cid, c_H in enumerate(CH):
            sol = ss.solve(c_H)
            an = analyze_cell(R_S, c_H, sol.f_S, sol.w_S, sol.lam2, conf)
            h = hs.solve(sol.f_S, cE)
            rows.append({"r": r, "sid": sid, "cid": cid, "mu_S": sol.mu_S,
                         "mu_H": h.mu_H if h.status == "optimal" else np.nan,
                         "b": an.n_bounded, "r_plus": an.r_plus,
                         "alpha": an.alpha, "s": an.s,
                         "n_excreted": an.n_excreted,
                         "qp_path": sol.qp_path, "lp_path": sol.lp_path})
        ss.close()
    return pd.DataFrame(rows)


def analysis_only(base_sol, conf):
    """Re-derive the analysis quantities from stored solutions (no re-solve)."""
    rows = []
    for (r, sid, R_S, CH), sols in base_sol:
        for cid, (c_H, sol) in enumerate(zip(CH, sols)):
            an = analyze_cell(R_S, c_H, sol.f_S, sol.w_S, sol.lam2, conf)
            rows.append({"r": r, "sid": sid, "cid": cid, "mu_S": sol.mu_S,
                         "mu_H": np.nan, "b": an.n_bounded, "r_plus": an.r_plus,
                         "alpha": an.alpha, "s": an.s,
                         "n_excreted": an.n_excreted,
                         "qp_path": sol.qp_path, "lp_path": sol.lp_path})
    return pd.DataFrame(rows)


print("baseline...", flush=True)
base = run(cfg)
stored = []
for r, sid, R_S, CH in CELLS:
    ss = SymbiontSolver(R_S, cfg)
    stored.append(((r, sid, R_S, CH), [ss.solve(c) for c in CH]))
    ss.close()


def compare(name, group, other, resolved):
    d = {"variant": name, "group": group, "resolved": resolved,
         "n": int(len(other))}
    for col, kind in (("mu_S", "num"), ("mu_H", "num"), ("s", "num"),
                      ("alpha", "num"), ("b", "int"), ("r_plus", "int"),
                      ("n_excreted", "int")):
        x = base[col].to_numpy(float)
        y = other[col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if kind == "int":
            d[f"{col}_frac_changed"] = float(np.mean(x[ok] != y[ok])) if ok.any() else np.nan
            d[f"{col}_max_abs_diff"] = float(np.max(np.abs(x[ok] - y[ok]))) if ok.any() else np.nan
        else:
            den = np.maximum(np.abs(x[ok]), 1e-12)
            d[f"{col}_max_abs_diff"] = float(np.max(np.abs(x[ok] - y[ok]))) if ok.any() else np.nan
            d[f"{col}_max_rel_diff"] = float(np.max(np.abs(x[ok] - y[ok]) / den)) if ok.any() else np.nan
            d[f"{col}_frac_changed"] = float(np.mean(np.abs(x[ok] - y[ok]) > 1e-9 * den)) if ok.any() else np.nan
    d["qp_fallback_rate"] = float((other["qp_path"] == "lp_vertex").mean())
    d["lp_fallback_rate"] = float((other["lp_path"] == "simplex_fallback").mean())
    return d


results = [compare("baseline", "reference", base, False)]

# --- analysis-only tolerances (no re-solve) ---------------------------------
for key, vals, group in (("tol_abs", [1e-7, 1e-8, 1e-9], "bounded_tol"),
                         ("tol_rel", [1e-5, 1e-6, 1e-7], "bounded_tol"),
                         ("tol_w", [1e-7, 1e-8, 1e-9], "active_mode_tol"),
                         ("tol_exc", [1e-7, 1e-8, 1e-9], "excretion_tol"),
                         ("rcond", [1e-8, 1e-10, 1e-12], "pinv_rcond")):
    for v in vals:
        c2 = copy.deepcopy(cfg)
        c2["tolerances"][key] = v
        results.append(compare(f"{key}={v:g}", group, analysis_only(stored, c2), False))
        print("  ", results[-1]["variant"], flush=True)

# --- zero-c^H rule (A14) -----------------------------------------------------
for rule in ("bounded_if_zero_flux", "always_unbounded"):
    c2 = copy.deepcopy(cfg)
    c2["flags"]["zero_cH_rule"] = rule
    results.append(compare(f"zero_cH_rule={rule}", "zero_cH_rule",
                           analysis_only(stored, c2), False))

# --- settings that need a re-solve ------------------------------------------
resolve_variants = {
    "eps_mu=1e-8": {"tolerances": {"eps_mu_abs": 1e-8, "eps_mu_rel": 1e-8}},
    "eps_mu=1e-9 (default)": {},
    "eps_mu=1e-10": {"tolerances": {"eps_mu_abs": 1e-10, "eps_mu_rel": 1e-10}},
    "ipm+crossover_on+presolve_on": {
        "solver": {"lp": dict(cfg["solver"]["lp"], run_crossover="on",
                              presolve="on")}},
    "simplex+presolve_on": {
        "solver": {"lp": {"solver": "simplex", "presolve": "on",
                          "time_limit": 60.0}}},
    "no_cH_normalisation": {"flags": {"normalize_cH": False}},
}
for name, patch in resolve_variants.items():
    c2 = copy.deepcopy(cfg)
    for kk, vv in patch.items():
        c2[kk].update(vv)
    results.append(compare(name, "solver_or_eps", run(c2), True))
    print("  ", results[-1]["variant"], flush=True)

out = pd.DataFrame(results)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.out, index=False)
print(out[["variant", "group", "b_frac_changed", "r_plus_frac_changed",
           "s_max_rel_diff", "mu_S_max_rel_diff",
           "n_excreted_frac_changed"]].to_string(index=False), flush=True)

# --- Level 3: MIP presolve off vs on ----------------------------------------
from grhd.milp import solve_host_evolution
rows = []
for r in [int(x) for x in a.milp_r.split(",")]:
    for sid in range(a.milp_n):
        R_S = make_R_S(cfg, r, sid)
        rec = {"r": r, "symbiont_id": sid}
        for label, pre in (("presolve_off", 0), ("presolve_on", 1)):
            c2 = copy.deepcopy(cfg)
            c2["solver"]["milp"]["presolve"] = pre
            s = solve_host_evolution(R_S, hd.R_H, cE, c2)
            rec[f"mu_H_{label}"] = s.mu_H
            rec[f"status_{label}"] = s.status
            rec[f"seconds_{label}"] = s.t_milp
        rec["abs_diff"] = abs(rec["mu_H_presolve_off"] - rec["mu_H_presolve_on"])
        rows.append(rec)
        print("  milp", rec["r"], rec["symbiont_id"],
              f"diff={rec['abs_diff']:.2e}", flush=True)
mp = pd.DataFrame(rows)
mp.to_csv(Path(a.out).with_name("V10_milp_presolve.csv"), index=False)
print("MILP presolve: max |mu_H diff| = %.3e over n=%d"
      % (mp["abs_diff"].max(), len(mp)), flush=True)
