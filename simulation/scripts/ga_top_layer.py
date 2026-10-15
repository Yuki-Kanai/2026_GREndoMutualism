"""GA for the top layer (Eq.3), as a cross-check on -- and an evolvability claim for -- the MILP.

Purpose
-------
Level 3 is currently solved as a single-stage MILP: the two lower levels are
replaced by their KKT systems and complementarity is expressed with SOS1 sets.
That is a global solver, but it says nothing about whether a *population* could
find such a c^H.  This script instead searches c^H directly with a genetic
algorithm, evaluating each candidate by actually solving the two lower levels:

    c^H  ->  Level 1 (symbiont maximises mu^S, min-norm tie-break)  ->  f^{S*}
         ->  Level 2 (host maximises mu^H given c^E - f^{S*})       ->  mu^H

Two things come out of this:

1. **A correctness check on the MILP.**  Any c^H the GA finds is feasible for
   Eq.3, so mu^H_GA is a lower bound on the MILP optimum (01_model_spec.md
   section 10.1-9).  If mu^H_GA > mu^H_MILP anywhere, the MILP is wrong.
2. **An evolvability statement.**  If a plain GA on c^H gets within a fraction of
   a percent of the global optimum in less wall-clock time than the MILP takes,
   then the optimal host regulation is reachable by mutation and selection, not
   just by a global solver.

Two modes
---------
full  (default, the one that carries the evolvability claim)
      The genome is c^H itself: m-1 non-negative numbers in the box [0, C_max].
      Nothing about the geometry of the problem is built in.

ray   (the cheaper variant, reported alongside)
      The genome is only the *direction* d, |d|_1 = 1.  Along a ray c^H = t*d the
      active set is invariant and f^{S*}(t d) = t f^{S*}(d), so Level 1 is solved
      once per individual, and t -> mu^H(t d) is concave piecewise linear and is
      maximised exactly by golden-section search using host LPs alone
      (01_model_spec.md section 10.1, points 6-9).  This is much cheaper per
      individual but bakes in structure a population would not know about --
      hence "sub".

Output
------
tables/GA_vs_MILP.csv        one row per (mode, r, symbiont): both optima, both times
tables/GA_traces.csv         best-so-far against evaluation count, for the figure
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grhd.cfgio import ROOT, load_config  # noqa: E402
from grhd.lp import HostSolver, SymbiontSolver  # noqa: E402
from grhd.pipeline import get_host, make_R_S  # noqa: E402
from grhd.rng import derive_seed  # noqa: E402


# --------------------------------------------------------------------------- #
class Evaluator:
    """c^H -> mu^H, by solving the two lower levels exactly.  Counts evaluations."""

    def __init__(self, R_S, R_H, cE, cfg):
        self.ss = SymbiontSolver(R_S, cfg)
        self.hs = HostSolver(R_H, cfg)
        self.cE = cE
        self.n_eval = 0
        self.n_host_lp = 0

    def mu_H(self, c_H):
        self.n_eval += 1
        sol = self.ss.solve(np.asarray(c_H, dtype=np.float64))
        if sol.status not in ("optimal", "qp_fallback"):
            return 0.0, None
        h = self.hs.solve(sol.f_S, self.cE)
        self.n_host_lp += 1
        # Level 2 infeasible: the host cannot run at all.  Score it below the
        # c^H = 0 baseline so selection walks away from it (01 section 4.1 keeps
        # NaN for the *data*; this is only the GA's fitness).
        if h.status == "infeasible" or not np.isfinite(h.mu_H):
            return 0.0, sol.f_S
        return float(h.mu_H), sol.f_S

    def mu_H_on_ray(self, f_S_dir, t):
        """mu^H at c^H = t*d, reusing f^{S*}(d): f^{S*}(t d) = t f^{S*}(d)."""
        self.n_host_lp += 1
        h = self.hs.solve(t * f_S_dir, self.cE)
        if h.status == "infeasible" or not np.isfinite(h.mu_H):
            return -np.inf
        return float(h.mu_H)

    def close(self):
        try:
            self.ss.close()
        except Exception:                                   # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
def _tournament(rng, fit, k=3):
    idx = rng.integers(0, len(fit), size=k)
    return int(idx[np.argmax(fit[idx])])


def ga_full(ev, m, rng, pop_size=100, generations=200, c_max=10.0,
            p_cross=0.9, p_mut=0.2, elite=2, trace=None,
            seed_c=None, n_seed_perturb=8):
    """GA on c^H itself.  No use of homogeneity, concavity or the active set.

    seed_c: an incumbent c^H (e.g. the MILP's, truncated or not) injected into
    the initial population together with n_seed_perturb jittered copies.  With
    elitism this makes the returned value >= mu^H(seed_c) by construction, so a
    seeded rescue can never be worse than the value it replaces (NEXT section
    2.4, option (c)).  Everything else about the search is unchanged.
    """
    n = m - 1
    # initial population: a spread of magnitudes and sparsities, nothing clever
    pop = np.zeros((pop_size, n))
    for i in range(pop_size):
        scale = 10.0 ** rng.uniform(-3, np.log10(c_max))
        v = rng.random(n) * scale
        v[rng.random(n) < rng.uniform(0.0, 0.7)] = 0.0       # random sparsity
        pop[i] = np.clip(v, 0.0, c_max)
    pop[0] = 0.0                                              # the c^H = 0 baseline
    if seed_c is not None:
        s0 = np.clip(np.asarray(seed_c, dtype=np.float64), 0.0, c_max)
        pop[1] = s0
        for i in range(2, min(2 + int(n_seed_perturb), pop_size)):
            v = s0 * np.exp(rng.normal(0.0, 0.3, n))
            v[rng.random(n) < 0.15] = 0.0                      # drop a component
            on = rng.random(n) < 0.15                          # switch one on
            v[on] = rng.random(int(on.sum())) * c_max
            pop[i] = np.clip(v, 0.0, c_max)
    fit = np.array([ev.mu_H(c)[0] for c in pop])
    best_i = int(np.argmax(fit))
    best, best_c = float(fit[best_i]), pop[best_i].copy()
    if trace is not None:
        trace.append((ev.n_eval, best))

    for _ in range(generations):
        order = np.argsort(fit)[::-1]
        new = [pop[order[j]].copy() for j in range(elite)]
        while len(new) < pop_size:
            a = pop[_tournament(rng, fit)]
            b = pop[_tournament(rng, fit)]
            if rng.random() < p_cross:
                if rng.random() < 0.5:                        # uniform crossover
                    mask = rng.random(n) < 0.5
                    child = np.where(mask, a, b)
                else:                                          # arithmetic blend
                    w = rng.random(n)
                    child = w * a + (1.0 - w) * b
            else:
                child = a.copy()
            mut = rng.random(n) < p_mut
            if mut.any():
                kind = rng.random(mut.sum())
                vals = child[mut]
                # multiplicative jitter | switch a component off | switch one on
                vals = np.where(kind < 0.6, vals * np.exp(rng.normal(0, 0.6, vals.shape)),
                       np.where(kind < 0.8, 0.0, rng.random(vals.shape) * c_max))
                child[mut] = vals
            new.append(np.clip(child, 0.0, c_max))
        pop = np.array(new)
        fit = np.array([ev.mu_H(c)[0] for c in pop])
        j = int(np.argmax(fit))
        if fit[j] > best:
            best, best_c = float(fit[j]), pop[j].copy()
        if trace is not None:
            trace.append((ev.n_eval, best))
    return best, best_c


def _ray_best(ev, f_S_dir, t_max, n_grid=12, n_iter=20):
    """Exact-ish max of the concave piecewise-linear t -> mu^H(t d)."""
    ts = np.linspace(0.0, t_max, n_grid)
    vals = np.array([ev.mu_H_on_ray(f_S_dir, t) for t in ts])
    if not np.isfinite(vals).any():
        return -np.inf, 0.0
    i = int(np.nanargmax(np.where(np.isfinite(vals), vals, -np.inf)))
    lo = ts[max(i - 1, 0)]
    hi = ts[min(i + 1, n_grid - 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = ev.mu_H_on_ray(f_S_dir, c), ev.mu_H_on_ray(f_S_dir, d)
    for _ in range(n_iter):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = ev.mu_H_on_ray(f_S_dir, c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = ev.mu_H_on_ray(f_S_dir, d)
    t = c if fc > fd else d
    v = max(fc, fd, float(np.nanmax(np.where(np.isfinite(vals), vals, -np.inf))))
    return v, float(t)


def ga_ray(ev, m, rng, pop_size=100, generations=60, c_max=10.0,
           p_cross=0.9, p_mut=0.2, elite=2, trace=None):
    """GA on the direction only; the scale along each ray is solved exactly."""
    n = m - 1

    def norm(v):
        v = np.clip(v, 0.0, None)
        s = v.sum()
        return v / s if s > 0 else np.full(n, 1.0 / n)

    def evaluate(d):
        mu, f_S = ev.mu_H(d)          # one Level 1 solve at |d|_1 = 1
        if f_S is None:
            return 0.0
        t_max = c_max / max(d.max(), 1e-12)
        v, _ = _ray_best(ev, f_S, t_max)
        return max(v, 0.0) if np.isfinite(v) else 0.0

    pop = np.array([norm(rng.random(n) * (rng.random(n) < rng.uniform(0.2, 1.0)))
                    for _ in range(pop_size)])
    fit = np.array([evaluate(d) for d in pop])
    best = float(fit.max()); best_d = pop[int(np.argmax(fit))].copy()
    if trace is not None:
        trace.append((ev.n_eval, best))
    for _ in range(generations):
        order = np.argsort(fit)[::-1]
        new = [pop[order[j]].copy() for j in range(elite)]
        while len(new) < pop_size:
            a, b = pop[_tournament(rng, fit)], pop[_tournament(rng, fit)]
            if rng.random() < p_cross:
                w = rng.random(n)
                child = w * a + (1.0 - w) * b
            else:
                child = a.copy()
            mut = rng.random(n) < p_mut
            if mut.any():
                kind = rng.random(mut.sum())
                vals = child[mut]
                vals = np.where(kind < 0.7, vals * np.exp(rng.normal(0, 0.6, vals.shape)),
                       np.where(kind < 0.85, 0.0, rng.random(vals.shape)))
                child[mut] = vals
            new.append(norm(child))
        pop = np.array(new)
        fit = np.array([evaluate(d) for d in pop])
        if fit.max() > best:
            best = float(fit.max()); best_d = pop[int(np.argmax(fit))].copy()
        if trace is not None:
            trace.append((ev.n_eval, best))
    return best, best_d


# --------------------------------------------------------------------------- #
def _one(args_tuple):
    (mode, r, sid, host_id, pop, gens, cfg_paths, seed) = args_tuple
    cfg = load_config(*cfg_paths)
    m = cfg["model"]["m"]
    hd, cE = get_host(cfg, host_id, m)
    R_S = make_R_S(cfg, r, sid, "paper", m, cfg["model"]["k"])
    ev = Evaluator(R_S, hd.R_H, cE, cfg)
    rng = np.random.default_rng(seed)
    trace = []
    t0 = time.perf_counter()
    if mode == "full":
        best, c = ga_full(ev, m, rng, pop_size=pop, generations=gens,
                          c_max=float(cfg["flags"]["c_H_box"]), trace=trace)
    else:
        best, c = ga_ray(ev, m, rng, pop_size=pop, generations=gens,
                         c_max=float(cfg["flags"]["c_H_box"]), trace=trace)
    dt = time.perf_counter() - t0
    ev.close()
    return ({"mode": mode, "r": r, "symbiont_id": sid, "host_id": host_id,
             "mu_H_ga": best, "mu_H_0": hd.mu_H_0, "seconds_ga": dt,
             "n_eval": ev.n_eval, "n_host_lp": ev.n_host_lp,
             "pop": pop, "generations": gens, "seed": seed},
            [{"mode": mode, "r": r, "symbiont_id": sid, "n_eval": e, "best": b}
             for e, b in trace])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="full,ray")
    ap.add_argument("--r-grid", default="5,10,13,20,30,50,80")
    ap.add_argument("--n", type=int, default=20, help="symbionts per r")
    ap.add_argument("--host-id", type=int, default=0)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--ray-generations", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "config" / "base.yaml")])
    ap.add_argument("--milp-dir", default=str(ROOT / "data" / "raw" / "post_paper"))
    ap.add_argument("--tables", default=str(ROOT / "tables"))
    ap.add_argument("--out-prefix", default="GA_vs_MILP")
    args = ap.parse_args()

    cfg = load_config(*args.config)
    rs = [int(x) for x in args.r_grid.split(",")]
    jobs = []
    for mode in args.modes.split(","):
        gens = args.generations if mode == "full" else args.ray_generations
        for r in rs:
            for sid in range(args.n):
                seed = derive_seed(cfg["master_seed"], "ga", mode=mode, r=r,
                                   symbiont_id=sid, host_id=args.host_id)
                jobs.append((mode, r, sid, args.host_id, args.pop, gens,
                             args.config, seed))

    t0 = time.perf_counter()
    if args.workers <= 1:
        out = [_one(j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(_one)(j) for j in jobs)
    res = pd.DataFrame([o[0] for o in out])
    traces = pd.DataFrame([t for o in out for t in o[1]])

    # MILP reference: already computed, with its own solve time
    import glob
    mf = sorted(glob.glob(os.path.join(args.milp_dir, "cells_*.parquet")))
    milp = pd.concat([pd.read_parquet(f, columns=["r", "symbiont_id", "mu_H",
                                                  "solve_time_milp_s", "solver_status"])
                      for f in mf], ignore_index=True)
    milp = milp.rename(columns={"mu_H": "mu_H_milp",
                                "solve_time_milp_s": "seconds_milp",
                                "solver_status": "milp_status"})
    res = res.merge(milp, on=["r", "symbiont_id"], how="left")
    res["ratio"] = res["mu_H_ga"] / res["mu_H_milp"].replace(0, np.nan)
    # Tolerance: the MILP's own accuracy is the limit.  G1 section 5 measured it by
    # re-solving the LP at the MILP's c^H*: |d mu^H| median 3.5e-11, max 1.8e-6.
    # A 1e-9 threshold is three orders finer than that and flags pure round-off --
    # it reported 11 "violations" whose largest excess was 2.3e-7 (relative 6.5e-7).
    TOL_ABS, TOL_REL = 1e-6, 1e-6
    res["excess"] = res["mu_H_ga"] - res["mu_H_milp"]
    res["ga_beats_milp"] = res["excess"] > TOL_ABS + TOL_REL * res["mu_H_milp"].abs()
    res["speedup"] = res["seconds_milp"] / res["seconds_ga"]

    tab = Path(args.tables); tab.mkdir(parents=True, exist_ok=True)
    res.to_csv(tab / f"{args.out_prefix}.csv", index=False)
    traces.to_csv(tab / f"{args.out_prefix}_traces.csv", index=False)

    print(f"{len(res)} runs in {time.perf_counter() - t0:.0f}s wall")
    g = res.groupby(["mode", "r"]).agg(
        n=("ratio", "size"),
        ratio_median=("ratio", "median"), ratio_min=("ratio", "min"),
        frac_within_1pct=("ratio", lambda x: float((x >= 0.99).mean())),
        frac_within_0p1pct=("ratio", lambda x: float((x >= 0.999).mean())),
        ga_s=("seconds_ga", "median"), milp_s=("seconds_milp", "median"),
        speedup=("speedup", "median"),
        beats=("ga_beats_milp", "sum"))
    print(g.round(4).to_string())
    optimal = res[res["milp_status"] == "optimal"]
    n_bad = int(optimal["ga_beats_milp"].sum())
    print(f"\nGA above a PROVEN-OPTIMAL MILP beyond tolerance: {n_bad} / {len(optimal)}"
          f"  -- any such case would mean the MILP is wrong (01 section 10.1-9)")
    if len(optimal):
        print(f"  largest excess over a proven optimum: {optimal['excess'].max():.2e}"
              f"  (the MILP's own re-solve accuracy reaches 1.8e-6, G1 section 5)")
    trunc = res[res["milp_status"] != "optimal"]
    if len(trunc):
        print(f"GA above a TRUNCATED MILP: {int((trunc['excess'] > 0).sum())} / {len(trunc)}"
              f"  median ratio {trunc['ratio'].median():.3f}  max {trunc['ratio'].max():.3f}"
              f"  -- expected, since both are only lower bounds")


if __name__ == "__main__":
    main()
