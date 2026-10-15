"""Level 3 — host evolution as a single-level MILP (01_model_spec.md section 5).

The lower level (Level 1 LP + the min-norm QP that makes its solution unique) is
replaced by its KKT system; complementarity is expressed with SOS1 sets, never
with big-M.  Because both KKT blocks together pin down a single lower-level
response, the bilevel optimistic/pessimistic ambiguity does not arise.

Solved in two stages (section 5.2.4):
  step 1  maximise mu_H
  step 2  minimise ||c^H||_1 subject to mu_H >= mu_H* - eps_H
so that the post-evolution structural quantities are not an artefact of which
optimal vertex the solver happened to return.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvolutionSolution:
    """Result of Level 3 for one host-symbiont pair (all NaN if CPLEX found nothing)."""
    mu_H: float               # host growth at the evolved c^H
    mu_S: float               # symbiont growth there
    c_H: np.ndarray           # (m-1,) evolved internal environment (after step 2)
    c_H_step1: np.ndarray     # (m-1,) c^H after step 1 only
    w_S: np.ndarray           # (r,) symbiont mode activities
    f_S: np.ndarray           # (m-1,) symbiont net uptake
    w_H: np.ndarray           # (r_H,) host mode activities
    lam1: np.ndarray          # (m,) duals of the symbiont LP
    status: str               # optimal | detlimit (time limit hit) | infeasible | ...
    mip_gap: float            # relative MIP gap (0 when optimal)
    mip_best_bound: float     # proven upper bound on mu^H from step 1, capped at 1
    flag_cmax_active: bool    # some c^H_i reached its artificial bound C_max
    flag_box_active: bool     # some variable reached its artificial bound
    box_report: dict          # per variable group, max value / its bound
    l1_step1: float           # ||c^H||_1 after step 1
    l1_step2: float           # ||c^H||_1 after step 2
    t_milp: float             # seconds, both steps
    detail: dict = field(default_factory=dict)   # step statuses, gaps, model size,
                                                  # milp_path, box_retries


def _status_name(cpx, prob) -> str:
    """CPLEX solution status as a short string: optimal, infeasible, unbounded, detlimit, ..."""
    st = prob.solution.get_status()
    txt = prob.solution.get_status_string().lower()
    if "optimal" in txt and "infeasible" not in txt:
        return "optimal"
    if "infeasible" in txt:
        return "infeasible"
    if "unbounded" in txt:
        return "unbounded"
    if "deterministic time limit" in txt or "dettime" in txt or "time limit" in txt:
        return "detlimit"
    return txt.replace(" ", "_")[:40]


class EvolutionSolver:
    """Builds the MILP once per (R^S, R^H) pair and runs the two stages."""

    def __init__(self, R_S, R_H, cE, cfg, solver="cplex", numerical_emphasis=0,
                 presolve=None):
        """Read the bounds, tolerances and CPLEX settings; the model is built in solve().

        numerical_emphasis and presolve override base.yaml; solve_host_evolution
        raises them when a first attempt finds no solution.
        """
        import cplex  # imported lazily so the LP-only pipeline never needs it

        self.cplex = cplex
        self.R_S = np.ascontiguousarray(R_S, dtype=np.float64)
        self.R_H = np.ascontiguousarray(R_H, dtype=np.float64)
        self.cE = np.asarray(cE, dtype=np.float64)
        self.cfg = cfg
        self.m, self.r = self.R_S.shape
        self.r_H = self.R_H.shape[1]

        fl = cfg["flags"]
        self.C_max = float(fl["c_H_box"])
        self.W_max = float(fl["w_S_box"])
        self.L_max = float(fl["lambda_box"])
        self.G_max = self.m * self.C_max
        mp = cfg["solver"]["milp"]
        self.det_time_limit = float(mp["det_time_limit"])
        self.presolve = int(mp["presolve"]) if presolve is None else int(presolve)
        # CPXPARAM_Emphasis_Numerical.  Off by default (the spec fixes presolve
        # off for reproducibility); raised only by the retry chain below.
        self.numerical_emphasis = int(numerical_emphasis)
        self.threads = int(mp.get("threads", 1))
        self.eps_H_abs = float(mp["eps_H_abs"])
        self.eps_H_rel = float(mp["eps_H_rel"])
        # knobs, all defaulting to the spec settings; swept by bench_solver.py
        self.mipgap = float(mp.get("mipgap", 1e-9))
        self.absmipgap = float(mp.get("absmipgap", 1e-12))
        # a lower bound on mu^S removes the large degenerate face w^S = 0 from
        # the feasible set.  0.0 = off (spec section 3: mu^S >= 0 is allowed).
        self.mu_S_floor = float(fl.get("mu_S_floor", 0.0))
        # optional column scaling of R^S (see the module docstring of the patch)
        self.col_normalize = str(fl.get("col_normalize", "none"))
        self.omega = np.ones(self.r)
        if self.col_normalize in ("exact", "model"):
            w1 = np.abs(self.R_S).sum(axis=0)
            # an all-zero column carries no flux; leave it alone
            self.omega = np.where(w1 > 1e-12, w1, 1.0)
            self.R_S = np.ascontiguousarray(self.R_S / self.omega)

    # ------------------------------------------------------------------ #
    def _index(self):
        """Position of every variable in the CPLEX variable vector.

        w_S  (r)    symbiont mode activities
        c_H  (m-1)  the internal environment, chosen by the host
        mu_S (1)    symbiont growth
        g    (m)    slacks: g_i = c^H_i - f^S_i (unused supply, i < m), g_m = 0
        lam1 (m)    duals of the symbiont LP (max mu^S); lam1_m fixed to 1
        lam2 (r)    duals of w^S >= 0 in that LP
        lam3 (m)    duals of the min-norm QP's constraints
        lam4 (r)    duals of w^S >= 0 in the QP
        w_H  (r_H)  host mode activities
        mu_H (1)    host growth, the objective of step 1
        """
        m, r, rH = self.m, self.r, self.r_H
        off = 0
        idx = {}
        idx["w_S"] = list(range(off, off + r)); off += r
        idx["c_H"] = list(range(off, off + m - 1)); off += m - 1
        idx["mu_S"] = off; off += 1
        idx["g"] = list(range(off, off + m)); off += m
        idx["lam1"] = list(range(off, off + m)); off += m
        idx["lam2"] = list(range(off, off + r)); off += r
        idx["lam3"] = list(range(off, off + m)); off += m
        idx["lam4"] = list(range(off, off + r)); off += r
        idx["w_H"] = list(range(off, off + rH)); off += rH
        idx["mu_H"] = off; off += 1
        idx["_n"] = off
        return idx

    def _build(self):
        """The single-level MILP of SI S5.2.

        The symbiont's problem (max mu^S, then min ||w^S||^2) is replaced by its
        KKT conditions: the linear constraints C1, C2, C4 below, and complementarity
        (a dual and its slack cannot both be non-zero) as SOS1 sets.  The host's
        own LP enters directly (C5, C6).  Every variable gets a finite artificial
        bound (C_max, W_max, L_max) because SOS1 branching needs one.
        Returns (CPLEX problem, variable index from _index()).
        """
        cplex = self.cplex
        m, r, rH = self.m, self.r, self.r_H
        R_S, R_H = self.R_S, self.R_H
        idx = self._index()
        n = idx["_n"]

        lb = np.zeros(n)
        ub = np.empty(n)
        names = [""] * n
        for j, v in enumerate(idx["w_S"]):
            ub[v] = self.W_max; names[v] = f"wS{j}"
        for i, v in enumerate(idx["c_H"]):
            ub[v] = self.C_max; names[v] = f"cH{i}"
        ub[idx["mu_S"]] = 1.0; names[idx["mu_S"]] = "muS"
        lb[idx["mu_S"]] = self.mu_S_floor
        for i, v in enumerate(idx["g"]):
            ub[v] = self.G_max; names[v] = f"g{i}"
        ub[idx["g"][m - 1]] = 0.0        # lam1_m = 1 > 0 forces g_m = 0
        for i, v in enumerate(idx["lam1"]):
            ub[v] = self.L_max; names[v] = f"l1_{i}"
        lb[idx["lam1"][m - 1]] = 1.0
        ub[idx["lam1"][m - 1]] = 1.0
        for j, v in enumerate(idx["lam2"]):
            ub[v] = self.L_max; names[v] = f"l2_{j}"
        for i, v in enumerate(idx["lam3"]):
            ub[v] = self.L_max; names[v] = f"l3_{i}"
        for j, v in enumerate(idx["lam4"]):
            ub[v] = self.L_max; names[v] = f"l4_{j}"
        for l, v in enumerate(idx["w_H"]):
            ub[v] = self.W_max; names[v] = f"wH{l}"
        ub[idx["mu_H"]] = 1.0; names[idx["mu_H"]] = "muH"

        prob = cplex.Cplex()
        prob.set_log_stream(None)
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        prob.set_error_stream(None)
        prob.variables.add(obj=[0.0] * n, lb=lb.tolist(), ub=ub.tolist(), names=names)

        rows, senses, rhs = [], [], []

        # (C1) g = [c^H; 0] + R^S w^S - [0; mu^S]
        for i in range(m):
            ind = [idx["g"][i]] + idx["w_S"]
            val = [1.0] + (-R_S[i, :]).tolist()
            if i < m - 1:
                ind.append(idx["c_H"][i]); val.append(-1.0)
            else:
                ind.append(idx["mu_S"]); val.append(1.0)
            rows.append([ind, val]); senses.append("E"); rhs.append(0.0)

        # (C2) (R^S)^T lam1 + lam2 = 0
        for j in range(r):
            ind = idx["lam1"] + [idx["lam2"][j]]
            val = R_S[:, j].tolist() + [1.0]
            rows.append([ind, val]); senses.append("E"); rhs.append(0.0)

        # (C4) -2 w^S + (R^S)^T lam3 + lam4 = 0
        # Under col_normalize="exact" the variable is w'_j = omega_j w_j and the
        # objective is still sum_j w_j^2 = sum_j (w'_j/omega_j)^2, so the
        # stationarity coefficient becomes -2/omega_j^2.  Under "model" the
        # objective is sum_j (w'_j)^2 and the coefficient stays -2.
        for j in range(r):
            cj = -2.0
            if self.col_normalize == "exact":
                cj = -2.0 / float(self.omega[j]) ** 2
            ind = [idx["w_S"][j]] + idx["lam3"] + [idx["lam4"][j]]
            val = [cj] + R_S[:, j].tolist() + [1.0]
            rows.append([ind, val]); senses.append("E"); rhs.append(0.0)

        # (C5) host balance: -R^S_{-m} w^S - R^H_{-m} w^H <= c^E   (f^H eliminated)
        for i in range(m - 1):
            ind = idx["w_S"] + idx["w_H"]
            val = (-R_S[i, :]).tolist() + (-R_H[i, :]).tolist()
            rows.append([ind, val]); senses.append("L"); rhs.append(float(self.cE[i]))

        # (C6) mu^H = R^H_m w^H
        rows.append([[idx["mu_H"]] + idx["w_H"],
                     [1.0] + (-R_H[m - 1, :]).tolist()])
        senses.append("E"); rhs.append(0.0)

        prob.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs)

        # SOS1 complementarity
        for i in range(m - 1):
            prob.SOS.add(type="1", SOS=[[idx["lam1"][i], idx["g"][i]], [1.0, 2.0]],
                         name=f"s1a{i}")
            prob.SOS.add(type="1", SOS=[[idx["lam3"][i], idx["g"][i]], [1.0, 2.0]],
                         name=f"s1b{i}")
        for j in range(r):
            prob.SOS.add(type="1", SOS=[[idx["lam2"][j], idx["w_S"][j]], [1.0, 2.0]],
                         name=f"s2a{j}")
            prob.SOS.add(type="1", SOS=[[idx["lam4"][j], idx["w_S"][j]], [1.0, 2.0]],
                         name=f"s2b{j}")

        # model size, for the CPLEX Community Edition 1000 var / 1000 row limits
        self.n_rows = prob.linear_constraints.get_num()
        self.n_sos = prob.SOS.get_num()

        prob.parameters.threads.set(self.threads)
        prob.parameters.dettimelimit.set(self.det_time_limit)
        prob.parameters.preprocessing.presolve.set(self.presolve)
        prob.parameters.emphasis.numerical.set(self.numerical_emphasis)
        prob.parameters.randomseed.set(int(self.cfg.get("master_seed", 1)) % 2_000_000_000)
        # tighter than default so that the MILP value agrees with an independent
        # LP re-solve at c^H* to ~1e-9 rather than ~1e-6 (V11)
        prob.parameters.simplex.tolerances.feasibility.set(1e-9)
        prob.parameters.simplex.tolerances.optimality.set(1e-9)
        prob.parameters.mip.tolerances.mipgap.set(self.mipgap)
        prob.parameters.mip.tolerances.absmipgap.set(self.absmipgap)
        return prob, idx

    def _box_report(self, x, idx) -> dict:
        """Which artificial boxes are active?  Any of them invalidates the run."""
        m = self.m
        tolb = 1e-6
        return {
            "c_H": float(np.max(x[idx["c_H"]]) / self.C_max) if m > 1 else 0.0,
            "w_S": float(np.max(x[idx["w_S"]]) / self.W_max),
            "w_H": float(np.max(x[idx["w_H"]]) / self.W_max),
            "g": float(np.max(x[idx["g"]]) / self.G_max),
            "lam1": float(np.max(x[idx["lam1"]][: m - 1]) / self.L_max),
            "lam2": float(np.max(x[idx["lam2"]]) / self.L_max),
            "lam3": float(np.max(x[idx["lam3"]]) / self.L_max),
            "lam4": float(np.max(x[idx["lam4"]]) / self.L_max),
        }

    # ------------------------------------------------------------------ #
    def solve(self) -> EvolutionSolution:
        """Step 1: max mu^H.  Step 2: min ||c^H||_1 with mu^H >= mu^H* - eps (SI S5.2).

        Step 2 picks one c^H among those reaching the same mu^H.  A time-limited
        step keeps the best solution found (status "detlimit").
        """
        m, r = self.m, self.r
        t0 = time.perf_counter()
        prob, idx = self._build()

        # ---- step 1: maximise mu_H
        prob.objective.set_sense(prob.objective.sense.maximize)
        prob.objective.set_linear([(idx["mu_H"], 1.0)])
        prob.solve()
        st1 = _status_name(self.cplex, prob)
        if prob.solution.get_solution_type() == prob.solution.type.none:
            return EvolutionSolution(
                np.nan, np.nan, np.full(m - 1, np.nan), np.full(m - 1, np.nan),
                np.full(r, np.nan), np.full(m - 1, np.nan), np.full(self.r_H, np.nan),
                np.full(m, np.nan), st1, np.nan, np.nan, False, False, {},
                np.nan, np.nan,
                time.perf_counter() - t0,
            )
        x1 = np.array(prob.solution.get_values())
        mu_H1 = float(x1[idx["mu_H"]])
        try:
            gap1 = float(prob.solution.MIP.get_mip_relative_gap())
        except Exception:
            gap1 = np.nan
        try:
            bb1 = float(prob.solution.MIP.get_best_objective())
        except Exception:
            bb1 = np.nan
        c_H1 = x1[idx["c_H"]].copy()

        # ---- step 2: minimise ||c^H||_1 at (essentially) the same mu_H
        eps_H = max(self.eps_H_abs, self.eps_H_rel * abs(mu_H1))
        prob.linear_constraints.add(
            lin_expr=[[[idx["mu_H"]], [1.0]]], senses=["G"], rhs=[mu_H1 - eps_H],
            names=["muH_floor"],
        )
        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.objective.set_linear([(idx["mu_H"], 0.0)]
                                  + [(v, 1.0) for v in idx["c_H"]])
        prob.solve()
        st2 = _status_name(self.cplex, prob)
        if prob.solution.get_solution_type() == prob.solution.type.none:
            x2, st2, gap2 = x1, st1 + "|step2_none", gap1
        else:
            x2 = np.array(prob.solution.get_values())
            try:
                gap2 = float(prob.solution.MIP.get_mip_relative_gap())
            except Exception:
                gap2 = np.nan
        t_milp = time.perf_counter() - t0

        c_H = x2[idx["c_H"]].copy()
        w_S = np.clip(x2[idx["w_S"]].copy(), 0.0, None) / self.omega
        w_H = np.clip(x2[idx["w_H"]].copy(), 0.0, None)
        mu_S = float(x2[idx["mu_S"]])
        mu_H = float(x2[idx["mu_H"]])
        f_S = -self.R_S[: m - 1, :] @ (w_S * self.omega)

        status = "optimal" if (st1 == "optimal" and st2 == "optimal") else (
            "detlimit" if "detlimit" in (st1, st2) else f"{st1}|{st2}")

        tolb = 1e-6
        cmax_active = bool(np.any(c_H >= self.C_max - tolb) or
                           np.any(c_H1 >= self.C_max - tolb))
        rep1 = self._box_report(x1, idx)
        rep2 = self._box_report(x2, idx)
        box_report = {kk: max(rep1[kk], rep2[kk]) for kk in rep1}
        box_active = any(v >= 1.0 - 1e-6 for v in box_report.values())
        return EvolutionSolution(
            mu_H=mu_H, mu_S=mu_S, c_H=c_H, c_H_step1=c_H1, w_S=w_S, f_S=f_S,
            w_H=w_H, lam1=x2[idx["lam1"]].copy(), status=status,
            mip_gap=float(np.nanmax([gap1, gap2])),
            # proven upper bound on mu^H from step 1; mu^H <= 1 always
            # (01_model_spec.md section 10-3), so cap it there
            mip_best_bound=float(min(bb1, 1.0)) if np.isfinite(bb1) else np.nan,
            flag_cmax_active=cmax_active, flag_box_active=box_active,
            box_report=box_report,
            l1_step1=float(np.sum(np.abs(c_H1))), l1_step2=float(np.sum(np.abs(c_H))),
            t_milp=t_milp,
            detail={"status_step1": st1, "status_step2": st2,
                    "mu_H_step1": mu_H1, "gap1": gap1, "gap2": gap2,
                    "n_vars": idx["_n"],
                    "n_rows": getattr(self, "n_rows", None),
                    "n_sos": getattr(self, "n_sos", None)},
        )


def solve_host_evolution(R_S, R_H, cE, cfg, max_box_retries: int = 2) -> EvolutionSolution:
    """Solve Level 3, escalating the artificial boxes if any of them is active.

    The boxes exist only so that SOS1 branching has finite bounds
    (01_model_spec.md section 5.1); a binding box would silently change the model,
    so we widen by 100x and re-solve rather than reporting the truncated answer.
    """
    import copy as _copy

    # The MILP is always feasible: c^H = 0, w^S = 0, mu^S = 0 with all slacks
    # zero satisfies every SOS1 set.  So an "integer infeasible" verdict is a
    # solver failure, not a property of the instance, and it must be retried
    # rather than recorded as a data point.  It shows up on fixed_repertoire
    # instances at r >= 20 (single-reaction modes), never on the paper arm.
    # CPXPARAM_Emphasis_Numerical, then presolve, each recover it; the spec
    # (01_model_spec.md section 8.2) says MIP presolve must not change the
    # optimum, so agreement between the settings is itself a check.
    SETTINGS = [
        ("default", dict()),
        ("numerical_emphasis", dict(numerical_emphasis=1)),
        ("presolve_on", dict(numerical_emphasis=1, presolve=1)),
    ]

    sol = None
    for path, kw in SETTINGS:
        work = cfg
        for attempt in range(max_box_retries + 1):
            sol = EvolutionSolver(R_S, R_H, cE, work, **kw).solve()
            if not (sol.flag_box_active or sol.flag_cmax_active)                     or attempt == max_box_retries:
                break
            work = _copy.deepcopy(work)
            for key in ("c_H_box", "w_S_box", "lambda_box"):
                work["flags"][key] = float(work["flags"][key]) * 100.0
        sol.detail["box_retries"] = attempt
        sol.detail["milp_path"] = path
        if np.isfinite(sol.mu_H):
            return sol
    return sol
