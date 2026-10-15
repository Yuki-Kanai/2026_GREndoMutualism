"""Level 1 (symbiont LP + min-norm QP) and Level 2 (host LP).

01_model_spec.md sections 3, 3.1, 4.

Implementation notes (deviations that are numerical, not modelling):

* The constraint matrix of each level depends only on the control matrix, so the
  solver objects are built once per R^S / R^H and re-used across the many right
  hand sides (c^H for Level 1, c^E - f^S* for Level 2).

* Level 1 is solved at unit scale.  The LP and the min-norm QP are positively
  homogeneous of degree 1 in c^H (01_model_spec.md V6), so we solve with
  c^H/|c^H|_1 and scale w back.  This keeps every instance in the same numerical
  range; the bounded/unbounded test is still applied in the original units with
  the configured tolerances.  Switch off with flags.normalize_cH: false.

* The min-norm tie-break tries several QP backends in the order given by
  solver.qp.backends; the one that succeeded is recorded per cell in
  SymbiontSolution.qp_path.  The default order is quadprog, then CPLEX, then
  HiGHS.
    - quadprog (Goldfarb-Idnani dual active set) is finite, exact and
      deterministic, needs only a positive-definite Hessian (ours is I), never
      failed in 2100 test instances, and agrees with CPLEX to 2.6e-13 median /
      6e-7 max in w and to 1.5e-12 in ||w||.  Licence: GPL-2.0+.
    - HiGHS's own active-set QP returns "Solve error" on 0-6 % of these
      degenerate instances (the rate is scale-independent: it is unchanged by
      the unit-scale normalisation above) and, worse, hangs at 100 % CPU when a
      failed Highs object is re-used.  It is therefore last in the chain and is
      always given a fresh object.
  See tables/QP_backends2.csv and tables/QP_backend_equivalence.csv.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import highspy

INF = highspy.kHighsInf


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class SymbiontSolution:
    """Result of SymbiontSolver.solve for one c^H (Level 1)."""
    mu_S: float                 # symbiont growth
    w_S: np.ndarray             # (r,) mode activities, after the min-norm tie-break
    f_S: np.ndarray             # (m-1,) net uptake from the host, -R^S_{-m} w^S
    lam1: np.ndarray            # R^m, lam1[m-1] == 1
    lam2: np.ndarray            # R^r, reduced costs
    status: str                 # optimal | qp_fallback | LP status if the LP failed
    qp_path: str = "cplex"      # quadprog | cplex | highs | lp_vertex | trivial | none
    lp_path: str = "ipm"        # ipm | simplex_fallback
    qp_fallback: bool = False   # True if every QP backend failed (w = LP vertex)
    t_lp: float = 0.0           # seconds
    t_qp: float = 0.0


@dataclass
class HostSolution:
    """Result of HostSolver.solve (Level 2)."""
    mu_H: float                 # host growth
    w_H: np.ndarray             # (r_H,) host mode activities
    f_H: np.ndarray             # (m-1,) f^S* plus the host's own net uptake (<= c^E)
    status: str
    t_lp: float = 0.0


_STATUS_MAP = {
    "Optimal": "optimal",
    "Infeasible": "infeasible",
    "Unbounded": "unbounded",
    "Primal infeasible": "infeasible",
    "Primal unbounded": "unbounded",
}


def _map_status(h) -> str:
    """HiGHS model status as a short lower-case string (optimal, infeasible, ...)."""
    s = h.modelStatusToString(h.getModelStatus())
    return _STATUS_MAP.get(s, s.lower().replace(" ", "_"))


def _dense_colwise(A: np.ndarray):
    """Column-wise CSC arrays for a dense matrix (exact zeros dropped)."""
    _, ncol = A.shape
    start = np.empty(ncol + 1, dtype=np.int32)
    idx, val = [], []
    start[0] = 0
    for j in range(ncol):
        nz = np.flatnonzero(A[:, j])
        idx.append(nz)
        val.append(A[nz, j])
        start[j + 1] = start[j] + nz.size
    index = np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)
    value = np.concatenate(val) if val else np.zeros(0)
    return start, index.astype(np.int32), value.astype(np.float64)


def _new_highs(opts: dict):
    """A silent, single-threaded HiGHS instance with the given options."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("threads", 1)
    for key, val in opts.items():
        h.setOptionValue(key, val)
    return h


# --------------------------------------------------------------------------- #
# Level 1
# --------------------------------------------------------------------------- #
class SymbiontSolver:
    """max R^S_m w  s.t.  -R^S_{-m} w <= c^H, w >= 0, then a min-norm tie-break."""

    def __init__(self, R_S: np.ndarray, cfg: dict):
        """Build the LP for one R^S; solve() then only changes the right-hand side c^H."""
        self.R_S = np.ascontiguousarray(R_S, dtype=np.float64)
        self.m, self.r = self.R_S.shape
        self.cfg = cfg
        tol = cfg["tolerances"]
        self.eps_mu_abs = float(tol["eps_mu_abs"])
        self.eps_mu_rel = float(tol["eps_mu_rel"])
        self.w_box = float(cfg["flags"]["w_S_box"])
        self.normalize = bool(cfg["flags"].get("normalize_cH", True))
        chain = cfg["solver"].get("qp", {}).get("backends",
                                                ["quadprog", "cplex", "highs"])
        self.qp_chain = [b for b in chain if b in self._QP_BACKENDS]
        if not self.qp_chain:
            raise ValueError(f"no known QP backend in {chain}")

        self.A = -self.R_S[: self.m - 1, :]          # (m-1) x r
        self.obj = self.R_S[self.m - 1, :].copy()    # row m

        self._h = _new_highs(dict(cfg["solver"]["lp"]))
        self._build_lp()
        self._h_fb = None
        self._qp = None
        self._qp_kind = None
        self._hq = None

    def close(self):
        """Release the CPLEX environment held by the min-norm QP."""
        if self._qp is not None:
            try:
                self._qp.end()
            except Exception:                          # noqa: BLE001
                pass
            self._qp = None

    def __del__(self):
        self.close()                                   # also release CPLEX on garbage collection

    # -- LP -------------------------------------------------------------- #
    def _lp_model(self):
        """max mu^S = R^S_m w as a HiGHS LP (minimise -mu^S); row bounds set per solve."""
        m, r = self.m, self.r
        lp = highspy.HighsLp()
        lp.num_col_ = r
        lp.num_row_ = m - 1
        lp.col_cost_ = -self.obj.copy()              # minimise -mu_S
        lp.col_lower_ = np.zeros(r)
        lp.col_upper_ = np.full(r, self.w_box)
        lp.row_lower_ = np.full(m - 1, -INF)
        lp.row_upper_ = np.zeros(m - 1)              # replaced per solve
        start, index, value = _dense_colwise(self.A)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = start
        lp.a_matrix_.index_ = index
        lp.a_matrix_.value_ = value
        lp.sense_ = highspy.ObjSense.kMinimize
        return lp

    def _build_lp(self):
        """Load the LP into the primary (IPM) HiGHS instance."""
        self._h.passModel(self._lp_model())

    def _fallback_lp(self, c_hat):
        """Simplex re-solve for the rare cells where IPM stalls at our tolerance."""
        if self._h_fb is None:
            opts = dict(self.cfg["solver"].get("lp_fallback",
                                               {"solver": "simplex", "presolve": "on"}))
            self._h_fb = _new_highs(opts)
            self._h_fb.passModel(self._lp_model())
        m = self.m
        self._h_fb.changeRowsBounds(m - 1, np.arange(m - 1, dtype=np.int32),
                                    np.full(m - 1, -INF), c_hat)
        self._h_fb.run()
        return self._h_fb, _map_status(self._h_fb)

    # -- QP (CPLEX) ------------------------------------------------------- #
    def _build_qp_cplex(self):
        """min ||w||^2  s.t.  -R^S_{-m} w <= c^H,  R^S_m w >= mu_lo,  w >= 0, in CPLEX.

        Built once; _solve_qp_cplex only changes the right-hand sides."""
        import cplex

        m, r = self.m, self.r
        p = cplex.Cplex()
        p.set_log_stream(None); p.set_results_stream(None)
        p.set_warning_stream(None); p.set_error_stream(None)
        p.variables.add(obj=[0.0] * r, lb=[0.0] * r, ub=[self.w_box] * r,
                        names=[f"w{j}" for j in range(r)])
        rows, senses, rhs = [], [], []
        for i in range(m - 1):
            nz = np.flatnonzero(self.A[i])
            rows.append([nz.tolist(), self.A[i, nz].tolist()])
            senses.append("L"); rhs.append(0.0)
        nz = np.flatnonzero(self.obj)
        rows.append([nz.tolist(), self.obj[nz].tolist()])
        senses.append("G"); rhs.append(0.0)
        p.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs)
        p.objective.set_quadratic([2.0] * r)         # c'x + 1/2 x'Qx  ->  ||w||^2
        p.objective.set_sense(p.objective.sense.minimize)
        p.parameters.threads.set(1)
        p.parameters.barrier.qcpconvergetol.set(1e-12)
        p.parameters.barrier.convergetol.set(1e-12)
        self._qp = p
        self._qp_kind = "cplex"

    def _build_qp_highs(self):
        """The same QP as _build_qp_cplex, as a fresh HiGHS model."""
        m, r = self.m, self.r
        Aq = np.vstack([self.A, self.obj[None, :]])
        h = _new_highs({"presolve": self.cfg["solver"]["qp"].get("presolve", "off")})
        lp = highspy.HighsLp()
        lp.num_col_ = r
        lp.num_row_ = m
        lp.col_cost_ = np.zeros(r)
        lp.col_lower_ = np.zeros(r)
        lp.col_upper_ = np.full(r, self.w_box)
        lp.row_lower_ = np.concatenate([np.full(m - 1, -INF), [0.0]])
        lp.row_upper_ = np.concatenate([np.zeros(m - 1), [INF]])
        start, index, value = _dense_colwise(Aq)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = start
        lp.a_matrix_.index_ = index
        lp.a_matrix_.value_ = value
        lp.sense_ = highspy.ObjSense.kMinimize
        hes = highspy.HighsHessian()
        hes.dim_ = r
        hes.format_ = highspy.HessianFormat.kTriangular
        hes.start_ = np.arange(r + 1, dtype=np.int32)
        hes.index_ = np.arange(r, dtype=np.int32)
        hes.value_ = np.full(r, 2.0)
        model = highspy.HighsModel()
        model.lp_ = lp
        model.hessian_ = hes
        h.passModel(model)
        return h

    def _solve_qp_quadprog(self, c_hat, mu_lo):
        """Goldfarb-Idnani dual active set: finite, exact, deterministic.

        min 1/2 w'w  s.t.  G w <= h, w >= 0, with G = [-R_{-m}; -R_m] and
        h = [c_hat; -mu_lo].  quadprog wants  min 1/2 x'Px - q'x  s.t.  C'x >= b,
        and needs P positive definite, which P = I is.
        """
        try:
            import quadprog
        except ImportError:
            return None
        m, r = self.m, self.r
        G = np.vstack([self.A, -self.obj[None, :]])
        h = np.concatenate([np.asarray(c_hat, dtype=float), [-float(mu_lo)]])
        C = np.hstack([-G.T, np.eye(r)])
        b = np.concatenate([-h, np.zeros(r)])
        try:
            w = quadprog.solve_qp(np.eye(r), np.zeros(r), C, b, 0)[0]
        except Exception:                              # noqa: BLE001
            return None
        return np.asarray(w, dtype=float)

    def _solve_qp_cplex(self, c_hat, mu_lo):
        """Min-norm w with CPLEX, or None on failure."""
        if self._qp is None:
            self._build_qp_cplex()
        p = self._qp
        m = self.m
        p.linear_constraints.set_rhs(
            [(i, float(c_hat[i])) for i in range(m - 1)] + [(m - 1, float(mu_lo))])
        try:
            p.solve()
            txt = p.solution.get_status_string().lower()
            if "optimal" not in txt or "infeasible" in txt:
                return None
            return np.array(p.solution.get_values(), dtype=np.float64)
        except Exception:                              # noqa: BLE001
            return None

    def _solve_qp_highs(self, c_hat, mu_lo):
        """Min-norm w with HiGHS, or None on failure."""
        # a fresh object every time: a failed HiGHS QP poisons the instance
        try:
            h = self._build_qp_highs()
            m = self.m
            h.changeRowsBounds(
                m, np.arange(m, dtype=np.int32),
                np.concatenate([np.full(m - 1, -INF), [mu_lo]]),
                np.concatenate([c_hat, [INF]]))
            h.run()
            if _map_status(h) != "optimal":
                return None
            return np.array(h.getSolution().col_value, dtype=np.float64)
        except Exception:                              # noqa: BLE001
            return None

    _QP_BACKENDS = {
        "quadprog": lambda self, c, mu: self._solve_qp_quadprog(c, mu),
        "cplex": lambda self, c, mu: self._solve_qp_cplex(c, mu),
        "highs": lambda self, c, mu: self._solve_qp_highs(c, mu),
    }

    # -- solve ------------------------------------------------------------ #
    def solve(self, c_H: np.ndarray) -> SymbiontSolution:
        """Level 1 for one internal environment c^H (SI S3, S5.1).

        (i) LP: mu^S* = max R^S_m w  s.t.  -R^S_{-m} w <= c^H, w >= 0 (HiGHS IPM,
        simplex if IPM fails).  (ii) QP: among the w reaching mu^S* (within eps),
        the one of minimum ||w||^2 (backends in solver.qp.backends, quadprog
        first).  Both are solved at c^H/||c^H||_1 and w is scaled back.
        """
        m, r = self.m, self.r
        c_H = np.asarray(c_H, dtype=np.float64)
        scale = float(np.sum(np.abs(c_H))) if self.normalize else 1.0
        if scale <= 0.0:
            return SymbiontSolution(
                0.0, np.zeros(r), np.zeros(m - 1),
                np.concatenate([np.zeros(m - 1), [1.0]]),
                -self.R_S[m - 1, :].copy(), "optimal", qp_path="trivial")
        c_hat = c_H / scale

        rows = np.arange(m - 1, dtype=np.int32)
        t0 = time.perf_counter()
        self._h.changeRowsBounds(m - 1, rows, np.full(m - 1, -INF), c_hat)
        self._h.run()
        status = _map_status(self._h)
        lp_path = "ipm"
        solver_obj = self._h
        if status != "optimal":
            solver_obj, status = self._fallback_lp(c_hat)
            lp_path = "simplex_fallback"
        t_lp = time.perf_counter() - t0
        if status != "optimal":
            return SymbiontSolution(
                np.nan, np.full(r, np.nan), np.full(m - 1, np.nan),
                np.full(m, np.nan), np.full(r, np.nan), status,
                qp_path="none", lp_path=lp_path, t_lp=t_lp)
        sol = solver_obj.getSolution()
        w_lp = np.array(sol.col_value, dtype=np.float64)
        mu_hat = float(self.obj @ w_lp)
        lam1 = np.empty(m)                             # duals are scale-invariant
        lam1[: m - 1] = -np.array(sol.row_dual, dtype=np.float64)
        lam1[m - 1] = 1.0
        lam2 = np.array(sol.col_dual, dtype=np.float64)

        eps = max(self.eps_mu_abs, self.eps_mu_rel * abs(mu_hat))
        t0 = time.perf_counter()
        w_hat, path = None, None
        for name in self.qp_chain:
            w_hat = self._QP_BACKENDS[name](self, c_hat, mu_hat - eps)
            if w_hat is not None:
                path = name
                break
        if w_hat is None:
            w_hat, path = w_lp, "lp_vertex"
        t_qp = time.perf_counter() - t0

        w = np.where(w_hat < 0.0, 0.0, w_hat) * scale
        mu_S = float(self.obj @ w)
        f_S = -self.R_S[: m - 1, :] @ w
        return SymbiontSolution(
            mu_S=mu_S, w_S=w, f_S=f_S, lam1=lam1, lam2=lam2,
            status="qp_fallback" if path == "lp_vertex" else "optimal",
            qp_path=path, lp_path=lp_path, qp_fallback=(path == "lp_vertex"),
            t_lp=t_lp, t_qp=t_qp,
        )


# --------------------------------------------------------------------------- #
# Level 2
# --------------------------------------------------------------------------- #
class HostSolver:
    """max R^H_m w  s.t.  -R^H_{-m} w <= c^E - f^S*, w >= 0."""

    def __init__(self, R_H: np.ndarray, cfg: dict):
        """Build the host LP for one R^H; solve() only changes the right-hand side."""
        self.R_H = np.ascontiguousarray(R_H, dtype=np.float64)
        self.m, self.r_H = self.R_H.shape
        self.A = -self.R_H[: self.m - 1, :]
        self.obj = self.R_H[self.m - 1, :].copy()
        self.w_box = float(cfg["flags"]["w_S_box"])
        self.cfg = cfg
        self._h = _new_highs(dict(cfg["solver"]["lp"]))
        self._h_fb = None
        self._model = None
        lp = highspy.HighsLp()
        lp.num_col_ = self.r_H
        lp.num_row_ = self.m - 1
        lp.col_cost_ = -self.obj.copy()
        lp.col_lower_ = np.zeros(self.r_H)
        lp.col_upper_ = np.full(self.r_H, self.w_box)
        lp.row_lower_ = np.full(self.m - 1, -INF)
        lp.row_upper_ = np.zeros(self.m - 1)
        start, index, value = _dense_colwise(self.A)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = start
        lp.a_matrix_.index_ = index
        lp.a_matrix_.value_ = value
        lp.sense_ = highspy.ObjSense.kMinimize
        self._model = lp
        self._h.passModel(lp)

    def solve(self, f_S: np.ndarray, cE: np.ndarray) -> HostSolution:
        """Level 2: max mu^H given what the symbiont takes, f^S* (SI S3, S5.1).

        The host can use what is left of the environment, c^E - f^S*.  With
        f_S = 0 this is the host without a symbiont, mu_H_0.
        """
        m = self.m
        rhs = np.asarray(cE, dtype=np.float64) - np.asarray(f_S, dtype=np.float64)
        rows = np.arange(m - 1, dtype=np.int32)
        t0 = time.perf_counter()
        self._h.changeRowsBounds(m - 1, rows, np.full(m - 1, -INF), rhs)
        self._h.run()
        status = _map_status(self._h)
        obj = self._h
        if status not in ("optimal", "infeasible"):
            if self._h_fb is None:
                self._h_fb = _new_highs(dict(self.cfg["solver"].get(
                    "lp_fallback", {"solver": "simplex", "presolve": "on"})))
                self._h_fb.passModel(self._model)
            self._h_fb.changeRowsBounds(m - 1, rows, np.full(m - 1, -INF), rhs)
            self._h_fb.run()
            obj = self._h_fb
            status = _map_status(self._h_fb)
        t_lp = time.perf_counter() - t0
        if status != "optimal":
            return HostSolution(np.nan, np.full(self.r_H, np.nan),
                                np.full(m - 1, np.nan), status, t_lp)
        w = np.array(obj.getSolution().col_value, dtype=np.float64)
        w = np.where(w < 0.0, 0.0, w)
        mu_H = float(self.obj @ w)
        f_H = np.asarray(f_S, dtype=np.float64) - self.R_H[: m - 1, :] @ w
        return HostSolution(mu_H, w, f_H, "optimal", t_lp)


def solve_host_once(R_H: np.ndarray, f_S: np.ndarray, cE: np.ndarray, cfg: dict) -> HostSolution:
    """Build a HostSolver and solve once (used for mu_H_0 during host sampling)."""
    return HostSolver(R_H, cfg).solve(f_S, cE)
