"""Cell-level derived quantities (01_model_spec.md section 7).

bounded / unbounded are implemented as two *independent* predicates so that the
identity b + u = m - 1 is a real check and not a tautology (V3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CellAnalysis:
    """Quantities of SI S6 for one solution (one row of the cells table)."""
    n_bounded: int        # b: metabolites whose supply constraint is active
    n_unbounded: int      # u = m - 1 - b
    ub: int               # u * b
    n_excreted: int       # #{i : f^S_i < -tol_exc}
    r_plus: int           # active regulatory modes, w^S_j > tol_w
    n_degenerate: int     # modes with w^S_j = 0 and zero reduced cost (pre only)
    alpha: float          # sqrt(b)
    s: float              # sensitivity, ||R_u+ (R_b+)^+||_F; 0 if b, u or r_plus is 0
    cond_b: float         # condition number of R_b+
    trivial_ub: bool      # b == 0 or u == 0
    bounded_mask: np.ndarray = field(repr=False)
    unbounded_mask: np.ndarray = field(repr=False)
    active_mask: np.ndarray = field(repr=False)


def bounded_predicates(c_H, f_S, tol_abs, tol_rel, zero_cH_rule="bounded_if_zero_flux"):
    """Return (bounded_mask, unbounded_mask) as independent predicates.

    bounded_i   <=>  c^H_i - f^S*_i <= tol_b
    unbounded_i <=>  c^H_i - f^S*_i >  tol_b
    tol_b = tol_abs + tol_rel * |c^H_i|

    zero_cH_rule handles c^H_i == 0, which is common after host evolution:
      "bounded_if_zero_flux" (default): bounded when f^S*_i == 0, unbounded when
          f^S*_i < 0 (the symbiont excretes into an unoffered metabolite).
      "always_unbounded": every c^H_i == 0 component counts as unbounded.
    """
    c_H = np.asarray(c_H, dtype=np.float64)
    f_S = np.asarray(f_S, dtype=np.float64)
    tol_b = tol_abs + tol_rel * np.abs(c_H)
    slack = c_H - f_S
    bounded = slack <= tol_b
    unbounded = slack > tol_b

    zero = np.abs(c_H) <= tol_abs
    if np.any(zero):
        if zero_cH_rule == "always_unbounded":
            bounded = np.where(zero, False, bounded)
            unbounded = np.where(zero, True, unbounded)
        elif zero_cH_rule == "bounded_if_zero_flux":
            excreting = zero & (f_S < -tol_abs)
            zero_flux = zero & ~excreting
            bounded = np.where(zero_flux, True, bounded)
            unbounded = np.where(zero_flux, False, unbounded)
            bounded = np.where(excreting, False, bounded)
            unbounded = np.where(excreting, True, unbounded)
        else:
            raise ValueError(f"unknown zero_cH_rule: {zero_cH_rule!r}")
    return bounded, unbounded


def analyze_cell(R_S, c_H, f_S, w_S, lam2, cfg) -> CellAnalysis:
    """Classify metabolites and modes at one solution and compute b, u, r_plus,
    alpha, s and the excretion count (SI S6), with the tolerances of base.yaml.

    lam2 (the LP reduced costs) is only available before evolution; pass None
    after evolution, and n_degenerate is then 0.
    """
    tol = cfg["tolerances"]
    tol_abs = float(tol["tol_abs"])
    tol_rel = float(tol["tol_rel"])
    tol_w = float(tol["tol_w"])
    tol_exc = float(tol["tol_exc"])
    rcond = float(tol["rcond"])
    rule = cfg["flags"].get("zero_cH_rule", "bounded_if_zero_flux")

    R_S = np.asarray(R_S, dtype=np.float64)
    m = R_S.shape[0]
    f_S = np.asarray(f_S, dtype=np.float64)
    w_S = np.asarray(w_S, dtype=np.float64)

    bounded, unbounded = bounded_predicates(c_H, f_S, tol_abs, tol_rel, rule)
    if np.any(bounded & unbounded) or np.any(~bounded & ~unbounded):
        raise AssertionError("bounded/unbounded predicates are not a partition")
    b = int(bounded.sum())
    u = int(unbounded.sum())
    if b + u != m - 1:
        raise AssertionError(f"b+u={b + u} != m-1={m - 1}")

    active = w_S > tol_w
    r_plus = int(active.sum())
    if lam2 is None:
        n_degenerate = 0
    else:
        lam2 = np.asarray(lam2, dtype=np.float64)
        n_degenerate = int(np.sum((w_S <= tol_w) & (np.abs(lam2) <= tol_w)))

    n_excreted = int(np.sum(f_S < -tol_exc))
    alpha = float(np.sqrt(b))

    trivial = (b == 0) or (u == 0) or (r_plus == 0)
    if trivial:
        s = 0.0
        cond_b = np.nan
    else:
        R_bp = R_S[:m - 1][bounded][:, active]        # b x r_plus
        R_up = R_S[:m - 1][unbounded][:, active]      # u x r_plus
        S = R_up @ np.linalg.pinv(R_bp, rcond=rcond)  # u x b
        s = float(np.linalg.norm(S, ord="fro"))
        try:
            sv = np.linalg.svd(R_bp, compute_uv=False)
            cond_b = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
        except np.linalg.LinAlgError:
            cond_b = np.nan

    return CellAnalysis(
        n_bounded=b, n_unbounded=u, ub=b * u, n_excreted=n_excreted,
        r_plus=r_plus, n_degenerate=n_degenerate, alpha=alpha, s=s,
        cond_b=cond_b, trivial_ub=bool((b == 0) or (u == 0)),
        bounded_mask=bounded, unbounded_mask=unbounded, active_mask=active,
    )


def diagnostic_counts(c_H, f_S, lam1, tol_abs, tol_exc):
    """Diagnostics that explain the post-evolution `b`, per 01_model_spec.md A19.

    n_offered        #{i : c^H_i > tol_abs}.  After host evolution the ||c^H||_1
                     tie-break drives unused offers to exactly zero, so `b`
                     (correctly, an active-set count) includes components the
                     host does not offer at all.  Diagnostic only: never a
                     replacement for `b`.
    n_weakly_active  #{i : c^H_i == 0 and lambda^1_i == 0}.  The constraint is
                     active but the dual price is zero, so the component does
                     not respond to perturbation: it inflates `b` without
                     contributing responsiveness.
    """
    c_H = np.asarray(c_H, dtype=np.float64)
    f_S = np.asarray(f_S, dtype=np.float64)
    zero = np.abs(c_H) <= tol_abs
    n_offered = int((~zero).sum())
    if lam1 is None:
        n_weakly_active = -1
    else:
        lam = np.asarray(lam1, dtype=np.float64)[: c_H.size]
        n_weakly_active = int(np.sum(zero & (np.abs(lam) <= tol_abs)))
    return n_offered, n_weakly_active


def sensitivity_matrix(R_S, bounded_mask, unbounded_mask, active_mask, rcond=1e-10):
    """S = R^S_{u+} (R^S_{b+})^+ — exposed for the A2-D5 Jacobian comparison."""
    R_S = np.asarray(R_S, dtype=np.float64)
    m = R_S.shape[0]
    R_bp = R_S[:m - 1][bounded_mask][:, active_mask]
    R_up = R_S[:m - 1][unbounded_mask][:, active_mask]
    if R_bp.size == 0 or R_up.size == 0:
        return np.zeros((int(unbounded_mask.sum()), int(bounded_mask.sum())))
    return R_up @ np.linalg.pinv(R_bp, rcond=rcond)


def check_identities(R_S, c_H, mu_S, f_S, w_S, mu_H=None, f_H=None, tol=1e-7):
    """01_model_spec.md section 10 identities. Returns a dict of violations."""
    R_S = np.asarray(R_S, dtype=np.float64)
    m = R_S.shape[0]
    out = {}
    out["colsum_R_S"] = float(np.max(np.abs(R_S.sum(axis=0))))
    out["mu_S_eq_sum_f_S"] = float(abs(mu_S - np.sum(f_S)))
    out["mu_S_le_l1_cH"] = float(max(0.0, mu_S - np.sum(np.abs(c_H))))
    out["f_S_balance"] = float(np.max(np.abs(f_S + R_S[:m - 1] @ w_S)))
    if mu_H is not None and f_H is not None and np.isfinite(mu_H):
        out["muH_muS_eq_sum_fH"] = float(abs(mu_H + mu_S - np.sum(f_H)))
        out["muH_muS_le_one"] = float(max(0.0, mu_H + mu_S - 1.0))
    out["violations"] = {k: v for k, v in out.items()
                         if isinstance(v, float) and v > tol}
    return out
