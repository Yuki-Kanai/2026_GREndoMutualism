"""Block-wise, resumable execution of the pre- and post-evolution stages.

A block is (stage, scheme, r, symbiont_id range).  Each block writes its own
parquet file, so a killed run resumes by skipping blocks whose output already
exists (05_conventions.md section 6).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import analyze_cell, check_identities, diagnostic_counts
from .generate import (c_E, draw_replicate, generate_R, generate_R_fixed,
                       reaction_pool, sample_c_H, sample_host)
from .lp import HostSolver, SymbiontSolver, solve_host_once
from .rng import derive

# Columns of the cells_*.parquet files: one row per host-symbiont pair (post)
# or per host-symbiont-environment triple (pre).  run_block_pre/post build each
# row as a tuple in exactly this order.
CELL_COLUMNS = [
    "run_id", "stage", "m", "k", "r", "r_H", "scheme", "n_rxn", "host_id",
    "symbiont_id", "c_id", "mu_H_0", "mu_S", "mu_H", "delta_mu_H",
    "n_bounded", "n_unbounded", "ub", "n_excreted", "r_plus", "n_degenerate",
    "alpha", "s", "cond_b", "trivial_ub", "flag_delta_small", "flag_infeasible",
    "flag_detlimit", "flag_cmax_active", "solver_status", "mip_gap", "c_H_l1",
    "solve_time_lp_s", "solve_time_qp_s", "solve_time_milp_s",
    "lp_path", "qp_path", "n_offered", "n_weakly_active", "mip_best_bound",
    "n_rxn_effective", "mode_size_mean", "mode_size_min", "mode_size_max",
    "milp_path",
]


# --------------------------------------------------------------------------- #
def world_pool(cfg: dict, world_id: int, m: int | None = None) -> np.ndarray:
    """The one tournament shared by the host and all symbionts of a world."""
    mm = cfg["model"]["m"] if m is None else m
    return reaction_pool(mm, derive(cfg["master_seed"], "world", m=mm,
                                    world_id=world_id))


def get_host(cfg: dict, host_id: int = 0, m: int | None = None,
             scheme: str = "paper"):
    """Deterministically re-derive a host from the master seed (cheap: ~10 LPs).

    scheme="shared_world": host_id is the world id, and every rejection-sampling
    attempt draws its modes from that world's tournament (tournament_scope="shared").
    """
    mm = cfg["model"]["m"] if m is None else m
    k, rH = cfg["model"]["k"], cfg["model"]["r_H"]
    cE = c_E(mm, cfg["model"]["c_E_first"])
    if scheme == "shared_world":
        rng = derive(cfg["master_seed"], "R_H_world", m=mm, k=k, r_H=rH,
                     world_id=host_id)
        pool = world_pool(cfg, host_id, mm)
    else:
        rng = derive(cfg["master_seed"], "R_H", m=mm, k=k, r_H=rH, host_id=host_id)
        pool = None
    hd = sample_host(
        mm, rH, k, rng, lambda R, f, c: solve_host_once(R, f, c, cfg), cE,
        cfg["model"]["host_accept_lo"], cfg["model"]["host_accept_hi"],
        cfg["model"]["max_attempts"], shared_pool=pool,
    )
    return hd, cE


def make_R_S_meta(cfg: dict, r: int, symbiont_id: int, scheme: str = "paper",
                  m: int | None = None, k: int | None = None,
                  n_rxn: int | None = None, world_id: int | None = None):
    """Build one symbiont's R^S and the bookkeeping A1 needs.

    Three schemes (03_additional_analyses.md A1):

      paper             k reactions per mode, r modes, k fixed
      k_sweep           identical generator, k varied.  The RNG key deliberately
                        says scheme="paper", so k_sweep at k = 4 reproduces the
                        paper arm bit-for-bit and does not have to be re-run
      fixed_repertoire  one replicate = one fixed reaction set Omega, one weight
                        per reaction, one order; r only changes how that ordered
                        list is cut into blocks.  The RNG key therefore contains
                        NEITHER r NOR k -- that invariance is the control.

    Returns (R_S, meta) where meta carries the columns n_rxn, n_rxn_effective
    (unique reactions actually present in R^S) and the mode-size summary.
    """
    mm = cfg["model"]["m"] if m is None else m
    if scheme == "fixed_repertoire":
        if n_rxn is None:
            raise ValueError("fixed_repertoire needs n_rxn")
        # namespace R_S_fixed, fields (m, n_rxn, rep_id) -- 05_conventions.md section 3.
        # r is deliberately absent: that is what makes the design matched.
        rng = derive(cfg["master_seed"], "R_S_fixed", m=mm, n_rxn=int(n_rxn),
                     rep_id=symbiont_id)
        R, modes = generate_R_fixed(mm, r, draw_replicate(mm, int(n_rxn), rng))
        kk = None
    elif scheme == "shared_world":
        # same generator as "paper", but the tournament is the world's, not a fresh one
        if world_id is None:
            raise ValueError("shared_world needs world_id")
        kk = cfg["model"]["k"] if k is None else int(k)
        rng = derive(cfg["master_seed"], "R_S_world", m=mm, k=kk, r=r,
                     world_id=int(world_id), symbiont_id=symbiont_id)
        R, modes = generate_R(mm, r, kk, world_pool(cfg, int(world_id), mm), rng,
                              return_idx=True)
    else:
        kk = cfg["model"]["k"] if k is None else int(k)
        rng = derive(cfg["master_seed"], "R_S", m=mm, k=kk, r=r, scheme="paper",
                     symbiont_id=symbiont_id)
        V = reaction_pool(mm, rng)
        R, modes = generate_R(mm, r, kk, V, rng, return_idx=True)
    sizes = np.array([len(x) for x in modes], dtype=float)
    used = np.unique(np.concatenate(modes)) if modes else np.array([], dtype=int)
    meta = {"k": kk, "n_rxn": (None if n_rxn is None else int(n_rxn)),
            "n_rxn_effective": int(used.size),
            "mode_size_mean": float(sizes.mean()),
            "mode_size_min": int(sizes.min()), "mode_size_max": int(sizes.max())}
    return R, meta


def make_R_S(cfg: dict, r: int, symbiont_id: int, scheme: str = "paper",
             m: int | None = None, k: int | None = None,
             n_rxn: int | None = None, world_id: int | None = None):
    """make_R_S_meta without the bookkeeping: just the m x r matrix R^S."""
    return make_R_S_meta(cfg, r, symbiont_id, scheme, m, k, n_rxn, world_id)[0]


def c_H_samples(cfg: dict, symbiont_id: int, n_c: int, m: int | None = None):
    """The n_c pre-evolution internal environments c^H of one symbiont (SI S2).

    Returns an (n_c, m-1) array, each row summing to model.c_H_l1 (0.01).  The
    seed depends on symbiont_id but not on r, so symbionts with the same id at
    different r see the same environments.
    """
    mm = cfg["model"]["m"] if m is None else m
    rng = derive(cfg["master_seed"], "c_H", m=mm, symbiont_id=symbiont_id, c_id=0)
    return sample_c_H(mm, n_c, rng, cfg["model"]["c_H_l1"])


# --------------------------------------------------------------------------- #
def run_block_pre(cfg, design, run_id, r, sid_lo, sid_hi, host_id=0,
                  save_vectors=True, m=None, k=None, scheme="paper", n_rxn=None):
    """Level 1 + Level 2 for symbionts [sid_lo, sid_hi) at a given r.

    Before host evolution: for each symbiont, for each of its N_c random c^H,
    solve the symbiont problem (max mu^S, then min ||w^S||^2) and then the host
    problem given the symbiont's uptake f^S*.  One output row per
    (symbiont, c^H), so a block of 10 symbionts with N_c = 100 gives 1000 rows.

    Returns (cells, vectors, violations): the rows as a DataFrame with
    CELL_COLUMNS, the solution vectors (or None), and a count of the identity
    checks of analyze.check_identities that failed, by name.
    """
    mm = cfg["model"]["m"] if m is None else m
    kk = cfg["model"]["k"] if k is None else k
    N_c = int(design["N_c"])
    hd, cE = get_host(cfg, host_id, mm, scheme)
    hs = HostSolver(hd.R_H, cfg)
    itol = float(cfg["tolerances"]["identity_tol"])

    rows, vecs = [], []
    viol = {}
    for sid in range(sid_lo, sid_hi):
        R_S, meta = make_R_S_meta(cfg, r, sid, scheme, mm, kk, n_rxn,
                                     world_id=host_id)
        ss = SymbiontSolver(R_S, cfg)
        CH = c_H_samples(cfg, sid, N_c, mm)
        for cid in range(N_c):
            c_H = CH[cid]
            sol = ss.solve(c_H)
            if sol.status not in ("optimal", "qp_fallback"):
                hsol = None
            else:
                hsol = hs.solve(sol.f_S, cE)
            an = analyze_cell(R_S, c_H, sol.f_S, sol.w_S, sol.lam2, cfg)
            mu_H = hsol.mu_H if hsol is not None else np.nan
            infeas = hsol is None or hsol.status == "infeasible"
            delta = mu_H - hd.mu_H_0
            ident = check_identities(R_S, c_H, sol.mu_S, sol.f_S, sol.w_S,
                                     mu_H if np.isfinite(mu_H) else None,
                                     hsol.f_H if hsol is not None else None, itol)
            for key in ident["violations"]:
                viol[key] = viol.get(key, 0) + 1
            rows.append((                                   # order = CELL_COLUMNS
                run_id, "pre", mm, meta["k"], r, cfg["model"]["r_H"], scheme,
                meta["n_rxn"], host_id,
                sid, cid, hd.mu_H_0, sol.mu_S, mu_H, delta,
                an.n_bounded, an.n_unbounded, an.ub, an.n_excreted, an.r_plus,
                an.n_degenerate, an.alpha, an.s, an.cond_b, an.trivial_ub,
                bool(np.isfinite(delta) and delta < 1e-2), bool(infeas),
                False, False,
                sol.status if not infeas else "infeasible", np.nan,
                float(np.sum(np.abs(c_H))),
                sol.t_lp, sol.t_qp, 0.0, sol.lp_path, sol.qp_path,
                *diagnostic_counts(c_H, sol.f_S, sol.lam1,
                                   cfg["tolerances"]["tol_abs"],
                                   cfg["tolerances"]["tol_exc"]),
                np.nan,
                meta["n_rxn_effective"], meta["mode_size_mean"],
                meta["mode_size_min"], meta["mode_size_max"], None,
            ))
            if save_vectors:
                vecs.append({
                    "r": r, "symbiont_id": sid, "c_id": cid,
                    "c_H": c_H.astype(np.float32).tolist(),
                    "f_S_star": sol.f_S.astype(np.float32).tolist(),
                    "w_S_star": sol.w_S.astype(np.float32).tolist(),
                    "bounded_mask": an.bounded_mask.tolist(),
                })
        ss.close()
    cells = pd.DataFrame(rows, columns=CELL_COLUMNS)
    vectors = pd.DataFrame(vecs) if save_vectors else None
    return cells, vectors, viol


# --------------------------------------------------------------------------- #
def run_block_post(cfg, design, run_id, r, sid_lo, sid_hi, host_id=0,
                   save_vectors=True, m=None, k=None, scheme="paper", n_rxn=None):
    """Level 3 (MILP) for symbionts [sid_lo, sid_hi) at a given r.

    After host evolution: for each symbiont, the host chooses c^H to maximise
    mu^H (milp.solve_host_evolution), and the structural quantities are
    computed at that c^H.  One output row per symbiont.  A pair the MILP could
    not solve keeps a row, with mu_H = NaN and flag_infeasible = True.

    Returns (cells, vectors, {}) as run_block_pre (no identity checks here).
    """
    from .milp import solve_host_evolution

    mm = cfg["model"]["m"] if m is None else m
    kk = cfg["model"]["k"] if k is None else k
    # the host, rebuilt from its seed: every block gets the identical R^H
    hd, cE = get_host(cfg, host_id, mm, scheme)

    rows, vecs = [], []
    for sid in range(sid_lo, sid_hi):
        # the symbiont, from its seed: the same R^S as in the pre-evolution run
        R_S, meta = make_R_S_meta(cfg, r, sid, scheme, mm, kk, n_rxn,
                                     world_id=host_id)
        # Level 3: the host's best c^H for this symbiont
        sol = solve_host_evolution(R_S, hd.R_H, cE, cfg)
        # b, u, s, alpha, ... at that c^H (SI S6); not defined without a solution
        if np.isfinite(sol.mu_H):
            an = analyze_cell(R_S, sol.c_H, sol.f_S, sol.w_S, None, cfg)
            b, u, ub, nex = an.n_bounded, an.n_unbounded, an.ub, an.n_excreted
            rp, ndeg, alpha, s, cond = (an.r_plus, an.n_degenerate, an.alpha,
                                        an.s, an.cond_b)
            triv = an.trivial_ub
            bmask = an.bounded_mask
        else:
            b = u = ub = nex = rp = ndeg = 0
            alpha = s = cond = np.nan
            triv = True
            bmask = np.zeros(mm - 1, dtype=bool)
        delta = sol.mu_H - hd.mu_H_0
        rows.append((                                       # order = CELL_COLUMNS
            run_id, "post", mm, meta["k"], r, cfg["model"]["r_H"], scheme,
            meta["n_rxn"], host_id,
            sid, None, hd.mu_H_0, sol.mu_S, sol.mu_H, delta,
            b, u, ub, nex, rp, ndeg, alpha, s, cond, triv,
            bool(np.isfinite(delta) and delta < 1e-2),
            bool(sol.status == "infeasible"),
            bool("detlimit" in sol.status), bool(sol.flag_cmax_active),
            sol.status, sol.mip_gap, float(np.sum(np.abs(sol.c_H))),
            0.0, 0.0, sol.t_milp, "milp", "milp",
            *diagnostic_counts(sol.c_H, sol.f_S, sol.lam1,
                               cfg["tolerances"]["tol_abs"],
                               cfg["tolerances"]["tol_exc"]),
            sol.mip_best_bound,
            meta["n_rxn_effective"], meta["mode_size_mean"],
            meta["mode_size_min"], meta["mode_size_max"],
            sol.detail.get("milp_path"),
        ))
        if save_vectors:
            vecs.append({
                "r": r, "symbiont_id": sid, "c_id": None,
                "c_H": np.asarray(sol.c_H, dtype=np.float32).tolist(),
                "c_H_step1": np.asarray(sol.c_H_step1, dtype=np.float32).tolist(),
                "f_S_star": np.asarray(sol.f_S, dtype=np.float32).tolist(),
                "w_S_star": np.asarray(sol.w_S, dtype=np.float32).tolist(),
                "bounded_mask": bmask.tolist(),
                "l1_step1": sol.l1_step1, "l1_step2": sol.l1_step2,
                "flag_box_active": sol.flag_box_active,
                "box_report": json.dumps(sol.box_report),
                "box_retries": sol.detail.get("box_retries", 0),
                "mu_H_step1": sol.detail.get("mu_H_step1"),
                "status_step1": sol.detail.get("status_step1"),
                "status_step2": sol.detail.get("status_step2"),
            })
    cells = pd.DataFrame(rows, columns=CELL_COLUMNS)
    vectors = pd.DataFrame(vecs) if save_vectors else None
    return cells, vectors, {}


# --------------------------------------------------------------------------- #
def block_paths(out_dir: Path, stage: str, tag: str, r: int, sid_lo: int):
    """The two output files of one block.

    A block is identified by its run folder (<stage>_<tag>), its r and its
    first symbiont id, e.g. stage "post", tag "paper", r 13, sid_lo 170 ->
        <out_dir>/post_paper/cells_r013_s00170.parquet
        <out_dir>/post_paper/vectors_r013_s00170.parquet
    execute_block uses these names both to write the results and, before
    solving anything, to check whether the block was already done.
    """
    base = out_dir / f"{stage}_{tag}"
    return (base / f"cells_r{r:03d}_s{sid_lo:05d}.parquet",
            base / f"vectors_r{r:03d}_s{sid_lo:05d}.parquet")


def execute_block(args):
    """Worker entry point: run one block unless its output is already there.

    `args` is one task tuple built by cli.cmd_run, in this order:
        (cfg, design, run_id, stage, tag, r, sid_lo, sid_hi, host_id, out_dir,
         save_vectors, m, k, scheme, n_rxn)
    Runs run_block_pre or run_block_post depending on `stage`, writes the
    block's parquet files (via a .tmp file and a rename, so an interrupted
    write never leaves a file that a rerun would mistake for a finished block)
    and returns a small summary dict that cli.cmd_run logs to progress.jsonl.
    """
    (cfg, design, run_id, stage, tag, r, sid_lo, sid_hi, host_id, out_dir,
     save_vectors, m, k, scheme, n_rxn) = args
    cell_p, vec_p = block_paths(Path(out_dir), stage, tag, r, sid_lo)
    if cell_p.exists() and (not save_vectors or vec_p.exists()):
        return {"r": r, "sid_lo": sid_lo, "skipped": True, "n": 0, "seconds": 0.0}
    cell_p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fn = run_block_pre if stage == "pre" else run_block_post
    cells, vectors, viol = fn(cfg, design, run_id, r, sid_lo, sid_hi, host_id,
                              save_vectors, m, k, scheme, n_rxn)
    tmp = cell_p.with_suffix(".parquet.tmp")
    cells.to_parquet(tmp, index=False)
    os.replace(tmp, cell_p)
    if vectors is not None:
        tmp = vec_p.with_suffix(".parquet.tmp")
        vectors.to_parquet(tmp, index=False)
        os.replace(tmp, vec_p)
    dt = time.perf_counter() - t0
    return {"r": r, "sid_lo": sid_lo, "skipped": False, "n": len(cells),
            "seconds": dt, "violations": viol,
            "status_counts": cells["solver_status"].value_counts().to_dict()}


def append_progress(run_dir: Path, rec: dict):
    """Append one block's summary (from execute_block) to runs/<run_id>/progress.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "progress.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
