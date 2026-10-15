"""Entry point.  Terminal output is one progress bar plus one line per block;
everything else goes to runs/<run_id>/log.txt (05_conventions.md section 6).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

# One BLAS/OpenMP thread per process: parallelism comes from --workers.  These
# are read when numpy is first imported, so they must be set before the
# imports below (hence "noqa: E402", import not at top of file).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .cfgio import ROOT, load_config, parse_r_grid, write_manifest  # noqa: E402
from .pipeline import append_progress, execute_block  # noqa: E402

MAX_WORKERS_DEFAULT = 4


def _setup_log(run_dir: Path) -> logging.Logger:
    """Send the "grhd" logger to runs/<run_id>/log.txt (not to the terminal)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("grhd")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    return log


def cmd_run(args):
    """`grhd run`: solve every block of one design.

    1. read base.yaml (cfg) and the design file (design); apply the
       command-line overrides
    2. name the run: output folder <stage>_<tag>, log folder runs/<run_id>/
    3. cut the work into blocks of --chunk symbionts at one r, keep this
       shard's share, and turn each block into a task tuple
    4. run pipeline.execute_block on every task, in --workers processes
    5. write runs/<run_id>/summary.md
    """
    # 1. settings: cfg = model, tolerances, solvers; design = what to compute
    cfg = load_config(*args.config)
    design = load_config(args.design)
    stage = design.get("stage", "pre")
    scheme = design.get("scheme", "paper")
    n_rxn = args.n_rxn if args.n_rxn is not None else design.get("n_rxn")
    n_rxn = None if n_rxn is None else int(n_rxn)
    r_grid = parse_r_grid(design["r_grid"])
    if args.r_grid:
        r_grid = parse_r_grid(args.r_grid.split(","))
    if scheme == "fixed_repertoire":
        if n_rxn is None:
            sys.exit("scheme fixed_repertoire requires n_rxn")
        # the repertoire is the ceiling: r > n_rxn would leave empty modes
        r_grid = [x for x in r_grid if x <= n_rxn]
    N_S = int(args.n_s or design["N_S"])
    chunk = int(args.chunk)
    save_vectors = (args.save_vectors or design.get("save_vectors", "all")) != "none"
    host_id = int(args.host_id if args.host_id is not None else design.get("host_id", 0))
    m = int(args.m or cfg["model"]["m"])
    k = int(args.k or design.get("k") or cfg["model"]["k"])
    # r^H must follow m in the R5-A design (08_r5_two_designs.md section 0), so it is
    # written straight back into cfg: get_host, the RNG key derive(..., r_H=...) and
    # the r_H cell column all read it from there, and stay consistent.
    r_H_override = args.r_h if args.r_h is not None else design.get("r_H")
    if r_H_override is not None:
        cfg["model"]["r_H"] = int(r_H_override)
    r_H = int(cfg["model"]["r_H"])

    # 2. names: tag -> data/raw/<stage>_<tag>/, run_id -> runs/<run_id>/
    suffix = (f"_n{n_rxn}" if scheme == "fixed_repertoire"
              else (f"_k{k}" if scheme == "k_sweep" else ""))
    # host 0 keeps the run-1 tag so that A4 reuses the existing blocks verbatim
    # (the symbiont RNG key does not contain host_id, 05_conventions.md section 3).
    hsuf = f"_h{host_id}" if host_id else ""
    if scheme == "shared_world":   # host_id is the world id; nothing to reuse
        hsuf = f"_w{host_id}"
    # an m / r^H sweep must not collide with the m=10, r^H=24 runs of run 1
    msuf = f"_m{m}_rH{r_H}" if (args.m is not None or r_H_override) else ""
    tag = args.tag or f"{scheme}{suffix}{msuf}{hsuf}"
    run_id = args.run_id or f"{stage}_{tag}_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = ROOT / "runs" / run_id
    out_dir = Path(args.out_dir or (ROOT / "data" / "raw"))
    log = _setup_log(run_dir)

    design_rec = {"stage": stage, "scheme": scheme, "r_grid": r_grid, "N_S": N_S,
                  "N_c": design.get("N_c"), "m": m, "k": k, "n_rxn": n_rxn,
                  "host_id": host_id, "r_H": r_H,
                  "chunk": chunk, "workers": args.workers,
                  "shard": [args.shard, args.n_shards]}
    write_manifest(run_dir, run_id, cfg, design_rec)

    # 3. blocks (r, first symbiont, end) -> this shard's tasks
    blocks = []
    for r in r_grid:
        for lo in range(0, N_S, chunk):
            blocks.append((r, lo, min(lo + chunk, N_S)))
    if args.n_shards > 1:
        blocks = [b for i, b in enumerate(blocks) if i % args.n_shards == args.shard]
    # a task carries everything one worker process needs for one block;
    # pipeline.execute_block unpacks it in this order
    tasks = [(cfg, design, run_id, stage, tag, r, lo, hi, host_id, str(out_dir),
              save_vectors, m, k, scheme, n_rxn) for (r, lo, hi) in blocks]

    log.info("run_id=%s stage=%s blocks=%d workers=%d", run_id, stage, len(tasks),
             args.workers)
    print(f"[{run_id}] {len(tasks)} blocks, {args.workers} workers -> {out_dir}",
          flush=True)

    # 4. run the tasks; tqdm only draws the progress bar.  With joblib, each
    #    result comes back as soon as any worker finishes it.
    from tqdm import tqdm
    results = []
    if args.workers <= 1:
        it = (execute_block(t) for t in tasks)
        for rec in tqdm(it, total=len(tasks), ncols=80, mininterval=5.0):
            results.append(rec)
            append_progress(run_dir, rec)
            log.info("block %s", rec)
    else:
        from joblib import Parallel, delayed
        with Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered") as par:
            gen = par(delayed(execute_block)(t) for t in tasks)
            for rec in tqdm(gen, total=len(tasks), ncols=80, mininterval=5.0):
                results.append(rec)
                append_progress(run_dir, rec)
                log.info("block %s", rec)

    # 5. totals over all blocks -> summary.md
    n_cells = sum(r0["n"] for r0 in results)
    secs = sum(r0["seconds"] for r0 in results)
    status_counts: dict = {}
    viol: dict = {}
    for r0 in results:
        for kk, vv in (r0.get("status_counts") or {}).items():
            status_counts[kk] = status_counts.get(kk, 0) + vv
        for kk, vv in (r0.get("violations") or {}).items():
            viol[kk] = viol.get(kk, 0) + vv
    summary = [
        f"# {run_id}", "",
        f"- stage: `{stage}`  scheme: `{scheme}`  m={m} r_H={r_H} "
        f"k={'n/a' if scheme == 'fixed_repertoire' else k} n_rxn={n_rxn}",
        f"- r grid: {r_grid[0]}..{r_grid[-1]} ({len(r_grid)} values), N_S={N_S}",
        f"- blocks: {len(tasks)} ({sum(1 for x in results if x['skipped'])} skipped)",
        f"- cells written: {n_cells}",
        f"- cumulative solve time: {secs / 3600:.2f} core-hours, workers={args.workers}",
        f"- solver_status: {json.dumps(status_counts)}",
        f"- identity violations (V12): {json.dumps(viol) if viol else 'none'}",
        f"- output: `{out_dir}/{stage}_{tag}/`",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary[2:]), flush=True)


def cmd_collect(args):
    """Concatenate block parquets into one table per stage."""
    src = Path(args.src)
    files = sorted(src.glob("cells_*.parquet"))
    if not files:
        sys.exit(f"no cell blocks under {src}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} cells -> {out}")
    if args.vectors:
        vfiles = sorted(src.glob("vectors_*.parquet"))
        if vfiles:
            vd = pd.concat([pd.read_parquet(f) for f in vfiles], ignore_index=True)
            vout = Path(args.vectors)
            vd.to_parquet(vout, index=False)
            print(f"{len(vd)} vector rows -> {vout}")


def main(argv=None):
    """Parse the command line and call cmd_run or cmd_collect."""
    p = argparse.ArgumentParser(prog="grhd")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser(
        "run", help="run one design",
        description="Solve every (r, symbiont) of a design, in blocks of --chunk "
                    "symbionts, writing data/raw/<stage>_<tag>/cells_rRRR_sSSSSS.parquet "
                    "(+ vectors_...). The stage (pre: Levels 1-2, post: Level 3 MILP) "
                    "and the scheme (paper, k_sweep, fixed_repertoire, shared_world) "
                    "are set in the design file. Existing blocks are skipped.")
    # what to compute
    r.add_argument("--config", nargs="+", default=[str(ROOT / "config" / "base.yaml")],
                   help="model constants, tolerances and solver settings; several "
                        "files are merged, later ones win (default: config/base.yaml)")
    r.add_argument("--design", required=True,
                   help="config/design_*.yaml: stage, scheme, r grid, N_S, N_c, host_id")
    r.add_argument("--r-grid", default=None,
                   help="comma-separated r values, replaces the design's r grid")
    r.add_argument("--n-s", type=int, default=None,
                   help="number of symbionts per r, replaces the design's N_S")
    r.add_argument("--m", type=int, default=None,
                   help="number of metabolites, replaces model.m (adds _m<m>_rH<rH> to the tag)")
    r.add_argument("--k", type=int, default=None,
                   help="reactions per regulatory mode, replaces model.k (scheme k_sweep)")
    r.add_argument("--n-rxn", type=int, default=None,
                   help="size of the fixed reaction set (scheme fixed_repertoire)")
    r.add_argument("--host-id", type=int, default=None,
                   help="which host to draw (adds _h<id> to the tag); the world id "
                        "for scheme shared_world (_w<id>)")
    r.add_argument("--r-h", type=int, default=None,
                   help="host genome size, replaces model.r_H (adds _m<m>_rH<rH> to the tag)")
    # where the output goes
    r.add_argument("--tag", default=None,
                   help="output folder name after <stage>_; default built from "
                        "scheme, k / n_rxn, m, r_H and host id, e.g. paper_h7")
    r.add_argument("--run-id", default=None,
                   help="name of the log folder runs/<run_id>/ (manifest, progress, "
                        "summary); default <stage>_<tag>_<timestamp>")
    r.add_argument("--out-dir", default=None,
                   help="root of the block files (default: data/raw)")
    r.add_argument("--save-vectors", default=None, choices=["all", "subset", "none"],
                   help="write the solution vectors too; 'none' skips them, "
                        "'subset' currently behaves as 'all' (default: from the design)")
    # how the work is split
    r.add_argument("--chunk", type=int, default=50,
                   help="symbionts per block; keep it unchanged when extending a run, "
                        "or blocks will not line up (post_paper used 10)")
    r.add_argument("--workers", type=int, default=MAX_WORKERS_DEFAULT,
                   help="blocks solved in parallel, one process and one core each")
    r.add_argument("--shard", type=int, default=0,
                   help="this process runs blocks i with i %% n_shards == shard "
                        "(to split one run over machines)")
    r.add_argument("--n-shards", type=int, default=1,
                   help="number of shards the run is split into")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("collect", help="concatenate a run's block files",
                       description="Concatenate data/raw/<stage>_<tag>/cells_*.parquet "
                                   "into one file (scripts/consolidate_parquet.py "
                                   "does the same with checks).")
    c.add_argument("--src", required=True, help="run folder, e.g. data/raw/post_paper")
    c.add_argument("--out", required=True, help="output parquet for the cells")
    c.add_argument("--vectors", default=None, help="output parquet for the vectors (optional)")
    c.set_defaults(func=cmd_collect)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
