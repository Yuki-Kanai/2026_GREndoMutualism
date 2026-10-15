"""Config loading and run manifests (05_conventions.md sections 2 and 5)."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import yaml

# the simulation/ directory (src/grhd/cfgio.py -> two levels up); runs/ and
# data/raw/ are created under it
ROOT = Path(__file__).resolve().parents[2]


def load_config(*paths) -> dict:
    """Load and shallow-merge YAML files (later files win, dicts merge one level)."""
    cfg: dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        for key, val in d.items():
            if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


def parse_r_grid(spec):
    """r values from a design: "1..100" (a range, both ends included) or a list."""
    if isinstance(spec, str) and ".." in spec:
        lo, hi = spec.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec]


def _pkg_versions() -> dict:
    """Versions of the Python packages and of the HiGHS / CPLEX solvers, for the manifest."""
    out = {}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "matplotlib",
                 "statsmodels", "highspy", "cplex", "yaml"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:                       # noqa: BLE001
            out[name] = f"unavailable ({exc.__class__.__name__})"
    try:
        import highspy
        out["HiGHS_solver"] = highspy.Highs().version()
    except Exception:
        pass
    try:
        import cplex
        c = cplex.Cplex()
        out["CPLEX_solver"] = c.get_version()
        c.end()
    except Exception:
        pass
    return out


def _git_info() -> dict:
    """Current git commit of the code, and whether it has uncommitted changes."""
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        if h.returncode == 0:
            return {"commit": h.stdout.strip(), "dirty": bool(d.stdout.strip())}
    except Exception:                                  # noqa: BLE001
        pass
    return {"commit": None, "dirty": None}


def write_manifest(run_dir: Path, run_id: str, cfg: dict, design: dict, extra: dict | None = None):
    """Record how a run was made, in runs/<run_id>/.

    manifest.json: start time, machine, Python and package versions, git commit,
    seed, flags, tolerances, solver settings, the design actually used (after
    command-line overrides) and the thread variables.  config.yaml: the merged
    configuration.  Called once per `grhd.cli run`.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    man = {
        "run_id": run_id,
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _pkg_versions(),
        "git": _git_info(),
        "master_seed": cfg.get("master_seed"),
        "flags": cfg.get("flags"),
        "tolerances": cfg.get("tolerances"),
        "solver": cfg.get("solver"),
        "design": design,
        "env": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS")},
    }
    if extra:
        man.update(extra)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, default=str)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return man
