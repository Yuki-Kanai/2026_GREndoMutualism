"""Phase A housekeeping: A3-c sanity checks, solver-path census, data inventory."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pre", default="data/raw/pre_paper")
ap.add_argument("--post", default="data/raw/post_paper")
ap.add_argument("--join", default="data/derived/plotdata_post_join.parquet")
ap.add_argument("--tables", default="tables")
ap.add_argument("--runs", default="runs")
a = ap.parse_args()

TAB = Path(a.tables); TAB.mkdir(parents=True, exist_ok=True)


def load(dirpath, cols=None):
    files = sorted(Path(dirpath).glob("cells_*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                     ignore_index=True), len(files)


# --------------------------------------------------------------------------- #
# solver-path census (05_conventions.md section 4.1)
# --------------------------------------------------------------------------- #
rows = []
for stage, path in (("pre", a.pre), ("post", a.post)):
    cols = ["r", "solver_status"]
    df, nfiles = load(path, None)
    rec = {"stage": stage, "n_cells": int(len(df)), "n_blocks": nfiles}
    for col in ("qp_path", "lp_path", "solver_status"):
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            for k, v in vc.items():
                rec[f"{col}::{k}"] = int(v)
    rows.append(rec)
census = pd.DataFrame(rows).fillna(0)
census.to_csv(TAB / "A3_solver_path_census.csv", index=False)
pd.set_option("display.width", 220)
print("=== solver-path census ===")
print(census.T.to_string())

pre, _ = load(a.pre, ["r", "mu_S", "mu_H", "mu_H_0", "delta_mu_H",
                      "flag_infeasible", "qp_path", "lp_path", "solver_status"])
post, _ = load(a.post, None)
mu_H_0 = float(post["mu_H_0"].iloc[0])

if "qp_path" in pre.columns:
    n_lpvertex = int((pre["qp_path"] == "lp_vertex").sum())
    n_simplex = int((pre["lp_path"] == "simplex_fallback").sum())
    print(f"\npre: qp_path == 'lp_vertex' : {n_lpvertex}   "
          f"lp_path == 'simplex_fallback' : {n_simplex}")

# --------------------------------------------------------------------------- #
# A3-c sanity
# --------------------------------------------------------------------------- #
j = pd.read_parquet(a.join)
checks = {
    "post_mu_H_lt_mu_H_0": int((post["mu_H"] < mu_H_0 - 1e-9).sum()),
    "post_mu_S_negative": int((post["mu_S"] < -1e-9).sum()),
    "pre_mu_S_negative": int((pre["mu_S"] < -1e-9).sum()),
    "post_mu_H_lt_pre_best": int((j["mu_H_post"] < j["mu_H_pre_max"] - 1e-7).sum()),
    "post_mu_H_plus_mu_S_gt_1": int(((post["mu_H"] + post["mu_S"]) > 1 + 1e-7).sum()),
    "pre_mu_S_gt_cH_l1": int((pre["mu_S"] > 0.01 + 1e-8 + 1e-6 * 0.01).sum()),
    "post_solver_unbounded": int((post["solver_status"] == "unbounded").sum()),
    "pre_solver_unbounded": int((pre["solver_status"] == "unbounded").sum()),
    "post_n_cells": int(len(post)), "pre_n_cells": int(len(pre)),
}
worst = float((mu_H_0 - post["mu_H"]).max())
checks["post_worst_deficit_vs_mu_H_0"] = max(0.0, worst)
checks["post_worst_deficit_vs_pre_best"] = float(
    np.nanmax(j["mu_H_pre_max"] - j["mu_H_post"]))
print("\n=== A3-c sanity ===")
for k, v in checks.items():
    print(f"  {k:34s} {v}")
json.dump(checks, open(TAB / "A3c_sanity.json", "w"), indent=2)

# --------------------------------------------------------------------------- #
# data inventory (needed by A2-D5 in Phase B)
# --------------------------------------------------------------------------- #
root = Path(".").resolve()


def du(p):
    try:
        out = subprocess.run(["du", "-sh", str(p)], capture_output=True,
                             text=True, timeout=600)
        return out.stdout.split()[0] if out.stdout else "?"
    except Exception:                                   # noqa: BLE001
        return "?"


lines = [
    "# data/raw の所在（消さないこと）", "",
    f"記録日時: {pd.Timestamp.now().isoformat(timespec='seconds')}",
    f"記録ホスト: {socket.gethostname()}", "",
    "`$HOME` は `/mnt2`（Public_NAS2, 25 T / 8 T free）上の共有 NFS で、",
    "**cell2 / cell3 / cell6 から同じパスで見える**。ホストごとのコピーは存在しない。", "",
    f"作業ルート: `{root}`", "",
    "| パス | 内容 | サイズ |",
    "|---|---|---|",
]
for rel, desc in (
        ("data/raw/pre_paper", "進化前 3.0×10⁶ cell。cells_*.parquet 600 個 + vectors_*.parquet 600 個"),
        ("data/raw/post_paper", "進化後 4,500 MILP。cells_*.parquet 450 個 + vectors_*.parquet 450 個"),
        ("data/derived", "間引き済みプロットデータ、規約再集計"),
        ("tables", "全 CSV"),
        ("figures", "PNG + PDF + meta.json"),
        ("runs", "manifest / log / progress / summary")):
    p = root / rel
    n = len(list(p.glob("*"))) if p.exists() else 0
    lines.append(f"| `{rel}` | {desc} | {du(p)}（{n} files） |")
lines += [
    "", "## ベクトル列（A2-D5 に必須）", "",
    "`data/raw/*/vectors_*.parquet` に `c_H`, `f_S_star`, `w_S_star`, `bounded_mask`",
    "（進化後はさらに `c_H_step1`, `box_report` ほか）が list 列で入っている。",
    "**これを消すと A2-D5（数値 Jacobian）と R3 が再計算できない。**", "",
    "## バックアップ", "",
    "同一 NFS 上の別ディレクトリへのコピーは冗長性にならないので取っていない。",
    "必要なら `data/raw` を別ファイルシステムへ rsync すること（約 1.7 GB）。",
]
Path(a.runs).mkdir(parents=True, exist_ok=True)
Path(a.runs, "DATA_LOCATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nwrote runs/DATA_LOCATION.md")
