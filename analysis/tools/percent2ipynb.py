#!/usr/bin/env python3
"""Turn a jupytext "percent format" R script into a Jupyter notebook.

    python tools/percent2ipynb.py notebooks/01_fig3_before_evolution.R

The .R file is the source of truth (it diffs cleanly and can be sourced with
Rscript); the .ipynb is generated from it.  Cell markers:

    # %% [markdown]
    # text, one '# ' per line

    # %%
    code

The format is the standard jupytext one, so `jupytext --to notebook` produces
the same thing if you would rather use that.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KERNEL = {
    "display_name": "R (GRHDanalysis)",
    "language": "R",
    "name": "ir_grhd",
}
MARKER = re.compile(r"^#\s*%%\s*(\[markdown\])?\s*$")


def split_cells(text: str):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            if kind is not None:
                cells.append((kind, buf))
            kind, buf = ("markdown" if m.group(1) else "code"), []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    return cells


def to_source(kind: str, lines: list[str]) -> list[str]:
    if kind == "markdown":
        lines = [ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln)
                 for ln in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:] if lines else []


def convert(src: Path, dst: Path | None = None) -> Path:
    dst = dst or src.with_suffix(".ipynb")
    cells = []
    for kind, lines in split_cells(src.read_text(encoding="utf-8")):
        source = to_source(kind, lines)
        if not source:
            continue
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": KERNEL,
                     "language_info": {"name": "R", "file_extension": ".r",
                                       "mimetype": "text/x-r-source"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(str(p) for p in Path("notebooks").glob("*.R"))
    for a in args:
        print("wrote", convert(Path(a)))
