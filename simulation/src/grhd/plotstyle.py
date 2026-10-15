"""Shared figure style: a compact, journal-like look (Helvetica metrics, thin
spines, outward ticks, no chartjunk).

Figures are written as 300 dpi PNG and vector PDF, with a sidecar .meta.json
naming the tables they were drawn from (05_conventions.md section 8).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Nimbus Sans / Liberation Sans are metric clones of Helvetica / Arial.
_PREFERRED = ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans",
              "FreeSans", "DejaVu Sans"]
_AVAILABLE = {f.name for f in font_manager.fontManager.ttflist}
SANS = next((f for f in _PREFERRED if f in _AVAILABLE), "DejaVu Sans")

# millimetres -> inches
MM = 1.0 / 25.4
COL1 = 89 * MM          # single column
COL15 = 120 * MM
COL2 = 183 * MM         # double column

C = {
    "scatter": "#C9CDD2",
    "scatter_edge": "none",
    "mean": "#1F4E79",
    "mean2": "#C0392B",
    "accent": "#E08214",
    "green": "#2E7D5B",
    "purple": "#6A51A3",
    "ref": "#4D4D4D",
    "band": "#9EB9D4",
    "grid": "#E6E6E6",
}
SERIES = ["#1F4E79", "#C0392B", "#2E7D5B", "#E08214", "#6A51A3", "#7F7F7F"]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [SANS, "DejaVu Sans"],
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 2.0,
    "axes.titlepad": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.3,
    "ytick.minor.size": 1.3,
    "lines.linewidth": 1.1,
    "lines.markersize": 2.2,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.25,
    "legend.borderpad": 0.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
}


def use():
    plt.rcParams.update(RC)


def panel_label(ax, letter, dx=-0.20, dy=1.06, size=9):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left")


def tidy(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    return ax


def save(fig, fig_id: str, figures_dir: Path, meta: dict | None = None):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{fig_id}.png"
    pdf = figures_dir / f"{fig_id}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    if meta is not None:
        with open(figures_dir / f"{fig_id}.meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
    plt.close(fig)
    return png
