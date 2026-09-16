"""Shared matplotlib style for the static figures embedded in the paper.

The interactive Plotly figures elsewhere in this package are dashboards: a
title, a subtitle, and a glossary of terms live on the figure itself, because
there is no caption to carry that text. A paper figure has a caption, so the
figure should carry only the data — no title, no subtitle, no footnote. This
module is the one place that decides what "minimal" looks like, so the four
paper figures read as one family instead of four one-off scripts:

  * a small, print-appropriate font and figure size,
  * no top/right spines, no figure-level title,
  * the same categorical/qualitative colours already validated for the
    Plotly figures (``CAT_COLORS`` etc. from ``km_line1_by_year``), so a
    treatment category or repetition kind keeps its colour across figures,
  * one ``savefig`` that writes a vector PDF to ``latex/figs/`` — the
    directory ``second_draft.tex`` already includes figures from.

Each of the four paper-figure scripts imports from here rather than
restyling matplotlib itself. It does *not* redefine palettes — each script
already owns a validated set of hex colours (``CAT_COLORS``, ``KIND_COLOR``,
...), either defined locally or imported from ``km_line1_by_year``; this
module only supplies the neutral ink/grid tokens those palettes sit on top
of, plus the layout mechanics (rcParams, panel labels, PDF export).
"""

from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_FIGS_DIR = REPO_ROOT / "latex" / "figs"

# Same neutral/ink tokens as the Plotly figures (dataviz reference palette,
# light mode), so a reader flipping between an exploratory HTML and the
# paper PDF sees the same language.
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"


def use_paper_style() -> None:
    """Set matplotlib rcParams for a minimalist, print-ready figure.

    Idempotent — safe to call at import time in every figure script.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": SEC,
            "ytick.color": SEC,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "text.color": INK,
            "axes.edgecolor": BASE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,  # embed as real text, not curves
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, letter: str) -> None:
    """Small (a)/(b)/(c) label in the top-left corner of a panel.

    Stands in for the Plotly figures' bold panel titles ("Patients",
    "Categories", ...) without adding words the caption should carry.
    """
    ax.text(
        -0.02,
        1.06,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def savefig(fig, name: str, **kwargs) -> Path:
    """Write ``fig`` as ``latex/figs/<name>.pdf`` and return the path."""
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIGS_DIR / f"{name}.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight", **kwargs)
    return out
