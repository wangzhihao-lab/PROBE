"""
Shared publication figure style for all program-generated paper figures.

Single source of truth for the paper's figure look. Import and call apply()
at the top of any figure script, after `import matplotlib`:

    import figstyle
    figstyle.apply()

Design contract (user requirements, 2026-07-17):
  1. NO in-figure titles  -> titles live in the LaTeX caption.
  2. Times New Roman type, one step smaller than the paper body text, so the
     figures typeset as if set in the manuscript font.
  3. Legend sits INSIDE the axes (never in a right-hand margin) and uses a
     light semi-transparent box so it never fully hides the data beneath it.
  4. Compact figure sizes (about the display width at \\linewidth) so nothing
     is scaled down to illegibility and no figure dominates the page.

Reference lines (deadline, chance diagonal, perfect calibration) use REF
(neutral grey) so they never consume a data-series color.

STYLING ONLY: this module changes appearance, never plotted values.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

NPG = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"]
TIER = {"top": "#E64B35", "mid": "#4DBBD5", "tail": "#00A087"}
REF = "#4A4A4A"
ACCENT = NPG[0]
ACCENT_2 = NPG[3]


def apply() -> None:
    """Install the shared rcParams (idempotent).

    Venue geometry: default sizing targets a single-column journal page
    (acmsmall, ~5 in display width). Setting FIGSTYLE_VENUE=ieee switches to
    IEEE double-column geometry (3.5 in column, 10pt body -> 8pt figure text)
    so figures render at 100% scale with no font shrinkage. Styling only;
    plotted values are never affected.
    """
    ieee = os.environ.get("FIGSTYLE_VENUE", "").lower() == "ieee"
    figsize = (3.5, 2.2) if ieee else (5.0, 3.1)
    base, tick = (8.0, 7.0) if ieee else (8.5, 7.5)
    mpl.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": figsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.prop_cycle": mpl.cycler(color=NPG),
        # Times New Roman, one step under the ~10pt body text.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": tick,
        # legend: inside, light, semi-transparent so it never occludes data
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.35,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "patch.linewidth": 0.5,
    })


def legend_inside(ax, **kwargs):
    """Place the legend INSIDE the axes, auto-positioned to avoid data, with a
    light semi-transparent box so any overlap does not hide the plot.
    """
    params = dict(loc="best", frameon=True, framealpha=0.8,
                  edgecolor="0.8", fancybox=False)
    params.update(kwargs)
    return ax.legend(**params)


# Back-compat alias: former call sites used legend_outside(); route them inside.
def legend_outside(ax, **kwargs):
    return legend_inside(ax, **kwargs)
