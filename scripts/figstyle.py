from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

NPG = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"]
TIER = {"top": "#E64B35", "mid": "#4DBBD5", "tail": "#00A087"}
REF = "#4A4A4A"
ACCENT = NPG[0]
ACCENT_2 = NPG[3]


def apply() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (5.0, 3.1),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.prop_cycle": mpl.cycler(color=NPG),

        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,

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
    params = dict(loc="best", frameon=True, framealpha=0.8,
                  edgecolor="0.8", fancybox=False)
    params.update(kwargs)
    return ax.legend(**params)


def legend_outside(ax, **kwargs):
    return legend_inside(ax, **kwargs)
