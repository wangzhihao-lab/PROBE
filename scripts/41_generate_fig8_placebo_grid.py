from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import figstyle

figstyle.apply()

PLACEBO_GREY = "#B0B4BA"

ROOT = Path(__file__).parent.parent
CSV = ROOT / "data" / "placebo_cutoff_rank.csv"
OUT_PDF = ROOT / "paper" / "figures" / "fig8_placebo_grid.pdf"


def main() -> int:
    df = pd.read_csv(CSV)
    top = df[df["tier"] == "top"].copy()
    top["cutoff_dt"] = pd.to_datetime(top["cutoff_date"])
    top = top.sort_values("cutoff_dt").reset_index(drop=True)
    top["abs_beta"] = top["beta2_level_change_pp"].abs()
    top["is_actual"] = top["is_actual_dec9_partition"].astype(bool)

    actual = top[top["is_actual"]].iloc[0]
    n = len(top)
    n_strict = int((top.loc[~top["is_actual"], "abs_beta"] > actual["abs_beta"]).sum())
    rank = int((top["abs_beta"] >= actual["abs_beta"]).sum())

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = [figstyle.ACCENT if a else PLACEBO_GREY for a in top["is_actual"]]

    for x, yv, c, a in zip(top["cutoff_dt"], top["beta2_level_change_pp"],
                           colors, top["is_actual"]):
        ax.plot([x, x], [0, yv], color=c, lw=2.2 if a else 1.4,
                zorder=5 if a else 2, solid_capstyle="round")
        ax.plot(x, yv, "o", color=c, markersize=7 if a else 4.5,
                zorder=6 if a else 3, markeredgecolor="black" if a else "none",
                markeredgewidth=0.5)

    ymin = float(top["beta2_level_change_pp"].min())
    ymax = float(top["beta2_level_change_pp"].max())
    ax.set_ylim(ymin - 4, ymax + 8)

    ax.annotate(
        f"actual deadline 2025-12-09\n"
        f"$\\hat\\beta_2={actual['beta2_level_change_pp']:.2f}$ pp; "
        f"rank {rank}/{n}",
        xy=(actual["cutoff_dt"], actual["beta2_level_change_pp"]),
        xytext=(0.30, 0.24), textcoords="axes fraction",
        ha="left", va="top", fontsize=8,
        arrowprops=dict(arrowstyle="->", color=figstyle.ACCENT, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=figstyle.ACCENT, lw=0.6),
    )

    largest = top.loc[top[~top["is_actual"]]["abs_beta"].idxmax()]
    ax.annotate(
        f"largest placebo $|\\hat\\beta_2|$\n{largest['cutoff_date']}: "
        f"$\\hat\\beta_2={largest['beta2_level_change_pp']:.2f}$ pp",
        xy=(largest["cutoff_dt"], largest["beta2_level_change_pp"]),
        xytext=(0.03, 0.30), textcoords="axes fraction",
        ha="left", va="top", fontsize=8,
        arrowprops=dict(arrowstyle="->", color=figstyle.REF, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=figstyle.REF, lw=0.6),
    )

    ax.axhline(0, color=figstyle.REF, lw=0.8)
    ax.set_ylabel(r"ITS level coefficient $\hat\beta_2$ (pp)")
    ax.set_xlabel("Candidate cutoff (all 28 admissible weekly Tuesdays; "
                  "actual = 2025-12-09)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax.text(
        0.02, 0.97,
        f"{n_strict} of {n - 1} placebo candidates have strictly larger "
        f"$|\\hat\\beta_2|$;\nDec-9 ranks {rank} of {n} "
        "(empirical specificity rank).",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="0.96",
                  edgecolor=figstyle.REF, lw=0.5),
    )

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")


    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=400, bbox_inches="tight")
    print(f"wrote {OUT_PDF} (+.png) ({n} candidates; Dec-9 rank {rank}/{n}, "
          f"{n_strict} strictly larger)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
