"""
TSE R1 revision, T5.2 + T5.3 — Regenerate the two RQ1 figures for the revised
manuscript: the 28-month trajectory (fig:rq1-timeline) and the 110-candidate
placebo grid (fig:placebo-grid). Written ONLY to paper_tse/figures/; the
as-submitted figures in paper/figures/ and the TOSEM tree stay untouched.

Style: scripts/figstyle.py with FIGSTYLE_VENUE=ieee (3.45in column geometry).

Inputs:  data/tse_r1_long_trajectory_monthly.csv, data/tse_r1_long_placebo_scan.csv
Outputs: paper_tse/figures/rq1_timeline_depsdev.pdf (+png),
         paper_tse/figures/fig8_placebo_grid.pdf (+png)

Run: FIGSTYLE_VENUE=ieee conda run --no-capture-output -n stage2c python scripts/69_gen_revision_figures.py
"""
from __future__ import annotations

import os
os.environ.setdefault("FIGSTYLE_VENUE", "ieee")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import figstyle
figstyle.apply()

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT = ROOT / "paper_tse" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

EVENTS = {  # dotted = other credential-policy events; dashed = anchor
    "2025-09-29": "Sep-29",
    "2025-11-05": "Nov-5",
    "2026-02-03": "Feb-3",
}
DEADLINE = "2025-12-09"
PROTOCOL = ("2025-08-01", "2026-05-10")


def timeline() -> None:
    m = pd.read_csv(DATA / "tse_r1_long_trajectory_monthly.csv", parse_dates=["period_start"])
    m = m[m["tier"].isin(["top", "mid", "tail"])]
    fig, ax = plt.subplots(figsize=(2.55, 1.60))
    ax.axvspan(pd.Timestamp(PROTOCOL[0]), pd.Timestamp(PROTOCOL[1]),
               color="0.92", zorder=0)
    for tier, mk in (("top", "o"), ("mid", "s"), ("tail", "^")):
        sub = m[m["tier"] == tier].sort_values("period_start")
        ax.plot(sub["period_start"], 100 * sub["rate"], marker=mk, markersize=2.6,
                linewidth=1.1, color=figstyle.TIER[tier], label=tier)
    ax.axvline(pd.Timestamp(DEADLINE), linestyle="--", color=figstyle.REF,
               alpha=0.75, linewidth=0.9)
    ax.text(pd.Timestamp(DEADLINE), 63, "Dec-9", color=figstyle.REF,
            fontsize=7, ha="center")
    for d, lbl in EVENTS.items():
        ax.axvline(pd.Timestamp(d), linestyle=":", color=figstyle.REF,
                   alpha=0.45, linewidth=0.7)
    ax.set_ylabel("Attestation rate (%)")
    ax.set_ylim(0, 68)
    # Ticks on January, May and September so the first label is the series start (2024-01).
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(loc="upper left", framealpha=0.85, edgecolor="0.8")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"rq1_timeline_depsdev.{ext}")
    plt.close(fig)
    print("✓ rq1_timeline_depsdev (28-month)", flush=True)


def placebo_grid() -> None:
    sc = pd.read_csv(DATA / "tse_r1_long_placebo_scan.csv", parse_dates=["cutoff"])
    top = sc[(sc["panel"] == "extended") & (sc["tier"] == "top")].sort_values("cutoff")
    assert len(top) == 110, f"expected 110 extended candidates, got {len(top)}"
    fig, ax = plt.subplots(figsize=(2.55, 1.70))
    grey = "0.62"
    for _, r in top.iterrows():
        a = bool(r["is_actual"])
        c = figstyle.ACCENT if a else grey
        ax.plot([r["cutoff"], r["cutoff"]], [0, r["beta2"]], color=c,
                lw=1.6 if a else 0.7, alpha=1.0 if a else 0.75,
                zorder=5 if a else 2)
        if a:
            ax.plot(r["cutoff"], r["beta2"], "o", color=c, markersize=4.5,
                    zorder=6, markeredgecolor="black", markeredgewidth=0.5)
    act = top[top["is_actual"]].iloc[0]
    ax.annotate(f"Dec-9  ({act['beta2']:+.1f} pp, rank 7/110)",
                xy=(act["cutoff"], act["beta2"]),
                xytext=(pd.Timestamp("2024-04-01"), 6.5),
                fontsize=7, color=figstyle.ACCENT,
                arrowprops=dict(arrowstyle="-", color=figstyle.ACCENT, lw=0.6,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_ylim(-26, 14)
    ax.axhline(0, color="0.5", linewidth=0.6)
    ax.set_ylabel(r"Level coefficient $\hat\beta_2$ (pp)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig8_placebo_grid.{ext}")
    plt.close(fig)
    print("✓ fig8_placebo_grid (110 candidates)", flush=True)


if __name__ == "__main__":
    timeline()
    placebo_grid()
    print("✅ revision figures complete", flush=True)
