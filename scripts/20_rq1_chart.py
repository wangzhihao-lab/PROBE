"""
Task 4 Step 4.4a — RQ1 chart: monthly attestation rate timeline.

Uses pubfig 0.3.0 for publication-grade rendering.

Reads: data/rq1_monthly_attestation_rate.csv
Outputs:
  paper/figures/rq1_timeline.png
  paper/figures/rq1_timeline.pdf

Chart design:
- 4 line series: tier='all' (bold), top, mid, tail
- X-axis: month (2025-08 to 2026-05)
- Y-axis: attestation rate (%)
- Vertical dashed line at 2025-12 marking npm Dec-9 deadline
- Title: "Monthly provenance attestation rate, 2025-08 – 2026-05"
- Caveat in caption: rates from 3000-package sample (Task 1); tier from
  ecosyste.ms current-state metadata pending BQ strata-shift check

Run: conda run --no-capture-output -n stage2c python scripts/20_rq1_chart.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pubfig

DATA_DIR = Path(__file__).parent.parent / "data"
PAPER_DIR = Path(__file__).parent.parent / "paper" / "figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
MONTHLY_CSV = DATA_DIR / "rq1_monthly_attestation_rate.csv"


def main() -> int:
    series: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with MONTHLY_CSV.open() as f:
        for r in csv.DictReader(f):
            series[r["tier"]].append((r["month"], float(r["rate"]) * 100))

    months = sorted({m for items in series.values() for m, _ in items})
    series_order = ["all", "top", "mid", "tail"]
    series_names = ["Overall", "Top tier (DPC>500)", "Mid tier (10<DPC≤500)", "Tail tier (DPC≤10)"]

    # pubfig.line expects data shape (n_points, n_series) for 2D input
    data = np.zeros((len(months), len(series_order)))
    for i, t in enumerate(series_order):
        m_to_r = dict(series[t])
        for j, m in enumerate(months):
            data[j, i] = m_to_r.get(m, np.nan)

    x = np.arange(len(months))
    fig = pubfig.line(
        data,
        x=x,
        series_names=series_names,
        x_label="Month",
        y_label="Attestation rate (%)",
        title="Provenance attestation rate per month, 2025-08 — 2026-05",
        show_y_grid=True,
        line_width=1.6,
        marker_size=4,
        width=720,
        height=420,
        legend_ncol=2,
    )
    ax = fig.axes[0]
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    # Mark Dec-9 deadline (2025-12 = index 4 in series); short annotation outside main band
    if "2025-12" in months:
        deadline_x = months.index("2025-12")
        ax.axvline(deadline_x, linestyle="--", color="#a02525", alpha=0.5, linewidth=1.0)
        ax.text(deadline_x + 0.1, 5, "Dec-9", color="#a02525", fontsize=7, va="bottom")
    ax.set_ylim(0, max(60, data.max() + 5))

    png_paths = pubfig.save_figure(fig, str(PAPER_DIR / "rq1_timeline.png"))
    pdf_paths = pubfig.save_figure(fig, str(PAPER_DIR / "rq1_timeline.pdf"))
    plt.close(fig)
    for p in png_paths + pdf_paths:
        print(f"  ✓ {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
