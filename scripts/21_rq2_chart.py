"""
Task 4 Step 4.4b — RQ2 chart: A-F class distribution by tier.

Uses pubfig 0.3.0 for publication-grade rendering.

Reads: data/classification.csv
Outputs:
  paper/figures/rq2_classification.png
  paper/figures/rq2_classification.pdf

Chart design:
- Grouped bar chart: x-axis = tier (top/mid/tail), groups = class (A/B/C/D/E)
- F excluded from the analytic A-E pool per kickoff
- Y-axis: % of tier's A-E pool
- Caveat in caption: C = possible external CI publish path; E = no visible
  package publish workflow in inspected sources (NOT asserted manual publish)

Run: conda run --no-capture-output -n stage2c python scripts/21_rq2_chart.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pubfig

DATA_DIR = Path(__file__).parent.parent / "data"
PAPER_DIR = Path(__file__).parent.parent / "paper" / "figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    cls: dict[tuple[str, str], str] = {}
    with (DATA_DIR / "classification.csv").open() as f:
        for r in csv.DictReader(f):
            cls[(r["package"], r["version"])] = r["class"]
    tier: dict[tuple[str, str], str] = {}
    with (DATA_DIR / "non_provenance_sample.csv").open() as f:
        for r in csv.DictReader(f):
            tier[(r["package"], r["version"])] = r["tier"]

    tier_class: dict[str, Counter] = {"top": Counter(), "mid": Counter(), "tail": Counter()}
    for k, c in cls.items():
        t = tier.get(k)
        if t in tier_class:
            tier_class[t][c] += 1

    classes = list("ABCDE")
    tiers = ["top", "mid", "tail"]
    # pubfig.bar expects data shape (n_categories, n_series) for grouped — i.e., one row per
    # category on x-axis (tier) and one column per series in legend (class).
    data = np.zeros((len(tiers), len(classes)))
    for i, t in enumerate(tiers):
        total_ae = sum(tier_class[t][c] for c in classes)
        if total_ae == 0:
            continue
        for j, c in enumerate(classes):
            data[i, j] = tier_class[t][c] / total_ae * 100

    # Short labels for legend (full descriptions in caption)
    class_labels = ["A", "B", "C", "D", "E"]

    fig = pubfig.bar(
        data,
        category_names=tiers,
        series_names=class_labels,
        x_label="Tier (by dependent_packages_count)",
        y_label="% of A-E pool",
        title="Release-path class distribution (RQ2), 500 non-prov sample",
        show_y_grid=True,
        value_nticks=8,
        grouped_total_span=0.7,
        width=720,
        height=420,
        legend_ncol=5,
    )
    png_paths = pubfig.save_figure(fig, str(PAPER_DIR / "rq2_classification.png"))
    pdf_paths = pubfig.save_figure(fig, str(PAPER_DIR / "rq2_classification.pdf"))
    plt.close(fig)
    for p in png_paths + pdf_paths:
        print(f"  ✓ {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
