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


    data = np.zeros((len(tiers), len(classes)))
    for i, t in enumerate(tiers):
        total_ae = sum(tier_class[t][c] for c in classes)
        if total_ae == 0:
            continue
        for j, c in enumerate(classes):
            data[i, j] = tier_class[t][c] / total_ae * 100


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
        print(f"   {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
